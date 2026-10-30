"""Benchmark tasks. Ground truth is computed from the data, never hard-coded."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

Data = dict[str, pd.DataFrame]

ANSWER_TYPES = ("number", "string", "bool", "list_ordered", "unanswerable")


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    difficulty: str  # easy | medium | hard
    question: str
    answer_type: str
    compute: Callable[[Data], Any]
    decimals: int | None = None  # for numbers: None means an exact integer is expected


def argmax_unique(series: pd.Series):
    """Index of the max value; raises on ties so no task has an ambiguous answer."""
    ordered = series.sort_values(ascending=False)
    if len(ordered) > 1 and np.isclose(ordered.iloc[0], ordered.iloc[1]):
        raise ValueError(f"tie between {ordered.index[0]!r} and {ordered.index[1]!r}")
    return ordered.index[0]


def revenue(sales: pd.DataFrame) -> pd.Series:
    return sales["units"] * sales["unit_price"] * (1 - sales["discount"].fillna(0))


def _sales_with_revenue(d: Data) -> pd.DataFrame:
    return d["sales"].assign(revenue=revenue(d["sales"]))


def _valid_sensors(d: Data) -> pd.DataFrame:
    return d["sensors"][d["sensors"]["temperature"] != -999]


def _month_units(d: Data) -> pd.Series:
    s = d["sales"]
    return s.groupby(pd.to_datetime(s["date"]).dt.month)["units"].sum()


def _employees_vs_manager(d: Data) -> int:
    e = d["employees"]
    reports = e.dropna(subset=["manager_id"]).astype({"manager_id": int})
    merged = reports.merge(
        e[["employee_id", "salary"]],
        left_on="manager_id",
        right_on="employee_id",
        suffixes=("", "_manager"),
    )
    return int((merged["salary"] > merged["salary_manager"]).sum())


def _missing_hours(d: Data, sensor: str) -> int:
    s = d["sensors"]
    ts = pd.to_datetime(s["timestamp"])
    expected = pd.date_range(ts.min(), ts.max(), freq="h")
    present = ts[s["sensor_id"] == sensor].nunique()
    return int(len(expected) - present)


def _region_winners(d: Data) -> int:
    units = d["sales"].groupby(["region", "product"])["units"].sum()
    winners = {argmax_unique(units.loc[region]) for region in units.index.levels[0]}
    return len(winners)


def _top_customers(d: Data, k: int) -> list[int]:
    totals = _sales_with_revenue(d).groupby("customer_id")["revenue"].sum()
    ordered = totals.sort_values(ascending=False)
    if np.isclose(ordered.iloc[k - 1], ordered.iloc[k]):
        raise ValueError("tie at the top-k boundary")
    return [int(c) for c in ordered.index[:k]]


REV = "Revenue per row is units * unit_price * (1 - discount), treating a missing discount as 0."

TASKS: list[Task] = [
    # --- aggregation -----------------------------------------------------------------------
    Task(
        "agg_001",
        "aggregation",
        "easy",
        "How many rows does sales.csv have?",
        "number",
        lambda d: len(d["sales"]),
    ),
    Task(
        "agg_002",
        "aggregation",
        "easy",
        "What is the total number of units sold across all rows of sales.csv?",
        "number",
        lambda d: int(d["sales"]["units"].sum()),
    ),
    Task(
        "agg_003",
        "aggregation",
        "easy",
        "What is the average unit_price in sales.csv? Round to 2 decimals.",
        "number",
        lambda d: d["sales"]["unit_price"].mean(),
        decimals=2,
    ),
    Task(
        "agg_004",
        "aggregation",
        "medium",
        f"What is the total revenue in sales.csv? {REV} Round to 2 decimals.",
        "number",
        lambda d: revenue(d["sales"]).sum(),
        decimals=2,
    ),
    # --- groupby ---------------------------------------------------------------------------
    Task(
        "grp_001",
        "groupby",
        "easy",
        "Which region has the highest total units sold?",
        "string",
        lambda d: argmax_unique(d["sales"].groupby("region")["units"].sum()),
    ),
    Task(
        "grp_002",
        "groupby",
        "medium",
        f"Which product generated the highest total revenue? {REV}",
        "string",
        lambda d: argmax_unique(_sales_with_revenue(d).groupby("product")["revenue"].sum()),
    ),
    Task(
        "grp_003",
        "groupby",
        "medium",
        f"What is the total revenue of the East region? {REV} Round to 2 decimals.",
        "number",
        lambda d: _sales_with_revenue(d).query("region == 'East'")["revenue"].sum(),
        decimals=2,
    ),
    Task(
        "grp_004",
        "groupby",
        "easy",
        "Which department in employees.csv has the highest average salary?",
        "string",
        lambda d: argmax_unique(d["employees"].groupby("department")["salary"].mean()),
    ),
    # --- filtering -------------------------------------------------------------------------
    Task(
        "flt_001",
        "filtering",
        "easy",
        "How many rows of sales.csv have a discount of exactly 0.2?",
        "number",
        lambda d: int((d["sales"]["discount"] == 0.2).sum()),
    ),
    Task(
        "flt_002",
        "filtering",
        "easy",
        "How many rows of sales.csv are Laptop sales with more than 8 units?",
        "number",
        lambda d: int(((d["sales"]["product"] == "Laptop") & (d["sales"]["units"] > 8)).sum()),
    ),
    Task(
        "flt_003",
        "filtering",
        "easy",
        "How many employees are remote and have the level Senior?",
        "number",
        lambda d: int((d["employees"]["remote"] & (d["employees"]["level"] == "Senior")).sum()),
    ),
    # --- time series -----------------------------------------------------------------------
    Task(
        "ts_001",
        "time_series",
        "medium",
        "In which month of 2024 were the most units sold? Answer with the month number (1-12).",
        "number",
        lambda d: int(argmax_unique(_month_units(d))),
    ),
    Task(
        "ts_002",
        "time_series",
        "medium",
        f"What was the total revenue in Q3 2024 (July to September)? {REV} Round to 2 decimals.",
        "number",
        lambda d: (
            _sales_with_revenue(d)
            .loc[lambda s: pd.to_datetime(s["date"]).dt.month.between(7, 9), "revenue"]
            .sum()
        ),
        decimals=2,
    ),
    Task(
        "ts_003",
        "time_series",
        "hard",
        "What is the percentage change in total units sold from March 2024 to April 2024? "
        "Answer as a percentage rounded to 2 decimals (e.g. -3.45 for a 3.45% decrease).",
        "number",
        lambda d: (_month_units(d)[4] - _month_units(d)[3]) / _month_units(d)[3] * 100,
        decimals=2,
    ),
    Task(
        "ts_004",
        "time_series",
        "easy",
        "How many employees were hired in 2021?",
        "number",
        lambda d: int((pd.to_datetime(d["employees"]["hire_date"]).dt.year == 2021).sum()),
    ),
    # --- joins -----------------------------------------------------------------------------
    Task(
        "join_001",
        "join",
        "medium",
        f"Using sales.csv and customers.csv, which customer segment generated the highest "
        f"total revenue? {REV}",
        "string",
        lambda d: argmax_unique(
            _sales_with_revenue(d)
            .merge(d["customers"], on="customer_id")
            .groupby("segment")["revenue"]
            .sum()
        ),
    ),
    Task(
        "join_002",
        "join",
        "medium",
        "How many distinct customers from Spain (country code ES) appear in sales.csv?",
        "number",
        lambda d: int(
            d["sales"]
            .merge(d["customers"], on="customer_id")
            .query("country == 'ES'")["customer_id"]
            .nunique()
        ),
    ),
    Task(
        "join_003",
        "join",
        "medium",
        f"What is the average revenue per row of sales.csv for Enterprise customers? {REV} "
        "Round to 2 decimals.",
        "number",
        lambda d: (
            _sales_with_revenue(d)
            .merge(d["customers"], on="customer_id")
            .query("segment == 'Enterprise'")["revenue"]
            .mean()
        ),
        decimals=2,
    ),
    Task(
        "join_004",
        "join",
        "hard",
        "How many employees earn a higher salary than their own manager?",
        "number",
        _employees_vs_manager,
    ),
    # --- data cleaning ---------------------------------------------------------------------
    Task(
        "cln_001",
        "cleaning",
        "medium",
        "sales.csv contains exact duplicate rows. How many rows remain after removing them?",
        "number",
        lambda d: len(d["sales"].drop_duplicates()),
    ),
    Task(
        "cln_002",
        "cleaning",
        "easy",
        "How many rows of sales.csv have a missing discount value?",
        "number",
        lambda d: int(d["sales"]["discount"].isna().sum()),
    ),
    Task(
        "cln_003",
        "cleaning",
        "hard",
        f"After removing exact duplicate rows from sales.csv, what is the total revenue? {REV} "
        "Round to 2 decimals.",
        "number",
        lambda d: revenue(d["sales"].drop_duplicates()).sum(),
        decimals=2,
    ),
    Task(
        "cln_004",
        "cleaning",
        "hard",
        "What is the mean temperature recorded by sensor S3, using only valid readings? "
        "Round to 2 decimals.",
        "number",
        lambda d: _valid_sensors(d).query("sensor_id == 'S3'")["temperature"].mean(),
        decimals=2,
    ),
    Task(
        "cln_005",
        "cleaning",
        "hard",
        "How many hourly readings are missing for sensor S2? Assume every sensor should report "
        "once per hour from the first to the last timestamp in sensors.csv.",
        "number",
        lambda d: _missing_hours(d, "S2"),
    ),
    # --- statistics ------------------------------------------------------------------------
    Task(
        "st_001",
        "statistics",
        "medium",
        "What is the Pearson correlation between salary and performance_score in employees.csv, "
        "ignoring rows with missing values? Round to 3 decimals.",
        "number",
        lambda d: d["employees"][["salary", "performance_score"]].dropna().corr().iloc[0, 1],
        decimals=3,
    ),
    Task(
        "st_002",
        "statistics",
        "easy",
        "What is the median salary of employees in the Engineering department?",
        "number",
        lambda d: d["employees"].query("department == 'Engineering'")["salary"].median(),
        decimals=1,
    ),
    Task(
        "st_003",
        "statistics",
        "medium",
        "What is the sample standard deviation (ddof=1) of salary across all employees? "
        "Round to 2 decimals.",
        "number",
        lambda d: d["employees"]["salary"].std(ddof=1),
        decimals=2,
    ),
    Task(
        "st_004",
        "statistics",
        "medium",
        "Is the average salary of remote employees higher than that of non-remote employees? "
        "Answer yes or no.",
        "bool",
        lambda d: bool(
            d["employees"].groupby("remote")["salary"].mean().loc[True]
            > d["employees"].groupby("remote")["salary"].mean().loc[False]
        ),
    ),
    # --- multi-step ------------------------------------------------------------------------
    Task(
        "ms_001",
        "multi_step",
        "hard",
        "For each region, find the product with the highest total units sold. How many distinct "
        "products appear among these per-region winners?",
        "number",
        _region_winners,
    ),
    Task(
        "ms_002",
        "multi_step",
        "hard",
        f"List the top 3 customer_id values by total revenue, from highest to lowest. {REV}",
        "list_ordered",
        lambda d: _top_customers(d, 3),
    ),
    Task(
        "ms_003",
        "multi_step",
        "hard",
        "Which sensor has the highest maximum temperature, using only valid readings?",
        "string",
        lambda d: argmax_unique(_valid_sensors(d).groupby("sensor_id")["temperature"].max()),
    ),
    Task(
        "ms_004",
        "multi_step",
        "hard",
        "On which date was the daily average temperature across all sensors the highest, using "
        "only valid readings? Answer in YYYY-MM-DD format.",
        "string",
        lambda d: argmax_unique(
            _valid_sensors(d).groupby(_valid_sensors(d)["timestamp"].str[:10])["temperature"].mean()
        ),
    ),
    # --- unanswerable (hallucination checks) -----------------------------------------------
    Task(
        "un_001",
        "unanswerable",
        "medium",
        "What is the average age of the customers in customers.csv?",
        "unanswerable",
        lambda d: None,
    ),
    Task(
        "un_002",
        "unanswerable",
        "medium",
        "Which sales representative closed the most orders in 2024?",
        "unanswerable",
        lambda d: None,
    ),
    Task(
        "un_003",
        "unanswerable",
        "hard",
        "What is the profit margin on Laptop sales?",
        "unanswerable",
        lambda d: None,
    ),
]

TASKS_BY_ID = {t.id: t for t in TASKS}


def select_tasks(spec: str) -> list[Task]:
    """Select tasks: 'all', 'category:<name>', or comma-separated task ids."""
    if spec == "all":
        return list(TASKS)
    if spec.startswith("category:"):
        category = spec.split(":", 1)[1]
        selected = [t for t in TASKS if t.category == category]
        if not selected:
            raise ValueError(f"No tasks in category '{category}'.")
        return selected
    ids = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [i for i in ids if i not in TASKS_BY_ID]
    if unknown:
        raise ValueError(f"Unknown task ids: {', '.join(unknown)}")
    return [TASKS_BY_ID[i] for i in ids]
