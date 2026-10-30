"""Benchmark tasks. Ground truth is computed from the data, never hard-coded."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from data_agent_bench.bench.datasets import SLA_HOURS, USD_TO_EUR

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


def _web_orders(d: Data) -> pd.DataFrame:
    """Web orders with the documented rules applied: test orders out, dates parsed, EUR."""
    w = d["web_orders"].assign(email=d["web_orders"]["customer_email"].str.strip().str.lower())
    w = w[~w["email"].str.endswith("@example.com")]
    raw = w["order_date"]
    date = (
        pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
        .fillna(pd.to_datetime(raw, format="%d/%m/%Y", errors="coerce"))
        .fillna(pd.to_datetime(raw, format="%b %d, %Y", errors="coerce"))
    )
    eur = w["amount"] * np.where(w["currency"] == "USD", USD_TO_EUR, 1.0)
    return w.assign(date=date, amount_eur=eur)


def _completed_web_orders(d: Data) -> pd.DataFrame:
    w = _web_orders(d)
    return w[w["status"] == "completed"]


def _tickets(d: Data) -> pd.DataFrame:
    t = d["tickets"]
    created = pd.to_datetime(t["created_at"])
    hours = (pd.to_datetime(t["resolved_at"]) - created).dt.total_seconds() / 3600
    return t.assign(
        priority=t["priority"].str.strip().str.capitalize(), created=created, hours=hours
    )


def _resolved_tickets(d: Data) -> pd.DataFrame:
    return _tickets(d).dropna(subset=["hours"])


def _sla_breaches(d: Data) -> int:
    t = _resolved_tickets(d)
    return int((t["hours"] > t["priority"].map(SLA_HOURS)).sum())


def _faster_team(t: pd.DataFrame) -> str:
    return argmax_unique(-t.groupby("team")["hours"].mean())


def _enterprise_laptop_share(d: Data) -> float:
    s = _sales_with_revenue(d).merge(d["customers"], on="customer_id")
    ent = s[s["segment"] == "Enterprise"]
    return ent.loc[ent["product"] == "Laptop", "revenue"].sum() / ent["revenue"].sum() * 100


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
    # --- v0.2: more tasks on the original tables ------------------------------------------
    Task(
        "grp_005",
        "groupby",
        "medium",
        "Which manager in employees.csv has the most direct reports? Answer with their "
        "employee_id.",
        "number",
        lambda d: int(argmax_unique(d["employees"]["manager_id"].value_counts())),
    ),
    Task(
        "join_005",
        "join",
        "medium",
        f"What is the total revenue from customers who signed up in 2023? {REV} "
        "Round to 2 decimals.",
        "number",
        lambda d: (
            _sales_with_revenue(d)
            .merge(d["customers"], on="customer_id")
            .loc[lambda s: s["signup_date"].str.startswith("2023"), "revenue"]
            .sum()
        ),
        decimals=2,
    ),
    Task(
        "join_006",
        "join",
        "medium",
        f"Customers from which country generated the highest total revenue? Answer with the "
        f"country code. {REV}",
        "string",
        lambda d: argmax_unique(
            _sales_with_revenue(d)
            .merge(d["customers"], on="customer_id")
            .groupby("country")["revenue"]
            .sum()
        ),
    ),
    Task(
        "ts_005",
        "time_series",
        "medium",
        "On which day of the week were the most units sold in sales.csv? Answer with the "
        "English day name (e.g. Monday).",
        "string",
        lambda d: argmax_unique(
            d["sales"].groupby(pd.to_datetime(d["sales"]["date"]).dt.day_name())["units"].sum()
        ),
    ),
    Task(
        "ts_006",
        "time_series",
        "medium",
        "How many employees had been with the company for at least 5 years on 2024-01-01?",
        "number",
        lambda d: int((pd.to_datetime(d["employees"]["hire_date"]) <= "2019-01-01").sum()),
    ),
    Task(
        "st_005",
        "statistics",
        "hard",
        "What is the Pearson correlation between temperature and humidity in sensors.csv, "
        "using only valid readings? Round to 3 decimals.",
        "number",
        lambda d: _valid_sensors(d)[["temperature", "humidity"]].corr().iloc[0, 1],
        decimals=3,
    ),
    Task(
        "cln_006",
        "cleaning",
        "medium",
        "Which sensor produced the most error readings?",
        "string",
        lambda d: argmax_unique(
            (d["sensors"]["temperature"] == -999).groupby(d["sensors"]["sensor_id"]).sum()
        ),
    ),
    Task(
        "cln_007",
        "cleaning",
        "medium",
        "After removing exact duplicate rows from sales.csv, what is the average number of "
        "units per row in the West region? Round to 3 decimals.",
        "number",
        lambda d: d["sales"].drop_duplicates().query("region == 'West'")["units"].mean(),
        decimals=3,
    ),
    Task(
        "ms_005",
        "multi_step",
        "hard",
        f"What percentage of the revenue from Enterprise customers came from Laptop sales? {REV} "
        "Answer as a percentage rounded to 2 decimals.",
        "number",
        _enterprise_laptop_share,
        decimals=2,
    ),
    # --- web_orders.csv: mixed formats, currencies and business rules ----------------------
    Task(
        "wo_001",
        "cleaning",
        "hard",
        "How many web orders were placed in March 2024?",
        "number",
        lambda d: int((_web_orders(d)["date"].dt.month == 3).sum()),
    ),
    Task(
        "wo_002",
        "business_rules",
        "hard",
        "What is the total revenue from web orders, in EUR? Round to 2 decimals.",
        "number",
        lambda d: _completed_web_orders(d)["amount_eur"].sum(),
        decimals=2,
    ),
    Task(
        "wo_003",
        "cleaning",
        "medium",
        "How many distinct customers placed web orders?",
        "number",
        lambda d: int(_web_orders(d)["email"].nunique()),
    ),
    Task(
        "wo_004",
        "filtering",
        "easy",
        "How many web orders have the status refunded?",
        "number",
        lambda d: int((_web_orders(d)["status"] == "refunded").sum()),
    ),
    Task(
        "wo_005",
        "time_series",
        "hard",
        "In which month of 2024 were the most completed web orders placed? Answer with the "
        "month number (1-12).",
        "number",
        lambda d: int(argmax_unique(_completed_web_orders(d)["date"].dt.month.value_counts())),
    ),
    Task(
        "wo_006",
        "business_rules",
        "medium",
        "What percentage of web orders were paid in USD? Answer as a percentage rounded to "
        "2 decimals.",
        "number",
        lambda d: (_web_orders(d)["currency"] == "USD").mean() * 100,
        decimals=2,
    ),
    Task(
        "wo_007",
        "multi_step",
        "hard",
        "Which customer spent the most on web orders, in EUR? Answer with their email address "
        "in lowercase.",
        "string",
        lambda d: argmax_unique(_completed_web_orders(d).groupby("email")["amount_eur"].sum()),
    ),
    # --- tickets.csv: inconsistent labels, open tickets, SLA rules ------------------------
    Task(
        "tk_001",
        "cleaning",
        "medium",
        "How many support tickets have High priority?",
        "number",
        lambda d: int((_tickets(d)["priority"] == "High").sum()),
    ),
    Task(
        "tk_002",
        "filtering",
        "easy",
        "How many support tickets are still open?",
        "number",
        lambda d: int(d["tickets"]["resolved_at"].isna().sum()),
    ),
    Task(
        "tk_003",
        "statistics",
        "medium",
        "What is the median resolution time of resolved support tickets, in hours? "
        "Round to 2 decimals.",
        "number",
        lambda d: _resolved_tickets(d)["hours"].median(),
        decimals=2,
    ),
    Task(
        "tk_004",
        "business_rules",
        "hard",
        "How many support tickets breached their SLA?",
        "number",
        _sla_breaches,
    ),
    Task(
        "tk_005",
        "groupby",
        "medium",
        "Which support team has the lower mean resolution time on resolved tickets? "
        "Answer Tier1 or Tier2.",
        "string",
        lambda d: _faster_team(_resolved_tickets(d)),
    ),
    Task(
        "tk_006",
        "time_series",
        "medium",
        "On which day of the week were the most support tickets created? Answer with the "
        "English day name (e.g. Monday).",
        "string",
        lambda d: argmax_unique(_tickets(d)["created"].dt.day_name().value_counts()),
    ),
    Task(
        "tk_007",
        "multi_step",
        "hard",
        "Considering only resolved tickets with Medium priority, which support team has the lower "
        "mean resolution time? Answer Tier1 or Tier2.",
        "string",
        lambda d: _faster_team(_resolved_tickets(d).query("priority == 'Medium'")),
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
    Task(
        "un_004",
        "unanswerable",
        "hard",
        "What was the total revenue in sales.csv in 2023?",
        "unanswerable",
        lambda d: None,
    ),
    Task(
        "un_005",
        "unanswerable",
        "medium",
        "What is the average temperature recorded by sensor S6?",
        "unanswerable",
        lambda d: None,
    ),
    Task(
        "un_006",
        "unanswerable",
        "hard",
        "What was the average temperature across all sensors in April 2024?",
        "unanswerable",
        lambda d: None,
    ),
    Task(
        "un_007",
        "unanswerable",
        "medium",
        "Which marketing channel brought in the most web orders?",
        "unanswerable",
        lambda d: None,
    ),
    Task(
        "un_008",
        "unanswerable",
        "medium",
        "What is the average delivery time of web orders, in days?",
        "unanswerable",
        lambda d: None,
    ),
    Task(
        "un_009",
        "unanswerable",
        "medium",
        "How many employees left the company in 2023?",
        "unanswerable",
        lambda d: None,
    ),
    Task(
        "un_010",
        "unanswerable",
        "medium",
        "What is the average satisfaction score of resolved support tickets?",
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
