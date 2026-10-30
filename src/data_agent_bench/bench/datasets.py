"""Deterministic synthetic datasets for the benchmark.

Synthetic data keeps every answer verifiable (ground truth is computed by code),
avoids licensing issues and cannot have leaked into any model's training data.
The data includes realistic traps: exact duplicates, missing values, a sentinel
error code and gaps in a time series, documented in DATA_DICTIONARY.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SEED = 42

REGIONS = ["North", "South", "East", "West"]
PRODUCTS = {
    "Laptop": 1100.0,
    "Phone": 750.0,
    "Tablet": 450.0,
    "Monitor": 280.0,
    "Headphones": 120.0,
}
SEGMENTS = ["Consumer", "SMB", "Enterprise"]
COUNTRIES = ["ES", "FR", "DE", "IT", "PT"]
DEPARTMENTS = {
    "Engineering": 55000,
    "Finance": 52000,
    "Sales": 45000,
    "Marketing": 43000,
    "Operations": 40000,
    "HR": 38000,
}
LEVELS = {"Junior": 0.75, "Mid": 1.0, "Senior": 1.35, "Lead": 1.7}
SENSORS = ["S1", "S2", "S3", "S4", "S5"]

DATA_DICTIONARY = """\
# Data dictionary

## sales.csv
One row per order line, year 2024.
- order_id: order identifier.
- date: order date (YYYY-MM-DD).
- region: sales region.
- product: product sold.
- customer_id: references customers.csv.
- units: units sold.
- unit_price: price per unit in EUR, before discount.
- discount: fractional discount applied (0.1 = 10%). A missing value means no discount was applied.

Known issue: the export process duplicated some rows (exact duplicates).

## customers.csv
- customer_id, segment (Consumer / SMB / Enterprise), country (ISO code), signup_date.

## employees.csv
- employee_id, department, level, hire_date, salary (EUR, yearly gross).
- manager_id: employee_id of the employee's manager (empty for department leads).
- remote: whether the employee works fully remote.
- performance_score: last review score from 1 to 5 (empty if not yet reviewed).

## sensors.csv
Hourly readings from 5 temperature sensors.
- timestamp (YYYY-MM-DD HH:MM:SS), sensor_id, temperature (Celsius), humidity (%).
- A temperature of -999 is an error code emitted by faulty sensors, not a real reading.
- Sensors should report once per hour; missing rows are readings that were lost.
"""


def generate(seed: int = DEFAULT_SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    customers = _customers(rng)
    return {
        "customers": customers,
        "sales": _sales(rng, customers),
        "employees": _employees(rng),
        "sensors": _sensors(rng),
    }


def write(out_dir: str | Path, seed: int = DEFAULT_SEED) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in generate(seed).items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "DATA_DICTIONARY.md").write_text(DATA_DICTIONARY, encoding="utf-8")
    return out_dir


def load(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the CSVs exactly as the agent will see them (round-tripped through CSV)."""
    return {p.stem: pd.read_csv(p) for p in sorted(Path(data_dir).glob("*.csv"))}


def _customers(rng: np.random.Generator) -> pd.DataFrame:
    n = 300
    signup = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 4 * 365, n), unit="D")
    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "segment": rng.choice(SEGMENTS, n, p=[0.6, 0.3, 0.1]),
            "country": rng.choice(COUNTRIES, n, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
            "signup_date": signup.strftime("%Y-%m-%d"),
        }
    )


def _sales(rng: np.random.Generator, customers: pd.DataFrame) -> pd.DataFrame:
    n = 2000
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    weights = np.where(days.month >= 11, 1.6, 1.0)  # holiday season peak
    dates = days[rng.choice(len(days), n, p=weights / weights.sum())]

    products = rng.choice(list(PRODUCTS), n, p=[0.15, 0.25, 0.2, 0.15, 0.25])
    base_price = np.array([PRODUCTS[p] for p in products])
    customer_ids = rng.integers(1, len(customers) + 1, n)
    segment = customers.set_index("customer_id").loc[customer_ids, "segment"].to_numpy()
    units = np.where(segment == "Enterprise", rng.integers(4, 11, n), rng.integers(1, 11, n))

    discount = rng.choice([0.0, 0.05, 0.1, 0.2], n, p=[0.5, 0.25, 0.15, 0.1])
    discount[rng.random(n) < 0.03] = np.nan

    df = pd.DataFrame(
        {
            "order_id": np.arange(100001, 100001 + n),
            "date": dates.strftime("%Y-%m-%d"),
            "region": rng.choice(REGIONS, n, p=[0.3, 0.2, 0.25, 0.25]),
            "product": products,
            "customer_id": customer_ids,
            "units": units,
            "unit_price": np.round(base_price * rng.normal(1.0, 0.08, n), 2),
            "discount": discount,
        }
    )
    duplicates = df.iloc[rng.choice(n, 15, replace=False)]
    df = pd.concat([df, duplicates]).sort_values(["date", "order_id"], kind="stable")
    return df.reset_index(drop=True)


def _employees(rng: np.random.Generator) -> pd.DataFrame:
    n = 250
    dept = rng.choice(list(DEPARTMENTS), n, p=[0.3, 0.12, 0.2, 0.15, 0.15, 0.08])
    level = rng.choice(list(LEVELS), n, p=[0.3, 0.35, 0.25, 0.1])
    for d in DEPARTMENTS:  # every department needs at least one lead to report to
        idx = np.flatnonzero(dept == d)
        if len(idx) and not (level[idx] == "Lead").any():
            level[idx[0]] = "Lead"

    base = np.array([DEPARTMENTS[d] for d in dept])
    mult = np.array([LEVELS[lv] for lv in level])
    salary = (np.round(base * mult * rng.normal(1.0, 0.15, n) / 100) * 100).astype(int)

    employee_id = np.arange(1, n + 1)
    manager_id = np.full(n, np.nan)
    for i in range(n):
        if level[i] != "Lead":
            leads = employee_id[(dept == dept[i]) & (level == "Lead")]
            manager_id[i] = rng.choice(leads)

    z = (salary - salary.mean()) / salary.std()
    score = np.clip(np.round(3 + 0.6 * z + rng.normal(0, 0.9, n)), 1, 5)
    score[rng.random(n) < 0.08] = np.nan

    hire = pd.Timestamp("2015-01-01") + pd.to_timedelta(rng.integers(0, 3468, n), unit="D")
    return pd.DataFrame(
        {
            "employee_id": employee_id,
            "department": dept,
            "level": level,
            "hire_date": hire.strftime("%Y-%m-%d"),
            "salary": salary,
            "manager_id": pd.array(manager_id, dtype="Int64"),
            "remote": rng.random(n) < 0.35,
            "performance_score": pd.array(score, dtype="Int64"),
        }
    )


def _sensors(rng: np.random.Generator) -> pd.DataFrame:
    ts = pd.date_range("2024-03-01", periods=30 * 24, freq="h")
    hours = ts.hour.to_numpy()
    frames = []
    for k, sensor in enumerate(SENSORS):
        temp = (
            18
            + 1.5 * k
            + 4 * np.sin(2 * np.pi * (hours - 9) / 24)
            + np.linspace(0, 3, len(ts))
            + rng.normal(0, 0.8, len(ts))
        )
        humidity = np.clip(55 - 1.2 * (temp - 20) + rng.normal(0, 4, len(ts)), 15, 95)
        df = pd.DataFrame(
            {
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "sensor_id": sensor,
                "temperature": np.round(temp, 2),
                "humidity": np.round(humidity, 1),
            }
        )
        drop_p = 0.05 if sensor == "S2" else 0.02
        df = df[rng.random(len(df)) >= drop_p].reset_index(drop=True)
        df.loc[rng.random(len(df)) < 0.01, "temperature"] = -999.0
        frames.append(df)
    out = pd.concat(frames).sort_values(["timestamp", "sensor_id"], kind="stable")
    return out.reset_index(drop=True)
