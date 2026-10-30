"""Deterministic synthetic datasets for the benchmark.

Synthetic data keeps every answer verifiable (ground truth is computed by code),
avoids licensing issues and cannot have leaked into any model's training data.
The data includes realistic traps: exact duplicates, missing values, a sentinel
error code, gaps in a time series, mixed date formats and currencies, inconsistent
labels and business rules, documented in DATA_DICTIONARY.md.
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
USD_TO_EUR = 0.92
SLA_HOURS = {"High": 8, "Medium": 24, "Low": 72}
TABLES = ("customers", "sales", "employees", "sensors", "web_orders", "tickets")

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

## web_orders.csv
Online shop orders from 2024, exported from a legacy system.
- order_ref: order identifier.
- order_date: order date. Three systems wrote this field, so formats are mixed:
  YYYY-MM-DD, DD/MM/YYYY (day first, European style) and "Mon DD, YYYY" (e.g. Mar 05, 2024).
- customer_email: the buyer's email. Emails are case-insensitive and some have stray
  spaces, so the same customer can appear with different spellings.
- amount: order total, in the currency given by `currency`.
- currency: EUR or USD. Convert USD to EUR at the fixed rate 1 USD = 0.92 EUR.
- status: completed, refunded or cancelled. Only completed orders count as revenue.

Orders whose email ends in @example.com are internal test orders and must be excluded
from every analysis.

## tickets.csv
Customer support tickets, June to August 2024.
- ticket_id: ticket identifier.
- created_at, resolved_at: UTC timestamps (YYYY-MM-DD HH:MM:SS). resolved_at is empty
  while the ticket is still open.
- priority: High, Medium or Low. Typed in by hand, so capitalisation and spacing vary.
- team: support team that handled the ticket (Tier1 or Tier2).

SLA: a resolved ticket breaches its SLA when it took longer than 8 hours (High),
24 hours (Medium) or 72 hours (Low) to resolve. Open tickets do not count as breaches.
"""


def generate(seed: int = DEFAULT_SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    customers = _customers(rng)
    return {
        "customers": customers,
        "sales": _sales(rng, customers),
        "employees": _employees(rng),
        "sensors": _sensors(rng),
        # Tables added later use their own generators, so the ones above never change.
        "web_orders": _web_orders(np.random.default_rng([seed, 1])),
        "tickets": _tickets(np.random.default_rng([seed, 2])),
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


def _web_orders(rng: np.random.Generator) -> pd.DataFrame:
    n, n_customers = 600, 150
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    weights = np.select([days.month == 12, days.month == 11], [1.8, 1.4], 1.0)  # holiday peak
    dates = days[np.sort(rng.choice(len(days), n, p=weights / weights.sum()))]

    weights = 1 / np.arange(1, n_customers + 1)  # a few customers buy much more often
    customer = rng.choice(n_customers, n, p=weights / weights.sum())
    domains = np.array(["gmail.com", "outlook.com", "yahoo.es"])
    emails = np.array([f"user{c:03d}@{domains[c % 3]}" for c in customer], dtype=object)
    variant = rng.random(n)
    emails[variant < 0.12] = [e.upper() for e in emails[variant < 0.12]]
    capitalised = (variant >= 0.12) & (variant < 0.2)
    emails[capitalised] = [e.capitalize() for e in emails[capitalised]]
    padded = (variant >= 0.2) & (variant < 0.25)
    emails[padded] = [f" {e} " for e in emails[padded]]

    date_style = rng.choice(["iso", "dmy", "text"], n, p=[0.6, 0.25, 0.15])
    order_date = np.where(
        date_style == "iso",
        dates.strftime("%Y-%m-%d"),
        np.where(date_style == "dmy", dates.strftime("%d/%m/%Y"), dates.strftime("%b %d, %Y")),
    )
    df = pd.DataFrame(
        {
            "order_date": order_date,
            "customer_email": emails,
            "amount": np.round(rng.lognormal(4.2, 0.6, n), 2),
            "currency": rng.choice(["EUR", "USD"], n, p=[0.75, 0.25]),
            "status": rng.choice(["completed", "refunded", "cancelled"], n, p=[0.85, 0.1, 0.05]),
        }
    )
    # Internal test orders: large amounts that distort totals if they are not excluded.
    n_test = 10
    test_dates = pd.Timestamp("2024-11-04") + pd.to_timedelta(rng.integers(0, 20, n_test), unit="D")
    tests = pd.DataFrame(
        {
            "order_date": test_dates.strftime("%Y-%m-%d"),
            "customer_email": [f"qa{i % 3 + 1}@example.com" for i in range(n_test)],
            "amount": np.round(rng.uniform(2000, 9000, n_test), 2),
            "currency": "EUR",
            "status": "completed",
        }
    )
    df = pd.concat([df, tests]).sample(frac=1, random_state=rng.integers(1 << 31))
    df.insert(0, "order_ref", [f"W{i:05d}" for i in range(1, len(df) + 1)])
    return df.reset_index(drop=True)


def _tickets(rng: np.random.Generator) -> pd.DataFrame:
    n = 400
    created = pd.Timestamp("2024-06-01") + pd.to_timedelta(
        rng.integers(0, 92 * 24 * 3600, n), unit="s"
    )
    created = created.sort_values()
    priority = rng.choice(list(SLA_HOURS), n, p=[0.25, 0.45, 0.3])
    # Tier2 takes most High tickets (which are fast) and few Low ones, so its overall mean
    # resolution time looks better even though it is slower on High and Medium tickets.
    p_tier2 = np.select([priority == "High", priority == "Medium"], [0.8, 0.4], 0.15)
    team = np.where(rng.random(n) < p_tier2, "Tier2", "Tier1")
    scale = np.select([priority == "High", priority == "Medium"], [5.0, 16.0], 45.0)
    hours = rng.exponential(scale * np.where(team == "Tier2", 1.3, 1.0))
    resolved = created + pd.to_timedelta(np.round(hours * 3600), unit="s")
    resolved_at = np.array(resolved.strftime("%Y-%m-%d %H:%M:%S"), dtype=object)
    resolved_at[rng.random(n) < 0.08] = None

    styles = rng.choice(
        ["as_is", "lower", "upper", "lead_space", "trail_space"], n, p=[0.6, 0.2, 0.1, 0.05, 0.05]
    )
    label = [
        {
            "as_is": p,
            "lower": p.lower(),
            "upper": p.upper(),
            "lead_space": f" {p.lower()}",
            "trail_space": f"{p} ",
        }[st]
        for p, st in zip(priority, styles, strict=True)
    ]
    return pd.DataFrame(
        {
            "ticket_id": np.arange(5001, 5001 + n),
            "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
            "resolved_at": resolved_at,
            "priority": label,
            "team": team,
        }
    )
