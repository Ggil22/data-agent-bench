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
