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
