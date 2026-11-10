# Data

No data files are committed. Both sources are produced on demand by
`src/cashflow_forecaster/data.py`.

## Synthetic corridors (default)

```python
from cashflow_forecaster import generate_flows
df = generate_flows("2021-01-01", "2025-12-31", seed=42)
```

Long format, one row per corridor (`MX`, `PH`, `IN`) and day:

| Column | Meaning |
|---|---|
| `volume` | simulated payout volume (USD), the product of the factor columns |
| `trend`, `weekday_factor`, `month_factor`, `annual_factor`, `holiday_factor`, `scheduled_factor`, `spike_factor`, `noise` | the true components |
| `is_spike` | whether the day is a random spike (separate regime, unpredictable) |
| `is_scheduled` | whether the day is a scheduled spike (biweekly payroll Friday) |
| `holiday_name`, `holiday_tier`, `holiday_offset` | position relative to the receiving country's holidays (`-3`..`-1` before a major holiday, `0` on the day, `1` after) |

Holiday calendars come from the `holidays` package. Mexico's Mother's Day (10 May) is
added by hand because it is not a public holiday. Which holidays count as "major" is set
in `features.MAJOR_HOLIDAYS`.

## U.S. Treasury (optional, public)

```python
from cashflow_forecaster import load_treasury_withdrawals
df = load_treasury_withdrawals(start="2021-01-01")
```

Daily Treasury Statement, Table II, from the Fiscal Data API (no key needed). `volume` is
the sum of withdrawal categories, excluding public-debt redemptions, on business days. It
is returned in the same long format as corridor `US`, with U.S. federal holiday windows.
`is_spike` and `is_scheduled` are unknown, so they are left empty. A few early-2021 days whose withdrawals net to zero or below (accounting adjustments) are dropped.
