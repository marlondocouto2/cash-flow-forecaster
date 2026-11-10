"""Calendar features: day-of-month buckets and holiday windows."""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

MONTH_BUCKETS = ["d01_03", "d04_13", "d14_16", "d17_late", "last3"]

# Holidays big enough to move money ahead of time. Everything else is "minor".
MAJOR_HOLIDAYS = {
    "MX": ("Año Nuevo", "Día de las Madres", "Navidad"),
    "PH": ("New Year's Day", "Christmas Day"),
    "IN": ("Diwali", "Holi"),
    "US": ("New Year's Day", "Thanksgiving", "Christmas Day"),
}
PRE_DAYS = 3  # major holidays: surge window of up to 3 days before

# Window offsets relative to a holiday: -3..-1 before (major only), 0 on the day, +1 after.
WINDOW_KEYS = [("major", -3), ("major", -2), ("major", -1), ("major", 0), ("major", 1),
               ("minor", 0), ("minor", 1)]


def month_bucket(dates: pd.DatetimeIndex | pd.Series) -> pd.Series:
    """Day-of-month bucket: 1-3, 4-13, 14-16, 17 to month end minus 3, last 3 days."""
    dates = pd.DatetimeIndex(dates)
    day, dim = dates.day.to_numpy(), dates.days_in_month.to_numpy()
    bucket = np.select(
        [day > dim - 3, day <= 3, day <= 13, day <= 16],
        ["last3", "d01_03", "d04_13", "d14_16"],
        default="d17_late",
    )
    return pd.Series(bucket, index=dates)


def future_dates(last_date: pd.Timestamp, horizon: int, business_days: bool) -> pd.DatetimeIndex:
    """The next `horizon` dates after `last_date` (calendar or business days)."""
    return pd.date_range(last_date, periods=horizon + 1, freq="B" if business_days else "D")[1:]


def holiday_calendar(country: str, years: range | list[int]) -> dict[date, tuple[str, str]]:
    """{date: (name, tier)} for one country, from the `holidays` package.

    Mexico's Mother's Day (10 May) is added by hand: it is not a public holiday, but it is
    one of the biggest remittance days of the year.
    """
    cal = holidays.country_holidays(country, years=list(years))
    out = {}
    for d, name in cal.items():
        tier = "major" if any(m in name for m in MAJOR_HOLIDAYS.get(country, ())) else "minor"
        out[d] = (name, tier)
    if country == "MX":
        for y in years:
            out[date(y, 5, 10)] = ("Día de las Madres", "major")
    return out


def holiday_window(dates: pd.DatetimeIndex, country: str) -> pd.DataFrame:
    """For each date, its position relative to the country's holidays.

    Columns: holiday_name, holiday_tier ("major"/"minor"/None), holiday_offset (int or NaN).
    Precedence when windows overlap: the holiday itself, then the day after a holiday,
    then the nearest upcoming major holiday within 3 days.
    """
    return _window_table(country).reindex(pd.DatetimeIndex(dates))


@lru_cache(maxsize=None)
def _window_table(country: str, first_year: int = 2001, last_year: int = 2035) -> pd.DataFrame:
    """Holiday-window table for every day of a long range, built once per country."""
    cal = holiday_calendar(country, range(first_year, last_year + 1))
    dates = pd.date_range(f"{first_year}-01-01", f"{last_year}-12-31", freq="D")
    names, tiers, offsets = [], [], []
    for ts in dates:
        d = ts.date()
        hit = None
        if d in cal:
            hit = (*cal[d], 0)
        elif d - timedelta(days=1) in cal:
            hit = (*cal[d - timedelta(days=1)], 1)
        else:
            for k in range(1, PRE_DAYS + 1):
                nxt = d + timedelta(days=k)
                if nxt in cal and cal[nxt][1] == "major":
                    hit = (*cal[nxt], -k)
                    break
        names.append(hit[0] if hit else None)
        tiers.append(hit[1] if hit else None)
        offsets.append(hit[2] if hit else np.nan)
    return pd.DataFrame(
        {"holiday_name": names, "holiday_tier": tiers, "holiday_offset": offsets}, index=dates
    )


def window_key(window: pd.DataFrame) -> pd.Series:
    """'major:-1' style key per date, or None outside any holiday window."""
    key = window["holiday_tier"] + ":" + window["holiday_offset"].astype("Int64").astype(str)
    return key.where(window["holiday_tier"].notna(), None)
