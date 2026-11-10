"""Synthetic remittance flows with spikes and holidays, plus an optional public-data loader.

Each corridor (US -> receiving country) is simulated multiplicatively from known parts,
and every true part is kept in its own column so tests can check recovery:

    volume = trend x weekday x month_position x annual x holiday x spike x noise

The first five parts are the Project 1 generator; spikes and holidays are new here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import holiday_window

RAW_WEEKDAY_SHAPE = np.array([1.10, 1.00, 0.95, 1.00, 1.25, 0.80, 0.70])
TRUE_WEEKDAY_FACTORS = RAW_WEEKDAY_SHAPE / RAW_WEEKDAY_SHAPE.mean()
NOISE_SIGMA = 0.06

# Corridor: (starting daily volume in USD, annual growth)
CORRIDORS = {
    "MX": (25_000_000.0, 0.12),
    "PH": (12_000_000.0, 0.10),
    "IN": (15_000_000.0, 0.18),
}

# Spikes: ~3 a month per corridor, 1.5-3x normal volume, independent of the calendar.
SPIKE_PROB = 3 / 30.44
SPIKE_RANGE = (1.5, 3.0)

# True multiplicative holiday effects by (tier, offset from the holiday).
TRUE_HOLIDAY_EFFECTS = {
    ("major", -3): 1.25,
    ("major", -2): 1.40,
    ("major", -1): 1.55,
    ("major", 0): 0.60,
    ("major", 1): 1.20,
    ("minor", 0): 0.60,
    ("minor", 1): 1.15,
}


def _last_business_day(dates: pd.DatetimeIndex) -> np.ndarray:
    month_end = dates + pd.offsets.MonthEnd(0)
    dow = month_end.day_of_week
    shift = np.where(dow == 5, 1, np.where(dow == 6, 2, 0))
    return (month_end - pd.to_timedelta(shift, unit="D")).to_numpy()


def month_position_effect(dates: pd.DatetimeIndex) -> np.ndarray:
    """Payday lifts around the 1st, the 15th and the last business day."""
    day = dates.day.to_numpy()
    effect = np.ones(len(dates))
    effect[day == 1] *= 1.25
    effect[day == 2] *= 1.12
    effect[day == 3] *= 1.05
    effect[day == 15] *= 1.20
    effect[(day == 14) | (day == 16)] *= 1.06
    days_to_lbd = (_last_business_day(dates) - dates.to_numpy()).astype("timedelta64[D]").astype(int)
    effect[days_to_lbd == 0] *= 1.30
    effect[days_to_lbd == 1] *= 1.08
    return effect


def annual_effect(dates: pd.DatetimeIndex) -> np.ndarray:
    """Mild yearly sine wave plus a December peak centred on the 20th."""
    doy = dates.day_of_year.to_numpy()
    sine = 1 + 0.04 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    dec20 = pd.to_datetime([f"{y}-12-20" for y in dates.year]).day_of_year.to_numpy()
    return sine * (1 + 0.12 * np.exp(-0.5 * ((doy - dec20) / 6.0) ** 2))


def generate_corridor(
    country: str, start: str, end: str, seed: int, base: float, growth: float
) -> pd.DataFrame:
    """One corridor's daily volume with all true components as columns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    years = (dates - dates[0]).days.to_numpy() / 365.25

    window = holiday_window(dates, country)
    keys = list(zip(window["holiday_tier"], window["holiday_offset"]))
    holiday = np.array([TRUE_HOLIDAY_EFFECTS.get(k, 1.0) for k in keys])

    is_spike = rng.random(len(dates)) < SPIKE_PROB
    spike = np.where(is_spike, rng.uniform(*SPIKE_RANGE, size=len(dates)), 1.0)
    noise = rng.lognormal(0.0, NOISE_SIGMA, size=len(dates))

    out = pd.DataFrame(
        {
            "date": dates,
            "corridor": country,
            "trend": base * (1 + growth) ** years,
            "weekday_factor": TRUE_WEEKDAY_FACTORS[dates.day_of_week],
            "month_factor": month_position_effect(dates),
            "annual_factor": annual_effect(dates),
            "holiday_factor": holiday,
            "spike_factor": spike,
            "noise": noise,
            "is_spike": is_spike,
            "holiday_name": window["holiday_name"].to_numpy(),
            "holiday_tier": window["holiday_tier"].to_numpy(),
            "holiday_offset": window["holiday_offset"].to_numpy(),
        }
    )
    parts = ["trend", "weekday_factor", "month_factor", "annual_factor", "holiday_factor",
             "spike_factor", "noise"]
    out.insert(2, "volume", out[parts].prod(axis=1))
    return out


def generate_flows(
    start: str = "2021-01-01", end: str = "2025-12-31", seed: int = 42,
    corridors: tuple[str, ...] = tuple(CORRIDORS),
) -> pd.DataFrame:
    """All corridors stacked in long format (one row per corridor and day)."""
    frames = [
        generate_corridor(c, start, end, seed + i, *CORRIDORS[c]) for i, c in enumerate(corridors)
    ]
    return pd.concat(frames, ignore_index=True)


FISCAL_DATA_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
    "v1/accounting/dts/deposits_withdrawals_operating_cash"
)


def load_treasury_withdrawals(start: str = "2021-01-01", end: str | None = None) -> pd.DataFrame:
    """Daily U.S. Treasury operating-cash withdrawals (USD), business days only.

    Daily Treasury Statement, Table II, via the Fiscal Data API: every withdrawal category
    except public-debt redemptions. Returned in the same long format, corridor "US".
    """
    import requests

    flt = f"record_date:gte:{start},transaction_type:eq:Withdrawals"
    if end:
        flt += f",record_date:lte:{end}"
    rows, page = [], 1
    while True:
        resp = requests.get(
            FISCAL_DATA_URL,
            params={"filter": flt, "fields": "record_date,transaction_catg,transaction_today_amt",
                    "page[size]": 10_000, "page[number]": page},
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
        rows.extend(payload["data"])
        if page >= payload["meta"]["total-pages"]:
            break
        page += 1

    raw = pd.DataFrame(rows)
    keep = (raw["transaction_catg"] != "null") & ~raw["transaction_catg"].str.startswith("Public Debt")
    raw = raw[keep].copy()
    raw["amount"] = pd.to_numeric(raw["transaction_today_amt"], errors="coerce") * 1e6
    out = raw.groupby("record_date", as_index=False)["amount"].sum()
    out.columns = ["date", "volume"]
    out["date"] = pd.to_datetime(out["date"])
    out.insert(1, "corridor", "US")
    window = holiday_window(pd.DatetimeIndex(out["date"]), "US")
    for col in window.columns:
        out[col] = window[col].to_numpy()
    out["is_spike"] = np.nan  # unknown on real data
    return out.sort_values("date").reset_index(drop=True)
