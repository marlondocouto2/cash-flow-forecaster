"""Walk-forward backtest, error metrics by day type, and dollar metrics."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import CONFIG
from .models import BaseModel, model_ladder

DAY_TYPES = ["normal", "spike", "scheduled", "holiday"]


def walk_forward(
    df: pd.DataFrame,
    model_factory: Callable[[], BaseModel],
    horizons: Iterable[int] = CONFIG.horizons,
    test_periods: int = CONFIG.backtest_days,
) -> pd.DataFrame:
    """Rolling-origin backtest for one series.

    The model is refit on rows up to and including the origin only, then forecasts the
    next max(horizons) steps. Models with `refit_every > 1` (Prophet) are refit every that
    many origins; in between, the latest fit is reused, so it never sees data after the
    origin either. Rows are kept where the target falls in the last `test_periods`
    observations and the step ahead is in `horizons`.
    """
    df = df[["date", "volume"]].sort_values("date").reset_index(drop=True)
    horizons = sorted(set(horizons))
    n, h_max = len(df), max(horizons)
    test_start = n - test_periods
    records = []
    first = test_start - h_max
    model = None
    for origin in range(first, n - 1):
        if model is None or (origin - first) % model.refit_every == 0:
            model = model_factory().fit(df.iloc[: origin + 1])
        targets = [origin + h for h in horizons if test_start <= origin + h < n]
        if not targets:
            continue
        dates = df.loc[targets, "date"]
        preds = model.predict(dates)
        for t, h in zip(targets, [t - origin for t in targets]):
            records.append({"model": model.name, "origin": df.at[origin, "date"],
                            "date": df.at[t, "date"], "horizon": h,
                            "forecast": float(preds.iloc[targets.index(t)]),
                            "actual": float(df.at[t, "volume"])})
    out = pd.DataFrame(records)
    out["error"] = out["actual"] - out["forecast"]  # positive = under-forecast (shortfall)
    return out


def _run(args) -> pd.DataFrame:
    corridor, series, factory, horizons, test_periods = args
    bt = walk_forward(series, factory, horizons, test_periods)
    bt.insert(0, "corridor", corridor)
    return bt


def backtest_all(
    flows: pd.DataFrame, horizons: Iterable[int] = CONFIG.horizons,
    test_periods: int = CONFIG.backtest_days, versions: Sequence[int] | None = None,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Run the chosen ladder versions on every corridor (in parallel) and label day types.

    Day type of each target day: holiday window > random spike > scheduled spike > normal.
    """
    horizons = tuple(horizons)
    tasks = []
    for corridor, g in flows.groupby("corridor", sort=False):
        g = g[["date", "volume"]].sort_values("date").reset_index(drop=True)
        for factory in model_ladder(corridor, versions):
            tasks.append((corridor, g, factory, horizons, test_periods))
    n_jobs = n_jobs or min(len(tasks), os.cpu_count() or 1)
    if n_jobs > 1:
        with ProcessPoolExecutor(n_jobs) as pool:
            frames = list(pool.map(_run, tasks))
    else:
        frames = [_run(t) for t in tasks]
    bt = pd.concat(frames, ignore_index=True)

    cols = ["corridor", "date", "is_spike", "is_scheduled", "holiday_tier", "holiday_offset"]
    truth = flows[[c for c in cols if c in flows.columns]]
    bt = bt.merge(truth, on=["corridor", "date"], how="left")
    flag = lambda c: bt[c].fillna(False).astype(bool) if c in bt else False  # noqa: E731
    bt["day_type"] = np.select(
        [bt["holiday_tier"].notna(), flag("is_spike"), flag("is_scheduled")],
        ["holiday", "spike", "scheduled"], default="normal",
    )
    return bt


def required_buffer(actual, forecast, q: float = CONFIG.buffer_quantile) -> float:
    """Cash held on top of the forecast to cover the shortfall on q of days."""
    shortfall = np.maximum(np.asarray(actual) - np.asarray(forecast), 0.0)
    return float(np.quantile(shortfall, q))


def idle_cash_cost(actual, forecast, annual_rate: float = CONFIG.annual_cost_of_capital) -> float:
    """Annual cost of over-forecasting: average daily over-forecast x (annual_rate / 365) x 365."""
    over = np.maximum(np.asarray(forecast) - np.asarray(actual), 0.0)
    return float(over.mean() * annual_rate)


def accuracy(actual, forecast) -> dict[str, float]:
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    err = a - f
    mse = float(np.mean(err**2))
    return {"MAE": float(np.mean(np.abs(err))), "MSE": mse, "RMSE": float(np.sqrt(mse))}


def daily_totals(bt: pd.DataFrame) -> pd.DataFrame:
    """Sum corridors per date: the amount treasury actually has to fund each day."""
    return bt.groupby(["model", "horizon", "date"], as_index=False)[["actual", "forecast"]].sum()


def ladder_table(bt: pd.DataFrame) -> pd.DataFrame:
    """One row per (horizon, model version): accuracy by day type plus dollar metrics.

    Accuracy is measured per corridor-day; dollar metrics on the daily total across
    corridors, since that is what has to be funded.
    """
    totals = daily_totals(bt)
    rows = []
    for (h, m), g in bt.groupby(["horizon", "model"]):
        row = {"horizon": h, "model": m, **accuracy(g["actual"], g["forecast"])}
        for t in DAY_TYPES:
            s = g[g["day_type"] == t]
            row[f"MAE_{t}"] = float(np.abs(s["error"]).mean())
        tot = totals[(totals["horizon"] == h) & (totals["model"] == m)]
        row["required_buffer"] = required_buffer(tot["actual"], tot["forecast"])
        # Like-for-like: cash set aside per day to cover 95% of days = forecast + buffer.
        row["avg_cash_held"] = float(tot["forecast"].mean() + row["required_buffer"])
        row["idle_cash_cost_per_year"] = idle_cash_cost(tot["actual"], tot["forecast"])
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["horizon", "model"]).reset_index(drop=True)
    by_h = out.groupby("horizon")
    # How much each version adds: MAE change vs the previous version and vs the baseline.
    out["MAE_vs_previous_pct"] = (by_h["MAE"].pct_change() * 100).round(1)
    out["MAE_vs_baseline_pct"] = ((out["MAE"] / by_h["MAE"].transform("first") - 1) * 100).round(1)
    base = by_h["idle_cash_cost_per_year"].transform("first")
    out["idle_cash_saving_vs_baseline"] = base - out["idle_cash_cost_per_year"]
    return out


def day_type_counts(bt: pd.DataFrame) -> pd.Series:
    one = bt[(bt["model"] == bt["model"].iloc[0]) & (bt["horizon"] == bt["horizon"].iloc[0])]
    return one["day_type"].value_counts()
