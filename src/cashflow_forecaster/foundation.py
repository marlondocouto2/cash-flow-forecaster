"""Version 6: Chronos-2, a pretrained time series foundation model, used zero-shot.

Chronos-2 (Amazon, 2025; Ansari et al., "Chronos-2: From Univariate to Universal
Forecasting", arXiv 2510.15821) is a 120M-parameter model pretrained on many real and
synthetic series. It is not fitted to these corridors at all: at each forecast origin it reads
the history up to that day and returns quantile forecasts.

It gets the same calendar knowledge the other versions use, as "known future covariates":
weekday, position in the month (the payday buckets of Project 1), and the holiday window
(e.g. "major:-2" = two days before a major holiday). Its group attention lets these covariates
inform the target series in context. It never sees spike or payroll labels.

Leakage: each origin's input is the history up to and including the origin, and covariates
are calendar facts known in advance, so nothing after the origin is used.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd

from .config import CONFIG
from .features import future_dates, holiday_window, month_bucket, window_key

MODEL_ID = "amazon/chronos-2"
NAME = "6. Chronos-2 (zero-shot)"
NAME_NO_COVARIATES = "Chronos-2 without covariates"


@lru_cache(maxsize=1)
def load_pipeline():
    """Download (first time only, ~0.5 GB, cached by Hugging Face) and load Chronos-2 on CPU."""
    import logging

    # Quiet the loading noise (a macOS torch note, Hugging Face progress bars and sign-in hints).
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
    import torch
    from chronos import Chronos2Pipeline
    from huggingface_hub.utils import disable_progress_bars
    from huggingface_hub.utils import logging as hf_logging
    from transformers.utils import logging as tf_logging

    disable_progress_bars()
    hf_logging.set_verbosity_error()
    tf_logging.set_verbosity_error()
    tf_logging.disable_progress_bar()
    torch.manual_seed(CONFIG.seed)
    return Chronos2Pipeline.from_pretrained(MODEL_ID, device_map="cpu")


def calendar_covariates(dates: pd.DatetimeIndex, country: str) -> dict[str, np.ndarray]:
    """Calendar facts known in advance for every date (categorical, as strings)."""
    dates = pd.DatetimeIndex(dates)
    key = window_key(holiday_window(dates, country)).fillna("none")
    return {
        "weekday": dates.day_name().to_numpy().astype(str),
        "month_position": month_bucket(dates).to_numpy().astype(str),
        "holiday_window": key.to_numpy().astype(str),
    }


def chronos_backtest(
    flows: pd.DataFrame,
    horizons: Iterable[int] = CONFIG.horizons,
    test_periods: int = CONFIG.backtest_days,
    covariates: bool = True,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Walk-forward backtest of Chronos-2 with the same origins, targets and output columns
    as `evaluate.walk_forward`, for every corridor in `flows`."""
    pipe = load_pipeline()
    horizons = sorted(set(horizons))
    h_max = max(horizons)
    name = NAME if covariates else NAME_NO_COVARIATES
    frames = []
    for corridor, g in flows.groupby("corridor", sort=False):
        g = g[["date", "volume"]].sort_values("date").reset_index(drop=True)
        n, test_start = len(g), len(g) - test_periods
        dates = pd.DatetimeIndex(g["date"])
        business_days = bool((dates.day_of_week < 5).all())  # U.S. Treasury data: weekdays only
        steps = future_dates(dates[-1], h_max, business_days)  # calendar dates past the data
        all_dates = dates.append(steps)
        cov = calendar_covariates(all_dates, corridor) if covariates else None
        y = g["volume"].to_numpy(float)
        origins = list(range(test_start - h_max, n - 1))
        inputs = []
        for o in origins:
            item = {"target": y[: o + 1]}
            if cov is not None:
                item["past_covariates"] = {k: v[: o + 1] for k, v in cov.items()}
                item["future_covariates"] = {k: v[o + 1: o + 1 + h_max] for k, v in cov.items()}
            inputs.append(item)
        _, median = pipe.predict_quantiles(inputs, prediction_length=h_max, quantile_levels=[0.5],
                                           batch_size=batch_size)
        records = []
        for o, m in zip(origins, median):
            path = m.reshape(-1).numpy()  # median forecast, steps 1..h_max
            for h in horizons:
                t = o + h
                if test_start <= t < n:
                    records.append({"corridor": corridor, "model": name, "origin": dates[o],
                                    "date": dates[t], "horizon": h, "forecast": float(path[h - 1]),
                                    "actual": float(y[t])})
        frames.append(pd.DataFrame(records))
    out = pd.concat(frames, ignore_index=True)
    out["error"] = out["actual"] - out["forecast"]
    return out
