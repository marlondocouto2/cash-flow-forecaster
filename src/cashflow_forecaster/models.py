"""The model ladder. Every model has fit(df) and predict(horizon or dates).

1. SeasonalWeekday  - Project 1: level x weekday factor x month-position factor.
2. SpikeAdjusted    - + a Gaussian mixture separates spike days; the baseline is
                      re-estimated on normal days only.
3. HolidayAdjusted  - + holiday effects estimated from history, applied on top.

`df` has columns `date` and `volume`, sorted, with nothing after the forecast origin.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from .config import CONFIG
from .features import MONTH_BUCKETS, future_dates, holiday_window, month_bucket, window_key


# --------------------------------------------------------------------------- helpers
def estimate_factors(
    y: pd.Series, business_days: bool, weekday_window_weeks: int, month_window_days: int,
    n_passes: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Weekday and month-position factors, each normalized to average 1 (Project 1 method)."""
    per_week = 5 if business_days else 7
    y_mp = y[y.index > y.index[-1] - pd.Timedelta(days=month_window_days)]
    weekday = pd.Series(1.0, index=range(7))
    month = pd.Series(1.0, index=MONTH_BUCKETS)
    for _ in range(n_passes):
        month = _ratio_factors(y_mp / _lookup(weekday, y_mp.index.day_of_week),
                               month_bucket(y_mp.index)).reindex(MONTH_BUCKETS).fillna(1.0)
        y_adj = y / _lookup(month, month_bucket(y.index))
        # Each day vs its centred one-week average, so trend and slow bumps cancel out.
        local = (y_adj / y_adj.rolling(per_week, center=True).mean()).dropna()
        local = local.iloc[-weekday_window_weeks * per_week:]
        weekday = _ratio_factors(local, local.index.day_of_week)
        weekday = (weekday / weekday.mean()).reindex(range(7))
    return weekday, month


def _ratio_factors(values: pd.Series, keys) -> pd.Series:
    keys = np.asarray(keys)
    factors = values.groupby(keys).mean() / values.mean()
    return factors / factors.reindex(keys).mean()


def _lookup(factors: pd.Series, keys) -> np.ndarray:
    return factors.reindex(np.asarray(keys)).fillna(1.0).to_numpy()


def seasonal(dates: pd.DatetimeIndex, weekday: pd.Series, month: pd.Series) -> np.ndarray:
    return _lookup(weekday, dates.day_of_week) * _lookup(month, month_bucket(dates))


def in_sample_fit(y: pd.Series, weekday: pd.Series, month: pd.Series, window: int = 28) -> pd.Series:
    """Historical 'what a normal day would have been': rolling-median level x seasonal factors.

    The median makes the level robust to spikes and holidays. Uses training data only.
    """
    s = seasonal(y.index, weekday, month)
    level = (y / s).rolling(window, center=True, min_periods=7).median()
    return level * s


# --------------------------------------------------------------------------- models
class BaseModel:
    name = "base"

    def fit(self, df: pd.DataFrame) -> "BaseModel":
        y = df.set_index("date")["volume"].astype(float).sort_index()
        self.last_date_ = y.index[-1]
        self.business_days_ = bool((y.index.day_of_week < 5).all())
        self._fit(y)
        return self

    def predict(self, horizon_or_dates: int | Sequence) -> pd.Series:
        if isinstance(horizon_or_dates, (int, np.integer)):
            dates = future_dates(self.last_date_, int(horizon_or_dates), self.business_days_)
        else:
            dates = pd.DatetimeIndex(horizon_or_dates)
        return pd.Series(self._predict(dates), index=dates, name=self.name)

    def _fit(self, y: pd.Series) -> None:
        raise NotImplementedError

    def _predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        raise NotImplementedError


class SeasonalWeekday(BaseModel):
    """Project 1 model: trailing level x weekday factor x month-position factor."""

    name = "1. Baseline"

    def __init__(self, level_window: int = CONFIG.level_window,
                 weekday_window_weeks: int = CONFIG.weekday_window_weeks,
                 month_window_days: int = CONFIG.month_window_days):
        self.level_window = level_window
        self.weekday_window_weeks = weekday_window_weeks
        self.month_window_days = month_window_days

    def _factors(self, y: pd.Series) -> tuple[pd.Series, pd.Series]:
        return estimate_factors(y, self.business_days_, self.weekday_window_weeks,
                                self.month_window_days)

    def _set_baseline(self, y: pd.Series) -> None:
        self.weekday_factors_, self.month_factors_ = self._factors(y)
        recent = y.iloc[-self.level_window:]
        self.level_ = float((recent / seasonal(recent.index, self.weekday_factors_,
                                               self.month_factors_)).mean())

    def _fit(self, y: pd.Series) -> None:
        self._set_baseline(y)

    def _predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        return self.level_ * seasonal(dates, self.weekday_factors_, self.month_factors_)


class SpikeAdjusted(SeasonalWeekday):
    """Baseline + spike decomposition with a Gaussian mixture.

    1. Residual = log(actual) - log(normal-day fit) over recent history.
    2. A Gaussian mixture on the residuals; the component with the highest mean is "spike".
    3. Spike days (and, in subclasses, holiday windows) are replaced by their normal-day
       fit, and the level and seasonal factors are re-estimated on the cleaned series.
    The point forecast is the clean baseline. `expected_spike` gives the extra volume to
    expect on average (spike probability x average spike size), for buffer sizing.
    """

    name = "2. + Spike GMM"

    def __init__(self, gmm_components: int = CONFIG.gmm_components,
                 gmm_history_days: int = CONFIG.gmm_history_days,
                 spike_threshold: float = CONFIG.spike_threshold,
                 seed: int = CONFIG.seed, **kwargs):
        super().__init__(**kwargs)
        self.gmm_components = gmm_components
        self.gmm_history_days = gmm_history_days
        self.spike_threshold = spike_threshold
        self.seed = seed

    def _calendar_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Days to keep out of the mixture and the baseline. None here; holidays later."""
        return np.zeros(len(index), dtype=bool)

    def _fit(self, y: pd.Series) -> None:
        weekday, month = self._factors(y)
        fitted = in_sample_fit(y, weekday, month, self.level_window)
        resid = (np.log(y) - np.log(fitted)).dropna()
        resid = resid[resid.index > y.index[-1] - pd.Timedelta(days=self.gmm_history_days)]
        on_calendar = self._calendar_mask(resid.index)

        gmm = GaussianMixture(self.gmm_components, random_state=self.seed)
        gmm.fit(resid[~on_calendar].to_numpy().reshape(-1, 1))
        spike_comp = int(np.argmax(gmm.means_.ravel()))
        post = gmm.predict_proba(resid.to_numpy().reshape(-1, 1))[:, spike_comp]
        flags = pd.Series((post > self.spike_threshold) & ~on_calendar, index=resid.index)

        self.gmm_, self.spike_component_, self.residuals_ = gmm, spike_comp, resid
        self.spike_flags_ = flags
        self.spike_prob_ = float(flags[~on_calendar].mean())
        self.spike_multiplier_ = float(np.exp(resid[flags]).mean()) if flags.any() else 1.0

        # Replace spikes and calendar days by what a normal day would have been, then refit.
        drop = flags.reindex(y.index, fill_value=False).to_numpy() | self._calendar_mask(y.index)
        self.clean_ = y.where(~drop, fitted).fillna(y)
        self._set_baseline(self.clean_)

    def expected_spike(self, dates: Sequence) -> pd.Series:
        """Expected extra volume from spikes on each date (for buffer sizing)."""
        base = SeasonalWeekday._predict(self, pd.DatetimeIndex(dates))
        return pd.Series(base * self.spike_prob_ * (self.spike_multiplier_ - 1), index=dates)


class HolidayAdjusted(SpikeAdjusted):
    """Spike-adjusted baseline x holiday effects.

    Effect for each (tier, offset) window position = median of actual / normal-day fit
    over past occurrences, shrunk toward 1.0 when there are few: 1 + n/(n+k) x (median - 1).
    """

    name = "3. + Holidays"

    def __init__(self, country: str, shrinkage: float = CONFIG.holiday_shrinkage, **kwargs):
        super().__init__(**kwargs)
        self.country = country
        self.shrinkage = shrinkage

    def _calendar_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        return holiday_window(index, self.country)["holiday_tier"].notna().to_numpy()

    def _fit(self, y: pd.Series) -> None:
        super()._fit(y)
        normal_day = in_sample_fit(self.clean_, self.weekday_factors_, self.month_factors_,
                                   self.level_window)
        keys = window_key(holiday_window(y.index, self.country))
        ratio = (y / normal_day)[keys.notna().to_numpy()].dropna()
        stats = ratio.groupby(keys.reindex(ratio.index)).agg(["median", "size"])
        n = stats["size"]
        self.holiday_effects_ = 1 + n / (n + self.shrinkage) * (stats["median"] - 1)
        self.holiday_counts_ = n

    def _predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        keys = window_key(holiday_window(dates, self.country))
        effect = self.holiday_effects_.reindex(keys.to_numpy()).fillna(1.0).to_numpy()
        return super()._predict(dates) * effect


def model_ladder(country: str) -> list:
    """Factories for the three versions, in ladder order."""
    return [SeasonalWeekday, SpikeAdjusted, lambda: HolidayAdjusted(country=country)]
