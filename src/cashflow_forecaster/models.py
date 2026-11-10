"""The model ladder. Every model has fit(df) and predict(horizon or dates).

1. SeasonalWeekday  - Project 1: level x weekday factor x month-position factor.
2. SpikeAdjusted    - + a Gaussian mixture separates spike days; the baseline is
                      re-estimated on normal days only.
3. HolidayAdjusted  - + holiday effects estimated from history, applied on top.
4. ProphetFourier   - a different family: Prophet trend + Fourier seasonalities (weekly,
                      biweekly, monthly, yearly) + holiday windows. Periodic spikes are
                      treated as seasonality, so scheduled spikes get forecast.
5. ProphetFourier(clean_spikes=True) - + the Gaussian mixture on Prophet's residuals, so
                      only spikes the seasonality can't explain are removed before refitting.

`df` has columns `date` and `volume`, sorted, with nothing after the forecast origin.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from .config import CONFIG
from .features import (
    MONTH_BUCKETS,
    future_dates,
    holiday_calendar,
    holiday_window,
    month_bucket,
    window_key,
)

for _name in ("cmdstanpy", "prophet"):
    logging.getLogger(_name).setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet.plot").disabled = True  # optional plotly import notice


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
    refit_every = 1  # backtest refit frequency, in origins

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

        gmm, spike_comp, flags = fit_spike_mixture(
            resid, on_calendar, self.gmm_components, self.spike_threshold, self.seed)

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


def fit_spike_mixture(resid: pd.Series, exclude: np.ndarray, n_components: int,
                      threshold: float, seed: int) -> tuple[GaussianMixture, int, pd.Series]:
    """Fit a Gaussian mixture on residuals; flag days in the highest-mean component."""
    gmm = GaussianMixture(n_components, random_state=seed)
    gmm.fit(resid[~exclude].to_numpy().reshape(-1, 1))
    comp = int(np.argmax(gmm.means_.ravel()))
    post = gmm.predict_proba(resid.to_numpy().reshape(-1, 1))[:, comp]
    return gmm, comp, pd.Series((post > threshold) & ~exclude, index=resid.index)


class ProphetFourier(BaseModel):
    """Prophet on log volume: trend + Fourier seasonalities + tiered holiday windows.

    Seasonalities (Fourier order in brackets): weekly (3), biweekly with period 14 (7),
    monthly with period 30.44 days (15), yearly (20). A periodic spike, such as a payroll
    run every other Friday, is absorbed by the biweekly terms and therefore forecast.

    With `clean_spikes=True`, a Gaussian mixture is fitted on the in-sample residuals
    (holiday windows excluded); days in the spike component are dropped and Prophet is
    refitted. Only spikes the seasonality can't explain are removed.
    """

    refit_every = 7  # Prophet fits take ~0.25s; refit weekly in the backtest

    def __init__(self, country: str, clean_spikes: bool = False, history_days: int = 1095,
                 gmm_components: int = CONFIG.gmm_components,
                 spike_threshold: float = CONFIG.spike_threshold, seed: int = CONFIG.seed):
        self.country = country
        self.clean_spikes = clean_spikes
        self.history_days = history_days
        self.gmm_components = gmm_components
        self.spike_threshold = spike_threshold
        self.seed = seed
        self.name = "5. Prophet + Fourier + GMM" if clean_spikes else "4. Prophet + Fourier"

    def _holidays(self, years: range) -> pd.DataFrame:
        cal = holiday_calendar(self.country, years)
        return pd.DataFrame(
            [{"ds": pd.Timestamp(d), "holiday": tier,
              "lower_window": -3 if tier == "major" else 0, "upper_window": 1}
             for d, (_, tier) in cal.items()]
        )

    def _new_prophet(self, years: range):
        from prophet import Prophet

        m = Prophet(growth="linear", yearly_seasonality=20, weekly_seasonality=3,
                    daily_seasonality=False, holidays=self._holidays(years))
        m.add_seasonality(name="biweekly", period=14, fourier_order=7)
        m.add_seasonality(name="monthly", period=30.4375, fourier_order=15)
        return m

    def _calendar_mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        return holiday_window(index, self.country)["holiday_tier"].notna().to_numpy()

    def _fit(self, y: pd.Series) -> None:
        y = y[y.index > y.index[-1] - pd.Timedelta(days=self.history_days)]
        years = range(y.index[0].year, y.index[-1].year + 3)
        data = pd.DataFrame({"ds": y.index, "y": np.log(y.to_numpy())})
        model = self._new_prophet(years).fit(data)
        if self.clean_spikes:
            fitted = model.predict(data[["ds"]])["yhat"].to_numpy()
            resid = pd.Series(data["y"].to_numpy() - fitted, index=y.index)
            on_calendar = holiday_window(y.index, self.country)["holiday_tier"].notna().to_numpy()
            self.gmm_, self.spike_component_, flags = fit_spike_mixture(
                resid, on_calendar, self.gmm_components, self.spike_threshold, self.seed)
            self.residuals_, self.spike_flags_ = resid, flags
            model = self._new_prophet(years).fit(data[~flags.to_numpy()])
        self.prophet_ = model

    def _predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        return np.exp(self.prophet_.predict(pd.DataFrame({"ds": dates}))["yhat"].to_numpy())


VERSION_NAMES = ["1. Baseline", "2. + Spike GMM", "3. + Holidays", "4. Prophet + Fourier",
                 "5. Prophet + Fourier + GMM"]


def model_ladder(country: str, versions: Sequence[int] | None = None) -> list:
    """Picklable factories for the ladder versions (1-5), in order."""
    factories = {
        1: partial(SeasonalWeekday),
        2: partial(SpikeAdjusted),
        3: partial(HolidayAdjusted, country=country),
        4: partial(ProphetFourier, country=country),
        5: partial(ProphetFourier, country=country, clean_spikes=True),
    }
    return [factories[v] for v in (versions or sorted(factories))]
