import numpy as np
import pandas as pd
import pytest

from cashflow_forecaster.data import TRUE_HOLIDAY_EFFECTS, generate_flows
from cashflow_forecaster.evaluate import (
    accuracy,
    backtest_all,
    idle_cash_cost,
    ladder_table,
    required_buffer,
    walk_forward,
)
from cashflow_forecaster.features import holiday_window
from cashflow_forecaster.models import HolidayAdjusted, SeasonalWeekday, SpikeAdjusted

CORRIDORS = ["MX", "PH", "IN"]


@pytest.fixture(scope="module")
def flows() -> pd.DataFrame:
    """The pipeline's default synthetic data (5 years, seed 42)."""
    return generate_flows()


@pytest.fixture(scope="module")
def fitted(flows) -> dict[str, HolidayAdjusted]:
    out = {}
    for c in CORRIDORS:
        series = flows[flows["corridor"] == c][["date", "volume"]]
        out[c] = HolidayAdjusted(country=c).fit(series)
    return out


def _series(flows, corridor):
    return flows[flows["corridor"] == corridor].reset_index(drop=True)


# ----------------------------------------------------------------- data and calendar
def test_generator_is_reproducible_and_volume_is_product_of_parts():
    a = generate_flows("2023-01-01", "2023-06-30", seed=3)
    b = generate_flows("2023-01-01", "2023-06-30", seed=3)
    pd.testing.assert_frame_equal(a, b)
    parts = ["trend", "weekday_factor", "month_factor", "annual_factor", "holiday_factor",
             "spike_factor", "noise"]
    np.testing.assert_allclose(a["volume"], a[parts].prod(axis=1))


def test_holiday_window_precedence():
    w = holiday_window(pd.to_datetime(["2025-05-07", "2025-05-09", "2025-05-10", "2025-05-11"]), "MX")
    assert w["holiday_offset"].tolist() == [-3, -1, 0, 1]
    assert (w["holiday_tier"] == "major").all()
    # Philippines: Christmas Eve is itself a (minor) holiday, so it is not a "day before".
    w = holiday_window(pd.to_datetime(["2025-12-23", "2025-12-24", "2025-12-25", "2025-12-26"]), "PH")
    assert w["holiday_tier"].tolist() == ["major", "minor", "major", "major"]
    assert w["holiday_offset"].tolist() == [-2, 0, 0, 1]


# ----------------------------------------------------------------- recovery
def test_weekday_factors_average_to_one(fitted):
    for m in fitted.values():
        assert m.weekday_factors_.mean() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("corridor", CORRIDORS)
def test_gmm_recovers_at_least_80pct_of_spike_days(flows, fitted, corridor):
    model = fitted[corridor]
    truth = _series(flows, corridor).set_index("date").loc[model.spike_flags_.index]
    outside_holidays = truth["holiday_tier"].isna()
    true_spikes = truth["is_spike"] & outside_holidays
    found = model.spike_flags_ & true_spikes
    assert found.sum() / true_spikes.sum() >= 0.80
    # And it isn't simply flagging everything.
    assert model.spike_flags_[outside_holidays].mean() < 0.15


@pytest.mark.parametrize("corridor", CORRIDORS)
def test_holiday_effects_within_15pct_of_truth(fitted, corridor):
    est = fitted[corridor].holiday_effects_
    assert len(est) == len(TRUE_HOLIDAY_EFFECTS)
    for key, value in est.items():
        tier, offset = key.split(":")
        truth = TRUE_HOLIDAY_EFFECTS[(tier, int(offset))]
        assert abs(value / truth - 1) < 0.15, (key, value, truth)


def test_spike_cleaning_lowers_the_level(flows):
    series = _series(flows, "MX")[["date", "volume"]]
    raw = SeasonalWeekday().fit(series)
    clean = SpikeAdjusted().fit(series)
    assert clean.level_ < raw.level_  # spikes no longer inflate the baseline
    assert clean.expected_spike(pd.date_range("2026-01-01", periods=7)).gt(0).all()


# ----------------------------------------------------------------- leakage
@pytest.mark.parametrize("factory", [SeasonalWeekday, SpikeAdjusted,
                                     lambda: HolidayAdjusted(country="MX")])
def test_no_leakage_future_data_does_not_change_forecasts(flows, factory):
    df = _series(flows, "MX")
    df = df[df["date"] >= "2023-01-01"].reset_index(drop=True)
    cutoff = pd.Timestamp("2025-10-31")
    altered = df.copy()
    future = altered["date"] > cutoff
    altered.loc[future, "volume"] = altered.loc[future, "volume"] * 3.0 + 1e7

    a = walk_forward(df, factory, horizons=(1, 7, 30), test_periods=90)
    b = walk_forward(altered, factory, horizons=(1, 7, 30), test_periods=90)
    known = a["origin"] <= cutoff
    assert known.sum() > 0 and (~known).sum() > 0
    np.testing.assert_allclose(a.loc[known, "forecast"], b.loc[known, "forecast"])
    assert not np.allclose(a.loc[~known, "forecast"], b.loc[~known, "forecast"])


# ----------------------------------------------------------------- metrics
def test_metric_definitions():
    m = accuracy([100.0, 200.0], [110.0, 180.0])
    assert m["MAE"] == pytest.approx(15.0) and m["MSE"] == pytest.approx(250.0)
    actual = np.full(20, 100.0)
    forecast = np.array([90.0] * 10 + [110.0] * 10)
    assert required_buffer(actual, forecast, q=0.95) == pytest.approx(10.0)
    # Over-forecast by 10 on half the days -> average over-forecast 5 -> 5 x 5% a year.
    assert idle_cash_cost(actual, forecast, annual_rate=0.05) == pytest.approx(0.25)


# ----------------------------------------------------------------- the ladder
@pytest.fixture(scope="module")
def ladder(flows) -> pd.DataFrame:
    """A shorter version of the pipeline's backtest: 7-day horizon, all corridors."""
    bt = backtest_all(flows, horizons=(7,), test_periods=365)
    return ladder_table(bt).set_index("model")


def test_each_step_reduces_error_on_its_target_days(ladder):
    base, gmm, full = ladder.loc["1. Baseline"], ladder.loc["2. + Spike GMM"], ladder.loc["3. + Holidays"]
    # Spike cleaning: spikes stop inflating the forecast on ordinary days.
    assert gmm["MAE_normal"] < 0.8 * base["MAE_normal"]
    # Holiday effects: error on holiday windows falls sharply.
    assert full["MAE_holiday"] < 0.8 * gmm["MAE_holiday"]
    # Overall, each step helps, and idle cash falls.
    assert base["MAE"] > gmm["MAE"] > full["MAE"]
    assert base["idle_cash_cost_per_year"] > gmm["idle_cash_cost_per_year"] > full["idle_cash_cost_per_year"]
