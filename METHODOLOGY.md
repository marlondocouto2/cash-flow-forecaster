# Methodology

This document explains how the cash-flow forecaster works: how the data is simulated, how each
model version is estimated, how it is evaluated, and what the results do and don't show. The
[README](README.md) has the headline results; the
[walkthrough notebook](notebooks/walkthrough.ipynb) shows each step with charts. All numbers
below come from `python scripts/run_pipeline.py` with the default settings in
[`config.py`](src/cashflow_forecaster/config.py).

## 1. Starting point: the weekday-seasonal base forecast

This project builds on the base forecast from
[Project 1: weekday-seasonal forecaster](https://github.com/marlondocouto2/weekday-seasonal-forecaster).
That model forecasts a day's volume as

```math
\hat{y}_t = L \times W_{\text{weekday}(t)} \times M_{\text{bucket}(t)}
```

where $`L`$ is a recent level, $`W`$ a day-of-week factor and $`M`$ a day-of-month (payday) factor.
It is small and explainable, and in Project 1 it beat naive baselines by 35–58%.

Its estimation code is carried over here unchanged as **version 1, the baseline**. Every later
version is measured against it. Versions 2 and 3 build directly on top of it. Versions 4 and 5
replace its core with Prophet, but keep the same interface, the same evaluation and the same
spike-detection idea, so the gain from each step can be read off one table.

The base forecast breaks on three kinds of day:

- **Random spikes:** one-off large flows. They inflate the level $`L`$, so every ordinary day is
  over-forecast.
- **Scheduled spikes:** a payroll run every other Friday. A weekday factor can only average
  payroll and ordinary Fridays together.
- **Holidays in the receiving country:** money is sent ahead of the holiday, volume drops on
  the day itself, and there is a catch-up afterwards.

Each version below targets one of these.

## 2. Data

### 2.0 Where the data comes from

No data files are stored in the repo. Every run creates the data in memory with
`generate_flows()` in [`data.py`](src/cashflow_forecaster/data.py), which takes under a
second, so the same seed always gives the same data on any machine.

The generator reuses Project 1's building blocks: trend, weekday, payday, annual pattern and
noise, with the same weekday, payday, annual and noise settings. The code is copied rather than
imported, so this repo runs on its own. The data itself is **not** the same as Project 1's:

| | Project 1 | This project |
|---|---|---|
| **Series** | one aggregate series | three corridors (MX, PH, IN), simulated separately |
| **Scale and growth** | \$50M a day, +15% a year | \$25M / \$12M / \$15M a day, +12% / +10% / +18% a year |
| **Seed** | 42 | 42, 43, 44 (one per corridor) |
| **Extra parts** | none | random spikes, biweekly payroll Fridays, holidays |

Both projects can optionally run on the same public series, U.S. Treasury withdrawals (2.3).

### 2.1 Synthetic corridors (default)

Three payout corridors are simulated (U.S. to Mexico, the Philippines and India), daily from
2021-01-01 to 2025-12-31, with seed 42. Each day's volume is a product of known parts:

```math
y_t = T_t \times W_t \times M_t \times A_t \times H_t \times S^{\text{sched}}_t \times S^{\text{rand}}_t \times \varepsilon_t
```

Every part is stored in its own column, so tests can check how well each model recovers it.

| Part | Definition |
|---|---|
| **Trend** $`T_t`$ | $`T_0 (1+g)^{\text{years}}`$, with $`T_0`$ = \$25M / \$12M / \$15M a day and $`g`$ = 12% / 10% / 18% a year (MX / PH / IN) |
| **Weekday** $`W_t`$ | Mon 1.10, Tue 1.00, Wed 0.95, Thu 1.00, Fri 1.25, Sat 0.80, Sun 0.70, rescaled to average 1 |
| **Paydays** $`M_t`$ | ×1.25 on the 1st, ×1.12 on the 2nd, ×1.05 on the 3rd; ×1.20 on the 15th, ×1.06 on the 14th and 16th; ×1.30 on the last business day, ×1.08 the day before |
| **Annual** $`A_t`$ | $`\left(1 + 0.04 \sin\frac{2\pi(d-80)}{365.25}\right)\left(1 + 0.12\, e^{-\frac{1}{2}\left(\frac{d - d_{\text{Dec 20}}}{6}\right)^2}\right)`$, where $`d`$ is the day of the year: a mild yearly wave plus a December peak |
| **Holidays** $`H_t`$ | see 2.2 |
| **Scheduled spikes** $`S^{\text{sched}}_t`$ | ×1.6 every 14 days, on Fridays, from 2021-01-08 (a biweekly payroll run) |
| **Random spikes** $`S^{\text{rand}}_t`$ | with probability $`3/30.44`$ a day (about 3 a month), a multiplier drawn uniformly from 1.5 to 3.0; otherwise 1 |
| **Noise** $`\varepsilon_t`$ | log-normal, $`\sigma = 0.06`$ |

Random and scheduled spikes are deliberately different. Random spikes are a separate,
unpredictable regime. Scheduled spikes repeat on a fixed 14-day cycle, so a model that
represents that cycle can forecast them.

### 2.2 Holiday calendar and effects

Calendars come from the [`holidays`](https://pypi.org/project/holidays/) package for each
receiving country. Mexico's Mother's Day (10 May) is added by hand: it isn't a public holiday,
but it is one of the biggest remittance days of the year. Holidays are split into two tiers.
**Major** holidays are Christmas, New Year, Mother's Day (MX), Diwali and Holi (IN). All other
public holidays are **minor**.

Each date gets a position relative to the nearest holiday, called its *window offset*. When
windows overlap, the first matching rule wins:

1. The date is itself a holiday: offset 0.
2. The previous day was a holiday: offset +1.
3. A major holiday falls 1–3 days later: offset −1 to −3, using the nearest one.

The true multiplicative effects are:

| Tier | −3 | −2 | −1 | 0 (holiday) | +1 |
|---|---|---|---|---|---|
| Major | ×1.25 | ×1.40 | ×1.55 | ×0.60 | ×1.20 |
| Minor | – | – | – | ×0.60 | ×1.15 |

The same calendar code is used to simulate the data and to build model features. Holiday dates
are known in advance, so using future holiday dates in a forecast is not leakage.

### 2.3 Public data (optional)

`--source treasury` runs the same pipeline on U.S. Treasury operating-cash withdrawals from the
Daily Treasury Statement (Fiscal Data API, no key needed). The series is the sum of all
withdrawal categories except public-debt redemptions, on business days only. Two early-2021
days whose totals are zero or negative (accounting adjustments) are dropped. The U.S. federal
calendar supplies the holiday windows. The true spike and holiday components are unknown for
this series, so it can only be used to compare forecast errors.

## 3. Models

Every version is a class with `fit(df)` and `predict(horizon or dates)`. `fit` receives only
rows up to the forecast origin.

### 3.1 Version 1: Baseline (the Project 1 base forecast)

- **Weekday factors $`W`$:** each day is first divided by its month-position factor, then by the
  centred 7-day mean around it. That mean contains exactly one of each weekday, so trend and
  slow seasonal bumps cancel out. The factors are the average ratio for each weekday over the
  last **12 weeks**, normalized to average 1. The centred mean only uses training data: the last
  three days of history simply have no centred value yet.
- **Month-position factors $`M`$:** the same ratio-to-average idea for five day-of-month buckets
  (days 1–3, 4–13, 14–16, 17 to month end minus 3, and the last 3 days), over the last
  **364 days**. They are normalized so the day-weighted average is 1.
- **Alternating estimation:** $`W`$ and $`M`$ distort each other (a payday that falls on a Friday
  inflates Friday), so they are estimated in turn for three passes.
- **Level $`L`$:** the mean of the last **28 days** after dividing out $`W`$ and $`M`$.
- **Forecast:** $`\hat{y}_t = L \cdot W_{\text{weekday}(t)} \cdot M_{\text{bucket}(t)}`$, flat
  in level over the horizon.

### 3.2 Version 2: + spike decomposition with a Gaussian mixture

1. **Normal-day fit.** For each past day, estimate what a normal day would have been:
   $`\tilde{y}_t = \operatorname{median}_{28}(y / (W M))_t \cdot W_t M_t`$. This is a centred
   28-day rolling *median* of the deseasonalized series times the seasonal factors. The median
   keeps spikes from pulling it up.
2. **Residuals.** $`r_t = \log y_t - \log \tilde{y}_t`$, over the last **730 days**.
3. **Mixture.** A two-component `sklearn.mixture.GaussianMixture` is fitted to $`r_t`$. The
   component with the higher mean is the spike regime. A day is labelled a spike when its
   posterior probability of belonging to that component is above **0.5**.
4. **Cleaning and refit.** Spike days are replaced by their normal-day fit $`\tilde{y}_t`$, and
   $`W`$, $`M`$ and $`L`$ are re-estimated on the cleaned series as in 3.1.
5. **Spike allowance.** The forecast itself is the clean baseline. Separately, the model reports
   the expected spike contribution $`\hat{y}_t \cdot p \cdot (\bar{m} - 1)`$, where $`p`$ is the
   share of days flagged and $`\bar{m}`$ the average spike multiplier. It is meant for sizing a
   buffer, not for the point forecast.

### 3.3 Version 3: + holiday effects

- **Excluding holiday windows.** Holiday windows are left out of the mixture fit and treated
  like spikes during cleaning, so pre-holiday surges aren't mistaken for random spikes and
  holidays don't distort the baseline.
- **Effects.** For each window position $`k`$ (tier and offset), the effect is estimated from
  history as the median ratio of actual to normal-day fit, shrunk toward 1 when there are few
  past occurrences:

```math
\hat{h}_k = 1 + \frac{n_k}{n_k + \kappa}\left(\operatorname{median}_{t \in k} \frac{y_t}{\tilde{y}_t} - 1\right), \qquad \kappa = 1
```

where $`n_k`$ is the number of past occurrences of position $`k`$.

- **Forecast:** $`\hat{y}_t = L \cdot W \cdot M \cdot \hat{h}_{k(t)}`$, with $`\hat{h} = 1`$
  outside holiday windows.

### 3.4 Version 4: Prophet + Fourier seasonality + holidays

A different model family, fitted with `prophet` on log volume over the last **3 years**:

```math
\log y_t = g(t) + \sum_{P} s_P(t) + h(t) + e_t
```

- **Trend $`g(t)`$:** Prophet's piecewise-linear trend.
- **Seasonalities $`s_P(t)`$:** Fourier series,
  $`s_P(t) = \sum_{n=1}^{N} \left[a_n \cos\frac{2\pi n t}{P} + b_n \sin\frac{2\pi n t}{P}\right]`$,
  with these periods $`P`$ and orders $`N`$:

  | Seasonality | Period $`P`$ (days) | Order $`N`$ |
  |---|---|---|
  | Weekly | 7 | 3 |
  | Biweekly | 14 | 7 |
  | Monthly | 30.44 | 15 |
  | Yearly | 365.25 | 20 |

  With order 7, the 14-day series can represent any repeating shape over the 14-day cycle,
  including a sharp peak on one day. So the payroll Fridays are learned as seasonality and
  forecast.
- **Holidays $`h(t)`$:** the same calendar, passed to Prophet by tier. Major holidays use a
  window from −3 to +1 days, minor ones from 0 to +1. Prophet learns one effect per tier and
  offset.
- **Forecast:** $`\hat{y}_t = \exp(\widehat{\log y_t})`$. This is a median-type forecast, which
  suits MAE.

### 3.5 Version 5: Prophet + Fourier + GMM

Version 4 with the mixture from 3.2, applied to *Prophet's* residuals:

1. Fit Prophet as in 3.4 and compute the in-sample residuals.
2. Fit the two-component mixture on those residuals, with holiday windows excluded.
3. Drop the days flagged as spikes and refit Prophet.

The order matters. Prophet's seasonality already explains the payroll Fridays, so they don't
look unusual in its residuals and the mixture leaves them alone. In Mexico it flags 0 of 69
payroll Fridays. Only spikes the seasonality can't explain are removed.

## 4. Evaluation design

### 4.1 Walk-forward backtest

- **Test period:** the last 365 days (calendar 2025), for each corridor separately.
- **Origins:** every day from 30 days before the test period to its end. At each origin, the
  model is fitted on data up to and including that day, and forecasts 1, 7 and 30 steps ahead.
  A forecast is kept when its target date falls inside the test period.
- **Refits:** versions 1–3 are refit at every origin. Prophet versions are refit every 7th
  origin, since each fit takes about 0.25 seconds; in between, the latest fit is reused. Either
  way, no model ever sees data after its origin.
- **Scale:** 1,095 scored corridor-days per version and horizon. The five versions run in
  parallel across corridors.

### 4.2 Day types

Each scored day is labelled using the simulated truth. When more than one label applies, the
first one in this order wins:

| Order | Day type | Days | Description |
|---|---|---|---|
| 1 | Holiday window | 105 | offsets −3 to +1 |
| 2 | Random spike | 82 | |
| 3 | Scheduled spike | 66 | payroll Fridays |
| 4 | Normal | 842 | everything else |

### 4.3 Metrics

Accuracy is measured per corridor-day, with error $`e_t = y_t - \hat{y}_t`$, where a positive
error is a shortfall:

- **MAE** $`= \overline{|e_t|}`$, **MSE** $`= \overline{e_t^2}`$, **RMSE** $`= \sqrt{\text{MSE}}`$,
  overall and for each day type.
- **Bias** on normal days: $`\overline{\hat{y}_t / y_t - 1}`$.

Dollar metrics are computed on the **daily total across corridors**, since that is what treasury
funds:

- **Required buffer** $`B = Q_{0.95}\left(\max(Y_t - \hat{Y}_t, 0)\right)`$: the cash to hold on
  top of the forecast to cover shortfalls on 95% of days.
- **Cash held** $`= \overline{\hat{Y}_t} + B`$: the like-for-like cost of 95% protection. The
  buffer alone rewards a forecast that is biased high, because the bias acts as a hidden
  cushion. Cash held doesn't.
- **Idle-cash cost** $`= \overline{\max(\hat{Y}_t - Y_t, 0)} \times 5\%`$ per year: the
  annualized cost of over-funding.

MAPE is not reported. Accuracy is compared in dollars, and the dollar metrics already capture
the business impact.

### 4.4 Leakage controls

- `walk_forward` passes each model only `df.iloc[: origin + 1]`.
- Rolling windows inside `fit`, including the centred ones, run on that slice only.
- Only the holiday calendar extends past the origin, and it is known in advance.
- Tests alter every value after a cutoff and check that all forecasts made at or before the
  cutoff are unchanged. This covers versions 1–3, and Prophet with its weekly refits.

## 5. Validation of the method

Because the data is simulated, each component can be checked against the truth. These checks
run as tests (`pytest`):

| Check | Result on the default data |
|---|---|
| **Spike recall**, version 3 (random spikes found, outside holiday windows) | MX 97%, PH 97%, IN 87% (test requires ≥ 80%) |
| **Spike recall**, version 5 | MX 100%, PH 100%, IN 96% |
| **Payroll Fridays wrongly flagged** as random spikes, version 5 | 0/69, 1/69, 1/60 (test requires < 10%) |
| **Holiday effects** vs truth, version 3 | mean absolute error 2–5%; worst case PH major holiday day, +14%; all within ±15% |
| **Payroll Friday ratio**, version 5 forecast | within 15% of the true ×1.6 (version 3 stays below ×1.25) |
| **Leakage** | forecasts unchanged when future data is altered, for all versions |

The worst holiday case has a known cause. Two of the ten past Philippine major-holiday days are
random spikes, and 1 January is also a payday whose true lift (×1.25) is larger than the
bucket's average lift. The estimate absorbs part of that difference, which is what the forecast
needs.

## 6. Results

Walk-forward, 7 days ahead. Daily totals average \$111M.

| Version | MAE | vs previous | Normal | Random spike | Scheduled | Holiday | Bias, normal days | Cash held | Idle cost / yr |
|---|---|---|---|---|---|---|---|---|---|
| 1. Baseline | \$8.80M | | \$5.07M | \$36.0M | \$11.0M | \$16.0M | +13.1% | \$174.7M | \$503k |
| 2. + Spike GMM | \$7.61M | −13.6% | \$2.98M | \$38.9M | \$15.4M | \$15.4M | +2.9% | \$174.2M | \$215k |
| 3. + Holidays | \$6.87M | −9.6% | \$2.77M | \$38.8M | \$15.4M | \$9.5M | +2.8% | \$170.7M | \$183k |
| 4. Prophet + Fourier | \$6.37M | −7.3% | \$3.15M | \$36.1M | \$6.2M | \$9.1M | +8.1% | \$165.5M | \$290k |
| 5. Prophet + Fourier + GMM | \$5.50M | −13.8% | \$1.98M | \$38.3M | \$4.3M | \$8.9M | +0.5% | \$164.2M | \$105k |

**How to read it:**

- **Each step fixes the day type it targets.** Spike cleaning helps normal days, holiday effects
  help holiday windows, and Fourier terms help scheduled spikes.
- **Version 4 still runs 8% high on normal days.** Random spikes inflate Prophet's fit, so a
  Fourier model alone is not enough. Version 5 combines both ideas: seasonality for what
  repeats, the mixture for what doesn't.
- **Random spike days don't improve in any version.** They are unpredictable by construction.
  Cleaning the level even makes them slightly worse, because an inflated level happened to sit
  closer to them.
- **The cash-held saving is real.** From version 1 to version 5, the same 95% protection needs
  \$10.5M less cash a day.

**Real data:** on U.S. Treasury withdrawals over the 12 months to November 2026, versions 2–3
are about 5% *worse* than the baseline. Version 4 is 12.5% better than version 3, and version
5 ends 9.7% below the baseline. The Treasury's large days are mostly scheduled (tax dates,
benefit payments), so treating them as random spikes hurts, and representing them as
seasonality helps. This matches the synthetic finding.

## 7. Assumptions and limitations

- **The data is simulated by design.** The spike and holiday structure is known, which makes
  recovery testable. Real flows have more regimes than these.
- **The holiday tiers and the Mother's Day addition are judgement calls.** On real data they
  would come from the business.
- **Prophet refits weekly in the backtest.** This slightly disadvantages versions 4–5 at the
  1-day horizon.
- **The required buffer is set by random spikes.** No point forecast removes that risk. The
  mixture's spike probability and size are the right inputs for sizing it separately, as cash
  or as a credit line. That comparison isn't built yet.
- **Fourier periods and orders are fixed.** They are chosen from the known structure (7, 14,
  30.44 and 365.25 days), not learned. On real data, the periods should be found by
  cross-validation or from the spectrum of the residuals.

## 8. Reproducing

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py      # all five versions, about a minute
pytest                              # about 45 seconds
```

Seeds, windows and horizons are set in [`config.py`](src/cashflow_forecaster/config.py), and
package versions are pinned in [`requirements.txt`](requirements.txt). The notebook version of
this document, [`notebooks/methodology.ipynb`](notebooks/methodology.ipynb), is generated from
this file by `python scripts/build_methodology_notebook.py`.
