# Daily cash-flow forecaster with spike and holiday handling

A money-transfer business has to pre-fund each payout corridor every morning. Two things
break a simple seasonal forecast. **Irregular spikes** (a large corporate payout, a promotion)
inflate the average the forecast is built on, so ordinary days get over-funded. **Holidays** in
the receiving country cause a surge before, a drop on the day, and a catch-up after. This
project starts from the simple weekday-seasonal model in
[Project 1](https://github.com/marlondocouto2/weekday-seasonal-forecaster) and fixes each
problem in turn: **diagnose → fix → measure → dollars saved**.

## Headline result

Across three corridors averaging $105M a day, the full model **cuts average forecast error
by 30%**, from $7.5M to $5.3M per corridor-day at 7 days ahead. Each step does its own job:

- **Spike cleaning:** removes a **10% over-forecast on ordinary days**. Error on normal days
  falls by 43%.
- **Holiday effects:** cut error in **holiday windows by 46%**.

Together they lower the **idle-cash cost from $418k to $100k a year**, a saving of **$318k a
year** at a 5% cost of capital.

![Error by model version and idle-cash cost](reports/figures/model_ladder.png)

## Data

All data is **simulated**: three corridors from the U.S. to Mexico, the Philippines and
India, daily from 2021-01-01 to 2025-12-31
([`data.py`](src/cashflow_forecaster/data.py)). Each is the product of known parts, and every
part is saved in its own column so tests can check recovery:

| Part | What is simulated |
|---|---|
| **Trend** | $25M / $12M / $15M a day at the start, growing 10–18% a year |
| **Weekday, paydays, annual** | the same as Project 1: Fri ×1.25, Sun ×0.70; lifts on the 1st, 15th and last business day; a December peak |
| **Spikes** | about 3 a month per corridor, at random; size 1.5–3× a normal day |
| **Holidays** | the receiving country's calendar from the [`holidays`](https://pypi.org/project/holidays/) package, plus Mexico's Mother's Day (10 May), added by hand |
| **Noise** | about 6% log-normal |

The true holiday effects depend on the holiday's tier:

| Tier | 3 days before | 2 days before | 1 day before | Holiday | Day after |
|---|---|---|---|---|---|
| **Major** (Christmas, New Year, Mother's Day, Diwali, Holi) | ×1.25 | ×1.40 | ×1.55 | ×0.60 | ×1.20 |
| **Minor** (all other public holidays) | – | – | – | ×0.60 | ×1.15 |

**Public data (optional):** `--source treasury` runs the same ladder on U.S. Treasury
operating-cash withdrawals, using the U.S. federal holiday calendar. Results are under
*Real data* below.

## Approach

Every version keeps the same `fit(df)` / `predict(horizon or dates)` interface
([`models.py`](src/cashflow_forecaster/models.py)).

1. **Baseline:** Project 1's `SeasonalWeekday`: a 28-day level × weekday factor ×
   day-of-month factor.
2. **+ Spike decomposition (Gaussian mixture):**
   - **Normal-day fit:** for each past day, what a normal day would have been (a rolling
     *median* level × the seasonal factors).
   - **Residuals:** the log of actual over normal-day fit, for the last two years.
   - **Mixture:** a two-component `GaussianMixture` is fitted on the residuals. Days most
     likely to belong to the high component are labelled spikes.
   - **Clean refit:** spike days are replaced by their normal-day value, and the level and
     factors are re-estimated, so spikes stop inflating the baseline.
   - **Spike allowance:** the model also reports the expected spike contribution (spike
     probability × average spike size), for sizing a buffer.
3. **+ Holiday effects:**
   - **Features:** days before and after each holiday in the receiving country, and the
     holiday's tier.
   - **Effects:** each window position's effect is the median of actual ÷ normal-day fit
     over past occurrences. It is shrunk toward 1.0 when there are few: `1 + n/(n+1) × (median − 1)`.
   - **Applied:** the effects multiply the clean forecast. Holiday windows are kept out of
     the mixture and the baseline, so pre-holiday surges aren't mistaken for spikes.
4. **No leakage:** every model is refit at every origin using data up to that day only.
   Holiday calendars are known in advance, so using future holiday dates is allowed.

![Residual histogram with fitted mixture components](reports/figures/residual_gmm.png)

## Evaluation

**Backtest:** a walk-forward backtest over the last 12 months (calendar 2025). Each version
is refit at every daily origin and forecasts 1, 7 and 30 days ahead, giving 1,095 scored
corridor-days per version and horizon. Code:
[`evaluate.py`](src/cashflow_forecaster/evaluate.py).

**Day types:** errors are broken down by the type of day being forecast:

- **Normal:** 908 days.
- **Spike:** 82 days.
- **Holiday window:** 105 days (3 days before through 1 day after).

**Dollar metrics:** computed on the daily total across corridors, since that's what gets
funded.

- **Required buffer:** the 95th percentile of the daily shortfall (`max(actual − forecast, 0)`).
- **Average cash held:** forecast + required buffer. This is the cash set aside to cover 95%
  of days.
- **Idle-cash cost:** average over-forecast × 5% a year.

**Model ladder, 7 days ahead** (all horizons are in [`reports/results.csv`](reports/results.csv)):

![Model ladder table](reports/figures/results_table.png)

### What each step fixed

- **Spikes were inflating the baseline.** About 1 day in 10 is a spike, at roughly twice
  normal volume. A level built from a plain average therefore runs about 10% high on every
  ordinary day. After the mixture removes them, the bias on normal days falls from **+10.4%
  to +0.3%**, and normal-day error drops from **$4.27M to $2.43M**.
- **Holidays were invisible to the baseline.** It predicted an ordinary Friday on the day
  before Mother's Day. The holiday module learns the surge, the drop and the catch-up from
  past years, and holiday-window error falls from **$14.9M to $8.1M**.

![Actual vs baseline vs full model around two holidays](reports/figures/holiday_zoom.png)

### What it can't fix, and why the buffer goes up

- **Spike days get slightly worse** ($33.4M → $35.9M). Spikes are random, so no calendar
  model can predict them. The baseline's inflated level just happened to sit closer to them.
- **The 95% buffer rises** ($57.0M → $64.2M). On about 23% of days at least one corridor
  spikes, so the 95th-percentile shortfall is set by spikes. The baseline's 10% over-forecast
  was quietly acting as part of that buffer. Judged on the buffer alone, a biased forecast
  looks better than it is.
- **Total cash held barely moves.** The like-for-like comparison is average cash held
  (forecast + buffer), which falls slightly from **$161.5M to $159.8M**.
- **Where the real saving is:** cash is now right-sized on the 75% of days that are normal,
  and the spike risk is a separate, explicit number (the GMM's spike probability × size).
  That's the input for deciding whether to hold spike cover as cash or as a credit line.

### Real data

On U.S. Treasury withdrawals (`--source treasury`, 12 months to November 2026), spike
cleaning lowers the 7-day required buffer from $53.1B to $48.0B. It also lowers average cash
held from $86.6B to $79.5B. However, MAE rises by about 5%.

Treasury's large days are mostly **scheduled**, not random: tax dates and benefit payment
days. A mixture model treats them as random spikes. The fix is to model them as known
calendar events, as the holiday module does. On this data the holiday step is roughly
neutral: the Treasury publishes no statement on federal holidays, so only the days around
them can be learned.

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
python scripts/run_pipeline.py && pytest
```

The pipeline takes about 35 seconds. It writes `reports/results.csv`, every forecast in
`reports/backtest_forecasts.csv.gz`, and the charts in `reports/figures/`. Add
`--source treasury` to run on public data (needs internet; results go to
`reports/treasury/`). Settings are in [`config.py`](src/cashflow_forecaster/config.py).
For a guided tour, see [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb).

### Tests

- **Spike recovery:** the mixture finds at least 80% of the true spike days in every
  corridor. Measured recall is 100%, 100% and 92.5%.
- **Holiday recovery:** every estimated holiday effect is within ±15% of the true effect.
  The mean error is 2–5%. The largest gap is the Philippines' major-holiday drop: 0.69
  against a true 0.60. Two of its ten examples are spike days, and 1 January is also a
  payday.
- **No leakage:** for all three versions, forecasts made at any origin are unchanged when
  every later value is altered.
- **The ladder:**
  - Spike cleaning cuts normal-day error by more than 20%.
  - Holiday effects cut holiday-window error by more than 20%.
  - Overall MAE and idle-cash cost fall at each step.
- **Calendar and metrics:** the holiday-window precedence rules and the metric definitions
  are checked directly.

## What I'd do next

- **Known big days:** model scheduled large days (tax dates, benefit payments) as calendar
  events. This is the main gap on the Treasury data.
- **Spike-aware buffer:** size a separate spike buffer from the mixture (probability ×
  size distribution) and compare holding it as cash against a credit line.
- **Library comparison:** compare with statsmodels `UnobservedComponents` or Prophet using
  the same holiday regressors.
- **Faster refits:** refit the mixture and holiday effects weekly rather than daily; they
  change slowly.
- **Next project:** use these volume forecasts as the input for FX margin
  (`fx-margin-forecaster`).
