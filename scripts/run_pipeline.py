"""One command: generate data -> fit every model version -> backtest -> figures and results.

    python scripts/run_pipeline.py                     # all 5 versions, synthetic data (~1 min)
    python scripts/run_pipeline.py --versions 1,3,5    # only some versions
    python scripts/run_pipeline.py --source treasury   # U.S. Treasury withdrawals (needs network)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Many tiny model fits: one thread each is as fast as many and leaves cores for parallel runs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from cashflow_forecaster import plots  # noqa: E402
from cashflow_forecaster.config import CONFIG  # noqa: E402
from cashflow_forecaster.data import generate_flows, load_treasury_withdrawals  # noqa: E402
from cashflow_forecaster.evaluate import backtest_all, day_type_counts, ladder_table  # noqa: E402
from cashflow_forecaster.models import VERSION_NAMES, ProphetFourier  # noqa: E402

HEADLINE_HORIZON = 7


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["synthetic", "treasury"], default="synthetic")
    parser.add_argument("--versions", default="1,2,3,4,5",
                        help="comma-separated versions to run: " +
                             "; ".join(VERSION_NAMES))
    args = parser.parse_args()
    versions = sorted({int(v) for v in args.versions.split(",")})

    if args.source == "synthetic":
        flows = generate_flows(CONFIG.start, CONFIG.end, CONFIG.seed, CONFIG.corridors)
        out_dir, test_periods = CONFIG.reports_dir, CONFIG.backtest_days
        holiday_panels = [("MX", "Mexico: Mother's Day (10 May 2025)", "2025-05-03", "2025-05-15"),
                          ("IN", "India: Diwali (20 Oct 2025)", "2025-10-13", "2025-10-25")]
        payroll_panels = [("MX", "Mexico: biweekly payroll Fridays (shaded)",
                           "2025-08-25", "2025-10-05")]
        gmm_corridor = "MX"
    else:
        flows = load_treasury_withdrawals(start=CONFIG.start)
        out_dir, test_periods = CONFIG.reports_dir / "treasury", 252
        holiday_panels = [("US", "U.S. Treasury: Christmas 2025", "2025-12-15", "2026-01-09")]
        if flows["date"].max() < pd.Timestamp(holiday_panels[0][3]):
            holiday_panels = []
        payroll_panels = []
        gmm_corridor = "US"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data: {flows['corridor'].nunique()} series, {len(flows):,} rows, "
          f"{flows['date'].min():%Y-%m-%d} to {flows['date'].max():%Y-%m-%d}")
    print("Versions:", ", ".join(VERSION_NAMES[v - 1] for v in versions))

    t0 = time.time()
    bt = backtest_all(flows, CONFIG.horizons, test_periods, versions=versions)
    print(f"Backtest: {len(bt):,} forecasts in {time.time() - t0:.0f}s")
    print("Scored corridor-days by type:", day_type_counts(bt).to_dict())

    ladder = ladder_table(bt)
    ladder.to_csv(out_dir / "results.csv", index=False, float_format="%.2f")
    bt.to_csv(out_dir / "backtest_forecasts.csv.gz", index=False, float_format="%.2f")

    plots.plot_accuracy_steps(ladder, HEADLINE_HORIZON, fig_dir / "accuracy_steps.png")
    plots.plot_ladder(ladder, HEADLINE_HORIZON, fig_dir / "model_ladder.png")
    plots.plot_table(ladder, HEADLINE_HORIZON, fig_dir / "results_table.png")
    present = [v for v in VERSION_NAMES if v in set(bt["model"])]
    compare = [v for v in (VERSION_NAMES[0], VERSION_NAMES[2], VERSION_NAMES[4]) if v in present]
    if holiday_panels:
        plots.plot_zoom(bt, holiday_panels, HEADLINE_HORIZON, compare, fig_dir / "holiday_zoom.png")
    if payroll_panels:
        plots.plot_zoom(bt, payroll_panels, HEADLINE_HORIZON, compare,
                        fig_dir / "scheduled_zoom.png", shade="scheduled")
    series = flows[flows["corridor"] == gmm_corridor][["date", "volume"]]
    final = ProphetFourier(country=gmm_corridor, clean_spikes=True).fit(series)
    plots.plot_residual_gmm(final, fig_dir / "residual_gmm.png",
                            f"Final model's residuals and fitted mixture, {gmm_corridor} corridor")

    h = ladder[ladder["horizon"] == HEADLINE_HORIZON]
    print(f"\nWhat each version adds ({HEADLINE_HORIZON} days ahead, MAE per corridor-day):")
    for _, r in h.iterrows():
        step = "" if pd.isna(r["MAE_vs_previous_pct"]) else f"  {r['MAE_vs_previous_pct']:+.1f}% vs previous"
        print(f"  {r['model']:<28} ${r['MAE'] / 1e6:6.2f}M{step}")
    show = ladder.drop(columns=["MSE"]).copy()
    money = [c for c in show.columns if c not in ("horizon", "model") and not c.endswith("_pct")]
    show[money] = (show[money] / 1e6).round(2)
    print("\nFull ladder, all horizons ($M):")
    print(show.to_string(index=False))
    print(f"\nSaved results and figures to {out_dir}")


if __name__ == "__main__":
    main()
