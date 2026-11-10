"""One command: generate data -> fit the model ladder -> backtest -> save figures and results.

    python scripts/run_pipeline.py                   # synthetic corridors (default, offline)
    python scripts/run_pipeline.py --source treasury # U.S. Treasury withdrawals (needs network)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Many tiny model fits: one thread each is as fast as many and leaves the machine free.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from cashflow_forecaster import plots  # noqa: E402
from cashflow_forecaster.config import CONFIG  # noqa: E402
from cashflow_forecaster.data import generate_flows, load_treasury_withdrawals  # noqa: E402
from cashflow_forecaster.evaluate import backtest_all, day_type_counts, ladder_table  # noqa: E402
from cashflow_forecaster.models import HolidayAdjusted  # noqa: E402

HEADLINE_HORIZON = 7


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["synthetic", "treasury"], default="synthetic")
    args = parser.parse_args()

    if args.source == "synthetic":
        flows = generate_flows(CONFIG.start, CONFIG.end, CONFIG.seed, CONFIG.corridors)
        out_dir, test_periods = CONFIG.reports_dir, CONFIG.backtest_days
        zoom = [("MX", "Mexico: Mother's Day (10 May 2025)", "2025-05-03", "2025-05-15"),
                ("IN", "India: Diwali (20 Oct 2025)", "2025-10-13", "2025-10-25")]
        gmm_corridor = "MX"
    else:
        flows = load_treasury_withdrawals(start=CONFIG.start)
        out_dir, test_periods = CONFIG.reports_dir / "treasury", 252
        last = flows["date"].max()
        zoom = [("US", "U.S. Treasury: Christmas 2025", "2025-12-15", "2026-01-09")]
        if last < pd.Timestamp(zoom[0][3]):
            zoom = []
        gmm_corridor = "US"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data: {flows['corridor'].nunique()} series, {len(flows):,} rows, "
          f"{flows['date'].min():%Y-%m-%d} to {flows['date'].max():%Y-%m-%d}")

    t0 = time.time()
    bt = backtest_all(flows, CONFIG.horizons, test_periods)
    print(f"Backtest: {len(bt):,} forecasts in {time.time() - t0:.0f}s")
    print("Scored corridor-days by type:", day_type_counts(bt).to_dict())

    ladder = ladder_table(bt)
    ladder.to_csv(out_dir / "results.csv", index=False, float_format="%.2f")
    bt.to_csv(out_dir / "backtest_forecasts.csv.gz", index=False, float_format="%.2f")

    plots.plot_ladder(ladder, HEADLINE_HORIZON, fig_dir / "model_ladder.png")
    plots.plot_table(ladder, HEADLINE_HORIZON, fig_dir / "results_table.png")
    if zoom:
        plots.plot_holiday_zoom(bt, zoom, HEADLINE_HORIZON, fig_dir / "holiday_zoom.png")
    series = flows[flows["corridor"] == gmm_corridor][["date", "volume"]]
    model = HolidayAdjusted(country=gmm_corridor).fit(series)
    plots.plot_residual_gmm(model, fig_dir / "residual_gmm.png",
                            f"Residuals and fitted mixture, {gmm_corridor} corridor "
                            f"(last {CONFIG.gmm_history_days} days)")

    show = ladder.drop(columns=["MSE"]).copy()
    money = [c for c in show.columns if c not in ("horizon", "model")]
    show[money] = (show[money] / 1e6).round(3)
    print("\nModel ladder ($M):")
    print(show.to_string(index=False))
    print(f"\nSaved results and figures to {out_dir}")


if __name__ == "__main__":
    main()
