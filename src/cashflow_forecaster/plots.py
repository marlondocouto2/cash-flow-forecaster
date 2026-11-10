"""Static charts for the README (PNG, matplotlib)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

# Fixed colour per model version (colour follows the entity, never its rank).
VERSIONS = ["1. Baseline", "2. + Spike GMM", "3. + Holidays"]
COLORS = {"1. Baseline": "#eb6834", "2. + Spike GMM": "#1baf7a", "3. + Holidays": "#2a78d6"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
    "legend.frameon": False, "legend.labelcolor": INK_2,
})


def _money(ax, axis: str = "y", unit: str = "M") -> None:
    scale = 1e6 if unit == "M" else 1e3
    fmt = plt.FuncFormatter(lambda v, _: f"${v / scale:,.0f}{unit}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ladder(ladder: pd.DataFrame, horizon: int, path: Path) -> Path:
    """Error by model version: MAE by day type, plus the annual idle-cash cost."""
    t = ladder[ladder["horizon"] == horizon].set_index("model").reindex(VERSIONS)
    groups = [("MAE", "All days"), ("MAE_normal", "Normal days"),
              ("MAE_spike", "Spike days"), ("MAE_holiday", "Holiday windows")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), width_ratios=[2.6, 1])
    x = np.arange(len(groups))
    w = 0.8 / len(VERSIONS)
    for i, v in enumerate(VERSIONS):
        vals = [t.at[v, c] for c, _ in groups]
        bars = ax.bar(x + (i - 1) * w, vals, w * 0.92, color=COLORS[v], label=v, zorder=2)
        for b, val in zip(bars, vals):
            ax.annotate(f"{val / 1e6:.1f}", (b.get_x() + b.get_width() / 2, val),
                        xytext=(0, 2), textcoords="offset points", ha="center",
                        fontsize=7.5, color=INK_2)
    ax.set_xticks(x, [g for _, g in groups])
    ax.tick_params(axis="x", length=0)
    ax.set_title(f"Average error per corridor-day, {horizon}-day-ahead ($M)")
    _money(ax)
    ax.legend(ncols=3, loc="upper left", bbox_to_anchor=(0, -0.08))

    cost = t["idle_cash_cost_per_year"]
    bars = ax2.bar(range(len(VERSIONS)), cost, 0.6, color=[COLORS[v] for v in VERSIONS], zorder=2)
    for b, val in zip(bars, cost):
        ax2.annotate(f"${val / 1e3:,.0f}k", (b.get_x() + b.get_width() / 2, val),
                     xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax2.set_xticks(range(len(VERSIONS)), ["Baseline", "+ Spike\nGMM", "+ Holidays"])
    ax2.tick_params(axis="x", length=0)
    ax2.set_title("Idle-cash cost per year")
    _money(ax2, unit="k")
    fig.text(0.125, 1.0, "Lower is better", ha="left", fontsize=9, color=INK_2)
    return _save(fig, path)


def plot_holiday_zoom(bt: pd.DataFrame, panels: list[tuple[str, str, str, str]],
                      horizon: int, path: Path) -> Path:
    """Actual vs baseline vs full model around holidays. panels: (corridor, title, start, end)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4), squeeze=False)
    for ax, (corridor, title, start, end) in zip(axes[0], panels):
        s = bt[(bt["corridor"] == corridor) & (bt["horizon"] == horizon)
               & bt["date"].between(start, end)]
        actual = s.drop_duplicates("date").set_index("date")
        ax.plot(actual.index, actual["actual"], color=INK, lw=1.4, marker="o", ms=3.5,
                label="Actual", zorder=4)
        for v in (VERSIONS[0], VERSIONS[2]):
            f = s[s["model"] == v].set_index("date")["forecast"]
            ax.plot(f.index, f.values, color=COLORS[v], lw=2, label=v, zorder=3)
        hol = actual[actual["holiday_offset"] == 0]
        for d in hol.index:
            ax.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12),
                       color=GRID, zorder=0)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
        _money(ax)
    axes[0][0].legend(loc="upper left", ncols=3, bbox_to_anchor=(0, -0.1))
    return _save(fig, path)


def plot_residual_gmm(model, path: Path, title: str) -> Path:
    """Histogram of log residuals with the fitted mixture components overlaid."""
    resid = model.residuals_
    on_cal = model._calendar_mask(resid.index)
    r = resid[~on_cal].to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4))
    bins = np.linspace(r.min() - 0.05, r.max() + 0.05, 80)
    ax.hist(r, bins=bins, color="#c9c8c2", edgecolor=SURFACE, linewidth=0.5, zorder=2,
            label="Residuals (non-holiday days)")
    grid = np.linspace(bins[0], bins[-1], 600)
    width = bins[1] - bins[0]
    gmm = model.gmm_
    for k in range(gmm.n_components):
        mu, sd, wt = gmm.means_[k, 0], np.sqrt(gmm.covariances_.ravel()[k]), gmm.weights_[k]
        is_spike = k == model.spike_component_
        color = COLORS["2. + Spike GMM"] if not is_spike else "#e34948"
        label = f"{'Spike' if is_spike else 'Normal'} component: {wt:.0%} of days, " \
                f"mean x{np.exp(mu):.2f}"
        ax.plot(grid, wt * len(r) * width * norm.pdf(grid, mu, sd), color=color, lw=2,
                label=label, zorder=3)
    ax.set_xlabel("log(actual) − log(normal-day fit)")
    ax.set_ylabel("Days")
    ax.set_title(title)
    ax.legend(loc="upper right")
    return _save(fig, path)


def plot_table(ladder: pd.DataFrame, horizon: int, path: Path) -> Path:
    """The model-ladder table for one horizon, as an image."""
    t = ladder[ladder["horizon"] == horizon].set_index("model").reindex(VERSIONS).reset_index()
    m = lambda v: f"${v / 1e6:,.2f}M"  # noqa: E731
    cols = [("model", "Version", str), ("MAE", "MAE", m), ("RMSE", "RMSE", m),
            ("MAE_normal", "MAE normal", m), ("MAE_spike", "MAE spike", m),
            ("MAE_holiday", "MAE holiday", m), ("required_buffer", "Req. buffer", m),
            ("avg_cash_held", "Avg cash held", m),
            ("idle_cash_cost_per_year", "Idle cost / yr", lambda v: f"${v / 1e3:,.0f}k")]
    cells = [[fmt(r[c]) for c, _, fmt in cols] for _, r in t.iterrows()]
    fig, ax = plt.subplots(figsize=(12, 1.5))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=[h for _, h, _ in cols], loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.6)
    table.auto_set_column_width(list(range(len(cols))))
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(SURFACE if r != len(cells) else "#eef4fc")
        cell.get_text().set_color(INK_2 if r == 0 else INK)
        if r == 0:
            cell.get_text().set_weight("bold")
    return _save(fig, path)
