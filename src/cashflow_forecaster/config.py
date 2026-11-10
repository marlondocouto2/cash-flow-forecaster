"""All tunable settings in one place."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    # Synthetic data
    start: str = "2021-01-01"
    end: str = "2025-12-31"
    seed: int = 42
    corridors: tuple[str, ...] = ("MX", "PH", "IN")  # receiving countries (US is the sender)

    # Seasonal baseline (as in Project 1)
    level_window: int = 28
    weekday_window_weeks: int = 12
    month_window_days: int = 364

    # Spike decomposition
    gmm_components: int = 2
    gmm_history_days: int = 730     # residual history the mixture is fitted on
    spike_threshold: float = 0.5    # posterior probability above which a day is a spike

    # Holiday effects
    holiday_shrinkage: float = 1.0  # pseudo-observations pulling each effect toward 1.0

    # Backtest
    horizons: tuple[int, ...] = (1, 7, 30)
    backtest_days: int = 365
    buffer_quantile: float = 0.95
    annual_cost_of_capital: float = 0.05

    # Output
    reports_dir: Path = field(default=PROJECT_ROOT / "reports")

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"


CONFIG = Config()
