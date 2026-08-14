"""Small, explicit configuration objects for the forecasting pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ForecastStrategy = Literal["recursive", "direct"]
TargetTransform = Literal["none", "auto", "difference_1", "difference_seasonal"]


@dataclass(frozen=True, slots=True)
class FrequencyProfile:
    """Feature defaults expressed in numbers of observations, not wall time."""

    seasonal_period: int
    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    date_features: tuple[str, ...]


FREQUENCY_PROFILES: dict[str, FrequencyProfile] = {
    "1h": FrequencyProfile(
        seasonal_period=24,
        lags=(1, 2, 3, 6, 12, 24, 48, 72, 168),
        rolling_windows=(24, 168),
        date_features=("hour", "weekday", "month"),
    ),
    "15m": FrequencyProfile(
        seasonal_period=96,
        lags=(1, 2, 4, 8, 12, 24, 48, 96, 192, 672),
        rolling_windows=(96, 672),
        date_features=("minute", "hour", "weekday", "month"),
    ),
    "1m": FrequencyProfile(
        seasonal_period=60,
        lags=(1, 2, 3, 5, 10, 15, 30, 60, 120, 1_440),
        rolling_windows=(60, 1_440),
        date_features=("minute", "hour", "weekday"),
    ),
    "1d": FrequencyProfile(
        seasonal_period=7,
        lags=(1, 2, 3, 7, 14, 28),
        rolling_windows=(7, 28),
        date_features=("weekday", "month"),
    ),
}


FAST_MODELS: tuple[str, ...] = (
    "LinearRegression",
    "Ridge",
    "GradientBoostingRegressor",
    "LGBMRegressor",
)


@dataclass(frozen=True, slots=True)
class ForecastConfig:
    """Controls model construction without hiding expensive behaviour."""

    freq: str = "1h"
    models: tuple[str, ...] = FAST_MODELS
    strategy: ForecastStrategy = "recursive"
    target_transform: TargetTransform = "none"
    interval_levels: tuple[int, ...] = (80, 95)
    interval_windows: int = 5
    random_state: int = 42
    n_jobs: int = 1
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.freq not in FREQUENCY_PROFILES:
            choices = ", ".join(FREQUENCY_PROFILES)
            raise ValueError(f"Unsupported frequency {self.freq!r}; choose one of: {choices}.")
        if self.interval_windows < 2:
            raise ValueError("interval_windows must be at least 2 for conformal calibration.")
        if self.strategy not in {"recursive", "direct"}:
            raise ValueError("strategy must be 'recursive' or 'direct'.")
        if self.target_transform not in {"none", "auto", "difference_1", "difference_seasonal"}:
            raise ValueError("Unsupported target_transform.")
        invalid_levels = [level for level in self.interval_levels if not 0 < level < 100]
        if invalid_levels:
            raise ValueError(f"Interval levels must be between 0 and 100: {invalid_levels}")

    @property
    def profile(self) -> FrequencyProfile:
        return FREQUENCY_PROFILES[self.freq]
