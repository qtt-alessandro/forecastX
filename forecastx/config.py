"""Small, explicit configuration objects for the forecasting pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
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
class RunConfiguration:
    """Validated execution settings loaded from ``config/run.json``."""

    frequency: str
    horizon: int
    step_size: int
    train_start: str
    backtest_start: str
    backtest_end: str
    training_window: int
    refit_every: int
    random_seed: int
    n_jobs: int

    def __post_init__(self) -> None:
        if self.frequency not in FREQUENCY_PROFILES:
            choices = ", ".join(FREQUENCY_PROFILES)
            raise ValueError(
                f"Unsupported frequency {self.frequency!r}; choose one of: {choices}."
            )
        positive_integers = {
            "horizon": self.horizon,
            "step_size": self.step_size,
            "training_window": self.training_window,
            "refit_every": self.refit_every,
        }
        invalid = [
            name
            for name, value in positive_integers.items()
            if type(value) is not int or value <= 0
        ]
        if invalid:
            raise ValueError(f"Run settings must be positive integers: {invalid}")
        if type(self.random_seed) is not int:
            raise ValueError("random_seed must be an integer.")
        if type(self.n_jobs) is not int or self.n_jobs == 0:
            raise ValueError("n_jobs must be a non-zero integer.")

        try:
            train_start = datetime.fromisoformat(self.train_start)
            backtest_start = datetime.fromisoformat(self.backtest_start)
            backtest_end = datetime.fromisoformat(self.backtest_end)
        except (TypeError, ValueError) as error:
            raise ValueError("Run dates must use ISO 8601 strings.") from error
        if not train_start < backtest_start <= backtest_end:
            raise ValueError(
                "Run dates must satisfy train_start < backtest_start <= backtest_end."
            )

    @property
    def seasonal_period(self) -> int:
        return FREQUENCY_PROFILES[self.frequency].seasonal_period

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk with a useful missing-file error."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a JSON object.")
    return payload


def load_run_configuration(path: str | Path) -> RunConfiguration:
    """Load and validate the complete backtest execution configuration."""

    source = Path(path)
    payload = read_json(source)
    expected = {item.name for item in fields(RunConfiguration)}
    missing = sorted(expected - payload.keys())
    unknown = sorted(payload.keys() - expected)
    if missing or unknown:
        raise ValueError(
            f"Invalid fields in {source}; missing={missing}, unknown={unknown}."
        )
    return RunConfiguration(**payload)


def load_model_configuration(
    manifest_path: str | Path,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Load the selected models and their individual parameter files."""

    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    entries = manifest.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{manifest_path} must contain a non-empty 'models' list.")

    models: list[str] = []
    parameters: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError(f"Every model in {manifest_path} needs a string 'name'.")
        if not isinstance(entry.get("file"), str):
            raise ValueError(f"Every model in {manifest_path} needs a string 'file'.")
        name = entry["name"]
        if name in parameters:
            raise ValueError(f"Duplicate model '{name}' in {manifest_path}.")
        model_path = manifest_path.parent / entry["file"]
        model_config = read_json(model_path)
        if model_config.get("name") != name:
            raise ValueError(f"Model name mismatch between {manifest_path} and {model_path}.")
        model_parameters = model_config.get("parameters")
        if not isinstance(model_parameters, dict):
            raise ValueError(f"{model_path} must contain a 'parameters' object.")
        models.append(name)
        parameters[name] = model_parameters
    return models, parameters


def load_feature_configuration(path: str | Path) -> tuple[list[str], list[str]]:
    """Load temporal and complete exogenous feature selections."""

    path = Path(path)
    config = read_json(path)
    base = config.get("base_exogenous")
    temporal = config.get("temporal")
    if (
        not isinstance(base, list)
        or not all(isinstance(name, str) for name in base)
        or not isinstance(temporal, list)
        or not all(isinstance(name, str) for name in temporal)
    ):
        raise ValueError(f"{path} must contain 'base_exogenous' and 'temporal' lists.")
    features = [*base, *temporal]
    if len(features) != len(set(features)):
        raise ValueError(f"Feature names in {path} must be unique.")
    return temporal, features


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
