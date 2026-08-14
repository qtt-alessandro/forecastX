"""Runtime checks for target leakage and safe external forecast schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from forecastx.engine import ForecastEngine
from forecastx.ensemble import point_model_columns
from forecastx.models import MODEL_REGISTRY


@dataclass(frozen=True, slots=True)
class LeakageAuditReport:
    passed: bool
    maximum_prediction_change: float
    checked_models: tuple[str, ...]
    explanation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def future_target_invariance(
    engine: ForecastEngine,
    future_df: pl.DataFrame,
    *,
    horizon: int,
    exog: list[str] | None = None,
    tolerance: float = 1e-10,
) -> LeakageAuditReport:
    """Predictions must not change when unavailable future targets are scrambled."""

    baseline = engine.predict(future_df, horizon=horizon, exog=exog, level=[])
    mutated = future_df.with_columns(
        (pl.col("y").reverse() * 10_003 + 17).alias("y")
    )
    alternate = engine.predict(mutated, horizon=horizon, exog=exog, level=[])
    models = point_model_columns(baseline)
    changes = [
        float(np.max(np.abs(baseline[model].to_numpy() - alternate[model].to_numpy())))
        for model in models
    ]
    maximum = max(changes, default=0.0)
    passed = maximum <= tolerance
    return LeakageAuditReport(
        passed=passed,
        maximum_prediction_change=maximum,
        checked_models=tuple(models),
        explanation=(
            "Forecasts are invariant to future target values."
            if passed
            else "Forecasts changed after scrambling unavailable future targets; investigate leakage."
        ),
    )


def external_forecast_columns(frame: pl.DataFrame, *, include_actual: bool = False) -> list[str]:
    """Allow forecasts and timing metadata but exclude features and training internals."""

    allowed_keys = ["unique_id", "ds", "cutoff", "forecast_start", "horizon_step", "window_id"]
    models = [name for name in MODEL_REGISTRY if name in frame.columns]
    if "ensemble" in frame.columns:
        models.append("ensemble")
    interval_columns = [
        column
        for column in frame.columns
        if ("-lo-" in column or "-hi-" in column)
        and any(column.startswith(f"{model}-") for model in models)
    ]
    columns = [column for column in allowed_keys if column in frame.columns]
    columns.extend(models)
    columns.extend(interval_columns)
    if include_actual and "y" in frame.columns:
        columns.append("y")
    return list(dict.fromkeys(columns))
