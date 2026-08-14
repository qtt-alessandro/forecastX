"""Leakage-aware forecast combination utilities."""

from __future__ import annotations

from functools import reduce

import numpy as np
import polars as pl


KEY_COLUMNS = {"unique_id", "ds", "y", "cutoff", "forecast_start", "horizon_step", "window_id"}


def is_interval_column(name: str) -> bool:
    return "-lo-" in name or "-hi-" in name


def point_model_columns(df: pl.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in KEY_COLUMNS and column != "ensemble" and not is_interval_column(column)
        and df.schema[column].is_numeric()
    ]


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    if not weights:
        raise ValueError("At least one ensemble weight is required.")
    if any(not np.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("Ensemble weights must be finite and non-negative.")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("At least one ensemble weight must be greater than zero.")
    return {name: weight / total for name, weight in weights.items()}


def add_weighted_ensemble(
    df: pl.DataFrame,
    weights: dict[str, float],
    *,
    name: str = "ensemble",
) -> pl.DataFrame:
    available = {model: weight for model, weight in weights.items() if model in df.columns}
    missing = sorted(set(weights) - set(available))
    if missing:
        raise ValueError(f"Ensemble models are absent from forecasts: {missing}")
    normalized = normalize_weights(available)
    expression = reduce(lambda left, right: left + right, (pl.col(model) * weight for model, weight in normalized.items()))
    return df.with_columns(expression.alias(name))


def inverse_error_weights(
    predictions: pl.DataFrame,
    *,
    models: list[str] | None = None,
    shrinkage: float = 0.2,
) -> dict[str, float]:
    """Fit inverse-MAE weights; caller must provide pre-test calibration forecasts."""

    if "y" not in predictions.columns:
        raise ValueError("Calibration predictions must include actual target column 'y'.")
    models = models or point_model_columns(predictions)
    if not models:
        raise ValueError("No point forecast columns were found for ensemble calibration.")
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be between zero and one.")
    errors: dict[str, float] = {}
    for model in models:
        mae = predictions.select((pl.col(model) - pl.col("y")).abs().mean()).item()
        if mae is not None and np.isfinite(mae):
            errors[model] = float(mae)
    if not errors:
        raise ValueError("No finite model errors were available for ensemble calibration.")
    inverse = normalize_weights({model: 1.0 / max(error, 1e-9) for model, error in errors.items()})
    equal = 1.0 / len(inverse)
    return normalize_weights(
        {model: (1 - shrinkage) * weight + shrinkage * equal for model, weight in inverse.items()}
    )
