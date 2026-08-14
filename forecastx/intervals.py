"""Simple horizon-conditional split-conformal intervals for any point forecast."""

from __future__ import annotations

import math

import numpy as np
import polars as pl


def add_conformal_intervals(
    forecasts: pl.DataFrame,
    *,
    calibration: pl.DataFrame,
    model: str,
    levels: list[int] | tuple[int, ...] = (80, 95),
    horizon_column: str = "horizon_step",
) -> pl.DataFrame:
    """Apply finite-sample, horizon-conditional absolute-residual intervals.

    Calibration rows must chronologically precede ``forecasts``. The function is
    backend-independent and is especially useful for weighted ensembles.
    """

    required = {"y", model, horizon_column}
    if missing := required - set(calibration.columns):
        raise ValueError(f"Calibration data is missing columns: {sorted(missing)}")
    if model not in forecasts.columns or horizon_column not in forecasts.columns:
        raise ValueError(f"Forecasts must include {model!r} and {horizon_column!r}.")
    result = forecasts
    steps = calibration[horizon_column].unique().sort().to_list()
    for level in levels:
        if not 0 < level < 100:
            raise ValueError("Interval levels must be between 0 and 100.")
        quantiles: list[dict[str, float | int]] = []
        for step in steps:
            group = calibration.filter(pl.col(horizon_column) == step)
            scores = (group["y"] - group[model]).abs().drop_nulls().to_numpy()
            if not len(scores):
                raise ValueError(f"No calibration residuals for horizon step {step}.")
            finite_sample_level = min(1.0, math.ceil((len(scores) + 1) * level / 100) / len(scores))
            quantiles.append(
                {
                    horizon_column: step,
                    "radius": float(np.quantile(scores, finite_sample_level, method="higher")),
                }
            )
        radii = pl.DataFrame(quantiles)
        result = (
            result.join(radii, on=horizon_column, how="left")
            .with_columns(
                (pl.col(model) - pl.col("radius")).alias(f"{model}-lo-{level}"),
                (pl.col(model) + pl.col("radius")).alias(f"{model}-hi-{level}"),
            )
            .drop("radius")
        )
    return result
