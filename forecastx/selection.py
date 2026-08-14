"""Pre-test comparison of expanding and sliding training histories."""

from __future__ import annotations

import polars as pl

from forecastx.backtest import backtest
from forecastx.metrics import evaluate_forecasts


def compare_training_windows(
    *,
    train_df: pl.DataFrame,
    validation_df: pl.DataFrame,
    candidates: list[int | None],
    horizon: int,
    step_size: int,
    freq: str,
    exog: list[str] | None = None,
    models: list[str] | None = None,
    target_transform: str = "none",
) -> pl.DataFrame:
    """Select history length exclusively on validation data preceding the final test."""

    rows: list[pl.DataFrame] = []
    seasonal_period = {"1h": 24, "15m": 96, "1m": 60, "1d": 7}[freq]
    for candidate in candidates:
        predictions, _ = backtest(
            train_df=train_df,
            test_df=validation_df,
            horizon=horizon,
            step_size=step_size,
            freq=freq,
            exog=exog,
            models=models,
            refit=True,
            target_transform=target_transform,
            training_window=candidate,
            level=[],
        )
        metrics = evaluate_forecasts(
            predictions,
            train_df=train_df,
            seasonal_period=seasonal_period,
        ).with_columns(
            pl.lit(candidate).cast(pl.Int64).alias("training_window")
            if candidate is not None
            else pl.lit(None).cast(pl.Int64).alias("training_window")
        )
        rows.append(metrics)
    return pl.concat(rows).sort(["model", "mae"])


def recent_regime_split(
    df: pl.DataFrame,
    *,
    observations_per_week: int,
    train_weeks: int = 2,
    test_weeks: int = 2,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create the requested recent two-week/two-week stress test per series."""

    required = observations_per_week * (train_weeks + test_weeks)
    train_parts: list[pl.DataFrame] = []
    test_parts: list[pl.DataFrame] = []
    for series in df.sort(["unique_id", "ds"]).partition_by("unique_id", maintain_order=True):
        if len(series) < required:
            raise ValueError(f"Recent regime stress test requires at least {required} rows per series.")
        recent = series.tail(required)
        split_at = observations_per_week * train_weeks
        train_parts.append(recent.head(split_at))
        test_parts.append(recent.tail(observations_per_week * test_weeks))
    return pl.concat(train_parts), pl.concat(test_parts)
