"""Leakage-safe calendar features shared by every ML model."""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl


def is_weekend(dates: pl.Expr) -> pl.Expr:
    return (dates.dt.weekday() >= 6).cast(pl.Int8)


def hour_sin(dates: pl.Expr) -> pl.Expr:
    return (2 * np.pi * dates.dt.hour() / 24).sin()


def hour_cos(dates: pl.Expr) -> pl.Expr:
    return (2 * np.pi * dates.dt.hour() / 24).cos()


def week_sin(dates: pl.Expr) -> pl.Expr:
    hour_of_week = (dates.dt.weekday() - 1) * 24 + dates.dt.hour()
    return (2 * np.pi * hour_of_week / 168).sin()


def week_cos(dates: pl.Expr) -> pl.Expr:
    hour_of_week = (dates.dt.weekday() - 1) * 24 + dates.dt.hour()
    return (2 * np.pi * hour_of_week / 168).cos()


def add_temperature_features(
    df: pl.DataFrame,
    *,
    column: str = "mean_temp",
    role: Literal["actual", "forecast"] | None = None,
    heating_balance: float = 15.0,
    cooling_balance: float = 18.0,
) -> pl.DataFrame:
    """Add deterministic piecewise temperature terms used in load forecasting.

    These values are safe in the future only when ``column`` itself is a weather
    forecast available at the origin.
    """

    if column not in df.columns:
        raise ValueError(f"Temperature column {column!r} is missing.")
    if role is None:
        squared = f"{column}_squared"
        heating = "heating_degree"
        cooling = "cooling_degree"
    else:
        suffix = f"_{role}"
        if not column.endswith(suffix):
            raise ValueError(f"Temperature column {column!r} must end with {suffix!r}.")
        base = column.removesuffix(suffix)
        squared = f"{base}_squared{suffix}"
        heating = f"heating_degree{suffix}"
        cooling = f"cooling_degree{suffix}"
    return df.with_columns(
        pl.col(column).pow(2).alias(squared),
        (pl.lit(heating_balance) - pl.col(column)).clip(lower_bound=0).alias(heating),
        (pl.col(column) - pl.lit(cooling_balance)).clip(lower_bound=0).alias(cooling),
    )
