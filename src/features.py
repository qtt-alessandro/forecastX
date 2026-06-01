import numpy as np
import polars as pl


def is_weekend(dates: pl.Expr) -> pl.Expr:
    """Returns 1 if the date falls on a weekend (Sat/Sun), else 0."""
    return (dates.dt.weekday() >= 5).cast(pl.Int8)


def hour_sin(dates: pl.Expr) -> pl.Expr:
    """Sine encoding of the hour of day (period = 24h)."""
    return (2 * np.pi * dates.dt.hour() / 24).sin()


def hour_cos(dates: pl.Expr) -> pl.Expr:
    """Cosine encoding of the hour of day (period = 24h)."""
    return (2 * np.pi * dates.dt.hour() / 24).cos()
