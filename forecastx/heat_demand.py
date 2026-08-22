"""Dataset-specific loading and calendar features for the heat-demand workflow."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from forecastx.data import frequency_delta, load_csv, resample


def load_heat_demand(path: str | Path, *, frequency: str = "1h") -> pl.DataFrame:
    """Load the source CSV and repair target outages from past weekly values."""

    frame = load_csv(
        path,
        time_col="UTC",
        target_col="realization",
        unique_id="heat_demand",
        exog=["mean_temp", "forecast"],
    )
    return resample(frame, frequency, target_fill="seasonal")


def add_synthetic_temperature_forecast(
    frame: pl.DataFrame,
    *,
    freq: str = "1h",
    lead_steps: int = 24,
    error_std: float = 1.5,
    random_state: int = 42,
) -> pl.DataFrame:
    """Separate measured temperature from a reproducible causal forecast proxy.

    This is demonstration data, not a replacement for archived weather vintages.
    The forecast for a target uses only the measurement at its issue timestamp,
    plus noise; it never uses the target timestamp's realized temperature.
    """

    if lead_steps < 1:
        raise ValueError("lead_steps must be positive.")
    if error_std < 0:
        raise ValueError("error_std cannot be negative.")
    if "mean_temp" not in frame.columns:
        raise ValueError("Input data is missing measured column 'mean_temp'.")

    ordered = frame.sort(["unique_id", "ds"]).rename(
        {"mean_temp": "mean_temp_actual"}
    )
    noise = np.random.default_rng(random_state).normal(0.0, error_std, len(ordered))
    lead = frequency_delta(freq) * lead_steps
    return (
        ordered.with_columns(pl.Series("_temperature_forecast_error", noise))
        .with_columns(
            (
                pl.col("mean_temp_actual").shift(lead_steps).over("unique_id")
                + pl.col("_temperature_forecast_error")
            ).alias("mean_temp_forecast"),
            (pl.col("ds") - lead).alias("mean_temp_forecast_vintage_ds"),
        )
        .drop_nulls("mean_temp_forecast")
        .drop("_temperature_forecast_error")
    )


def easter_sunday(year: int) -> date:
    """Return Gregorian Easter Sunday for holiday feature generation."""

    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def danish_holidays(year: int) -> set[date]:
    """Return Danish public holidays relevant to the available demand history."""

    easter = easter_sunday(year)
    return {
        date(year, 1, 1),
        easter - timedelta(days=3),
        easter - timedelta(days=2),
        easter,
        easter + timedelta(days=1),
        easter + timedelta(days=39),
        easter + timedelta(days=49),
        easter + timedelta(days=50),
        date(year, 12, 25),
        date(year, 12, 26),
    }


def add_temporal_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Create deterministic calendar inputs known for every forecast horizon."""

    years = frame["ds"].dt.year().unique().to_list()
    holidays = sorted(day for year in years for day in danish_holidays(int(year)))

    timestamp = pl.col("ds")
    hour = timestamp.dt.hour()
    weekday = timestamp.dt.weekday()
    day_of_year = timestamp.dt.ordinal_day()
    hour_of_week = (weekday - 1) * 24 + hour
    is_weekend = weekday >= 6
    is_holiday = timestamp.dt.date().is_in(holidays)

    return frame.with_columns(
        hour.cast(pl.Int8).alias("calendar_hour"),
        weekday.cast(pl.Int8).alias("calendar_weekday"),
        timestamp.dt.month().cast(pl.Int8).alias("calendar_month"),
        day_of_year.cast(pl.Int16).alias("calendar_day_of_year"),
        is_weekend.cast(pl.Int8).alias("calendar_is_weekend"),
        ((~is_weekend) & (~is_holiday)).cast(pl.Int8).alias("calendar_is_working_day"),
        ((weekday <= 5) & hour.is_between(7, 17))
        .cast(pl.Int8)
        .alias("calendar_is_business_hour"),
        is_holiday.cast(pl.Int8).alias("calendar_is_holiday"),
        (2 * np.pi * hour / 24).sin().alias("calendar_hour_sin"),
        (2 * np.pi * hour / 24).cos().alias("calendar_hour_cos"),
        (2 * np.pi * hour_of_week / 168).sin().alias("calendar_week_sin"),
        (2 * np.pi * hour_of_week / 168).cos().alias("calendar_week_cos"),
        (2 * np.pi * (day_of_year - 1) / 365.2425).sin().alias("calendar_year_sin"),
        (2 * np.pi * (day_of_year - 1) / 365.2425).cos().alias("calendar_year_cos"),
    )


def validate_feature_columns(frame: pl.DataFrame, features: list[str]) -> None:
    """Ensure every configured feature was produced before model fitting."""

    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Configured features are missing after preprocessing: {missing}")
