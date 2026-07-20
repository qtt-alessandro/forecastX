import re
import warnings
from datetime import datetime, timedelta, timezone
from typing import cast

import polars as pl

_FREQ_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _freq_to_timedelta(freq: str) -> timedelta:
    match = re.fullmatch(r"(\d+)([smhd])", freq)
    if match is None:
        raise ValueError(f"Unsupported frequency string: {freq!r} (expected e.g. '1m', '15m', '1h', '1d')")
    value, unit = match.groups()
    return timedelta(seconds=int(value) * _FREQ_UNIT_SECONDS[unit])


def estimate_freq(df: pl.DataFrame) -> timedelta:
    """Estimate the native frequency of df as the most common ds step per series."""
    diffs = (
        df.sort(["unique_id", "ds"])
        .select(pl.col("ds").diff().over("unique_id").alias("dt"))["dt"]
        .drop_nulls()
    )
    if diffs.is_empty():
        raise ValueError("Cannot estimate data frequency: need at least two rows per series.")
    return cast(timedelta, diffs.mode().min())


def resample(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    """
    Align df to the requested frequency.

    If the data is finer than `freq`, aggregate it (mean for numeric
    columns, first value otherwise) and emit a warning. If the data is
    coarser than `freq`, raise, since finer data cannot be invented.
    """
    target = _freq_to_timedelta(freq)
    estimated = estimate_freq(df)

    if estimated == target:
        return df
    if estimated > target:
        raise ValueError(
            f"Data frequency ({estimated}) is coarser than requested {freq!r}; cannot upsample."
        )

    warnings.warn(
        f"Data frequency ({estimated}) is finer than requested {freq!r}; "
        f"resampling to {freq!r} by mean."
    )
    aggs = [
        pl.col(c).mean() if df.schema[c].is_numeric() else pl.col(c).first()
        for c in df.columns
        if c not in ("unique_id", "ds")
    ]
    return (
        df.sort(["unique_id", "ds"])
        .group_by_dynamic("ds", every=freq, group_by="unique_id")
        .agg(aggs)
    )


def split(
    df: pl.DataFrame,
    train_start: str | datetime,
    test_start: str | datetime,
    train_end: str | datetime | None = None,
    test_end: str | datetime | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split df into train and test sets with explicit boundaries."""
    ds_tz = df.schema["ds"].time_zone if hasattr(df.schema["ds"], "time_zone") else None

    def _parse(x):
        dt = datetime.fromisoformat(x) if isinstance(x, str) else x
        if ds_tz and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    train_start = _parse(train_start)
    test_start  = _parse(test_start)
    train_end   = _parse(train_end) if train_end is not None else test_start
    test_end    = _parse(test_end)  if test_end  is not None else cast(datetime, df["ds"].max())

    train_df = df.filter((pl.col("ds") >= train_start) & (pl.col("ds") < train_end))
    test_df  = df.filter((pl.col("ds") >= test_start)  & (pl.col("ds") <= test_end))

    print(f"Train: {train_df['ds'].min()} → {train_df['ds'].max()}  ({len(train_df)} rows)")
    print(f"Test:  {test_df['ds'].min()}  → {test_df['ds'].max()}   ({len(test_df)} rows)")

    return train_df, test_df