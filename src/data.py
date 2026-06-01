import polars as pl

from config import HORIZON
from datetime import datetime
from typing import cast


def load_data(path: str) -> pl.DataFrame:
    """
    Load raw CSV, select relevant columns, forward-fill gaps,
    and upsample to a continuous hourly index.
    """
    return (
        pl.read_csv(path, try_parse_dates=True)
        .select(["UTC", "realization", "mean_temp", "forecast"])
        .rename({"UTC": "ds", "realization": "y"})
        .sort("ds")
        .upsample(time_column="ds", every="1h")
        .with_columns([
            pl.col("y").forward_fill(),
            pl.col("mean_temp").forward_fill(),
            pl.col("forecast").forward_fill(),
        ])
        .with_columns(pl.lit("heat_demand").alias("unique_id"))
    )


def split(
    df: pl.DataFrame,
    train_start: str | datetime,
    test_start: str | datetime,
    train_end: str | datetime | None = None,
    test_end: str | datetime | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split df into train and test sets with explicit boundaries.

    Defaults
    --------
    train_end : test_start  (train ends where test begins)
    test_end  : end of df   (use all remaining data)
    """
    def _parse(x):
        return datetime.fromisoformat(x) if isinstance(x, str) else x

    train_start = _parse(train_start)
    test_start  = _parse(test_start)
    train_end   = _parse(train_end) if train_end is not None else test_start
    if test_end is not None:
        test_end = _parse(test_end)
    else:
        test_end = _parse(test_end) if test_end is not None else cast(datetime, df["ds"].max())

    train_df = df.filter((pl.col("ds") >= train_start) & (pl.col("ds") < train_end))
    test_df  = df.filter((pl.col("ds") >= test_start)  & (pl.col("ds") <= test_end))

    print(f"Train: {train_df['ds'].min()} → {train_df['ds'].max()}  ({len(train_df)} rows)")
    print(f"Test:  {test_df['ds'].min()}  → {test_df['ds'].max()}   ({len(test_df)} rows)")

    return train_df, test_df
