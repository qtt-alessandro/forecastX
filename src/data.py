import polars as pl
from datetime import datetime, timezone
from typing import cast


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