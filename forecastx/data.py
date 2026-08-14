"""Canonical ingestion, validation, resampling, and temporal splitting."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol, cast

import polars as pl


REQUIRED_COLUMNS = ("unique_id", "ds", "y")
_FREQ_SECONDS = {"s": 1, "m": 60, "h": 3_600, "d": 86_400}


class Loader(Protocol):
    """Minimal contract for local, object-store, database, or API loaders."""

    def load(self) -> pl.DataFrame: ...


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    rows: int
    series: int
    start: datetime
    end: datetime
    inferred_frequency_seconds: float
    duplicate_keys: int
    missing_timestamps: int
    null_counts: dict[str, int]

    @property
    def is_valid(self) -> bool:
        return self.duplicate_keys == 0 and self.missing_timestamps == 0 and not any(
            self.null_counts.get(column, 0) for column in REQUIRED_COLUMNS
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def frequency_delta(freq: str) -> timedelta:
    match = re.fullmatch(r"(\d+)([smhd])", freq)
    if match is None:
        raise ValueError(f"Unsupported frequency {freq!r}; expected e.g. '15m', '1h', or '1d'.")
    value, unit = match.groups()
    return timedelta(seconds=int(value) * _FREQ_SECONDS[unit])


def load_csv(
    path: str | Path,
    *,
    time_col: str,
    target_col: str,
    unique_id: str,
    exog: list[str] | None = None,
    timezone: str | None = None,
) -> pl.DataFrame:
    """Load a CSV into Nixtla's canonical long schema.

    Target values are deliberately never imputed here. Missing target handling is
    a modelling decision and silently forward-filling it would create fake load.
    """

    selected = [time_col, target_col, *(exog or [])]
    frame = pl.read_csv(path, try_parse_dates=True).select(selected)
    if not isinstance(frame.schema[time_col], pl.Datetime):
        frame = frame.with_columns(pl.col(time_col).str.to_datetime(strict=True))
    frame = frame.rename({time_col: "ds", target_col: "y"})
    if timezone is not None:
        dtype = frame.schema["ds"]
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
            frame = frame.with_columns(pl.col("ds").dt.replace_time_zone(timezone))
        else:
            frame = frame.with_columns(pl.col("ds").dt.convert_time_zone(timezone))
    return (
        frame.with_columns(
            pl.lit(unique_id).alias("unique_id"),
            pl.col("y").cast(pl.Float64),
        )
        .select([*REQUIRED_COLUMNS, *(exog or [])])
        .sort(["unique_id", "ds"])
    )


def estimate_frequency(df: pl.DataFrame) -> timedelta:
    validate_schema(df)
    diffs = (
        df.sort(["unique_id", "ds"])
        .select(pl.col("ds").diff().over("unique_id").alias("delta"))["delta"]
        .drop_nulls()
    )
    if diffs.is_empty():
        raise ValueError("At least two timestamps are required to estimate frequency.")
    return cast(timedelta, diffs.mode().min())


def validate_schema(
    df: pl.DataFrame,
    *,
    exog: list[str] | None = None,
    require_complete: bool = False,
) -> None:
    missing = [column for column in [*REQUIRED_COLUMNS, *(exog or [])] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not isinstance(df.schema["ds"], pl.Datetime):
        raise TypeError("Column 'ds' must be a Polars Datetime.")
    if not df.schema["y"].is_numeric():
        raise TypeError("Column 'y' must be numeric.")
    if require_complete:
        required = [*REQUIRED_COLUMNS, *(exog or [])]
        nulls = df.select(required).null_count().row(0, named=True)
        nonzero = {name: count for name, count in nulls.items() if count}
        if nonzero:
            raise ValueError(f"Null values remain in model inputs: {nonzero}")


def quality_report(df: pl.DataFrame, *, freq: str | None = None) -> DataQualityReport:
    validate_schema(df)
    duplicate_keys = (
        df.group_by(["unique_id", "ds"]).len().filter(pl.col("len") > 1).select(pl.sum("len") - pl.len()).item()
        or 0
    )
    inferred = estimate_frequency(df)
    expected = frequency_delta(freq) if freq else inferred
    missing_timestamps = 0
    for series in df.partition_by("unique_id", maintain_order=True):
        start = cast(datetime, series["ds"].min())
        end = cast(datetime, series["ds"].max())
        expected_rows = int((end - start) / expected) + 1
        missing_timestamps += max(0, expected_rows - series["ds"].n_unique())
    null_counts = {name: int(value) for name, value in df.null_count().row(0, named=True).items()}
    return DataQualityReport(
        rows=len(df),
        series=df["unique_id"].n_unique(),
        start=cast(datetime, df["ds"].min()),
        end=cast(datetime, df["ds"].max()),
        inferred_frequency_seconds=inferred.total_seconds(),
        duplicate_keys=int(duplicate_keys),
        missing_timestamps=missing_timestamps,
        null_counts=null_counts,
    )


def resample(
    df: pl.DataFrame,
    freq: str,
    *,
    target_aggregation: Literal["mean", "sum"] = "mean",
    exog_fill: Literal["forward", "interpolate", "none"] = "forward",
    target_fill: Literal["none", "forward", "interpolate", "seasonal"] = "none",
    seasonal_period: int | None = None,
) -> pl.DataFrame:
    """Align a canonical frame with an explicit, auditable target policy.

    ``seasonal`` uses only a past seasonal lag and is therefore safe at training
    time. Interpolation uses a future neighbour and should be restricted to
    offline data repair, never live feature generation.
    """

    validate_schema(df)
    target = frequency_delta(freq)
    native = estimate_frequency(df)
    if native > target:
        raise ValueError(
            f"Data frequency {native} is coarser than requested {freq}; upsampling would invent targets."
        )
    exog = [column for column in df.columns if column not in (*REQUIRED_COLUMNS, "y_imputed")]
    frame = df.sort(["unique_id", "ds"])
    if native < target:
        y_agg = pl.col("y").sum() if target_aggregation == "sum" else pl.col("y").mean()
        aggregations: list[pl.Expr] = [y_agg]
        aggregations.extend(
            pl.col(column).mean() if frame.schema[column].is_numeric() else pl.col(column).first()
            for column in exog
        )
        frame = frame.group_by_dynamic("ds", every=freq, group_by="unique_id").agg(aggregations)
    report = quality_report(frame, freq=freq)
    if report.missing_timestamps:
        frame = frame.upsample(time_column="ds", every=freq, group_by="unique_id")
    was_imputed = pl.col("y_imputed").fill_null(False) if "y_imputed" in frame.columns else pl.lit(False)
    frame = frame.with_columns((was_imputed | pl.col("y").is_null()).alias("y_imputed"))
    if target_fill == "seasonal":
        default_periods = {"1h": 168, "15m": 672, "1m": 1_440, "1d": 7}
        period = seasonal_period or default_periods.get(freq)
        if period is None:
            raise ValueError("seasonal_period is required for seasonal target filling at this frequency.")
        frame = frame.with_columns(
            pl.when(pl.col("y").is_null())
            .then(pl.col("y").shift(period).over("unique_id"))
            .otherwise(pl.col("y"))
            .alias("y")
        )
    elif target_fill == "forward":
        frame = frame.with_columns(pl.col("y").forward_fill().over("unique_id"))
    elif target_fill == "interpolate":
        frame = frame.with_columns(pl.col("y").interpolate().over("unique_id"))
    if exog and exog_fill != "none":
        expressions = []
        for column in exog:
            expression = pl.col(column)
            if frame.schema[column].is_numeric() and exog_fill == "interpolate":
                expression = expression.interpolate()
            else:
                expression = expression.forward_fill()
            expressions.append(expression.over("unique_id").alias(column))
        frame = frame.with_columns(expressions)
    remaining = frame["y"].null_count()
    if target_fill != "none" and remaining:
        raise ValueError(
            f"Target fill policy {target_fill!r} left {remaining} missing values; provide more history or another policy."
        )
    return frame.sort(["unique_id", "ds"])


def split(
    df: pl.DataFrame,
    *,
    train_start: str | datetime,
    test_start: str | datetime,
    train_end: str | datetime | None = None,
    test_end: str | datetime | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create non-overlapping half-open train and test intervals."""

    validate_schema(df)
    dtype = df.schema["ds"]
    timezone = dtype.time_zone if isinstance(dtype, pl.Datetime) else None

    def parse(value: str | datetime) -> datetime:
        result = datetime.fromisoformat(value) if isinstance(value, str) else value
        if timezone is not None and result.tzinfo is None:
            from zoneinfo import ZoneInfo

            result = result.replace(tzinfo=ZoneInfo(timezone))
        return result

    parsed_train_start = parse(train_start)
    parsed_test_start = parse(test_start)
    parsed_train_end = parse(train_end) if train_end is not None else parsed_test_start
    parsed_test_end = parse(test_end) if test_end is not None else cast(datetime, df["ds"].max()) + timedelta(microseconds=1)
    if not parsed_train_start < parsed_train_end <= parsed_test_start < parsed_test_end:
        raise ValueError("Expected train_start < train_end <= test_start < test_end.")
    train = df.filter(pl.col("ds").is_between(parsed_train_start, parsed_train_end, closed="left"))
    test = df.filter(pl.col("ds").is_between(parsed_test_start, parsed_test_end, closed="left"))
    if train.is_empty() or test.is_empty():
        raise ValueError("The requested split produced an empty train or test frame.")
    return train, test
