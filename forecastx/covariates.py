"""As-of selection and validation for historical and future covariates."""

from __future__ import annotations

from datetime import datetime
from typing import Mapping, Sequence

import polars as pl

from forecastx.data import frequency_delta


def resolve_covariate_names(
    *,
    exog: Sequence[str] | None = None,
    hist_exog: Sequence[str] | None = None,
    futr_exog: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve the legacy ``exog`` alias and reject ambiguous role assignments."""

    if exog is not None and (hist_exog is not None or futr_exog is not None):
        raise ValueError("Use either exog or hist_exog/futr_exog, not both.")
    historical = tuple(hist_exog or ())
    future = tuple(exog if exog is not None else (futr_exog or ()))
    if len(historical) != len(set(historical)) or len(future) != len(set(future)):
        raise ValueError("Covariate names must be unique within each role.")
    if overlap := sorted(set(historical) & set(future)):
        raise ValueError(
            f"Historical and future covariates must use distinct columns: {overlap}"
        )
    if "y" in {*historical, *future}:
        raise ValueError("The target column 'y' cannot be a covariate.")
    if invalid := [name for name in historical if name.endswith("_forecast")]:
        raise ValueError(f"Forecast columns cannot be historical-only covariates: {invalid}")
    if invalid := [name for name in future if name.endswith("_actual")]:
        raise ValueError(f"Actual columns cannot be future covariates: {invalid}")
    return historical, future


def validate_vintage_mapping(
    futr_exog: Sequence[str],
    futr_exog_vintages: Mapping[str, str] | None,
) -> dict[str, str]:
    """Require explicit vintage metadata for every ``*_forecast`` column."""

    future = tuple(futr_exog)
    vintages = dict(futr_exog_vintages or {})
    if unknown := sorted(set(vintages) - set(future)):
        raise ValueError(f"Vintage metadata was provided for non-future covariates: {unknown}")
    if missing := [name for name in future if name.endswith("_forecast") and name not in vintages]:
        raise ValueError(f"Forecast covariates are missing vintage metadata: {missing}")
    return vintages


def as_of_covariates(
    df: pl.DataFrame,
    *,
    cutoff: datetime,
    horizon: int,
    freq: str,
    hist_exog: Sequence[str] | None = None,
    futr_exog: Sequence[str] | None = None,
    futr_exog_vintages: Mapping[str, str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return covariates genuinely available at one forecast cutoff.

    Historical covariates stop at ``cutoff``. Future covariates cover exactly the
    requested horizon. Forecast-type future covariates carry a vintage timestamp;
    deterministic future-known inputs such as calendars do not need one.
    """

    if horizon < 1:
        raise ValueError("horizon must be positive.")
    historical, future = resolve_covariate_names(
        hist_exog=hist_exog,
        futr_exog=futr_exog,
    )
    vintages = validate_vintage_mapping(future, futr_exog_vintages)

    vintage_columns = tuple(dict.fromkeys(vintages.values()))
    required = ["unique_id", "ds", *historical, *future, *vintage_columns]
    if missing := [column for column in required if column not in df.columns]:
        raise ValueError(f"Covariate source is missing required columns: {missing}")
    if not isinstance(df.schema["ds"], pl.Datetime):
        raise TypeError("Column 'ds' must be a Polars Datetime.")
    if df.select(pl.struct(["unique_id", "ds"]).is_duplicated().any()).item():
        raise ValueError("Duplicate (unique_id, ds) covariate rows are not allowed.")
    for column in vintage_columns:
        if not isinstance(df.schema[column], pl.Datetime):
            raise TypeError(f"Forecast vintage column {column!r} must be a Polars Datetime.")

    delta = frequency_delta(freq)
    past_frames: list[pl.DataFrame] = []
    future_frames: list[pl.DataFrame] = []
    ordered = df.sort(["unique_id", "ds"])
    for series in ordered.partition_by("unique_id", maintain_order=True):
        unique_id = series["unique_id"][0]
        past = series.filter(pl.col("ds") <= cutoff)
        expected = [cutoff + delta * step for step in range(1, horizon + 1)]
        future_window = series.filter(pl.col("ds").is_in(expected)).sort("ds")

        if past.is_empty():
            raise ValueError(f"No historical covariates exist at cutoff {cutoff} for {unique_id!r}.")
        if len(future_window) != horizon or future_window["ds"].to_list() != expected:
            raise ValueError(
                f"No complete future-covariate horizon exists at cutoff {cutoff} "
                f"for {unique_id!r}."
            )
        if past.filter(pl.col("ds") > cutoff).height:
            raise ValueError("Historical covariates contain timestamps after the cutoff.")
        if future_window.filter(pl.col("ds") <= cutoff).height:
            raise ValueError("Future covariates contain timestamps at or before the cutoff.")

        for vintage_column in vintage_columns:
            historical_vintages = past.select(["ds", vintage_column])
            future_vintages = future_window.select(["ds", vintage_column])
            if historical_vintages[vintage_column].null_count():
                raise ValueError(f"Historical vintage column {vintage_column!r} contains nulls.")
            if future_vintages[vintage_column].null_count():
                raise ValueError(
                    f"No valid forecast vintage in {vintage_column!r} covers cutoff {cutoff}."
                )
            if historical_vintages.filter(pl.col(vintage_column) >= pl.col("ds")).height:
                raise ValueError(
                    f"Vintage column {vintage_column!r} was not issued before its target."
                )
            if future_vintages.filter(pl.col(vintage_column) > cutoff).height:
                raise ValueError(
                    f"Vintage column {vintage_column!r} contains an issue after cutoff {cutoff}."
                )
            if future_vintages.filter(pl.col(vintage_column) >= pl.col("ds")).height:
                raise ValueError(
                    f"Vintage column {vintage_column!r} was not issued before its target."
                )

        past_columns = list(dict.fromkeys(["unique_id", "ds", *historical, *future, *vintage_columns]))
        future_columns = list(dict.fromkeys(["unique_id", "ds", *future, *vintage_columns]))
        past_values = [*historical, *future]
        if past_values:
            nonzero = {
                name: count
                for name, count in past.select(past_values)
                .null_count()
                .row(0, named=True)
                .items()
                if count
            }
            if nonzero:
                raise ValueError(f"Null values remain in historical covariates: {nonzero}")
        if future:
            nonzero = {
                name: count
                for name, count in future_window.select(future)
                .null_count()
                .row(0, named=True)
                .items()
                if count
            }
            if nonzero:
                raise ValueError(f"Null values remain in future covariates: {nonzero}")
        past_frames.append(past.select(past_columns))
        future_frames.append(future_window.select(future_columns))

    return (
        pl.concat(past_frames).sort(["unique_id", "ds"]),
        pl.concat(future_frames).sort(["unique_id", "ds"]),
    )
