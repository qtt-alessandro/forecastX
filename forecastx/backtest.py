"""Operationally faithful rolling-origin backtesting."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, cast

import polars as pl
from statsforecast import StatsForecast

from forecastx.config import FAST_MODELS, ForecastConfig
from forecastx.covariates import as_of_covariates, resolve_covariate_names
from forecastx.data import frequency_delta, validate_schema
from forecastx.engine import ForecastEngine, build_mlf, fit
from forecastx.ensemble import add_weighted_ensemble, inverse_error_weights
from forecastx.intervals import add_conformal_intervals
from forecastx.models import MODEL_REGISTRY


def _tail_per_series(df: pl.DataFrame, rows: int | None) -> pl.DataFrame:
    if rows is None:
        return df
    return (
        df.sort(["unique_id", "ds"])
        .group_by("unique_id", maintain_order=True)
        .tail(rows)
        .sort(["unique_id", "ds"])
    )


def _should_refit(refit: bool | int, window_id: int) -> bool:
    if window_id == 0:
        return True
    if refit is True:
        return True
    if isinstance(refit, int) and not isinstance(refit, bool):
        if refit < 1:
            raise ValueError("Integer refit cadence must be positive.")
        return window_id % refit == 0
    return False


def _calibration_split(
    train_df: pl.DataFrame,
    *,
    horizon: int,
    step_size: int,
    n_windows: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    test_rows = horizon + step_size * (n_windows - 1)
    train_parts: list[pl.DataFrame] = []
    test_parts: list[pl.DataFrame] = []
    for series in train_df.partition_by("unique_id", maintain_order=True):
        if len(series) <= test_rows + horizon * 3:
            raise ValueError("Not enough training history for leakage-free ensemble calibration.")
        train_parts.append(series.head(len(series) - test_rows))
        test_parts.append(series.tail(test_rows))
    return pl.concat(train_parts), pl.concat(test_parts)


def statistical_backtest(
    *,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    horizon: int,
    models: list[str] | tuple[str, ...],
    exog: list[str] | None = None,
    step_size: int | None = None,
    freq: str = "1h",
    refit: bool | int = False,
    level: list[int] | None = None,
    ensemble_weights: dict[str, float] | None = None,
    n_jobs: int = 1,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Efficient Nixtla state-updating CV for StatsForecast-only model lists."""

    exog = exog or []
    step_size = step_size or horizon
    names = tuple(models)
    invalid = [name for name in names if MODEL_REGISTRY[name].backend != "statsforecast"]
    if invalid:
        raise ValueError(f"statistical_backtest only accepts StatsForecast models: {invalid}")
    validate_schema(train_df, exog=exog, require_complete=True)
    validate_schema(test_df, exog=exog, require_complete=True)
    delta = frequency_delta(freq)
    test_start = cast(datetime, test_df["ds"].min())
    test_end = cast(datetime, test_df["ds"].max())
    starts: list[datetime] = []
    start = test_start
    while start + delta * (horizon - 1) <= test_end:
        starts.append(start)
        start += delta * step_size
    if not starts:
        raise ValueError("Test data is shorter than the forecast horizon.")
    used_end = starts[-1] + delta * (horizon - 1)
    used_test = test_df.filter(pl.col("ds") <= used_end)
    config = ForecastConfig(freq=freq, models=names, interval_levels=tuple(level or ()), n_jobs=n_jobs)
    frames: list[pl.DataFrame] = []
    groups = (
        [name for name in names if MODEL_REGISTRY[name].supports_future_exog],
        [name for name in names if not MODEL_REGISTRY[name].supports_future_exog],
    )
    for group in groups:
        if not group:
            continue
        columns = ["unique_id", "ds", "y"]
        if MODEL_REGISTRY[group[0]].supports_future_exog:
            columns.extend(exog)
        combined = pl.concat([train_df.select(columns), used_test.select(columns)])
        forecaster = StatsForecast(
            models=[MODEL_REGISTRY[name].factory(config) for name in group],
            freq=freq,
            n_jobs=n_jobs,
        )
        frame = forecaster.cross_validation(
            df=combined,
            h=horizon,
            n_windows=len(starts),
            step_size=step_size,
            refit=refit,
            level=level or None,
        )
        frames.append(frame if isinstance(frame, pl.DataFrame) else pl.from_pandas(frame))
    result = frames[0]
    for frame in frames[1:]:
        extra = [column for column in frame.columns if column not in {"unique_id", "ds", "cutoff", "y"}]
        result = result.join(frame.select(["unique_id", "ds", "cutoff", *extra]), on=["unique_id", "ds", "cutoff"])
    result = result.with_columns(
        (pl.col("cutoff") + delta).alias("forecast_start"),
        pl.col("cutoff").rank("dense").cast(pl.Int32).sub(1).alias("window_id"),
        pl.col("ds").rank("ordinal").over(["unique_id", "cutoff"]).cast(pl.Int32).alias("horizon_step"),
    ).sort(["forecast_start", "unique_id", "ds"])
    if ensemble_weights is not None:
        result = add_weighted_ensemble(result, ensemble_weights)
    windows = pl.DataFrame(
        {
            "window_id": list(range(len(starts))),
            "cutoff": [value - delta for value in starts],
            "forecast_start": starts,
            "forecast_end": [value + delta * (horizon - 1) for value in starts],
            "refitted": [_should_refit(refit, index) for index in range(len(starts))],
            "horizon": [horizon] * len(starts),
            "step_size": [step_size] * len(starts),
        }
    )
    return result, windows


def backtest(
    *,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    horizon: int,
    exog: list[str] | None = None,
    hist_exog: list[str] | None = None,
    futr_exog: list[str] | None = None,
    futr_exog_vintages: dict[str, str] | None = None,
    step_size: int | None = None,
    freq: str = "1h",
    refit: bool | int = False,
    ensemble_weights: dict[str, float] | Literal["performance"] | None = None,
    level: list[int] | None = None,
    models: list[str] | tuple[str, ...] | None = None,
    strategy: Literal["recursive", "direct"] = "recursive",
    target_transform: str = "none",
    training_window: int | None = None,
    ensemble_calibration_windows: int = 5,
    interval_calibration_windows: int = 20,
    random_state: int = 42,
    n_jobs: int = 1,
    model_params: dict[str, dict[str, object]] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Backtest exactly ``horizon`` steps every ``step_size`` observations.

    Earlier test targets become available only after their timestamps, matching an
    online forecast service. With ``step_size=1`` this predicts H steps every step;
    with ``step_size=H`` it predicts non-overlapping H-step blocks.
    """

    role_aware_covariates = any(
        value is not None for value in (hist_exog, futr_exog, futr_exog_vintages)
    )
    historical, future = resolve_covariate_names(
        exog=exog,
        hist_exog=hist_exog,
        futr_exog=futr_exog,
    )
    vintages = dict(futr_exog_vintages or {})
    models = tuple(models or FAST_MODELS)
    step_size = step_size or horizon
    if horizon < 1 or step_size < 1:
        raise ValueError("horizon and step_size must be positive.")
    validate_schema(train_df, exog=[*historical, *future], require_complete=True)
    validate_schema(test_df, exog=[*historical, *future], require_complete=True)
    if set(train_df["unique_id"].unique()) != set(test_df["unique_id"].unique()):
        raise ValueError("Train and test series identifiers must match.")
    delta = frequency_delta(freq)
    test_start = cast(datetime, test_df["ds"].min())
    test_end = cast(datetime, test_df["ds"].max())
    forecast_starts: list[datetime] = []
    start = test_start
    while start + delta * (horizon - 1) <= test_end:
        forecast_starts.append(start)
        start += delta * step_size
    if not forecast_starts:
        raise ValueError("Test data is shorter than the forecast horizon.")

    if (
        not role_aware_covariates
        and all(MODEL_REGISTRY[name].backend == "statsforecast" for name in models)
    ):
        if ensemble_weights == "performance":
            raise ValueError("Use explicit pre-test weights with statistical_backtest.")
        return statistical_backtest(
            train_df=train_df,
            test_df=test_df,
            horizon=horizon,
            models=models,
            exog=list(future),
            step_size=step_size,
            freq=freq,
            refit=refit,
            level=level,
            ensemble_weights=ensemble_weights if isinstance(ensemble_weights, dict) else None,
            n_jobs=n_jobs,
        )

    fully_refits = refit is True or (
        isinstance(refit, int) and not isinstance(refit, bool) and refit == 1
    )
    provisional = build_mlf(
        freq=freq,
        models=models,
        strategy=strategy,
        target_transform=target_transform,
        interval_levels=tuple(level or ()),
        random_state=random_state,
        n_jobs=n_jobs,
        model_params=model_params,
    )
    if provisional.requires_refit_for_new_origin and not fully_refits and len(forecast_starts) > 1:
        raise ValueError(
            "Statistical models require refit=True for multiple origins in a mixed-backend run. "
            "Use statistical_backtest for efficient fixed-parameter state updates. "
            "MLForecast and NeuralForecast models can reuse fixed weights with fresh history."
        )

    requested_levels = list(level or [])
    interval_calibration: pl.DataFrame | None = None
    engine_levels = requested_levels
    if requested_levels and not fully_refits and len(forecast_starts) > 1:
        interval_train, interval_test = _calibration_split(
            train_df,
            horizon=horizon,
            step_size=step_size,
            n_windows=interval_calibration_windows,
        )
        interval_calibration, _ = backtest(
            train_df=interval_train,
            test_df=interval_test,
            horizon=horizon,
            hist_exog=list(historical),
            futr_exog=list(future),
            futr_exog_vintages=vintages,
            step_size=step_size,
            freq=freq,
            refit=refit,
            ensemble_weights=None,
            level=[],
            models=models,
            strategy=strategy,
            target_transform=target_transform,
            training_window=training_window,
            random_state=random_state,
            n_jobs=n_jobs,
            model_params=model_params,
        )
        engine_levels = []

    resolved_weights: dict[str, float] | None
    if ensemble_weights == "performance":
        calibration_train, calibration_test = _calibration_split(
            train_df,
            horizon=horizon,
            step_size=step_size,
            n_windows=ensemble_calibration_windows,
        )
        calibration_predictions, _ = backtest(
            train_df=calibration_train,
            test_df=calibration_test,
            horizon=horizon,
            hist_exog=list(historical),
            futr_exog=list(future),
            futr_exog_vintages=vintages,
            step_size=step_size,
            freq=freq,
            refit=True if provisional.requires_refit_for_new_origin else refit,
            ensemble_weights=None,
            level=[],
            models=models,
            strategy=strategy,
            target_transform=target_transform,
            training_window=training_window,
            random_state=random_state,
            n_jobs=n_jobs,
            model_params=model_params,
        )
        resolved_weights = inverse_error_weights(calibration_predictions, models=list(models))
    elif isinstance(ensemble_weights, dict):
        resolved_weights = ensemble_weights
    else:
        resolved_weights = None

    engine: ForecastEngine | None = None
    predictions: list[pl.DataFrame] = []
    windows: list[dict[str, object]] = []
    source = pl.concat([train_df, test_df], how="vertical_relaxed").sort(
        ["unique_id", "ds"]
    )
    for window_id, forecast_start in enumerate(forecast_starts):
        cutoff = forecast_start - delta
        past_covariates, future_covariates = as_of_covariates(
            source,
            cutoff=cutoff,
            horizon=horizon,
            freq=freq,
            hist_exog=historical,
            futr_exog=future,
            futr_exog_vintages=vintages,
        )
        observed_targets = source.filter(pl.col("ds") <= cutoff).select(
            ["unique_id", "ds", "y"]
        )
        observed = observed_targets.join(
            past_covariates,
            on=["unique_id", "ds"],
            how="inner",
            validate="1:1",
        ).sort(["unique_id", "ds"])
        history = _tail_per_series(observed, training_window)
        refitted = _should_refit(refit, window_id)
        if refitted:
            engine = build_mlf(
                freq=freq,
                models=models,
                strategy=strategy,
                target_transform=target_transform,
                interval_levels=tuple(engine_levels),
                random_state=random_state,
                n_jobs=n_jobs,
                model_params=model_params,
            )
            fit(
                engine,
                history,
                horizon=horizon,
                hist_exog=list(historical),
                futr_exog=list(future),
                futr_exog_vintages=vintages,
            )
            history_arg = None
        else:
            if engine is None:
                raise RuntimeError("Backtest engine was not initialized.")
            history_arg = history
        truth = future_covariates.select(["unique_id", "ds"]).join(
            source.select(["unique_id", "ds", "y"]),
            on=["unique_id", "ds"],
            how="left",
            validate="1:1",
        )
        assert engine is not None
        forecast = engine.predict(
            future_covariates,
            horizon=horizon,
            hist_exog=list(historical),
            futr_exog=list(future),
            history_df=history_arg,
            level=engine_levels,
        )
        forecast = forecast.join(
            truth,
            on=["unique_id", "ds"],
            how="left",
            validate="1:1",
        ).with_columns(
            pl.lit(window_id).alias("window_id"),
            pl.lit(cutoff).alias("cutoff"),
            pl.lit(forecast_start).alias("forecast_start"),
            (pl.col("ds").rank("ordinal").over("unique_id")).cast(pl.Int32).alias("horizon_step"),
        )
        predictions.append(forecast)
        windows.append(
            {
                "window_id": window_id,
                "cutoff": cutoff,
                "forecast_start": forecast_start,
                "forecast_end": forecast_start + delta * (horizon - 1),
                "train_start": history["ds"].min(),
                "train_end": history["ds"].max(),
                "train_rows": len(history),
                "refitted": refitted,
                "training_window": training_window,
                "horizon": horizon,
                "step_size": step_size,
            }
        )
    result = pl.concat(predictions, how="vertical_relaxed").sort(["forecast_start", "unique_id", "ds"])
    if resolved_weights is not None:
        result = add_weighted_ensemble(result, resolved_weights)
        for window in windows:
            window["ensemble_weights"] = resolved_weights
    if interval_calibration is not None:
        for model in models:
            result = add_conformal_intervals(
                result,
                calibration=interval_calibration,
                model=model,
                levels=requested_levels,
            )
        if resolved_weights is not None:
            calibrated_ensemble = add_weighted_ensemble(interval_calibration, resolved_weights)
            result = add_conformal_intervals(
                result,
                calibration=calibrated_ensemble,
                model="ensemble",
                levels=requested_levels,
            )
    return result, pl.DataFrame(windows)
