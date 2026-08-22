"""Unified multi-backend forecasting engine with a compact functional API."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any, cast

import polars as pl
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExponentiallyWeightedMean, RollingMean, RollingStd
from mlforecast.target_transforms import Differences
from mlforecast.utils import PredictionIntervals
from statsforecast import StatsForecast

from forecastx.config import FAST_MODELS, ForecastConfig
from forecastx.covariates import resolve_covariate_names, validate_vintage_mapping
from forecastx.data import frequency_delta, validate_schema
from forecastx.diagnostics import StationarityReport, stationarity_report
from forecastx.ensemble import add_weighted_ensemble
from forecastx.features import hour_cos, hour_sin, is_weekend, week_cos, week_sin
from forecastx.models import MODEL_REGISTRY, validate_model_names


def _as_polars(frame: Any) -> pl.DataFrame:
    return frame if isinstance(frame, pl.DataFrame) else pl.from_pandas(frame)


def _validate_historical_vintages(
    frame: pl.DataFrame,
    vintages: dict[str, str],
    *,
    context: str,
) -> None:
    for vintage_column in dict.fromkeys(vintages.values()):
        if vintage_column not in frame.columns:
            raise ValueError(f"{context} is missing forecast vintage column {vintage_column!r}.")
        if not isinstance(frame.schema[vintage_column], pl.Datetime):
            raise TypeError(f"Forecast vintage column {vintage_column!r} must be a Polars Datetime.")
        if frame[vintage_column].null_count():
            raise ValueError(f"{context} vintage column {vintage_column!r} contains nulls.")
        if frame.filter(pl.col(vintage_column) >= pl.col("ds")).height:
            raise ValueError(
                f"{context} vintage column {vintage_column!r} was not issued before its target."
            )


class ForecastEngine:
    """One fitted forecast origin across sklearn, statistical, and neural models."""

    def __init__(self, config: ForecastConfig):
        validate_model_names(config.models)
        unused_parameters = sorted(set(config.model_params) - set(config.models))
        if unused_parameters:
            raise ValueError(f"model_params were provided for unselected models: {unused_parameters}")
        self.config = config
        self.horizon: int | None = None
        self.hist_exog: tuple[str, ...] = ()
        self.futr_exog: tuple[str, ...] = ()
        self.exog: tuple[str, ...] = ()
        self.futr_exog_vintages: dict[str, str] = {}
        self.stationarity: list[StationarityReport] = []
        self.selected_transform: str = config.target_transform
        self._mlf: MLForecast | None = None
        self._stats_exog: StatsForecast | None = None
        self._stats_univariate: StatsForecast | None = None
        self._neural: Any | None = None
        self._train_df: pl.DataFrame | None = None
        self._fit_end: dict[str, datetime] = {}
        self._intervals_calibrated = False

    @property
    def model_names(self) -> list[str]:
        return list(self.config.models)

    @property
    def requires_refit_for_new_origin(self) -> bool:
        return any(MODEL_REGISTRY[name].backend == "statsforecast" for name in self.config.models)

    def _resolve_transform(self, train_df: pl.DataFrame) -> list[Differences] | None:
        requested = self.config.target_transform
        self.stationarity = stationarity_report(
            train_df,
            seasonal_period=self.config.profile.seasonal_period,
        )
        if requested == "auto":
            recommendations = {report.recommended_transform for report in self.stationarity}
            requested = recommendations.pop() if len(recommendations) == 1 else "none"
        self.selected_transform = requested
        if requested == "difference_1":
            return [Differences([1])]
        if requested == "difference_seasonal":
            return [Differences([self.config.profile.seasonal_period])]
        return None

    def _build_mlforecast(self, target_transforms: list[Differences] | None) -> MLForecast | None:
        names = [name for name in self.config.models if MODEL_REGISTRY[name].backend == "mlforecast"]
        if not names:
            return None
        estimators: dict[str, Any] = {}
        for name in names:
            estimator = MODEL_REGISTRY[name].factory(self.config)
            parameters = self.config.model_params.get(name, {})
            if parameters:
                estimator.set_params(**parameters)
            estimators[name] = estimator
        rolling: list[Any] = []
        for window in self.config.profile.rolling_windows:
            rolling.extend(
                [
                    RollingMean(window_size=window, min_samples=max(2, window // 2)),
                    RollingStd(window_size=window, min_samples=max(2, window // 2)),
                ]
            )
        rolling.append(ExponentiallyWeightedMean(alpha=0.1))
        date_features: list[Any] = [*self.config.profile.date_features, is_weekend]
        if self.config.freq in {"1h", "15m", "1m"}:
            date_features.extend([hour_sin, hour_cos, week_sin, week_cos])
        return MLForecast(
            models=estimators,
            freq=self.config.freq,
            lags=list(self.config.profile.lags),
            lag_transforms={1: rolling},
            date_features=date_features,
            target_transforms=target_transforms,
            num_threads=max(1, self.config.n_jobs if self.config.n_jobs > 0 else 1),
        )

    def _fit_statsforecast(self, train_df: pl.DataFrame) -> None:
        stat_names = [name for name in self.config.models if MODEL_REGISTRY[name].backend == "statsforecast"]
        if not stat_names:
            return
        exog_names = [name for name in stat_names if MODEL_REGISTRY[name].supports_future_exog]
        univariate_names = [name for name in stat_names if not MODEL_REGISTRY[name].supports_future_exog]
        jobs = self.config.n_jobs
        if exog_names:
            self._stats_exog = StatsForecast(
                models=[MODEL_REGISTRY[name].factory(self.config) for name in exog_names],
                freq=self.config.freq,
                n_jobs=jobs,
            )
            columns = ["unique_id", "ds", "y", *self.futr_exog]
            self._stats_exog.fit(train_df.select(columns))
        if univariate_names:
            self._stats_univariate = StatsForecast(
                models=[MODEL_REGISTRY[name].factory(self.config) for name in univariate_names],
                freq=self.config.freq,
                n_jobs=jobs,
            )
            self._stats_univariate.fit(train_df.select(["unique_id", "ds", "y"]))

    def _fit_neuralforecast(self, train_df: pl.DataFrame, horizon: int) -> None:
        if "LSTM" not in self.config.models:
            return
        try:
            import torch
            from neuralforecast import NeuralForecast
            from neuralforecast.losses.pytorch import DistributionLoss
            from neuralforecast.models import LSTM
        except ImportError as exc:
            raise ImportError(
                "LSTM requires the optional dependency: install this package with `pip install 'fcastx[neural]'`."
            ) from exc
        # Multiple native OpenMP runtimes (LightGBM/XGBoost/PyTorch) can crash
        # macOS processes during LSTM backpropagation. A small load model is also
        # faster and more predictable when it does not oversubscribe CPU cores.
        torch.set_num_threads(max(1, self.config.n_jobs if self.config.n_jobs > 0 else 1))
        loss = DistributionLoss(distribution="Normal", level=list(self.config.interval_levels))
        parameters: dict[str, Any] = {
            "h": horizon,
            "input_size": max(horizon * 3, self.config.profile.seasonal_period * 7),
            "encoder_n_layers": 1,
            "encoder_hidden_size": 32,
            "max_steps": 150,
            "early_stop_patience_steps": 3,
            "val_check_steps": 25,
            "scaler_type": "robust",
            "hist_exog_list": list(self.hist_exog),
            "futr_exog_list": list(self.futr_exog),
            "loss": loss,
            "random_seed": self.config.random_state,
            "alias": "LSTM",
            "enable_progress_bar": False,
            "logger": False,
        }
        parameters.update(self.config.model_params.get("LSTM", {}))
        model = LSTM(**parameters)
        self._neural = NeuralForecast(models=[model], freq=self.config.freq)
        columns = ["unique_id", "ds", "y", *self.hist_exog, *self.futr_exog]
        self._neural.fit(
            df=train_df.select(columns).to_pandas(),
            val_size=max(horizon * 2, 24),
        )

    def fit(
        self,
        train_df: pl.DataFrame,
        *,
        horizon: int,
        exog: list[str] | None = None,
        hist_exog: list[str] | None = None,
        futr_exog: list[str] | None = None,
        futr_exog_vintages: dict[str, str] | None = None,
        training_window: int | None = None,
    ) -> "ForecastEngine":
        if horizon < 1:
            raise ValueError("horizon must be positive.")
        historical, future = resolve_covariate_names(
            exog=exog,
            hist_exog=hist_exog,
            futr_exog=futr_exog,
        )
        vintages = validate_vintage_mapping(future, futr_exog_vintages)
        validate_schema(train_df, exog=[*historical, *future], require_complete=True)
        _validate_historical_vintages(train_df, vintages, context="Training data")
        if training_window is not None:
            if training_window <= max(self.config.profile.lags) + horizon:
                raise ValueError("training_window must exceed the largest lag plus the forecast horizon.")
            train_df = (
                train_df.sort(["unique_id", "ds"])
                .group_by("unique_id", maintain_order=True)
                .tail(training_window)
                .sort(["unique_id", "ds"])
            )
        if train_df.select(pl.struct(["unique_id", "ds"]).is_duplicated().any()).item():
            raise ValueError("Duplicate (unique_id, ds) rows are not allowed.")
        self.horizon = horizon
        self.hist_exog = historical
        self.futr_exog = future
        self.exog = future  # Backwards-compatible name for the legacy future-only API.
        self.futr_exog_vintages = vintages
        self._train_df = train_df.sort(["unique_id", "ds"])
        self._fit_end = {
            str(series["unique_id"][0]): cast(datetime, series["ds"].max())
            for series in self._train_df.partition_by("unique_id", maintain_order=True)
        }
        target_transforms = self._resolve_transform(self._train_df)
        self._mlf = self._build_mlforecast(target_transforms)
        if self._mlf is not None:
            minimum_rows = min(len(series) for series in self._train_df.partition_by("unique_id"))
            max_lag = max(self.config.profile.lags)
            possible_windows = (minimum_rows - max_lag - horizon - 1) // horizon
            interval_windows = min(self.config.interval_windows, possible_windows)
            prediction_intervals = None
            if self.config.interval_levels and interval_windows >= 2:
                prediction_intervals = PredictionIntervals(n_windows=interval_windows, h=horizon)
                self._intervals_calibrated = True
            fit_kwargs: dict[str, Any] = {
                "static_features": [],
                "prediction_intervals": prediction_intervals,
                "validate_data": True,
                "as_numpy": True,
            }
            if self.config.strategy == "direct":
                fit_kwargs["max_horizon"] = horizon
            columns = ["unique_id", "ds", "y", *self.futr_exog]
            self._mlf.fit(self._train_df.select(columns), **fit_kwargs)
        self._fit_statsforecast(self._train_df)
        self._fit_neuralforecast(self._train_df, horizon)
        return self

    def _future_window(self, future_df: pl.DataFrame, history_df: pl.DataFrame, horizon: int) -> pl.DataFrame:
        vintage_columns = list(dict.fromkeys(self.futr_exog_vintages.values()))
        required = ["unique_id", "ds", *self.futr_exog, *vintage_columns]
        missing = [column for column in required if column not in future_df.columns]
        if missing:
            raise ValueError(f"Future data is missing required columns: {missing}")
        if not isinstance(future_df.schema["ds"], pl.Datetime):
            raise TypeError("Future column 'ds' must be a Polars Datetime.")
        nulls = future_df.select(required).null_count().row(0, named=True)
        if nonzero := {name: count for name, count in nulls.items() if count}:
            raise ValueError(f"Null values remain in future inputs: {nonzero}")
        delta = frequency_delta(self.config.freq)
        windows: list[pl.DataFrame] = []
        ids = history_df["unique_id"].unique(maintain_order=True).to_list()
        for unique_id in ids:
            history = history_df.filter(pl.col("unique_id") == unique_id)
            last = cast(datetime, history["ds"].max())
            expected = [last + delta * step for step in range(1, horizon + 1)]
            window = (
                future_df.filter(pl.col("unique_id") == unique_id)
                .filter(pl.col("ds").is_in(expected))
                .sort("ds")
            )
            if len(window) != horizon or window["ds"].to_list() != expected:
                raise ValueError(
                    f"Future data for {unique_id!r} must contain exactly {horizon} contiguous rows after {last}."
                )
            if window.filter(pl.col("ds") <= last).height:
                raise ValueError("Future model inputs contain timestamps at or before the forecast cutoff.")
            for feature, vintage_column in self.futr_exog_vintages.items():
                if window.filter(pl.col(vintage_column) > last).height:
                    raise ValueError(
                        f"Future covariate {feature!r} uses a vintage issued after cutoff {last}."
                    )
                if window.filter(pl.col(vintage_column) >= pl.col("ds")).height:
                    raise ValueError(
                        f"Future covariate {feature!r} contains a vintage not issued before its target."
                    )
            windows.append(window)
        return pl.concat(windows).sort(["unique_id", "ds"])

    def predict(
        self,
        future_df: pl.DataFrame,
        *,
        horizon: int | None = None,
        exog: list[str] | None = None,
        hist_exog: list[str] | None = None,
        futr_exog: list[str] | None = None,
        history_df: pl.DataFrame | None = None,
        level: list[int] | None = None,
        ensemble_weights: dict[str, float] | None = None,
    ) -> pl.DataFrame:
        if self._train_df is None or self.horizon is None:
            raise RuntimeError("Fit the engine before predicting.")
        horizon = horizon or self.horizon
        if self.config.strategy == "direct" and horizon != self.horizon:
            raise ValueError("A direct model can only predict the horizon used during fitting.")
        requested_historical, requested_future = resolve_covariate_names(
            exog=exog,
            hist_exog=hist_exog,
            futr_exog=futr_exog,
        )
        if requested_historical != self.hist_exog or requested_future != self.futr_exog:
            raise ValueError(
                "Prediction covariate roles differ from the fitted roles: "
                f"hist={self.hist_exog}, futr={self.futr_exog}."
            )
        history = history_df.sort(["unique_id", "ds"]) if history_df is not None else self._train_df
        validate_schema(
            history,
            exog=[*self.hist_exog, *self.futr_exog],
            require_complete=True,
        )
        _validate_historical_vintages(
            history,
            self.futr_exog_vintages,
            context="Prediction history",
        )
        changed_origin = any(
            cast(datetime, series["ds"].max()) != self._fit_end[str(series["unique_id"][0])]
            for series in history.partition_by("unique_id", maintain_order=True)
        )
        if changed_origin and self.requires_refit_for_new_origin:
            raise ValueError("Selected statistical/neural models require refitting at a changed forecast origin.")
        window = self._future_window(future_df, history, horizon)
        levels = level if level is not None else list(self.config.interval_levels)
        frames: list[pl.DataFrame] = []
        x_df = window.select(["unique_id", "ds", *self.futr_exog]) if self.futr_exog else None
        if self._mlf is not None:
            kwargs: dict[str, Any] = {"h": horizon, "X_df": x_df}
            if history_df is not None:
                kwargs["new_df"] = history.select(
                    ["unique_id", "ds", "y", *self.futr_exog]
                )
            if levels and self._intervals_calibrated:
                kwargs["level"] = levels
            frames.append(_as_polars(self._mlf.predict(**kwargs)))
        if self._stats_exog is not None:
            frames.append(_as_polars(self._stats_exog.predict(h=horizon, X_df=x_df, level=levels or None)))
        if self._stats_univariate is not None:
            frames.append(_as_polars(self._stats_univariate.predict(h=horizon, level=levels or None)))
        if self._neural is not None:
            neural_future = window.select(["unique_id", "ds", *self.futr_exog]).to_pandas()
            neural_kwargs: dict[str, Any] = {"futr_df": neural_future}
            if history_df is not None:
                neural_kwargs["df"] = history.select(
                    ["unique_id", "ds", "y", *self.hist_exog, *self.futr_exog]
                ).to_pandas()
            frames.append(_as_polars(self._neural.predict(**neural_kwargs)))
        if not frames:
            raise RuntimeError("No forecast backends were fitted.")
        forecasts = frames[0]
        for frame in frames[1:]:
            forecasts = forecasts.join(frame, on=["unique_id", "ds"], how="inner")
        if "y" in window.columns:
            forecasts = forecasts.join(
                window.select(["unique_id", "ds", "y"]),
                on=["unique_id", "ds"],
                how="left",
            )
        if ensemble_weights is not None:
            forecasts = add_weighted_ensemble(forecasts, ensemble_weights)
        return forecasts.sort(["unique_id", "ds"])


def build_mlf(
    *,
    freq: str = "1h",
    models: list[str] | tuple[str, ...] | None = None,
    strategy: str = "recursive",
    target_transform: str = "none",
    interval_levels: tuple[int, ...] = (80, 95),
    interval_windows: int = 5,
    random_state: int = 42,
    n_jobs: int = 1,
    model_params: dict[str, dict[str, Any]] | None = None,
) -> ForecastEngine:
    """Build the unified engine while preserving the original public call shape."""

    config = ForecastConfig(
        freq=freq,
        models=tuple(models or FAST_MODELS),
        strategy=cast(Any, strategy),
        target_transform=cast(Any, target_transform),
        interval_levels=interval_levels,
        interval_windows=interval_windows,
        random_state=random_state,
        n_jobs=n_jobs,
        model_params=model_params or {},
    )
    return ForecastEngine(config)


def fit(
    mlf: ForecastEngine,
    train_df: pl.DataFrame,
    *,
    horizon: int,
    exog: list[str] | None = None,
    hist_exog: list[str] | None = None,
    futr_exog: list[str] | None = None,
    futr_exog_vintages: dict[str, str] | None = None,
    training_window: int | None = None,
) -> ForecastEngine:
    return mlf.fit(
        train_df,
        horizon=horizon,
        exog=exog,
        hist_exog=hist_exog,
        futr_exog=futr_exog,
        futr_exog_vintages=futr_exog_vintages,
        training_window=training_window,
    )


def predict(
    mlf: ForecastEngine,
    test_df: pl.DataFrame,
    *,
    horizon: int,
    exog: list[str] | None = None,
    hist_exog: list[str] | None = None,
    futr_exog: list[str] | None = None,
    level: list[int] | None = None,
    ensemble_weights: dict[str, float] | None = None,
    history_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    return mlf.predict(
        test_df,
        horizon=horizon,
        exog=exog,
        hist_exog=hist_exog,
        futr_exog=futr_exog,
        level=level,
        ensemble_weights=ensemble_weights,
        history_df=history_df,
    )
