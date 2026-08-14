"""Central model registry spanning sklearn, StatsForecast, and NeuralForecast."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsforecast.models import ARIMA, AutoARIMA, MSTL, SeasonalNaive
from xgboost import XGBRegressor

from forecastx.config import ForecastConfig


Backend = Literal["mlforecast", "statsforecast", "neuralforecast"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    backend: Backend
    complexity: Literal["low", "medium", "high"]
    supports_future_exog: bool
    default: bool
    description: str
    factory: Callable[[ForecastConfig], Any]


def _linear(_: ForecastConfig) -> LinearRegression:
    return LinearRegression()


def _ridge(_: ForecastConfig):
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def _lasso(_: ForecastConfig):
    return make_pipeline(StandardScaler(), Lasso(alpha=0.001, max_iter=5_000))


def _elastic_net(_: ForecastConfig):
    return make_pipeline(StandardScaler(), ElasticNet(alpha=0.001, l1_ratio=0.3, max_iter=5_000))


def _random_forest(config: ForecastConfig) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        max_features=0.8,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
    )


def _extra_trees(config: ForecastConfig) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        max_features=0.9,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
    )


def _gradient_boosting(config: ForecastConfig) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(n_estimators=200, learning_rate=0.04, random_state=config.random_state)


def _hist_gradient_boosting(config: ForecastConfig) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        l2_regularization=0.1,
        early_stopping=True,
        random_state=config.random_state,
    )


def _lightgbm(config: ForecastConfig) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=350,
        learning_rate=0.035,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
        verbosity=-1,
    )


def _xgboost(config: ForecastConfig) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=350,
        learning_rate=0.035,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        n_jobs=config.n_jobs,
        random_state=config.random_state,
        verbosity=0,
    )


def _knn(_: ForecastConfig):
    return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=12, weights="distance"))


def _mlp(config: ForecastConfig):
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            early_stopping=True,
            max_iter=300,
            random_state=config.random_state,
        ),
    )


def _gaussian_process(config: ForecastConfig):
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    return make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            alpha=1e-5,
            n_restarts_optimizer=0,
            random_state=config.random_state,
        ),
    )


def _seasonal_naive(config: ForecastConfig):
    return SeasonalNaive(season_length=config.profile.seasonal_period, alias="SeasonalNaive")


def _auto_arima(config: ForecastConfig):
    return AutoARIMA(
        season_length=config.profile.seasonal_period,
        max_p=3,
        max_q=3,
        max_P=1,
        max_Q=1,
        max_order=5,
        stepwise=True,
        nmodels=30,
        approximation=True,
        alias="AutoARIMA",
    )


def _arimax(_: ForecastConfig):
    return ARIMA(order=(2, 1, 2), alias="ARIMAX")


def _sarimax(config: ForecastConfig):
    return ARIMA(
        order=(2, 0, 2),
        season_length=config.profile.seasonal_period,
        seasonal_order=(1, 1, 1),
        alias="SARIMAX",
    )


def _mstl(config: ForecastConfig):
    periods = [config.profile.seasonal_period]
    weekly = config.profile.seasonal_period * 7
    if weekly <= 1_000:
        periods.append(weekly)
    return MSTL(season_length=periods, alias="MSTL")


def _lstm(_: ForecastConfig) -> None:
    return None


MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec("LinearRegression", "mlforecast", "low", True, True, "Ordinary least squares baseline.", _linear),
        ModelSpec("Ridge", "mlforecast", "low", True, True, "Scaled L2-regularized linear model.", _ridge),
        ModelSpec("Lasso", "mlforecast", "low", True, False, "Sparse scaled linear model.", _lasso),
        ModelSpec("ElasticNet", "mlforecast", "low", True, False, "Mixed L1/L2 linear model.", _elastic_net),
        ModelSpec("RandomForestRegressor", "mlforecast", "medium", True, True, "Bagged nonlinear trees.", _random_forest),
        ModelSpec("ExtraTreesRegressor", "mlforecast", "medium", True, False, "Randomized tree ensemble.", _extra_trees),
        ModelSpec("GradientBoostingRegressor", "mlforecast", "medium", True, False, "Classical boosted trees.", _gradient_boosting),
        ModelSpec("HistGradientBoostingRegressor", "mlforecast", "medium", True, True, "Fast histogram boosting.", _hist_gradient_boosting),
        ModelSpec("LGBMRegressor", "mlforecast", "medium", True, True, "LightGBM gradient boosting.", _lightgbm),
        ModelSpec("XGBRegressor", "mlforecast", "medium", True, True, "XGBoost gradient boosting.", _xgboost),
        ModelSpec("KNeighborsRegressor", "mlforecast", "medium", True, False, "Scaled local-neighbour regression.", _knn),
        ModelSpec("MLPRegressor", "mlforecast", "medium", True, False, "Small feed-forward neural network.", _mlp),
        ModelSpec("GaussianProcessRegressor", "mlforecast", "high", True, False, "Probabilistic kernel model; cubic fit cost.", _gaussian_process),
        ModelSpec("SeasonalNaive", "statsforecast", "low", False, True, "Mandatory seasonal persistence benchmark.", _seasonal_naive),
        ModelSpec("AutoARIMA", "statsforecast", "medium", True, False, "Nixtla automatic ARIMA benchmark with exogenous support.", _auto_arima),
        ModelSpec("ARIMAX", "statsforecast", "medium", True, False, "Fixed non-seasonal ARIMA with exogenous regressors.", _arimax),
        ModelSpec("SARIMAX", "statsforecast", "high", True, False, "Fixed daily seasonal ARIMA with exogenous regressors.", _sarimax),
        ModelSpec("MSTL", "statsforecast", "medium", False, False, "Multi-seasonal decomposition benchmark.", _mstl),
        ModelSpec("LSTM", "neuralforecast", "high", True, False, "Nixtla encoder-decoder LSTM; optional dependency.", _lstm),
    )
}


def list_models() -> list[dict[str, object]]:
    return [
        {
            "name": spec.name,
            "backend": spec.backend,
            "complexity": spec.complexity,
            "supports_future_exog": spec.supports_future_exog,
            "default": spec.default,
            "description": spec.description,
        }
        for spec in MODEL_REGISTRY.values()
    ]


def validate_model_names(names: tuple[str, ...] | list[str]) -> None:
    unknown = sorted(set(names) - MODEL_REGISTRY.keys())
    if unknown:
        raise ValueError(f"Unknown models {unknown}. Available: {sorted(MODEL_REGISTRY)}")
