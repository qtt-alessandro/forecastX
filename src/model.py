import polars as pl
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from mlforecast.target_transforms import Differences
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from xgboost import XGBRegressor

from config import HORIZON
from src.features import hour_cos, hour_sin, is_weekend

_RESERVED = {"unique_id", "ds", "y"}

_FREQ_CONFIGS = {
    "1h": dict(
        lags=[1, 2, 3, 6, 12, 24, 48, 72, 168],
        lag_transforms={
            24:  [RollingMean(window_size=24), RollingStd(window_size=24, min_samples=1)],
            168: [RollingMean(window_size=168, min_samples=1)],
        },
        target_transforms=[Differences([24])],
        date_features=["hour", "weekday", "month", "year", is_weekend, hour_sin, hour_cos],
    ),
    "15m": dict(
        lags=[1, 2, 3, 4, 8, 12, 16, 24, 48, 96, 672],
        lag_transforms={
            4:  [RollingMean(window_size=4), RollingStd(window_size=4, min_samples=1)],
            96: [RollingMean(window_size=96, min_samples=1)],
        },
        target_transforms=[Differences([1])],
        date_features=["minute", "hour", "weekday", is_weekend, hour_sin, hour_cos],
    ),
    "1m": dict(
        lags=[1, 2, 3, 5, 8, 10, 12, 15, 20, 30, 40, 50, 60],
        lag_transforms={
            15: [RollingMean(window_size=15), RollingStd(window_size=15, min_samples=1)],
            60: [RollingMean(window_size=60, min_samples=1)],
        },
        target_transforms=[Differences([1])],
        date_features=["minute", "hour", "weekday"],
    ),
}


# Available models, by name. build_mlf() only includes DEFAULT_MODELS unless
# the caller asks for specific ones -- e.g. GaussianProcessRegressor is O(n^3)
# in training rows, so it's opt-in per pipeline rather than always fit.
MODEL_REGISTRY = {
    "LinearRegression": lambda: LinearRegression(),
    "Ridge": lambda: Ridge(alpha=1.0),
    "RandomForestRegressor": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBRegressor": lambda: XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "GaussianProcessRegressor": lambda: GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
        normalize_y=True,
        alpha=1e-6,
        random_state=42,
    ),
}

DEFAULT_MODELS = ["LinearRegression", "Ridge", "RandomForestRegressor", "XGBRegressor"]


def build_mlf(freq: str = "1h", models: list[str] | None = None) -> MLForecast:
    """
    Instantiate MLForecast for the given frequency.
    Supported: '1h' (hourly), '15m' (quarter-hourly), '1m' (minutely).

    `models` selects which entries from MODEL_REGISTRY to fit, by name.
    Defaults to DEFAULT_MODELS.
    """
    cfg = _FREQ_CONFIGS[freq]
    names = models or DEFAULT_MODELS
    return MLForecast(
        models=[MODEL_REGISTRY[name]() for name in names],
        freq=freq,
        **cfg,
    )


def fit(
    mlf: MLForecast,
    train_df: pl.DataFrame,
    horizon: int = HORIZON,
    exog: list[str] | None = None,
) -> MLForecast:
    keep = _RESERVED | set(exog or [])
    mlf.fit(
        train_df.select([c for c in train_df.columns if c in keep]),
        static_features=[],
        validate_data=True,
    )
    return mlf


def predict(
    mlf: MLForecast,
    test_df: pl.DataFrame,
    horizon: int = HORIZON,
    exog: list[str] | None = None,
    level: list[int] | None = None,
) -> pl.DataFrame:
    X_df = test_df.select(["unique_id", "ds"] + exog) if exog else None
    forecast_df = mlf.predict(h=horizon, X_df=X_df, level=level or [95])
    test_tz = test_df.schema["ds"].time_zone if hasattr(test_df.schema["ds"], "time_zone") else None
    if test_tz:
        forecast_df = forecast_df.with_columns(pl.col("ds").dt.replace_time_zone(test_tz))
    return forecast_df.join(
        test_df.select(["ds", "y"]),
        on="ds",
        how="left",
    )


def cross_validate(
    mlf: MLForecast,
    df: pl.DataFrame,
    horizon: int = HORIZON,
    exog: list[str] | None = None,
    n_windows: int = 20,
    step_size: int | None = None,
) -> pl.DataFrame:
    if step_size is None:
        step_size = horizon
    keep = ["unique_id", "ds", "y"] + (exog or [])
    return mlf.cross_validation(
        df=df.select(keep),
        h=horizon,
        n_windows=n_windows,
        step_size=step_size,
        static_features=[],
        refit=False,
    )