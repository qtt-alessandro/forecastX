import polars as pl
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from mlforecast.target_transforms import Differences
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from statsforecast.utils import ConformalIntervals
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from config import HORIZON
from src.features import hour_cos, hour_sin, is_weekend


def build_mlf() -> MLForecast:
    """
    Instantiate an MLForecast object with all four models and the
    full feature set (lags, rolling statistics, date features).
    """
    return MLForecast(
        models=[
            LinearRegression(),
            Ridge(alpha=1.0),
            RandomForestRegressor(n_estimators=100, random_state=42),
            XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            #LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1),
        ],
        freq="1h",
        lags=[1, 2, 3, 6, 12, 24, 48, 72, 168],
        lag_transforms={
            24:  [RollingMean(window_size=24), RollingStd(window_size=24, min_samples=1)],
            168: [RollingMean(window_size=168, min_samples=1)],
        },
        target_transforms=[Differences([24])],
        date_features=[
            "hour", "weekday", "month", "year",
            is_weekend, hour_sin, hour_cos,
        ],
    )


def fit(
    mlf: MLForecast,
    train_df: pl.DataFrame,
    horizon: int = HORIZON,
) -> MLForecast:
    """
    Fit *mlf* on *train_df*, enabling conformal prediction intervals.

    The 'forecast' column is dropped before fitting because it is a
    future-looking feature not available at training time.

    Returns the fitted MLForecast instance (mutated in-place, but also
    returned for convenience).
    """
    mlf.fit(
        train_df.drop("forecast"),
        static_features=[],
        validate_data=True,
        #prediction_intervals=ConformalIntervals(n_windows=10, h=horizon),
    )
    return mlf


def predict(
    mlf: MLForecast,
    test_df: pl.DataFrame,
    horizon: int = HORIZON,
    level: list[int] | None = None,
) -> pl.DataFrame:
    """
    Generate a *horizon*-step-ahead forecast and join actuals/baseline.

    Parameters
    ----------
    mlf     : fitted MLForecast instance
    test_df : held-out data that includes 'mean_temp', 'y', 'forecast'
    horizon : number of steps to forecast
    level   : confidence levels for prediction intervals, e.g. [95]

    Returns
    -------
    Polars DataFrame with forecast columns, actuals, and the raw
    'forecast' baseline joined on 'ds'.
    """
    if level is None:
        level = [95]

    X_df = test_df.select(["unique_id", "ds", "mean_temp"])
    forecast_df = mlf.predict(h=horizon, X_df=X_df, level=level)

    return forecast_df.join(
        test_df.select(["ds", "y", "forecast"]),
        on="ds",
        how="left",
    )


def cross_validate(
    mlf: MLForecast,
    df: pl.DataFrame,
    horizon: int = HORIZON,
    n_windows: int = 20,
    step_size: int | None = None,
) -> pl.DataFrame:
    """
    Run time-series cross-validation with *n_windows* expanding windows.

    Parameters
    ----------
    mlf       : MLForecast instance (need not be pre-fitted)
    df        : full dataset (train + test)
    horizon   : forecast horizon per window
    n_windows : number of CV folds
    step_size : gap between window cutoffs (defaults to *horizon*)

    Returns
    -------
    Polars DataFrame with columns: unique_id, ds, cutoff, y, <model>…
    """
    if step_size is None:
        step_size = horizon

    return mlf.cross_validation(
        df=df.select(["unique_id", "ds", "y", "mean_temp"]),
        h=horizon,
        n_windows=n_windows,
        step_size=step_size,
        static_features=[],
        refit=False,
    )
