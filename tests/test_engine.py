from __future__ import annotations

import polars as pl

from forecastx.audit import future_target_invariance
from forecastx.engine import build_mlf, fit, predict
from forecastx.outputs import ForecastArtifact


def _split(frame):
    return frame.head(500), frame.tail(120)


def test_mlforecast_predicts_and_cannot_see_future_target(hourly_frame):
    train, future = _split(hourly_frame)
    engine = build_mlf(
        freq="1h",
        models=["LinearRegression", "Ridge"],
        target_transform="none",
        interval_levels=(),
        n_jobs=1,
    )
    fit(engine, train, horizon=24, exog=["mean_temp"])
    forecast = predict(engine, future, horizon=24, exog=["mean_temp"])
    assert forecast.shape == (24, 5)
    audit = future_target_invariance(engine, future, horizon=24, exog=["mean_temp"])
    assert audit.passed
    assert audit.maximum_prediction_change == 0


def test_true_future_does_not_require_unknown_target(hourly_frame):
    train, future = _split(hourly_frame)
    engine = build_mlf(
        freq="1h",
        models=["Ridge"],
        target_transform="none",
        interval_levels=(),
        n_jobs=1,
    )
    fit(engine, train, horizon=24, exog=["mean_temp"])
    result = predict(
        engine,
        future.drop("y"),
        horizon=24,
        exog=["mean_temp"],
    )
    assert len(result) == 24
    assert "y" not in result.columns


def test_direct_strategy(hourly_frame):
    train, future = _split(hourly_frame)
    engine = build_mlf(
        freq="1h",
        models=["Ridge"],
        strategy="direct",
        target_transform="none",
        interval_levels=(),
        n_jobs=1,
    )
    fit(engine, train, horizon=10, exog=["mean_temp"])
    result = predict(engine, future, horizon=10, exog=["mean_temp"])
    assert len(result) == 10


def test_statsforecast_benchmark(hourly_frame):
    train, future = _split(hourly_frame)
    engine = build_mlf(
        freq="1h",
        models=["SeasonalNaive", "ARIMAX"],
        target_transform="none",
        interval_levels=(80,),
        n_jobs=1,
    )
    fit(engine, train, horizon=24, exog=["mean_temp"])
    result = predict(engine, future, horizon=24, exog=["mean_temp"])
    assert {"SeasonalNaive", "ARIMAX", "ARIMAX-lo-80", "ARIMAX-hi-80"} <= set(result.columns)


def test_safe_json_excludes_actuals_and_features(hourly_frame):
    frame = hourly_frame.head(2).with_columns(
        pl.lit(3.2).alias("Ridge"),
        pl.lit(1.0).alias("mean_temp-lo-80"),
    )
    artifact = ForecastArtifact(frame)
    safe = artifact.safe_frame()
    assert "y" not in safe.columns
    assert "mean_temp" not in safe.columns
    assert "Ridge" in safe.columns
    assert "mean_temp-lo-80" not in safe.columns
    assert '"forecasts"' in artifact.to_json()
