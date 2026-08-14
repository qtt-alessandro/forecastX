from __future__ import annotations

from forecastx.backtest import backtest, statistical_backtest
from forecastx.metrics import evaluate_forecasts


def test_horizon_and_stride_are_independent(hourly_frame):
    train, test = hourly_frame.head(500), hourly_frame.tail(120)
    common = dict(
        train_df=train,
        test_df=test,
        horizon=10,
        freq="1h",
        exog=["mean_temp"],
        models=["Ridge"],
        target_transform="none",
        level=[],
        n_jobs=1,
    )
    every_step, every_step_windows = backtest(step_size=1, refit=False, **common)
    every_ten, every_ten_windows = backtest(step_size=10, refit=False, **common)
    assert len(every_step_windows) == 111
    assert len(every_ten_windows) == 12
    assert every_step["horizon_step"].min() == 1
    assert every_step["horizon_step"].max() == 10


def test_performance_ensemble_is_calibrated_before_test(hourly_frame):
    train, test = hourly_frame.head(500), hourly_frame.tail(120)
    predictions, windows = backtest(
        train_df=train,
        test_df=test,
        horizon=12,
        step_size=12,
        freq="1h",
        exog=["mean_temp"],
        models=["LinearRegression", "Ridge"],
        target_transform="none",
        refit=False,
        ensemble_weights="performance",
        ensemble_calibration_windows=3,
        level=[],
        n_jobs=1,
    )
    assert "ensemble" in predictions.columns
    assert "ensemble_weights" in windows.columns
    metrics = evaluate_forecasts(predictions, train_df=train, seasonal_period=24)
    assert "ensemble" in metrics["model"].to_list()


def test_fixed_model_intervals_use_pretest_calibration(hourly_frame):
    train, test = hourly_frame.head(500), hourly_frame.tail(120)
    predictions, _ = backtest(
        train_df=train,
        test_df=test,
        horizon=10,
        step_size=10,
        freq="1h",
        exog=["mean_temp"],
        models=["Ridge"],
        target_transform="none",
        refit=False,
        level=[80, 95],
        interval_calibration_windows=10,
        n_jobs=1,
    )
    assert {"Ridge-lo-80", "Ridge-hi-80", "Ridge-lo-95", "Ridge-hi-95"} <= set(predictions.columns)


def test_statistical_state_update_backtest(hourly_frame):
    train, test = hourly_frame.head(500), hourly_frame.tail(120)
    predictions, windows = statistical_backtest(
        train_df=train,
        test_df=test,
        horizon=12,
        step_size=12,
        freq="1h",
        exog=["mean_temp"],
        models=["SeasonalNaive", "ARIMAX"],
        refit=False,
        level=[80],
        n_jobs=1,
    )
    assert len(windows) == 10
    assert {"SeasonalNaive", "ARIMAX", "horizon_step"} <= set(predictions.columns)
