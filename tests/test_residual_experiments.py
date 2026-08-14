from datetime import datetime, timedelta

import polars as pl

from scripts.experiment_residual_improvements import (
    THERMAL_EXOG,
    add_adaptive_ensemble,
    add_thermal_dynamics,
)


def test_thermal_features_do_not_depend_on_target() -> None:
    start = datetime(2025, 1, 1)
    frame = pl.DataFrame(
        {
            "unique_id": ["load"] * 48,
            "ds": [start + timedelta(hours=index) for index in range(48)],
            "y": [float(index) for index in range(48)],
            "mean_temp": [5.0 + index / 24 for index in range(48)],
            "mean_temp_squared": [25.0] * 48,
            "heating_degree": [10.0] * 48,
            "cooling_degree": [0.0] * 48,
        }
    )
    scrambled = frame.with_columns(pl.col("y").reverse())
    expected = add_thermal_dynamics(frame).select(THERMAL_EXOG)
    observed = add_thermal_dynamics(scrambled).select(THERMAL_EXOG)
    assert expected.equals(observed)
    assert observed.null_count().sum_horizontal().item() == 0


def test_adaptive_weights_do_not_use_current_window_actuals() -> None:
    start = datetime(2025, 1, 1)
    frame = pl.DataFrame(
        {
            "unique_id": ["load"] * 6,
            "ds": [start + timedelta(hours=index) for index in range(6)],
            "window_id": [0, 0, 1, 1, 2, 2],
            "horizon_step": [1, 2, 1, 2, 1, 2],
            "y": [1.0, 2.0, 2.0, 3.0, 3.0, 4.0],
            "model_a": [1.1, 2.1, 2.2, 3.2, 3.3, 4.3],
            "model_b": [2.0, 3.0, 3.0, 4.0, 4.0, 5.0],
        }
    )
    changed = frame.with_columns(
        pl.when(pl.col("window_id") == 2)
        .then(pl.lit(10_000.0))
        .otherwise(pl.col("y"))
        .alias("y")
    )
    kwargs = {
        "models": ["model_a", "model_b"],
        "initial_weights": {"model_a": 0.5, "model_b": 0.5},
        "lookback_windows": 2,
        "name": "adaptive",
    }
    expected = add_adaptive_ensemble(frame, **kwargs)
    observed = add_adaptive_ensemble(changed, **kwargs)
    assert expected.filter(pl.col("window_id") == 2)["adaptive"].equals(
        observed.filter(pl.col("window_id") == 2)["adaptive"]
    )
