from __future__ import annotations

import importlib.util

import pytest

from forecastx.engine import build_mlf, fit, predict


@pytest.mark.skipif(importlib.util.find_spec("neuralforecast") is None, reason="neural extra is not installed")
def test_small_lstm_smoke(hourly_frame):
    train, future = hourly_frame.head(500), hourly_frame.tail(120)
    engine = build_mlf(
        freq="1h",
        models=["LSTM"],
        interval_levels=(),
        n_jobs=1,
        model_params={
            "LSTM": {
                "max_steps": 1,
                "val_check_steps": 1,
                "early_stop_patience_steps": -1,
            }
        },
    )
    fit(engine, train, horizon=10, exog=["mean_temp"])
    result = predict(engine, future, horizon=10, exog=["mean_temp"])
    assert len(result) == 10
    assert "LSTM" in result.columns
