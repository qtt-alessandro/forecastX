from __future__ import annotations

import polars as pl

from forecastx.reporting import model_comparison_markdown


def test_model_comparison_report_ranks_models_and_explains_ensemble():
    metrics = pl.DataFrame(
        {
            "model": ["Ridge", "XGBRegressor", "ensemble"],
            "observations": [100, 100, 100],
            "mae": [1.0, 1.05, 1.2],
            "rmse": [1.4, 1.3, 1.5],
            "smape": [0.12, 0.11, 0.10],
            "mase": [0.8, 0.85, 0.9],
            "bias": [0.1, -0.02, 0.03],
        }
    )

    report = model_comparison_markdown(
        metrics,
        ensemble_weights={"Ridge": 0.6, "XGBRegressor": 0.4},
        configuration={
            "horizon": 24,
            "backtest_start": "2025-01-01",
            "backtest_end": "2025-02-01",
        },
        elapsed_seconds=120.0,
    )

    assert "**Ridge** has the lowest MAE" in report
    assert "🟢 **1.0000**" in report
    assert "🟡 Within 10%" in report
    assert "🔴 Above 10%" in report
    assert "| Ridge | 60.00% |" in report
    assert "20% shrinkage toward equal weighting" in report
