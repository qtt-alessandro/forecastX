"""Backtest every usable daily origin across the available boiler history.

This is a retrospective stability analysis, not a replacement for the untouched
April model-selection holdout. Ensemble weights use only the initial calibration
period and are then frozen for the entire January-to-September simulation.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import polars as pl

from forecastx import (
    ForecastArtifact,
    add_temperature_features,
    evaluate_forecasts,
    forecast_dashboard,
    residual_diagnostics,
    split,
)
from forecastx.ensemble import add_weighted_ensemble, inverse_error_weights
from forecastx.heat_demand import load_heat_demand
from scripts.benchmark_heat_demand import ML_MODELS, STAT_MODELS, evaluate_period

DATA_PATH = "data/heat_demand_features_set_old.csv"
INITIAL_START = "2024-12-12"
CALIBRATION_START = "2025-01-02"
BACKTEST_START = "2025-01-09"
BACKTEST_END = "2025-09-24 01:00"


def monthly_metrics(predictions: pl.DataFrame, train_df: pl.DataFrame) -> pl.DataFrame:
    rows: list[pl.DataFrame] = []
    labelled = predictions.with_columns(pl.col("ds").dt.strftime("%Y-%m").alias("month"))
    for month in labelled["month"].unique().sort().to_list():
        metrics = evaluate_forecasts(
            labelled.filter(pl.col("month") == month).drop("month"),
            train_df=train_df,
            seasonal_period=24,
        ).with_columns(pl.lit(month).alias("month"))
        rows.append(metrics)
    return pl.concat(rows).select(
        ["month", "model", "observations", "mae", "rmse", "bias", "smape", "mase"]
    )


def main() -> None:
    data = add_temperature_features(
        load_heat_demand(DATA_PATH),
        heating_balance=15.0,
        cooling_balance=20.0,
    )

    # All ensemble calibration timestamps precede the first reported origin.
    calibration_train, calibration = split(
        data,
        train_start=INITIAL_START,
        test_start=CALIBRATION_START,
        test_end=BACKTEST_START,
    )
    calibration_predictions = evaluate_period(calibration_train, calibration, levels=[])
    models = [*ML_MODELS, *STAT_MODELS]
    weights = inverse_error_weights(calibration_predictions, models=models)

    train, test = split(
        data,
        train_start=INITIAL_START,
        test_start=BACKTEST_START,
        test_end=BACKTEST_END,
    )
    predictions = evaluate_period(train, test, levels=[])
    predictions = add_weighted_ensemble(predictions, weights)
    predictions = predictions.join(
        test.select(["unique_id", "ds", "forecast"]).rename(
            {"forecast": "ProvidedForecast"}
        ),
        on=["unique_id", "ds"],
        how="left",
    )

    overall = evaluate_forecasts(predictions, train_df=train, seasonal_period=24)
    by_month = monthly_metrics(predictions, train)
    residuals = residual_diagnostics(predictions)
    output = Path("output")
    output.mkdir(parents=True, exist_ok=True)

    ForecastArtifact(
        predictions,
        metadata={
            "purpose": "retrospective full-history stability analysis",
            "initial_training_start": INITIAL_START,
            "backtest_start": BACKTEST_START,
            "backtest_end": BACKTEST_END,
            "frequency": "1h",
            "horizon": 24,
            "step_size": 24,
            "initial_training_rows": len(train),
            "forecast_rows": len(predictions),
            "training_window": 672,
            "refit_cadence": 14,
            "ensemble_calibration_end": BACKTEST_START,
            "ensemble_weights": weights,
            "models": models,
            "exogenous_status": (
                "oracle realized temperature; replace with weather-forecast vintages"
            ),
            "caveat": (
                "The final hyperparameters were selected later in the dataset; "
                "use the April holdout report for selection-unbiased comparison."
            ),
        },
    ).write_json(output / "full_history_forecasts.json")
    forecast_dashboard(
        predictions,
        title="Full-history electric-boiler backtest",
    ).write_html(
        output / "full_history_forecast_explorer.html",
        include_plotlyjs=True,
    )
    report = {
        "overall_metrics": overall.to_dicts(),
        "monthly_metrics": by_month.to_dicts(),
        "residual_diagnostics": [asdict(report) for report in residuals],
        "ensemble_weights": weights,
    }
    (output / "full_history_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(overall)
    print(by_month.filter(pl.col("model").is_in(["ensemble", "ProvidedForecast"])))


if __name__ == "__main__":
    main()
