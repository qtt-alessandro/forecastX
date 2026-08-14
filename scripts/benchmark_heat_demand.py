"""Reproducible electric-boiler benchmark using only pre-test model selection."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import polars as pl

from forecastx import (
    ForecastArtifact,
    add_conformal_intervals,
    add_temperature_features,
    backtest,
    evaluate_forecasts,
    forecast_dashboard,
    interval_metrics,
    residual_diagnostics,
    split,
    statistical_backtest,
)
from forecastx.ensemble import add_weighted_ensemble, inverse_error_weights
from loaders.heat_demand import load

DATA_PATH = "data/heat_demand_features_set_old.csv"
HORIZON = 24
STEP_SIZE = 24
FREQ = "1h"
WEATHER_EXOG = ["mean_temp", "mean_temp_squared", "heating_degree", "cooling_degree"]
STATISTICAL_EXOG = ["mean_temp"]
ML_MODELS = ["Ridge", "GradientBoostingRegressor", "LGBMRegressor"]
STAT_MODELS = ["SeasonalNaive", "AutoARIMA", "SARIMAX"]
RIDGE_PARAMS = {"Ridge": {"ridge__alpha": 100.0}}


def merge_backends(ml: pl.DataFrame, statistical: pl.DataFrame) -> pl.DataFrame:
    keys = ["unique_id", "ds", "cutoff"]
    timing = {"y", "forecast_start", "window_id", "horizon_step"}
    extra = [column for column in statistical.columns if column not in {*keys, *timing}]
    return ml.join(statistical.select([*keys, *extra]), on=keys, how="inner")


def evaluate_period(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    *,
    levels: list[int],
) -> pl.DataFrame:
    ml, _ = backtest(
        train_df=train_df,
        test_df=test_df,
        horizon=HORIZON,
        step_size=STEP_SIZE,
        freq=FREQ,
        exog=WEATHER_EXOG,
        models=ML_MODELS,
        target_transform="none",
        training_window=672,
        refit=14,
        level=levels,
        n_jobs=1,
        model_params=RIDGE_PARAMS,
    )
    statistical, _ = statistical_backtest(
        train_df=train_df,
        test_df=test_df,
        horizon=HORIZON,
        step_size=STEP_SIZE,
        freq=FREQ,
        exog=STATISTICAL_EXOG,
        models=STAT_MODELS,
        refit=False,
        level=levels,
        n_jobs=1,
    )
    return merge_backends(ml, statistical)


def main() -> None:
    data = add_temperature_features(load(DATA_PATH), heating_balance=15.0, cooling_balance=20.0)

    # Model and ensemble selection: ends before the final April evaluation period.
    validation_train, validation = split(
        data,
        train_start="2025-01-01",
        test_start="2025-03-15",
        test_end="2025-04-15",
    )
    validation_predictions = evaluate_period(validation_train, validation, levels=[])
    ensemble_models = [*ML_MODELS, *STAT_MODELS]
    weights = inverse_error_weights(validation_predictions, models=ensemble_models)
    validation_predictions = add_weighted_ensemble(validation_predictions, weights)

    # Final period: hyperparameters and weights are now frozen.
    train, test = split(
        data,
        train_start="2025-01-01",
        test_start="2025-04-15",
        test_end="2025-04-28 01:00",
    )
    predictions = evaluate_period(train, test, levels=[80, 95])
    predictions = add_weighted_ensemble(predictions, weights)
    predictions = add_conformal_intervals(
        predictions,
        calibration=validation_predictions,
        model="ensemble",
        levels=[80, 95],
    )
    predictions = predictions.join(
        test.select(["unique_id", "ds", "forecast"]).rename({"forecast": "ProvidedForecast"}),
        on=["unique_id", "ds"],
        how="left",
    )

    metrics = evaluate_forecasts(predictions, train_df=train, seasonal_period=24)
    coverage = interval_metrics(predictions)
    residuals = residual_diagnostics(predictions)
    output = Path("artifacts")
    output.mkdir(parents=True, exist_ok=True)
    artifact = ForecastArtifact(
        predictions,
        metadata={
            "frequency": FREQ,
            "horizon": HORIZON,
            "step_size": STEP_SIZE,
            "training_window": 672,
            "refit_cadence": 14,
            "models": ensemble_models,
            "ensemble_weights": weights,
            "exogenous_columns": WEATHER_EXOG,
            "statistical_exogenous_columns": STATISTICAL_EXOG,
            "exogenous_status": "oracle realized temperature; replace with weather-forecast vintages",
            "target_imputed_rows": int(data["y_imputed"].sum()),
        },
    )
    artifact.write_json(output / "heat_demand_forecast.json")
    forecast_dashboard(predictions, title="Electric-boiler heat-demand benchmark").write_html(
        output / "heat_demand_forecast_explorer.html",
        include_plotlyjs=True,
    )
    report = {
        "validation_metrics": evaluate_forecasts(
            validation_predictions,
            train_df=validation_train,
            seasonal_period=24,
        ).to_dicts(),
        "final_metrics": metrics.to_dicts(),
        "interval_metrics": coverage.to_dicts(),
        "residual_diagnostics": [asdict(report) for report in residuals],
        "ensemble_weights": weights,
    }
    (output / "heat_demand_benchmark.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(metrics)
    print(coverage)
    for report in residuals:
        print(report)


if __name__ == "__main__":
    main()
