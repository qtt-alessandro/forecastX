"""Full-series heat-demand comparison across configured models and ensemble."""

#%% Imports
from __future__ import annotations

from pathlib import Path
from time import perf_counter

from forecastx import add_temperature_features, backtest, split
from forecastx.config import (
    load_covariate_configuration,
    load_model_configuration,
    load_run_configuration,
)
from forecastx.heat_demand import (
    add_synthetic_temperature_forecast,
    add_temporal_features,
    load_heat_demand,
    validate_feature_columns,
)
from forecastx.metrics import evaluate_backtest
from forecastx.outputs import export_backtest_results, print_backtest_summary


#%% Run configuration — edit backtest parameters here
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "heat_demand_features_set_old.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
CONFIG_DIR = PROJECT_ROOT / "config"
MODEL_MANIFEST_PATH = CONFIG_DIR / "models.json"
FEATURE_CONFIG_PATH = CONFIG_DIR / "features.json"
RUN_CONFIG_PATH = CONFIG_DIR / "run.json"


#%% Execute the pipeline
if __name__ == "__main__":
    run_config = load_run_configuration(RUN_CONFIG_PATH)
    selected_models, selected_model_params = load_model_configuration(
        MODEL_MANIFEST_PATH
    )
    hist_exog, futr_exog, futr_exog_vintages = load_covariate_configuration(
        FEATURE_CONFIG_PATH
    )

    raw_df = load_heat_demand(DATA_PATH, frequency=run_config.frequency)
    weather_df = add_synthetic_temperature_forecast(
        raw_df,
        freq=run_config.frequency,
        lead_steps=run_config.horizon,
        random_state=run_config.random_seed,
    )
    actual_temperature_df = add_temperature_features(
        weather_df,
        column="mean_temp_actual",
        role="actual",
        heating_balance=15.0,
        cooling_balance=20.0,
    )
    temperature_df = add_temperature_features(
        actual_temperature_df,
        column="mean_temp_forecast",
        role="forecast",
        heating_balance=15.0,
        cooling_balance=20.0,
    )
    model_df = add_temporal_features(temperature_df)
    validate_feature_columns(
        model_df,
        [*hist_exog, *futr_exog, *futr_exog_vintages.values()],
    )

    train_df, test_df = split(
        model_df,
        train_start=run_config.train_start,
        test_start=run_config.backtest_start,
        test_end=run_config.backtest_end,
    )

    started = perf_counter()
    predictions_df, windows_df = backtest(
        train_df=train_df,
        test_df=test_df,
        horizon=run_config.horizon,
        step_size=run_config.step_size,
        freq=run_config.frequency,
        hist_exog=hist_exog,
        futr_exog=futr_exog,
        futr_exog_vintages=futr_exog_vintages,
        models=selected_models,
        refit=run_config.refit_every,
        training_window=run_config.training_window,
        ensemble_weights="performance",
        ensemble_calibration_windows=5,
        level=[80, 95],
        random_state=run_config.random_seed,
        n_jobs=run_config.n_jobs,
        model_params=selected_model_params,
    )
    training_seconds = perf_counter() - started
    if "LSTM-median" in predictions_df.columns:
        predictions_df = predictions_df.drop("LSTM-median")

    metrics_by_scope = evaluate_backtest(
        predictions_df,
        train_df=train_df,
        seasonal_period=run_config.seasonal_period,
    )
    run_configuration = {
        **run_config.to_dict(),
        "models": selected_models,
        "model_params": selected_model_params,
        "hist_exog": hist_exog,
        "futr_exog": futr_exog,
        "futr_exog_vintages": futr_exog_vintages,
    }
    output_paths = export_backtest_results(
        predictions_df,
        windows_df,
        metrics_by_scope,
        output_dir=OUTPUT_DIR,
        configuration=run_configuration,
        elapsed_seconds=training_seconds,
        chart_title=None,
        weather_note=(
            "This run uses measured temperature only as historical input and a "
            f"causal synthetic {run_config.horizon}-step forecast proxy. "
            "Production evaluation still "
            "requires archived weather-forecast vintages."
        ),
    )
    print_backtest_summary(
        predictions_df,
        windows_df,
        metrics_by_scope,
        elapsed_seconds=training_seconds,
        paths=output_paths,
    )

#%%
