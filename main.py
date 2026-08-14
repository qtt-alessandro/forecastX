"""Full-series heat-demand comparison across configured models and ensemble."""

#%% Imports
from __future__ import annotations

from pathlib import Path
from time import perf_counter

from forecastx import add_temperature_features, backtest, split
from forecastx.config import (
    load_feature_configuration,
    load_model_configuration,
    load_run_configuration,
)
from forecastx.heat_demand import (
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
    temporal_features, exogenous_features = load_feature_configuration(
        FEATURE_CONFIG_PATH
    )

    raw_df = load_heat_demand(DATA_PATH, frequency=run_config.frequency)
    temperature_df = add_temperature_features(
        raw_df,
        heating_balance=15.0,
        cooling_balance=20.0,
    )
    model_df = add_temporal_features(temperature_df)
    validate_feature_columns(model_df, exogenous_features)

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
        exog=exogenous_features,
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
        "features": exogenous_features,
    }
    output_paths = export_backtest_results(
        predictions_df,
        windows_df,
        metrics_by_scope,
        output_dir=OUTPUT_DIR,
        configuration=run_configuration,
        elapsed_seconds=training_seconds,
        chart_title=(
            f"Heat demand: {', '.join(selected_models)} and calibrated ensemble"
        ),
        weather_note=(
            "This retrospective run uses realized mean_temp. Production evaluation "
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
