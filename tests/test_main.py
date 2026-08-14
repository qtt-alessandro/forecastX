from __future__ import annotations

from datetime import date, datetime

import polars as pl

import main
from forecastx.config import (
    load_feature_configuration,
    load_model_configuration,
    load_run_configuration,
)
from forecastx.heat_demand import add_temporal_features, easter_sunday


def test_main_configuration_loads_ordered_model_parameter_files():
    models, parameters = load_model_configuration(main.MODEL_MANIFEST_PATH)

    assert models == [
        "SeasonalNaive",
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "HistGradientBoostingRegressor",
        "LGBMRegressor",
        "XGBRegressor",
        "SARIMAX",
        "LSTM",
    ]
    assert set(parameters) == set(models)
    assert parameters["LSTM"]["encoder_hidden_size"] > 0
    assert parameters["XGBRegressor"]["n_estimators"] > 0


def test_main_configuration_loads_run_settings():
    run = load_run_configuration(main.RUN_CONFIG_PATH)

    assert run.frequency == "1h"
    assert run.horizon == 24
    assert run.training_window == 1_344
    assert run.seasonal_period == 24


def test_main_temporal_features_include_danish_holiday():
    frame = pl.DataFrame(
        {"ds": [datetime(2025, 4, 17, 8), datetime(2025, 4, 22, 8)]}
    )

    temporal_features, exogenous_features = load_feature_configuration(
        main.FEATURE_CONFIG_PATH
    )
    result = add_temporal_features(frame)

    assert easter_sunday(2025) == date(2025, 4, 20)
    assert set(temporal_features) <= set(result.columns)
    assert set(temporal_features) <= set(exogenous_features)
    assert result["calendar_is_holiday"].to_list() == [1, 0]
    assert result["calendar_is_working_day"].to_list() == [0, 1]
