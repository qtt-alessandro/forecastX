"""Heat-demand example following the stable functional API."""

from pathlib import Path

import loaders.heat_demand as loader

from forecastx import (
    ForecastArtifact,
    backtest,
    build_mlf,
    evaluate_forecasts,
    fit,
    forecast_dashboard,
    predict,
    resample,
    split,
)

DATA_PATH = "data/heat_demand_features_set_old.csv"
FREQ = "1h"
HORIZON = 24
EXOG = ["mean_temp"]
MODELS = [
    "LinearRegression",
    "Ridge",
    "GradientBoostingRegressor",
    "LGBMRegressor",
]


def main() -> None:
    df = resample(loader.load(DATA_PATH), FREQ)
    train_df, test_df = split(
        df,
        train_start="2025-01-01",
        test_start="2025-04-15",
        test_end="2025-04-28 01:00",
    )

    # Fit and fixed-origin prediction.
    mlf = build_mlf(freq=FREQ, models=MODELS)
    fit(mlf, train_df, horizon=HORIZON, exog=EXOG)
    forecast_df = predict(mlf, test_df, horizon=HORIZON, exog=EXOG)

    # Predict H steps every H steps. Set step_size=1 for H steps every step.
    predictions_df, windows_df = backtest(
        train_df=train_df,
        test_df=test_df,
        horizon=HORIZON,
        step_size=HORIZON,
        freq=FREQ,
        exog=EXOG,
        models=MODELS,
        refit=False,
        ensemble_weights="performance",
        level=[80, 95],
    )

    output = Path("artifacts")
    ForecastArtifact(
        forecast_df,
        metadata={"frequency": FREQ, "horizon": HORIZON, "models": MODELS, "exog": EXOG},
    ).write_json(output / "example_fixed_origin_forecast.json")
    forecast_dashboard(predictions_df, title="Electric-boiler heat demand").write_html(
        output / "example_backtest.html",
        include_plotlyjs="cdn",
    )
    print(evaluate_forecasts(predictions_df, train_df=train_df, seasonal_period=24))
    print(windows_df)


if __name__ == "__main__":
    main()
