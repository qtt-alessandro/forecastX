# forecastx

Production-oriented, Polars-first load forecasting built on Nixtla's MLForecast,
StatsForecast, and optional NeuralForecast. The package makes forecast origin,
horizon, stride, target repair, refitting, future exogenous availability, and
interval calibration explicit so that backtests match deployment.

## What is included

- Canonical ingestion and data-quality reports for `unique_id`, `ds`, `y`
- Explicit resampling and leakage-safe past-seasonal target repair
- ADF + KPSS stationarity classification and chronological drift diagnostics
- 19 registered models with four stable, complementary ML defaults
- Recursive or one-model-per-horizon direct forecasting
- Fixed-origin prediction and rolling-origin backtesting
- `horizon=10, step_size=1` or `horizon=10, step_size=10` schedules
- Leakage-free inverse-MAE ensembles calibrated only on pre-test windows
- Horizon-aware conformal intervals plus statistical model intervals
- SeasonalNaive, AutoARIMA, ARIMAX, SARIMAX, and MSTL benchmarks
- Optional Nixtla LSTM (`uv sync --extra neural`)
- MAE, RMSE, bias, sMAPE, MASE, coverage, residual diagnostics
- Interactive Plotly forecast explorer and privacy-conscious JSON export

## Quick start

```python
import loaders.heat_demand as loader

from forecastx import backtest, build_mlf, fit, predict, resample, split

DATA_PATH = "data/heat_demand_features_set_old.csv"
FREQ = "1h"
HORIZON = 24
EXOG = ["mean_temp"]
MODELS = ["LinearRegression", "Ridge", "RandomForestRegressor", "XGBRegressor"]

df = resample(loader.load(DATA_PATH), FREQ)
train_df, test_df = split(
    df,
    train_start="2025-01-01",
    test_start="2025-04-15",
    test_end="2025-04-28 01:00",
)

mlf = build_mlf(freq=FREQ, models=MODELS)
fit(mlf, train_df, horizon=HORIZON, exog=EXOG)
forecast_df = predict(mlf, test_df, horizon=HORIZON, exog=EXOG)

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
```

Use `training_window=672` in `fit(...)` to train only on the latest four weeks of
hourly history. In backtests, the same option is applied independently at every
forecast origin.

`forecast_df` and `predictions_df` are Polars dataframes. Use
`ForecastArtifact(forecast_df).write_json(...)` for an external JSON artifact and
`forecast_dashboard(predictions_df)` for the interactive explorer.

## Operational warning about weather

Future exogenous values must be available at the forecast origin. Historical
realized `mean_temp` is an oracle input in an offline backtest unless operations
will truly know that value. For a deployable score, replace it with the weather
forecast vintage that was available at each historical origin.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run python main.py
```

The scientific and architectural decisions are in
[`docs/research_and_architecture.md`](docs/research_and_architecture.md).

## Full-history stability backtest

The April split above is the protected model-selection holdout. To evaluate
seasonal stability over all usable data after the initial training and ensemble
calibration period, run:

```bash
uv run python scripts/backtest_full_history.py
```

This simulates 258 daily forecast origins (6,192 hourly predictions) from
2025-01-09 through 2025-09-24. It freezes ensemble weights before the first
reported origin, reports aggregate and monthly metrics, and writes:

- `artifacts/full_history_metrics.json`
- `artifacts/full_history_forecasts.json`
- `artifacts/full_history_forecast_explorer.html`

The full-history report is a retrospective stability analysis because the final
hyperparameters were selected later in the dataset. Keep the protected April
result for a selection-unbiased comparison; use the long report to expose seasonal
regimes and operational degradation.

## Residual-improvement ablation

Run the reproducible comparison with:

```bash
uv run python scripts/experiment_residual_improvements.py \
  --candidates recursive_28d_refit14d recursive_28d_refit7d \
  recursive_28d_refit3d recursive_28d_refit1d \
  recursive_28d_refit1d_thermal direct_28d_refit14d
```

Across the 258 full-history origins, the best experiment was recursive Ridge
refitted at every daily origin on the latest 28 days, with temperature-change,
thermal-memory, extreme-cold, and hour/heating interaction features. It reached
0.297 MAE versus 0.323 for the original frozen ensemble. Direct Ridge, longer or
shorter histories, and adding the improved Ridge back to the stale ensemble were
worse. These are retrospective results using realized temperature and require
prospective confirmation with archived weather-forecast vintages.
