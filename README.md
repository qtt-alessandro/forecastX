# forecastx

`forecastx` is a Polars-first forecasting library built around Nixtla's
MLForecast, StatsForecast, and NeuralForecast. It provides chronological data
splits, rolling-origin backtests, future exogenous variables, calibrated
ensembles, prediction intervals, metrics, and interactive Plotly output.

The repository also includes a complete heat-demand workflow in
[`main.py`](main.py). Its model selection and parameters are controlled through
JSON files, so changing an experiment does not require editing pipeline code.

## Installation

Install the standard ML and statistical dependencies:

```bash
uv sync
```

Install LSTM support and development tools:

```bash
uv sync --extra neural --extra dev
```

Python 3.12 or newer is required.

## Run the configured heat-demand workflow

```bash
uv run python main.py
```

The workflow performs these operations in order:

1. Loads and validates the JSON configuration.
2. Loads and resamples the hourly heat-demand data.
3. Creates temperature and calendar features.
4. Makes a chronological train/test split.
5. Runs a rolling-origin backtest.
6. Calibrates a performance-weighted ensemble using only pre-test windows.
7. Evaluates overall, monthly, and horizon-level accuracy.
8. Exports forecasts, metrics, intervals, an interactive chart, and a ranked
   Markdown model comparison.

Results are written to:

- `output/forecasts.csv`
- `output/windows.csv`
- `output/metrics.json`
- `output/forecast_explorer.html`
- `output/model_comparison.md`

## Configuration files

| File | Purpose |
|---|---|
| `config/run.json` | Frequency, horizon, stride, dates, training window, refit cadence, seed, and workers |
| `config/models.json` | Selected models and the path to each parameter file |
| `config/models/*.json` | Parameters for one model per file |
| `config/features.json` | Temperature and temporal features passed to the models |

In `run.json`, `horizon`, `step_size`, and `training_window` are measured in
observations. `refit_every` is measured in forecast origins. With hourly data,
`horizon: 24` and `step_size: 24` produce one non-overlapping day-ahead forecast
per origin.

Feature names in `features.json` must be produced by the preprocessing code.
Adding an arbitrary name to the JSON does not create that feature; its calculation
must also be added to `forecastx/features.py` or `forecastx/heat_demand.py`.

## Select and add models

`config/models.json` is only the active model manifest:

```json
{
  "models": [
    {
      "name": "LSTM",
      "file": "models/lstm.json"
    },
    {
      "name": "XGBRegressor",
      "file": "models/xgboost.json"
    }
  ]
}
```

To add a model:

1. Choose its exact name from the catalog below.
2. Create a JSON file under `config/models/`.
3. Add its name and relative file path to `config/models.json`.
4. Run `uv run python main.py`.

For example, add LightGBM to the manifest:

```json
{
  "name": "LGBMRegressor",
  "file": "models/lightgbm.json"
}
```

Then create `config/models/lightgbm.json`:

```json
{
  "name": "LGBMRegressor",
  "parameters": {
    "n_estimators": 400,
    "learning_rate": 0.035,
    "num_leaves": 31,
    "subsample": 0.9,
    "colsample_bytree": 0.9
  }
}
```

Use an empty object to retain a model's registered defaults:

```json
{
  "name": "ExtraTreesRegressor",
  "parameters": {}
}
```

The name inside the parameter file must exactly match its name in the manifest.
For scaled linear pipelines, parameters use scikit-learn's nested syntax. Ridge,
for example, uses `ridge__alpha`:

```json
{
  "name": "Ridge",
  "parameters": {
    "ridge__alpha": 10.0
  }
}
```

Every selected model is evaluated separately. When multiple models are selected,
the workflow also produces an `ensemble` forecast using inverse-error weights
calibrated before the reported backtest period.

## Available models

| Backend | Model | Intended use | Cost |
|---|---|---|---|
| MLForecast | `LinearRegression` | Ordinary linear baseline | Low |
| MLForecast | `Ridge` | Stable regularized linear baseline | Low |
| MLForecast | `Lasso` | Sparse linear model and feature selection | Low |
| MLForecast | `ElasticNet` | Combined L1/L2 regularization | Low |
| MLForecast | `RandomForestRegressor` | Robust bagged decision trees | Medium |
| MLForecast | `ExtraTreesRegressor` | More randomized tree ensemble | Medium |
| MLForecast | `GradientBoostingRegressor` | Classical sequential boosting | Medium |
| MLForecast | `HistGradientBoostingRegressor` | Fast histogram-based boosting | Medium |
| MLForecast | `LGBMRegressor` | LightGBM gradient boosting | Medium |
| MLForecast | `XGBRegressor` | XGBoost gradient boosting | Medium |
| MLForecast | `KNeighborsRegressor` | Similar-history regression | Medium |
| MLForecast | `MLPRegressor` | Feed-forward neural network | Medium |
| MLForecast | `GaussianProcessRegressor` | Probabilistic kernel regression | High |
| StatsForecast | `SeasonalNaive` | Seasonal persistence benchmark | Low |
| StatsForecast | `AutoARIMA` | Automatically selected ARIMA | Medium |
| StatsForecast | `ARIMAX` | Fixed ARIMA with exogenous variables | Medium |
| StatsForecast | `SARIMAX` | Seasonal ARIMA with exogenous variables | High |
| StatsForecast | `MSTL` | Multiple-season decomposition | Medium |
| NeuralForecast | `LSTM` | Recurrent sequence model | High |

The model registry and defaults are defined in
[`forecastx/models.py`](forecastx/models.py). They can also be inspected from
Python:

```python
from forecastx import list_models

for model in list_models():
    print(model)
```

### Backend constraints

- MLForecast and LSTM models can reuse fitted weights while receiving the latest
  history at each forecast origin.
- A mixed run containing any StatsForecast model must use `"refit_every": 1` in
  `config/run.json`.
- The configured workflow uses a performance-weighted ensemble. For an
  StatsForecast-only model list, call `statistical_backtest` directly or change
  the ensemble setting in `main.py`; performance calibration is not used by the
  StatsForecast-only path.
- StatsForecast parameter JSON files currently use an empty `parameters` object;
  their defaults are controlled by their factories in `forecastx/models.py`.
- LSTM requires `uv sync --extra neural`.
- Gaussian process, SARIMAX, and LSTM are the most computationally expensive
  choices.
- More models do not guarantee a better ensemble. Prefer models with different
  error patterns rather than many nearly identical estimators.

A practical diverse starting set is `Ridge`, `XGBRegressor`, `LGBMRegressor`, and
`LSTM`. Run `SeasonalNaive` as a benchmark, keeping the refit requirement above in
mind.

## Use forecastx as a Python library

Input data uses Nixtla's canonical long schema:

| Column | Meaning |
|---|---|
| `unique_id` | Time-series identifier |
| `ds` | Polars datetime timestamp |
| `y` | Numeric target |
| Other columns | Optional exogenous variables |

A minimal rolling backtest looks like this:

```python
from forecastx import backtest, evaluate_forecasts, load_csv, resample, split

frame = load_csv(
    "data/load.csv",
    time_col="timestamp",
    target_col="demand",
    unique_id="site_1",
    exog=["mean_temp"],
)
frame = resample(frame, "1h", target_fill="seasonal")

train_df, test_df = split(
    frame,
    train_start="2024-12-12",
    test_start="2025-01-09",
    test_end="2025-03-01",
)

predictions, windows = backtest(
    train_df=train_df,
    test_df=test_df,
    horizon=24,
    step_size=24,
    freq="1h",
    exog=["mean_temp"],
    models=["Ridge", "XGBRegressor"],
    training_window=1_344,
    refit=7,
    ensemble_weights="performance",
    level=[80, 95],
    model_params={
        "Ridge": {"ridge__alpha": 10.0},
        "XGBRegressor": {"n_estimators": 400, "max_depth": 6},
    },
)

metrics = evaluate_forecasts(
    predictions,
    train_df=train_df,
    seasonal_period=24,
)
```

The package also exposes `build_mlf`, `fit`, and `predict` for a single fixed
forecast origin; `statistical_backtest` for StatsForecast-only experiments; and
diagnostic, interval, selection, serialization, and visualization utilities from
the top-level `forecastx` package.

## Forecasting safeguards

- Training and evaluation are chronological; future targets are never used as
  features.
- Ensemble weights and conformal intervals are calibrated on pre-test windows.
- Role-aware calls use Nixtla's `hist_exog`/`futr_exog` notation. Forecast-type
  future inputs require a vintage timestamp no later than each fold's cutoff.
- The main heat-demand run keeps `mean_temp_actual` historical-only and builds a
  reproducible causal `mean_temp_forecast` proxy from the measurement available
  24 hours earlier. Production evaluation still requires archived forecasts.

## Tests

```bash
uv sync --extra neural --extra dev
uv run pytest
```

Additional design and validation notes are available in
[`docs/research_and_architecture.md`](docs/research_and_architecture.md).
