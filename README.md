# forecastx

Time-series forecasting toolkit built around [MLForecast](https://nixtlaverse.nixtla.io/mlforecast/index.html). Trains an ensemble of ML models on hourly energy/heat-demand data, runs rolling backtests, and optionally attaches conformal prediction intervals.

## Features

- **Ensemble models** — LinearRegression, Ridge, RandomForest, XGBoost (LightGBM optional)
- **Lag features** — hourly lags up to 168 h (1 week) + rolling mean/std
- **Cyclic date features** — hour sin/cos encoding, weekend flag
- **Rolling backtest** — configurable horizon, step size, and optional refit
- **Weighted ensemble** — combines model outputs into a single forecast
- **Interactive plots** — Plotly figures for backtest results and CV diagnostics

## Setup

```bash
uv sync
```

Requires Python 3.12+. Place your data CSV in `data/` (not tracked by git).

## Usage

```bash
# run the full pipeline (data → fit → backtest → plot)
uv run python main.py
```

Data must follow the schema expected by `src/data.load_data`:
columns `UTC`, `realization`, `mean_temp`, `forecast` (hourly resolution).

## Project layout

```
forecastx/
├── config.py          # HORIZON, color palette
├── main.py            # entry point
├── src/
│   ├── data.py        # load_data, split
│   ├── features.py    # cyclic encodings, weekend flag
│   ├── model.py       # build_mlf, fit, predict, cross_validate
│   ├── backtest.py    # rolling backtest engine
│   └── plot.py        # Plotly visualisations
└── data/              # local data files (git-ignored)
```
