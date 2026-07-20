# %%
from config import HORIZON
from src.data import split, resample
from src.model import build_mlf, fit, predict, cross_validate
from src.backtest import backtest
from src.plot import plot_backtest
import loaders.heat_demand as loader

DATA_PATH = "data/heat_demand_features_set_old.csv"
FREQ      = "1h"
EXOG      = ["mean_temp"]

# ── Data ─────────────────────────────────────────────────────────────
df = resample(loader.load(DATA_PATH), FREQ)
train_df, test_df = split(df, train_start="2025-01-01", test_start="2025-04-15", test_end="2025-04-28")

# ── Fit & predict ────────────────────────────────────────────────────
mlf = build_mlf(freq=FREQ)
fit(mlf, train_df, horizon=HORIZON, exog=EXOG)
forecast_df = predict(mlf, test_df, horizon=HORIZON, exog=EXOG)

# ── Backtest ─────────────────────────────────────────────────────────
predictions_df, windows_df = backtest(
    train_df=train_df,
    test_df=test_df,
    horizon=HORIZON,
    exog=EXOG,
    refit=False,
    ensemble_weights={"LinearRegression": 1, "Ridge": 1, "RandomForestRegressor": 1, "XGBRegressor": 1},
)
plot_backtest(predictions_df)

# %%
