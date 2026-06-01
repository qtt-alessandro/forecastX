# %%
from src.data import split
from src.model import build_mlf, fit, predict
from src.backtest import backtest
from src.plot import plot_backtest
import loaders.afrr as loader

# %%
BIDDING_ZONE = "DK1"
HORIZON      = 15
EXOG         = None

# %%
# ── Data ─────────────────────────────────────────────────────────────
df = loader.load(BIDDING_ZONE, time_from="2025-04-14 00:00")

train_df, test_df = split(df, train_start="2025-04-14", test_start="2025-04-15", test_end="2025-04-15 06:00")

# %%
# ── Fit & predict ────────────────────────────────────────────────────
mlf = build_mlf(freq="1m")
fit(mlf, train_df, horizon=HORIZON, exog=EXOG)
forecast_df = predict(mlf, test_df, horizon=HORIZON, exog=EXOG)

# %%
# ── Backtest ─────────────────────────────────────────────────────────
predictions_df, windows_df = backtest(
    train_df=train_df,
    test_df=test_df,
    horizon=HORIZON,
    freq="1m",
    exog=EXOG,
    refit=True,
    ensemble_weights={"LinearRegression": 1, "Ridge": 1, "RandomForestRegressor": 1, "XGBRegressor": 1},
)

# %%
plot_backtest(predictions_df)
# %%
