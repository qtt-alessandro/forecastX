# %%
from src.data import split, resample
from src.model import build_mlf, fit, predict
from src.backtest import backtest
from src.plot import plot_backtest
import loaders.afrr as loader

# %%
BIDDING_ZONE = "DK1"
HORIZON      = 15
FREQ         = "15m"
EXOG         = None

# %%
# ── Data ─────────────────────────────────────────────────────────────
df = resample(loader.load(BIDDING_ZONE, time_from="2025-04-14 00:00"), FREQ)

train_df, test_df = split(df, train_start="2025-05-01 00:00", test_start="2025-07-01 00:00", test_end="2025-07-15 00:00")

# %%
# ── Fit & predict ────────────────────────────────────────────────────
mlf = build_mlf(freq=FREQ)
fit(mlf, train_df, horizon=HORIZON, exog=EXOG)
forecast_df = predict(mlf, test_df, horizon=HORIZON, exog=EXOG)

# %%
# ── Backtest ─────────────────────────────────────────────────────────
predictions_df, windows_df = backtest(
    train_df=train_df,
    test_df=test_df,
    horizon=HORIZON,
    freq="15m",
    exog=EXOG,
    refit=False,
    ensemble_weights={"LinearRegression": 1, "Ridge": 1, "RandomForestRegressor": 1, "XGBRegressor": 1},
)

# %%
plot_backtest(predictions_df)
# %%
