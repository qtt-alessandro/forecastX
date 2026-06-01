#%%
from config import HORIZON
from src.data import load_data, split
from src.model import build_mlf, fit, predict, cross_validate
from src.plot import plot_backtest


#%%
DATA_PATH = "data/heat_demand_features_set_old.csv"

#%%
# ── Data ────────────────────────────────────────────────────────────
df = load_data(DATA_PATH)
train_df, test_df = split(
    df,
    train_start="2025-01-01", train_end="2025-04-15",
    test_start="2025-04-15", #test_end="2025-02-28",
)

#%%
# ── Model ───────────────────────────────────────────────────────────
mlf = build_mlf()
fit(mlf, train_df, horizon=HORIZON)
#%%
# ── Evaluation ──────────────────────────────────────────────────────
forecast_df = predict(mlf, test_df, horizon=HORIZON, level=[95])
cv_df       = cross_validate(mlf, df, horizon=HORIZON, n_windows=20)
#%%
# ── Plots ───────────────────────────────────────────────────────────
#plot_forecast(forecast_df)
#plot_cv(cv_df, model="XGBRegressor", n_windows=5)
#plot_cv_metrics(cv_df)


# %%
from src.backtest import backtest

predictions_df, windows_df = backtest(
    train_df=train_df,
    test_df=test_df,
    horizon=24,
    refit=False,
    ensemble_weights={"LinearRegression": 1, "Ridge": 1, "RandomForestRegressor": 1, "XGBRegressor": 1},
)

plot_backtest(predictions_df)
# %%
