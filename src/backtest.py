from datetime import timedelta, datetime
from functools import reduce

import polars as pl

from config import HORIZON
from src.model import build_mlf, fit

# Columns that are never exogenous
_RESERVED = {"unique_id", "ds", "y", "cutoff"}


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def _add_ensemble(df: pl.DataFrame, weights: dict[str, float]) -> pl.DataFrame:
    weights = _normalize_weights(weights)
    terms = [pl.col(m) * w for m, w in weights.items() if m in df.columns]
    return df.with_columns(reduce(lambda a, b: a + b, terms).alias("ensemble"))


_FREQ_DELTA = {
    "1h":  timedelta(hours=1),
    "15m": timedelta(minutes=15),
    "1m":  timedelta(minutes=1),
}


def backtest(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    horizon: int = HORIZON,
    step_size: int = HORIZON,
    freq: str = "1h",
    exog: list[str] | None = None,
    refit: bool | int = False,
    ensemble_weights: dict[str, float] | None = None,
    level: list[int] | None = None,
    models: list[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Rolling-window backtest.

    Parameters
    ----------
    train_df         : initial training data
    test_df          : held-out data to evaluate on
    horizon          : forecast horizon in steps
    step_size        : how many steps to advance per window
    exog             : exogenous column names, e.g. ["mean_temp"].
                       Pass None (default) for a purely univariate backtest.
                       Must match the columns present in both train_df and test_df.
    refit            : False → static model, reuses history for lag features only.
                       True  → refit on expanding window at every cutoff.
    ensemble_weights : optional dict mapping model names to weights for a
                       blended 'ensemble' column, e.g.
                       {"LinearRegression": 1, "Ridge": 1, "XGBRegressor": 2}
    level            : prediction interval levels, e.g. [95]
    models           : model names from src.model.MODEL_REGISTRY to fit.
                       Defaults to src.model.DEFAULT_MODELS.

    Returns
    -------
    predictions_df : all per-window forecasts with actuals joined
    windows_df     : metadata for every backtest window
    """
    if isinstance(refit, int) and refit > 1:
        raise NotImplementedError(f"refit={refit} not yet implemented.")

    exog = exog or []
    unit = _FREQ_DELTA[freq]

    # Build cutoffs
    test_min, test_max = test_df["ds"].min(), test_df["ds"].max()
    assert isinstance(test_min, datetime) and isinstance(test_max, datetime)
    cutoffs, t = [], test_min
    while t + unit * (horizon - 1) <= test_max:
        cutoffs.append(t)
        t += unit * step_size

    print(f"Backtesting {len(cutoffs)} windows | refit={refit} | exog={exog or 'none'}")

    # Initial fit on train_df only
    mlf = build_mlf(freq=freq, models=models)
    fit(mlf, train_df, horizon=horizon, exog=exog)

    all_preds, windows = [], []

    for i, cutoff in enumerate(cutoffs):
        test_end = cutoff + unit * (horizon - 1)

        if refit is True:
            rolling_train = pl.concat([
                train_df,
                test_df.filter(pl.col("ds") < cutoff),
            ])
            mlf = build_mlf(freq=freq, models=models)
            fit(mlf, rolling_train, horizon=horizon, exog=exog)
            new_df_arg = None
            train_end = rolling_train["ds"].max()
            train_rows = len(rolling_train)
        else:
            # Provide history up to cutoff so lag features can be computed
            history_cols = ["unique_id", "ds", "y"] + exog
            history = pl.concat([
                train_df.select(history_cols),
                test_df.filter(pl.col("ds") < cutoff).select(history_cols),
            ])
            new_df_arg = history
            train_end = train_df["ds"].max()
            train_rows = len(train_df)

        window = test_df.filter(
            (pl.col("ds") >= cutoff) & (pl.col("ds") <= test_end)
        )
        if window.is_empty():
            continue

        X_df = window.select(["unique_id", "ds"] + exog) if exog else None
        predict_kwargs: dict = dict(h=horizon, X_df=X_df, new_df=new_df_arg)
        if level is not None:
            predict_kwargs["level"] = level

        preds = mlf.predict(**predict_kwargs)
        test_tz = window.schema["ds"].time_zone if hasattr(window.schema["ds"], "time_zone") else None
        if test_tz:
            preds = preds.with_columns(pl.col("ds").dt.replace_time_zone(test_tz))
        preds = preds.join(window.select(["ds", "y"]), on="ds", how="left")
        preds = preds.with_columns(pl.lit(cutoff).alias("cutoff"))
        all_preds.append(preds)

        windows.append({
            "window_id":   i,
            "train_start": train_df["ds"].min(),
            "train_end":   train_end,
            "train_rows":  train_rows,
            "test_start":  cutoff,
            "test_end":    test_end,
        })

    predictions_df = pl.concat(all_preds)
    if ensemble_weights is not None:
        predictions_df = _add_ensemble(predictions_df, ensemble_weights)

    return predictions_df, pl.DataFrame(windows)