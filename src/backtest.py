from datetime import timedelta
from functools import reduce
from datetime import datetime, timedelta
import polars as pl

from config import HORIZON
from src.model import build_mlf, fit


def _normalize_weights(weights):
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def _add_ensemble(df, weights):
    weights = _normalize_weights(weights)
    terms = [pl.col(m) * w for m, w in weights.items() if m in df.columns]
    return df.with_columns(reduce(lambda a, b: a + b, terms).alias("ensemble"))


def backtest(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    horizon: int = HORIZON,
    step_size: int = HORIZON,
    refit: bool | int = False,
    ensemble_weights: dict[str, float] | None = None,
    level: list[int] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if isinstance(refit, int) and refit > 1:
        raise NotImplementedError(f"refit={refit} not yet implemented.")

    # cutoffs every step_size hours through test_df
    test_min, test_max = test_df["ds"].min(), test_df["ds"].max()
    assert isinstance(test_min, datetime) and isinstance(test_max, datetime)
    cutoffs, t = [], test_min

    while t + timedelta(hours=horizon - 1) <= test_max:
        cutoffs.append(t)
        t += timedelta(hours=step_size)

    print(f"Backtesting {len(cutoffs)} windows | refit={refit}")

    # initial fit on train_df only — never sees test data
    mlf = build_mlf()
    fit(mlf, train_df, horizon=horizon)

    all_preds, windows = [], []

    for i, cutoff in enumerate(cutoffs):
        test_end = cutoff + timedelta(hours=horizon - 1)

        if refit is True:
            # expand training set with test data preceding the cutoff
            rolling_train = pl.concat([
                train_df,
                test_df.filter(pl.col("ds") < cutoff),
            ])
            mlf = build_mlf()
            fit(mlf, rolling_train, horizon=horizon)
            new_df_arg = None
            train_end = rolling_train["ds"].max()
            train_rows = len(rolling_train)
        else:
            # static models, but provide history up to cutoff for lag features
            history = pl.concat([
                train_df,
                test_df.filter(pl.col("ds") < cutoff),
            ]).drop("forecast")
            new_df_arg = history
            train_end = train_df["ds"].max()  # model was fit on this
            train_rows = len(train_df)

        window = test_df.filter(
            (pl.col("ds") >= cutoff) & (pl.col("ds") <= test_end)
        )
        if window.is_empty():
            continue

        X_df = window.select(["unique_id", "ds", "mean_temp"])
        predict_kwargs = dict(h=horizon, X_df=X_df, new_df=new_df_arg)
        if level is not None:
            predict_kwargs["level"] = level
        preds = mlf.predict(**predict_kwargs)
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
