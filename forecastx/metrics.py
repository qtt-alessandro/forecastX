"""Point, interval, and residual diagnostics for rolling forecasts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox

from forecastx.ensemble import point_model_columns


@dataclass(frozen=True, slots=True)
class ResidualReport:
    model: str
    observations: int
    mean: float
    standard_deviation: float
    jarque_bera_pvalue: float
    ljung_box_pvalue: float
    approximately_gaussian: bool
    approximately_uncorrelated: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _seasonal_scale(train_df: pl.DataFrame, seasonal_period: int) -> float:
    scales: list[float] = []
    for series in train_df.partition_by("unique_id", maintain_order=True):
        values = series.sort("ds")["y"].to_numpy()
        if len(values) > seasonal_period:
            scales.extend(np.abs(values[seasonal_period:] - values[:-seasonal_period]).tolist())
    return float(np.mean(scales)) if scales else float("nan")


def evaluate_forecasts(
    predictions: pl.DataFrame,
    *,
    train_df: pl.DataFrame,
    seasonal_period: int,
    by_horizon: bool = False,
) -> pl.DataFrame:
    """Return long-form metrics; MASE is scaled only from training observations."""

    if "y" not in predictions.columns:
        raise ValueError("Predictions must include actual column 'y'.")
    models = point_model_columns(predictions)
    if "ensemble" in predictions.columns:
        models.append("ensemble")
    scale = _seasonal_scale(train_df, seasonal_period)
    rows: list[dict[str, object]] = []
    groups = predictions.partition_by("horizon_step", as_dict=True) if by_horizon else {("all",): predictions}
    for key, group in groups.items():
        horizon_step = key[0]
        actual = group["y"].to_numpy()
        for model in models:
            forecast = group[model].to_numpy()
            residual = actual - forecast
            denominator = np.abs(actual) + np.abs(forecast)
            smape = np.mean(np.divide(2 * np.abs(residual), denominator, out=np.zeros_like(residual), where=denominator > 0))
            rows.append(
                {
                    "model": model,
                    "horizon_step": horizon_step,
                    "observations": len(group),
                    "mae": float(np.mean(np.abs(residual))),
                    "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                    "bias": float(np.mean(forecast - actual)),
                    "smape": float(smape),
                    "mase": float(np.mean(np.abs(residual)) / scale) if scale > 0 else None,
                }
            )
    return pl.DataFrame(rows).sort(["horizon_step", "mae"])


def interval_metrics(predictions: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for column in predictions.columns:
        if "-lo-" not in column:
            continue
        model, level_text = column.rsplit("-lo-", 1)
        high = f"{model}-hi-{level_text}"
        if high not in predictions.columns:
            continue
        level = int(level_text)
        coverage = predictions.select(
            ((pl.col("y") >= pl.col(column)) & (pl.col("y") <= pl.col(high))).mean()
        ).item()
        width = predictions.select((pl.col(high) - pl.col(column)).mean()).item()
        rows.append({"model": model, "level": level, "coverage": coverage, "mean_width": width})
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={"model": pl.String, "level": pl.Int64, "coverage": pl.Float64, "mean_width": pl.Float64}
    )


def residual_diagnostics(predictions: pl.DataFrame, *, lags: int = 24) -> list[ResidualReport]:
    reports: list[ResidualReport] = []
    models = point_model_columns(predictions)
    if "ensemble" in predictions.columns:
        models.append("ensemble")
    for model in models:
        residuals = (predictions["y"] - predictions[model]).drop_nulls().to_numpy()
        if len(residuals) < max(32, lags + 5):
            continue
        jb = jarque_bera(residuals)
        lb = acorr_ljungbox(residuals, lags=[min(lags, len(residuals) // 5)], return_df=True)
        lb_pvalue = float(lb["lb_pvalue"].iloc[-1])
        reports.append(
            ResidualReport(
                model=model,
                observations=len(residuals),
                mean=float(np.mean(residuals)),
                standard_deviation=float(np.std(residuals)),
                jarque_bera_pvalue=float(jb.pvalue),
                ljung_box_pvalue=lb_pvalue,
                approximately_gaussian=bool(jb.pvalue >= 0.05),
                approximately_uncorrelated=bool(lb_pvalue >= 0.05),
            )
        )
    return reports
