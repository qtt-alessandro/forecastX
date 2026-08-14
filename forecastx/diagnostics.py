"""Stationarity, distribution-shift, and validation-window diagnostics."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import polars as pl
from scipy.stats import ks_2samp, wasserstein_distance
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, acf, kpss

from forecastx.data import validate_schema


@dataclass(frozen=True, slots=True)
class StationarityReport:
    unique_id: str
    observations: int
    adf_statistic: float
    adf_pvalue: float
    kpss_statistic: float
    kpss_pvalue: float
    lag_1_acf: float
    seasonal_acf: float
    weekly_acf: float | None
    classification: Literal["stationary", "non_stationary", "trend_stationary", "inconclusive"]
    recommended_transform: Literal["none", "difference_1", "difference_seasonal"]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DriftMetric:
    column: str
    ks_statistic: float
    ks_pvalue: float
    standardized_wasserstein: float
    mean_ratio: float
    std_ratio: float
    material_shift: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ValidationRecommendation:
    strategy: Literal["expanding", "sliding"]
    candidate_training_windows: tuple[int | None, ...]
    include_recent_regime_stress_test: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _clean(values: pl.Series) -> np.ndarray:
    array = values.drop_nulls().cast(pl.Float64).to_numpy()
    array = array[np.isfinite(array)]
    if len(array) < 32:
        raise ValueError("Stationarity and drift diagnostics require at least 32 finite observations.")
    return array


def stationarity_report(
    df: pl.DataFrame,
    *,
    seasonal_period: int,
    alpha: float = 0.05,
) -> list[StationarityReport]:
    """Combine ADF and KPSS, whose null hypotheses point in opposite directions."""

    validate_schema(df)
    reports: list[StationarityReport] = []
    for series in df.partition_by("unique_id", maintain_order=True):
        values = _clean(series["y"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            adf_result = adfuller(values, autolag="AIC")
            kpss_result = kpss(values, regression="c", nlags="auto")
        adf_rejects_unit_root = float(adf_result[1]) < alpha
        kpss_rejects_stationarity = float(kpss_result[1]) < alpha
        if adf_rejects_unit_root and not kpss_rejects_stationarity:
            classification = "stationary"
            transform = "none"
        elif not adf_rejects_unit_root and kpss_rejects_stationarity:
            classification = "non_stationary"
            transform = "difference_seasonal" if seasonal_period > 1 else "difference_1"
        elif adf_rejects_unit_root and kpss_rejects_stationarity:
            classification = "trend_stationary"
            transform = "difference_1"
        else:
            classification = "inconclusive"
            transform = "difference_seasonal" if seasonal_period > 1 else "difference_1"
        correlations = acf(values, nlags=min(max(seasonal_period * 7, 1), len(values) // 3), fft=True)
        weekly_lag = seasonal_period * 7
        reports.append(
            StationarityReport(
                unique_id=str(series["unique_id"][0]),
                observations=len(values),
                adf_statistic=float(adf_result[0]),
                adf_pvalue=float(adf_result[1]),
                kpss_statistic=float(kpss_result[0]),
                kpss_pvalue=float(kpss_result[1]),
                lag_1_acf=float(correlations[1]),
                seasonal_acf=float(correlations[seasonal_period]),
                weekly_acf=float(correlations[weekly_lag]) if weekly_lag < len(correlations) else None,
                classification=classification,
                recommended_transform=transform,
            )
        )
    return reports


def distribution_drift(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    *,
    columns: list[str] | None = None,
    alpha: float = 0.01,
    effect_threshold: float = 0.25,
) -> list[DriftMetric]:
    """Detect statistical *and* practically meaningful distribution changes."""

    columns = columns or ["y"]
    metrics: list[DriftMetric] = []
    for column in columns:
        if column not in reference.columns or column not in current.columns:
            raise ValueError(f"Drift column {column!r} is missing from reference or current data.")
        ref = _clean(reference[column])
        cur = _clean(current[column])
        ks_result = ks_2samp(ref, cur)
        scale = float(np.std(ref))
        standardized_wasserstein = float(wasserstein_distance(ref, cur) / max(scale, 1e-12))
        ref_mean = float(np.mean(ref))
        ref_std = float(np.std(ref))
        metrics.append(
            DriftMetric(
                column=column,
                ks_statistic=float(ks_result.statistic),
                ks_pvalue=float(ks_result.pvalue),
                standardized_wasserstein=standardized_wasserstein,
                mean_ratio=float(np.mean(cur) / max(abs(ref_mean), 1e-12)),
                std_ratio=float(np.std(cur) / max(ref_std, 1e-12)),
                material_shift=bool(ks_result.pvalue < alpha and standardized_wasserstein >= effect_threshold),
            )
        )
    return metrics


def recommend_validation(
    df: pl.DataFrame,
    *,
    seasonal_period: int,
    exog: list[str] | None = None,
) -> ValidationRecommendation:
    """Recommend expanding or sliding training based only on chronological history."""

    validate_schema(df, exog=exog)
    ordered = df.sort(["unique_id", "ds"])
    midpoint = len(ordered) // 2
    drift = distribution_drift(
        ordered.head(midpoint),
        ordered.tail(len(ordered) - midpoint),
        columns=["y", *(exog or [])],
    )
    shifted = [metric.column for metric in drift if metric.material_shift]
    minimum = max(seasonal_period * 14, 128)
    candidates = (minimum, minimum * 2, minimum * 4, None)
    if shifted:
        return ValidationRecommendation(
            strategy="sliding",
            candidate_training_windows=candidates,
            include_recent_regime_stress_test=True,
            reason=(
                f"Material chronological drift detected in {', '.join(shifted)}. Select a sliding "
                "window by pre-test rolling-origin validation; keep expanding history as a benchmark."
            ),
        )
    return ValidationRecommendation(
        strategy="expanding",
        candidate_training_windows=(None, minimum * 4, minimum * 2),
        include_recent_regime_stress_test=False,
        reason="No material chronological shift was detected; expanding history is the lower-variance default.",
    )
