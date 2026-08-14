from __future__ import annotations

import polars as pl

from forecastx.diagnostics import distribution_drift, recommend_validation, stationarity_report


def test_stationarity_and_drift_reports(hourly_frame):
    reports = stationarity_report(hourly_frame, seasonal_period=24)
    assert len(reports) == 1
    assert reports[0].observations == len(hourly_frame)
    shifted = hourly_frame.with_columns((pl.col("y") + 10).alias("y"))
    drift = distribution_drift(hourly_frame, shifted, columns=["y"])
    assert drift[0].material_shift
    recommendation = recommend_validation(
        pl.concat([hourly_frame.head(310), shifted.tail(310)]),
        seasonal_period=24,
        exog=["mean_temp"],
    )
    assert recommendation.strategy == "sliding"
