from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from forecastx.data import quality_report, resample, split


def test_seasonal_target_repair_is_marked_and_past_only():
    start = datetime(2025, 1, 1)
    frame = pl.DataFrame(
        {
            "unique_id": ["x"] * 5,
            "ds": [start + timedelta(hours=i) for i in [0, 1, 2, 4, 5]],
            "y": [1.0, 2.0, 3.0, 5.0, 6.0],
        }
    )
    repaired = resample(frame, "1h", target_fill="seasonal", seasonal_period=2)
    missing = repaired.filter(pl.col("ds") == start + timedelta(hours=3))
    assert missing["y"].item() == 2.0
    assert missing["y_imputed"].item() is True


def test_split_is_half_open(hourly_frame):
    train, test = split(
        hourly_frame,
        train_start="2025-01-01",
        test_start="2025-01-11",
        test_end="2025-01-12",
    )
    assert train["ds"].max() == datetime(2025, 1, 10, 23)
    assert test["ds"].min() == datetime(2025, 1, 11)
    assert test["ds"].max() == datetime(2025, 1, 11, 23)


def test_quality_report_detects_frequency(hourly_frame):
    report = quality_report(hourly_frame, freq="1h")
    assert report.is_valid
    assert report.inferred_frequency_seconds == 3_600
