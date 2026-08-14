from __future__ import annotations

import polars as pl

from forecastx.intervals import add_conformal_intervals
from forecastx.visualization import forecast_dashboard


def test_backend_independent_conformal_intervals():
    calibration = pl.DataFrame(
        {
            "horizon_step": [1, 1, 1, 2, 2, 2],
            "y": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
            "ensemble": [1.1, 2.2, 2.7, 1.5, 3.1, 4.3],
        }
    )
    forecasts = pl.DataFrame({"horizon_step": [1, 2], "ensemble": [5.0, 6.0]})
    result = add_conformal_intervals(
        forecasts,
        calibration=calibration,
        model="ensemble",
        levels=[80],
    )
    assert "ensemble-lo-80" in result.columns
    assert "ensemble-hi-80" in result.columns
    assert (result["ensemble-hi-80"] >= result["ensemble-lo-80"]).all()


def test_dashboard_accepts_interval_colors():
    predictions = pl.DataFrame(
        {
            "unique_id": ["x", "x"],
            "ds": [1, 2],
            "y": [1.0, 2.0],
            "Ridge": [1.1, 1.9],
            "Ridge-lo-80": [0.5, 1.4],
            "Ridge-hi-80": [1.7, 2.4],
            "horizon_step": [1, 2],
        }
    )
    figure = forecast_dashboard(predictions)
    assert len(figure.data) >= 4
