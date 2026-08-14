"""Polars-first forecast artifact and privacy-conscious JSON export."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from forecastx.audit import external_forecast_columns
from forecastx.visualization import forecast_dashboard


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


@dataclass(slots=True)
class ForecastArtifact:
    """A forecast remains a Polars frame in memory and can be safely serialized."""

    frame: pl.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    def safe_frame(self, *, include_actual: bool = False) -> pl.DataFrame:
        return self.frame.select(external_forecast_columns(self.frame, include_actual=include_actual))

    def to_json(self, *, include_actual: bool = False, indent: int | None = 2) -> str:
        payload = {
            "metadata": self.metadata,
            "forecasts": self.safe_frame(include_actual=include_actual).to_dicts(),
        }
        return json.dumps(payload, default=_json_default, indent=indent, allow_nan=False)

    def write_json(
        self,
        path: str | Path,
        *,
        include_actual: bool = False,
        indent: int | None = 2,
    ) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(include_actual=include_actual, indent=indent), encoding="utf-8")
        return destination


def export_backtest_results(
    predictions: pl.DataFrame,
    windows: pl.DataFrame,
    metrics: dict[str, pl.DataFrame],
    *,
    output_dir: str | Path,
    configuration: dict[str, Any],
    elapsed_seconds: float,
    chart_title: str,
    weather_note: str,
) -> dict[str, Path]:
    """Write the complete backtest result set to one output directory."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "forecasts": destination / "forecasts.csv",
        "windows": destination / "windows.csv",
        "metrics": destination / "metrics.json",
        "chart": destination / "forecast_explorer.html",
    }
    predictions.write_csv(paths["forecasts"])
    windows.drop("ensemble_weights").write_csv(paths["windows"])
    forecast_dashboard(predictions, title=chart_title).write_html(
        paths["chart"],
        include_plotlyjs=True,
    )

    report = {
        "configuration": configuration,
        "ensemble_weights": windows["ensemble_weights"][0],
        "elapsed_seconds": elapsed_seconds,
        "overall": metrics["overall"].to_dicts(),
        "monthly": metrics["monthly"].to_dicts(),
        "by_horizon": metrics["by_horizon"].to_dicts(),
        "weather_note": weather_note,
    }
    paths["metrics"].write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return paths


def print_backtest_summary(
    predictions: pl.DataFrame,
    windows: pl.DataFrame,
    metrics: dict[str, pl.DataFrame],
    *,
    elapsed_seconds: float,
    paths: dict[str, Path],
) -> None:
    """Print the compact terminal summary for a completed backtest."""

    print(metrics["overall"])
    print(f"\nForecast rows: {len(predictions):,}")
    print(f"Forecast origins: {len(windows):,}")
    print(f"Elapsed time: {elapsed_seconds / 60:.2f} minutes")
    print(f"Ensemble weights: {windows['ensemble_weights'][0]}")
    print(f"Interactive chart: {paths['chart']}")
