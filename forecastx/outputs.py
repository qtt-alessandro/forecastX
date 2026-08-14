"""Polars-first forecast artifact and privacy-conscious JSON export."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from forecastx.audit import external_forecast_columns


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
