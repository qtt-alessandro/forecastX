from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def hourly_frame() -> pl.DataFrame:
    rows = 620
    start = datetime(2025, 1, 1)
    index = np.arange(rows)
    temperature = 8 + 7 * np.sin(2 * np.pi * index / (24 * 30))
    target = (
        4
        - 0.18 * temperature
        + 0.8 * np.sin(2 * np.pi * index / 24)
        + 0.3 * np.sin(2 * np.pi * index / 168)
    )
    return pl.DataFrame(
        {
            "unique_id": ["load"] * rows,
            "ds": [start + timedelta(hours=int(step)) for step in index],
            "y": target,
            "mean_temp": temperature,
        }
    )
