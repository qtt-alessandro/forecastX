import polars as pl

from forecastx.data import load_csv, resample


def load(path: str) -> pl.DataFrame:
    """Load heat demand and repair known outages from past weekly values."""
    return resample(
        load_csv(
            path,
            time_col="UTC",
            target_col="realization",
            unique_id="heat_demand",
            exog=["mean_temp", "forecast"],
        ),
        "1h",
        target_fill="seasonal",
    )
