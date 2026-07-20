import polars as pl


def load(path: str = "data/inflow_and_weather_data.parquet") -> pl.DataFrame:
    """Load inflow and weather (precipitation, wind, humidity) data."""
    return (
        pl.read_parquet(path)
        .rename({"time": "ds", "inflow": "y"})
        # mlforecast drops tz info internally when computing future dates for
        # prediction, which then clashes with tz-aware X_df at predict time.
        # Keep ds tz-naive (UTC wall-clock), matching loaders/heat_demand.py.
        .with_columns(pl.col("ds").dt.replace_time_zone(None))
        .sort("ds")
        .upsample(time_column="ds", every="1h")
        .with_columns([
            pl.col("y").forward_fill(),
            pl.col("precip_past1min").forward_fill(),
            pl.col("wind_speed").forward_fill(),
            pl.col("humidity").forward_fill(),
        ])
        .with_columns(pl.lit("inflow").alias("unique_id"))
    )
