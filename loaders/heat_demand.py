import polars as pl


def load(path: str) -> pl.DataFrame:
    """Load heat demand CSV."""
    return (
        pl.read_csv(path, try_parse_dates=True)
        .select(["UTC", "realization", "mean_temp", "forecast"])
        .rename({"UTC": "ds", "realization": "y"})
        .sort("ds")
        .upsample(time_column="ds", every="1h")
        .with_columns([
            pl.col("y").forward_fill(),
            pl.col("mean_temp").forward_fill(),
            pl.col("forecast").forward_fill(),
        ])
        .with_columns(pl.lit("heat_demand").alias("unique_id"))
    )
