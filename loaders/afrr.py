import polars as pl
from requests_cache import CachedSession


def load(bidding_zone: str, time_from: str | None = None) -> pl.DataFrame:
    """Load aFRR activations from Energinet API."""
    url = "https://api.energidataservice.dk/dataset/PowerSystemRightNow"
    params = {}
    if time_from is not None:
        params["start"] = (
            pl.Series([time_from])
            .str.to_datetime()
            .dt.replace_time_zone("CET")
            .dt.convert_time_zone("UTC")
            .dt.strftime("%Y-%m-%dT%H:%M")[0]
        )

    session = CachedSession(cache_name=".cache", expire_after=300)
    response = session.get(url, params=params)
    response.raise_for_status()
    records = response.json()["records"]

    return (
        pl.DataFrame({
            "ds": [r["Minutes1UTC"] for r in records],
            "y":  [r[f"aFRR_Activated{bidding_zone}"] for r in records],
        })
        .with_columns([
            pl.col("ds").str.to_datetime().dt.replace_time_zone("UTC"),
            pl.col("y").cast(pl.Float64),
        ])
        .drop_nulls()
        .sort("ds")
        .upsample(time_column="ds", every="1m")
        .with_columns(pl.col("y").forward_fill())
        .with_columns(pl.lit(f"aFRR_{bidding_zone}").alias("unique_id"))
    )