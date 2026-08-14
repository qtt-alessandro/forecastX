"""Repeat deterministic leakage, schema, and exposure audits on real data."""

from __future__ import annotations

import json

from forecastx import (
    ForecastArtifact,
    build_mlf,
    fit,
    future_target_invariance,
    quality_report,
    split,
)
from forecastx.heat_demand import load_heat_demand


def main() -> None:
    data = load_heat_demand("data/heat_demand_features_set_old.csv")
    train, future = split(
        data,
        train_start="2025-01-01",
        test_start="2025-04-15",
        test_end="2025-04-16",
    )
    audits = []
    for seed in (7, 42, 101):
        engine = build_mlf(
            freq="1h",
            models=["LinearRegression", "Ridge", "LGBMRegressor"],
            target_transform="none",
            interval_levels=(),
            random_state=seed,
            n_jobs=1,
        )
        fit(engine, train, horizon=24, exog=["mean_temp"])
        audits.append(future_target_invariance(engine, future, horizon=24, exog=["mean_temp"]).to_dict())
    exported = ForecastArtifact(
        engine.predict(future, horizon=24, exog=["mean_temp"], level=[])
    ).safe_frame()
    report = {
        "quality": quality_report(data, freq="1h").to_dict(),
        "future_target_invariance": audits,
        "external_columns": exported.columns,
        "external_contains_target": "y" in exported.columns,
        "external_contains_temperature": "mean_temp" in exported.columns,
    }
    print(json.dumps(report, indent=2, default=str))
    if not all(audit["passed"] for audit in audits):
        raise SystemExit("Leakage audit failed.")
    if report["external_contains_target"] or report["external_contains_temperature"]:
        raise SystemExit("Exposure audit failed.")


if __name__ == "__main__":
    main()
