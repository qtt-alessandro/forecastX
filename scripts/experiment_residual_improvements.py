"""Leakage-safe ablations targeting the boiler forecast's residual structure.

The experiment keeps every forecast origin chronological. Model candidates see
only observations before their origin; adaptive weights and corrections use only
errors that had already been observed. Results are retrospective because the
available history has already been inspected and must be confirmed on new data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from forecastx import (
    ForecastArtifact,
    add_temperature_features,
    backtest,
    evaluate_forecasts,
    forecast_dashboard,
    residual_diagnostics,
    split,
)
from forecastx.heat_demand import load_heat_demand

DATA_PATH = "data/heat_demand_features_set_old.csv"
TRAIN_START = "2024-12-12"
BACKTEST_START = "2025-01-09"
BACKTEST_END = "2025-09-24 01:00"
HORIZON = 24
STEP_SIZE = 24
RIDGE_PARAMS = {"Ridge": {"ridge__alpha": 100.0}}
BASE_EXOG = ["mean_temp", "mean_temp_squared", "heating_degree", "cooling_degree"]
THERMAL_EXOG = [
    *BASE_EXOG,
    "mean_temp_delta_1h",
    "mean_temp_delta_6h",
    "mean_temp_delta_24h",
    "mean_temp_mean_6h",
    "mean_temp_mean_24h",
    "extreme_cold_degree",
    "heating_hour_sin",
    "heating_hour_cos",
]


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    strategy: str
    training_window: int
    refit: int
    thermal_features: bool = False


CANDIDATES = {
    candidate.name: candidate
    for candidate in [
        Candidate("recursive_28d_refit14d", "recursive", 672, 14),
        Candidate("recursive_28d_refit7d", "recursive", 672, 7),
        Candidate("recursive_28d_refit3d", "recursive", 672, 3),
        Candidate("recursive_28d_refit1d", "recursive", 672, 1),
        Candidate("recursive_14d_refit1d", "recursive", 336, 1),
        Candidate("recursive_28d_refit1d_thermal", "recursive", 672, 1, True),
        Candidate("recursive_28d_refit7d_thermal", "recursive", 672, 7, True),
        Candidate("recursive_56d_refit7d", "recursive", 1_344, 7),
        Candidate("recursive_56d_refit1d", "recursive", 1_344, 1),
        Candidate("recursive_84d_refit7d", "recursive", 2_016, 7),
        Candidate("direct_28d_refit14d", "direct", 672, 14),
        Candidate("direct_56d_refit7d", "direct", 1_344, 7),
        Candidate("direct_56d_refit7d_thermal", "direct", 1_344, 7, True),
    ]
}


def add_thermal_dynamics(df: pl.DataFrame) -> pl.DataFrame:
    """Features derivable from a weather forecast available at the origin."""

    hour_angle = 2 * np.pi * pl.col("ds").dt.hour() / 24
    return df.with_columns(
        (pl.col("mean_temp") - pl.col("mean_temp").shift(1).over("unique_id")).alias(
            "mean_temp_delta_1h"
        ),
        (pl.col("mean_temp") - pl.col("mean_temp").shift(6).over("unique_id")).alias(
            "mean_temp_delta_6h"
        ),
        (pl.col("mean_temp") - pl.col("mean_temp").shift(24).over("unique_id")).alias(
            "mean_temp_delta_24h"
        ),
        pl.col("mean_temp")
        .rolling_mean(window_size=6, min_samples=1)
        .over("unique_id")
        .alias("mean_temp_mean_6h"),
        pl.col("mean_temp")
        .rolling_mean(window_size=24, min_samples=1)
        .over("unique_id")
        .alias("mean_temp_mean_24h"),
        (pl.lit(5.0) - pl.col("mean_temp")).clip(lower_bound=0).alias(
            "extreme_cold_degree"
        ),
        (pl.col("heating_degree") * hour_angle.sin()).alias("heating_hour_sin"),
        (pl.col("heating_degree") * hour_angle.cos()).alias("heating_hour_cos"),
    ).with_columns(
        pl.col(column).fill_null(0.0).alias(column)
        for column in THERMAL_EXOG
        if column not in BASE_EXOG
    )


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    base = add_temperature_features(
        load_heat_demand(DATA_PATH),
        heating_balance=15.0,
        cooling_balance=20.0,
    )
    return base, add_thermal_dynamics(base)


def metric_row(
    predictions: pl.DataFrame,
    *,
    model: str,
    train_df: pl.DataFrame,
    candidate: str,
    segment: str,
) -> dict[str, object]:
    selected = predictions.select(["unique_id", "ds", "y", model]).rename(
        {model: "candidate_forecast"}
    )
    metric = evaluate_forecasts(
        selected,
        train_df=train_df,
        seasonal_period=24,
    ).row(0, named=True)
    return {"candidate": candidate, "segment": segment, **metric}


def segmented_metrics(
    predictions: pl.DataFrame,
    *,
    model: str,
    train_df: pl.DataFrame,
    candidate: str,
) -> list[dict[str, object]]:
    periods = {
        "all": (BACKTEST_START, BACKTEST_END),
        "cold": ("2025-01-09", "2025-03-15"),
        "transition": ("2025-03-15", "2025-05-15"),
        "warm": ("2025-05-15", BACKTEST_END),
    }
    rows: list[dict[str, object]] = []
    for segment, (start, end) in periods.items():
        subset = predictions.filter(
            pl.col("ds").is_between(
                datetime.fromisoformat(start),
                datetime.fromisoformat(end),
                closed="left",
            )
        )
        rows.append(
            metric_row(
                subset,
                model=model,
                train_df=train_df,
                candidate=candidate,
                segment=segment,
            )
        )
    return rows


def run_model_candidate(
    candidate: Candidate,
    *,
    base: pl.DataFrame,
    thermal: pl.DataFrame,
) -> tuple[pl.DataFrame, list[dict[str, object]]]:
    data = thermal if candidate.thermal_features else base
    exog = THERMAL_EXOG if candidate.thermal_features else BASE_EXOG
    train_df, test_df = split(
        data,
        train_start=TRAIN_START,
        test_start=BACKTEST_START,
        test_end=BACKTEST_END,
    )
    predictions, _ = backtest(
        train_df=train_df,
        test_df=test_df,
        horizon=HORIZON,
        step_size=STEP_SIZE,
        freq="1h",
        exog=exog,
        models=["Ridge"],
        strategy=candidate.strategy,
        target_transform="none",
        training_window=candidate.training_window,
        refit=candidate.refit,
        level=[],
        n_jobs=1,
        model_params=RIDGE_PARAMS,
    )
    return predictions, segmented_metrics(
        predictions,
        model="Ridge",
        train_df=train_df,
        candidate=candidate.name,
    )


def load_existing_predictions(base: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, float]]:
    path = Path("output/full_history_forecasts.json")
    if not path.exists():
        raise FileNotFoundError("Run scripts/backtest_full_history.py before adaptive experiments.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions = pl.DataFrame(payload["forecasts"]).with_columns(
        pl.col("ds").str.to_datetime(),
        pl.col("forecast_start").str.to_datetime(),
        pl.col("cutoff").str.to_datetime(),
    )
    actual = base.select(["unique_id", "ds", "y"])
    return predictions.join(actual, on=["unique_id", "ds"]), payload["metadata"][
        "ensemble_weights"
    ]


def add_adaptive_ensemble(
    predictions: pl.DataFrame,
    *,
    models: list[str],
    initial_weights: dict[str, float],
    lookback_windows: int,
    name: str,
) -> pl.DataFrame:
    """Update inverse-MAE weights using only fully observed prior origins."""

    ordered = predictions.sort(["window_id", "ds"])
    windows = ordered["window_id"].to_numpy()
    values = ordered.select(models).to_numpy()
    actual = ordered["y"].to_numpy()
    forecasts = np.empty(len(ordered), dtype=float)
    initial = np.asarray([initial_weights[model] for model in models], dtype=float)
    initial /= initial.sum()
    for window_id in np.unique(windows):
        current = windows == window_id
        history = (windows < window_id) & (windows >= window_id - lookback_windows)
        if history.any():
            errors = np.mean(np.abs(actual[history, None] - values[history]), axis=0)
            weights = 1.0 / np.maximum(errors, 1e-9)
            weights /= weights.sum()
        else:
            weights = initial
        forecasts[current] = values[current] @ weights
    return ordered.with_columns(pl.Series(name, forecasts))


def add_online_horizon_correction(
    predictions: pl.DataFrame,
    *,
    source: str,
    lookback_windows: int,
    shrinkage: float,
    name: str,
) -> pl.DataFrame:
    """Correct each horizon using only residuals from prior completed origins."""

    ordered = predictions.sort(["window_id", "ds"])
    windows = ordered["window_id"].to_numpy()
    horizons = ordered["horizon_step"].to_numpy()
    actual = ordered["y"].to_numpy()
    source_values = ordered[source].to_numpy()
    corrected = source_values.copy()
    for window_id in np.unique(windows):
        current_window = windows == window_id
        history_window = (windows < window_id) & (windows >= window_id - lookback_windows)
        for horizon_step in range(1, HORIZON + 1):
            current = current_window & (horizons == horizon_step)
            history = history_window & (horizons == horizon_step)
            if history.any():
                correction = np.mean(actual[history] - source_values[history])
                corrected[current] += shrinkage * correction
    return ordered.with_columns(pl.Series(name, corrected))


def run_online_candidates(
    base: pl.DataFrame,
) -> tuple[pl.DataFrame, list[dict[str, object]]]:
    predictions, initial_weights = load_existing_predictions(base)
    train_df, _ = split(
        base,
        train_start=TRAIN_START,
        test_start=BACKTEST_START,
        test_end=BACKTEST_END,
    )
    models = list(initial_weights)
    candidate_columns = ["ensemble"]
    for lookback in [3, 7, 14, 28]:
        name = f"adaptive_weights_{lookback}d"
        predictions = add_adaptive_ensemble(
            predictions,
            models=models,
            initial_weights=initial_weights,
            lookback_windows=lookback,
            name=name,
        )
        candidate_columns.append(name)
    for lookback in [7, 14, 28]:
        for shrinkage in [0.5, 1.0]:
            name = f"horizon_correction_{lookback}d_s{shrinkage:g}"
            predictions = add_online_horizon_correction(
                predictions,
                source="ensemble",
                lookback_windows=lookback,
                shrinkage=shrinkage,
                name=name,
            )
            candidate_columns.append(name)
    rows: list[dict[str, object]] = []
    for name in candidate_columns:
        rows.extend(
            segmented_metrics(
                predictions,
                model=name,
                train_df=train_df,
                candidate=name,
            )
        )
    return predictions, rows


def run_daily_ridge_hybrids(
    online_predictions: pl.DataFrame,
    daily_ridge: pl.DataFrame,
    *,
    base: pl.DataFrame,
) -> tuple[pl.DataFrame, list[dict[str, object]]]:
    """Replace the stale Ridge member, then recompute static/adaptive ensembles."""

    combined = online_predictions.join(
        daily_ridge.select(["unique_id", "ds", "Ridge"]).rename({"Ridge": "RidgeDaily"}),
        on=["unique_id", "ds"],
        how="inner",
    )
    _, initial_weights = load_existing_predictions(base)
    hybrid_weights = {
        ("RidgeDaily" if model == "Ridge" else model): weight
        for model, weight in initial_weights.items()
    }
    models = list(hybrid_weights)
    static_name = "daily_ridge_static_ensemble"
    combined = combined.with_columns(
        pl.sum_horizontal(
            [pl.col(model) * hybrid_weights[model] for model in models]
        ).alias(static_name)
    )
    adaptive_name = "daily_ridge_adaptive_weights_7d"
    combined = add_adaptive_ensemble(
        combined,
        models=models,
        initial_weights=hybrid_weights,
        lookback_windows=7,
        name=adaptive_name,
    )
    corrected_name = "daily_ridge_adaptive_7d_correction14_s0.5"
    combined = add_online_horizon_correction(
        combined,
        source=adaptive_name,
        lookback_windows=14,
        shrinkage=0.5,
        name=corrected_name,
    )
    train_df, _ = split(
        base,
        train_start=TRAIN_START,
        test_start=BACKTEST_START,
        test_end=BACKTEST_END,
    )
    rows: list[dict[str, object]] = []
    for name in [static_name, adaptive_name, corrected_name]:
        rows.extend(
            segmented_metrics(
                combined,
                model=name,
                train_df=train_df,
                candidate=name,
            )
        )
    return combined, rows


def export_best_model(
    name: str,
    predictions: pl.DataFrame,
    *,
    candidate: Candidate,
    data: pl.DataFrame,
) -> None:
    """Write auditable diagnostics and privacy-conscious outputs for a winner."""

    train_df, _ = split(
        data,
        train_start=TRAIN_START,
        test_start=BACKTEST_START,
        test_end=BACKTEST_END,
    )
    model_name = "DailyThermalRidge" if candidate.thermal_features else "DailyRidge"
    winner = predictions.rename({"Ridge": model_name})
    overall = evaluate_forecasts(winner, train_df=train_df, seasonal_period=24)
    by_horizon = evaluate_forecasts(
        winner,
        train_df=train_df,
        seasonal_period=24,
        by_horizon=True,
    )
    monthly_rows: list[pl.DataFrame] = []
    labelled = winner.with_columns(pl.col("ds").dt.strftime("%Y-%m").alias("month"))
    for month in labelled["month"].unique().sort().to_list():
        monthly_rows.append(
            evaluate_forecasts(
                labelled.filter(pl.col("month") == month).drop("month"),
                train_df=train_df,
                seasonal_period=24,
            ).with_columns(pl.lit(month).alias("month"))
        )
    monthly = pl.concat(monthly_rows).select(
        ["month", "model", "observations", "mae", "rmse", "bias", "smape", "mase"]
    )
    residual = winner["y"].to_numpy() - winner[model_name].to_numpy()
    acf = {
        str(lag): float(np.corrcoef(residual[:-lag], residual[lag:])[0, 1])
        for lag in [1, 24, 168]
    }
    reports = residual_diagnostics(winner)
    output = Path("output")
    metadata = {
        "candidate": name,
        "frequency": "1h",
        "horizon": HORIZON,
        "step_size": STEP_SIZE,
        "strategy": candidate.strategy,
        "training_window": candidate.training_window,
        "refit_cadence": candidate.refit,
        "thermal_features": candidate.thermal_features,
        "exogenous_status": (
            "oracle realized temperature; replace with origin-time weather vintages"
        ),
        "selection_status": "retrospective; requires prospective confirmation",
    }
    ForecastArtifact(winner, metadata=metadata).write_json(
        output / "best_experiment_forecasts.json"
    )
    forecast_dashboard(
        winner,
        title="Best residual-improvement experiment",
    ).write_html(
        output / "best_experiment_forecast_explorer.html",
        include_plotlyjs=True,
    )
    (output / "best_experiment_metrics.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "overall_metrics": overall.to_dicts(),
                "monthly_metrics": monthly.to_dicts(),
                "horizon_metrics": by_horizon.to_dicts(),
                "residual_diagnostics": [report.to_dict() for report in reports],
                "residual_autocorrelation": acf,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        nargs="*",
        choices=sorted(CANDIDATES),
        default=[],
        help="Ridge configurations to run; omit to run only fast online ablations.",
    )
    parser.add_argument(
        "--output",
        default="output/residual_improvement_experiments.json",
    )
    parser.add_argument(
        "--export-best",
        action="store_true",
        help="Export the best requested Ridge candidate and its diagnostics.",
    )
    args = parser.parse_args()
    base, thermal = load_data()
    online_predictions, rows = run_online_candidates(base)
    model_predictions: dict[str, pl.DataFrame] = {}
    for name in args.candidates:
        candidate = CANDIDATES[name]
        print(f"Running {name}...", flush=True)
        predictions, candidate_rows = run_model_candidate(candidate, base=base, thermal=thermal)
        model_predictions[name] = predictions
        rows.extend(candidate_rows)
        overall = next(row for row in candidate_rows if row["segment"] == "all")
        print(f"{name}: MAE={overall['mae']:.6f}, bias={overall['bias']:.6f}", flush=True)

    daily_name = "recursive_28d_refit1d"
    if daily_name in model_predictions:
        _, hybrid_rows = run_daily_ridge_hybrids(
            online_predictions,
            model_predictions[daily_name],
            base=base,
        )
        rows.extend(hybrid_rows)

    if args.export_best:
        if not model_predictions:
            raise ValueError("--export-best requires at least one model candidate.")
        model_names = set(model_predictions)
        best_row = min(
            (
                row
                for row in rows
                if row["segment"] == "all" and row["candidate"] in model_names
            ),
            key=lambda row: float(row["mae"]),
        )
        best_name = str(best_row["candidate"])
        best_candidate = CANDIDATES[best_name]
        export_best_model(
            best_name,
            model_predictions[best_name],
            candidate=best_candidate,
            data=thermal if best_candidate.thermal_features else base,
        )

    results = sorted(rows, key=lambda row: (str(row["segment"]), float(row["mae"])))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "design": {
                    "train_start": TRAIN_START,
                    "backtest_start": BACKTEST_START,
                    "backtest_end": BACKTEST_END,
                    "horizon": HORIZON,
                    "step_size": STEP_SIZE,
                    "weather_caveat": (
                        "Realized temperature is used consistently for comparison; "
                        "production evaluation requires origin-time weather vintages."
                    ),
                    "selection_caveat": (
                        "Available history was previously inspected; confirm the winner "
                        "prospectively or on newly collected data."
                    ),
                },
                "candidate_definitions": [asdict(candidate) for candidate in CANDIDATES.values()],
                "derived_candidates": {
                    "adaptive_weights_Nd": "Inverse-MAE model weights from only the prior N origins.",
                    "horizon_correction_Nd_sX": (
                        "Per-horizon mean prior residual over N origins, multiplied by X."
                    ),
                    "daily_ridge_static_ensemble": (
                        "Original frozen weights with Ridge replaced by daily-refitted Ridge."
                    ),
                    "daily_ridge_adaptive_weights_7d": (
                        "Daily-refitted Ridge plus other members, weighted on seven prior origins."
                    ),
                },
                "metrics": results,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    table = pl.DataFrame(results).filter(pl.col("segment") == "all").sort("mae")
    print(table.select(["candidate", "mae", "rmse", "bias", "mase"]), flush=True)


if __name__ == "__main__":
    main()
