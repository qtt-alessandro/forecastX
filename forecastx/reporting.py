"""Human-readable reports for completed forecasting runs."""

from __future__ import annotations

from math import isclose
from typing import Any

import polars as pl


METRICS = (
    ("mae", "MAE ↓", False),
    ("rmse", "RMSE ↓", False),
    ("smape", "sMAPE ↓", False),
    ("mase", "MASE ↓", False),
    ("bias", "Bias ≈ 0", True),
)


def _format_metric(value: float, *, percent: bool = False, best: bool = False) -> str:
    formatted = f"{value * 100:.2f}%" if percent else f"{value:.4f}"
    return f"🟢 **{formatted}**" if best else formatted


def _mae_signal(value: float, best: float) -> str:
    if isclose(value, best, rel_tol=1e-12, abs_tol=1e-12):
        return "🟢 Best"
    if value <= best * 1.10:
        return "🟡 Within 10%"
    return "🔴 Above 10%"


def model_comparison_markdown(
    overall_metrics: pl.DataFrame,
    *,
    ensemble_weights: dict[str, float],
    configuration: dict[str, Any],
    elapsed_seconds: float,
) -> str:
    """Render a ranked model table and a transparent ensemble explanation."""

    required = {"model", "observations", *(name for name, _, _ in METRICS)}
    missing = sorted(required - set(overall_metrics.columns))
    if missing:
        raise ValueError(f"Model comparison metrics are missing columns: {missing}")
    if overall_metrics.is_empty():
        raise ValueError("Model comparison requires at least one metric row.")

    rows = overall_metrics.sort("mae").to_dicts()
    best_values = {
        name: min(abs(float(row[name])) if absolute else float(row[name]) for row in rows)
        for name, _, absolute in METRICS
    }
    best_model = str(rows[0]["model"])
    ensemble_rank = next(
        (rank for rank, row in enumerate(rows, start=1) if row["model"] == "ensemble"),
        None,
    )
    observations = int(rows[0]["observations"])
    horizon = configuration.get("horizon", "unknown")
    backtest_start = configuration.get("backtest_start", "unknown")
    backtest_end = configuration.get("backtest_end", "unknown")

    lines = [
        "# Model comparison",
        "",
        (
            f"Rolling-origin evaluation over **{observations:,} forecasts**, with a "
            f"**{horizon}-step horizon**, from **{backtest_start}** to "
            f"**{backtest_end}**. Runtime: **{elapsed_seconds / 60:.2f} minutes**."
        ),
        "",
        f"The primary ranking uses MAE. **{best_model}** has the lowest MAE.",
    ]
    if ensemble_rank is not None:
        lines.append(f"The performance-weighted ensemble ranks **#{ensemble_rank} by MAE**.")
    lines.extend(
        [
            "",
            "Color guide: 🟢 best MAE, 🟡 within 10% of the best MAE, "
            "🔴 more than 10% above it. A green value marks the winner for that metric.",
            "",
            "| Rank | MAE band | Model | MAE ↓ | RMSE ↓ | sMAPE ↓ | MASE ↓ | Bias ≈ 0 | Ensemble weight |",
            "|---:|:---|:---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for rank, row in enumerate(rows, start=1):
        model = str(row["model"])
        values: list[str] = []
        for name, _, absolute in METRICS:
            value = float(row[name])
            score = abs(value) if absolute else value
            values.append(
                _format_metric(
                    value,
                    percent=name == "smape",
                    best=isclose(score, best_values[name], rel_tol=1e-12, abs_tol=1e-12),
                )
            )
        weight = ensemble_weights.get(model)
        weight_text = f"{weight:.2%}" if weight is not None else "—"
        lines.append(
            f"| {rank} | {_mae_signal(float(row['mae']), best_values['mae'])} | "
            f"{model} | {' | '.join(values)} | {weight_text} |"
        )

    lines.extend(
        [
            "",
            "## How the ensemble works",
            "",
            "The ensemble is not another fitted forecasting model. It combines the "
            "individual point forecasts using a weighted average:",
            "",
            "`ensemble forecast = Σ(model weight × model forecast)`",
            "",
            "1. Each model is evaluated on rolling windows immediately before the "
            "reported backtest period.",
            "2. A model's initial score is the inverse of its calibration MAE, so a "
            "smaller error produces a larger weight.",
            "3. The weights receive 20% shrinkage toward equal weighting. This limits "
            "the influence of one unusually strong calibration result.",
            "4. The non-negative weights are normalized to sum to 100% and are then "
            "held fixed throughout the reported evaluation period.",
            "",
            "The weights come from pre-test calibration data, whereas the ranking "
            "above comes from the held-out backtest. Their ordering can therefore differ.",
            "",
            "| Component model | Weight |",
            "|:---|---:|",
        ]
    )
    for model, weight in sorted(ensemble_weights.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| {model} | {weight:.2%} |")
    lines.extend(
        [
            "| **Total** | **100.00%** |",
            "",
            "## Metric guide",
            "",
            "- **MAE:** average absolute forecast error; the main ranking metric.",
            "- **RMSE:** penalizes large misses more strongly than MAE.",
            "- **sMAPE:** scale-independent percentage error.",
            "- **MASE:** error relative to a seasonal-naive baseline; below 1 is better "
            "than that baseline on the scaling sample.",
            "- **Bias:** signed average error; the value closest to zero is best.",
            "",
        ]
    )
    return "\n".join(lines)
