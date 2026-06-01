import re

import plotly.graph_objects as go

from config import COLORS

RESERVED_COLS = {"ds", "unique_id", "cutoff", "y", "ensemble"}


def _is_ci_col(col: str) -> bool:
    return bool(re.search(r"-(lo|hi)-\d+$", col))


def _ci_bands(columns, model: str) -> list[tuple[str, str, int]]:
    bands = []
    for col in columns:
        m = re.match(rf"^{re.escape(model)}-lo-(\d+)$", col)
        if m:
            hi_col = f"{model}-hi-{m.group(1)}"
            if hi_col in columns:
                bands.append((col, hi_col, int(m.group(1))))
    return bands


def plot_backtest(predictions_df, show_ensemble=True):
    preds = predictions_df.sort("ds")

    cols = preds.columns
    models = [c for c in cols if c not in RESERVED_COLS and not _is_ci_col(c)]
    colors = (COLORS * ((len(models) // len(COLORS)) + 1))[:len(models)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=preds["ds"], y=preds["y"],
        name="Actual", line=dict(color="black", width=1.5),
    ))

    for model, color in zip(models, colors):
        for lo_col, hi_col, level in _ci_bands(cols, model):
            fig.add_trace(go.Scatter(
                x=list(preds["ds"]) + list(preds["ds"])[::-1],
                y=list(preds[hi_col]) + list(preds[lo_col])[::-1],
                fill="toself", fillcolor=color, opacity=0.15,
                line=dict(width=0), name=f"{model} {level}% CI",
                legendgroup=model,
            ))
        fig.add_trace(go.Scatter(
            x=preds["ds"], y=preds[model],
            name=model, line=dict(color=color, width=1),
            opacity=0.8, legendgroup=model,
        ))

    if show_ensemble and "ensemble" in preds.columns:
        fig.add_trace(go.Scatter(
            x=preds["ds"], y=preds["ensemble"],
            name="Ensemble", line=dict(color="gold", width=2, dash="dash"),
        ))

    fig.update_layout(
        title="Backtest — rolling forecasts vs actuals",
        xaxis_title="Time", yaxis_title="Heat Demand",
        hovermode="x unified",
    )
    fig.show()
