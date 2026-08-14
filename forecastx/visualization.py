"""Interactive forecast and diagnostics dashboards."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from forecastx.ensemble import point_model_columns


COLORS = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#d97706",
    "#0891b2",
    "#db2777",
    "#4f46e5",
    "#65a30d",
    "#475569",
)


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def forecast_dashboard(
    predictions: pl.DataFrame,
    *,
    title: str | None = "Forecast explorer",
) -> go.Figure:
    """Zoomable point, interval, residual, and horizon-error exploration."""

    if "y" not in predictions.columns:
        raise ValueError("Dashboard requires actual target column 'y'.")
    models = point_model_columns(predictions)
    if "ensemble" in predictions.columns:
        models.append("ensemble")
    if not models:
        raise ValueError("No point forecast columns were found.")
    ordered = predictions.sort(["ds", "horizon_step"] if "horizon_step" in predictions.columns else "ds")
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        row_heights=[0.58, 0.22, 0.20],
        subplot_titles=("Actual and forecasts", "Residuals", "Absolute error by horizon"),
    )
    actual = ordered.unique(subset=["unique_id", "ds"], keep="last").sort("ds")
    fig.add_trace(
        go.Scatter(
            x=actual["ds"],
            y=actual["y"],
            name="Actual",
            line={"color": "#0f172a", "width": 2},
            hovertemplate="%{x}<br>actual=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for index, model in enumerate(models):
        color = COLORS[index % len(COLORS)]
        interval_levels = sorted(
            int(column.rsplit("-lo-", 1)[1])
            for column in ordered.columns
            if column.startswith(f"{model}-lo-")
        )
        if interval_levels:
            level = max(interval_levels)
            low, high = f"{model}-lo-{level}", f"{model}-hi-{level}"
            fig.add_trace(
                go.Scatter(x=ordered["ds"], y=ordered[high], mode="lines", line={"width": 0}, showlegend=False, legendgroup=model),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=ordered["ds"],
                    y=ordered[low],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=_rgba(color, 0.13),
                    name=f"{model} {level}% interval",
                    legendgroup=model,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=ordered["ds"],
                y=ordered[model],
                name=model,
                line={"color": color, "width": 1.4},
                legendgroup=model,
                hovertemplate=f"%{{x}}<br>{model}=%{{y:.3f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        residual = ordered["y"] - ordered[model]
        fig.add_trace(
            go.Scatter(
                x=ordered["ds"],
                y=residual,
                name=f"{model} residual",
                mode="markers",
                marker={"color": color, "size": 4, "opacity": 0.55},
                legendgroup=model,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        horizon = ordered["horizon_step"] if "horizon_step" in ordered.columns else pl.Series(np.arange(1, len(ordered) + 1))
        fig.add_trace(
            go.Scatter(
                x=horizon,
                y=residual.abs(),
                name=f"{model} absolute error",
                mode="markers",
                marker={"color": color, "size": 5, "opacity": 0.45},
                legendgroup=model,
                showlegend=False,
            ),
            row=3,
            col=1,
        )
    fig.add_hline(y=0, line={"color": "#94a3b8", "dash": "dot"}, row=2, col=1)
    fig.update_xaxes(
        rangeslider={"visible": True, "thickness": 0.06},
        rangeselector={
            "buttons": [
                {"count": 1, "label": "1d", "step": "day", "stepmode": "backward"},
                {"count": 7, "label": "7d", "step": "day", "stepmode": "backward"},
                {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
                {"step": "all", "label": "All"},
            ]
        },
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Forecast horizon step", row=3, col=1)
    fig.update_yaxes(title_text="Load", row=1, col=1)
    fig.update_yaxes(title_text="Actual − forecast", row=2, col=1)
    fig.update_yaxes(title_text="Absolute error", row=3, col=1)
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        height=900,
        legend={"orientation": "h", "y": 1.04, "x": 0},
        margin={"l": 70, "r": 30, "t": 80 if title is None else 110, "b": 50},
    )
    return fig
