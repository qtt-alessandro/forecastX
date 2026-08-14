"""forecastx: modular, leakage-aware energy load forecasting."""

from forecastx.audit import LeakageAuditReport, future_target_invariance
from forecastx.backtest import backtest, statistical_backtest
from forecastx.config import FAST_MODELS, ForecastConfig
from forecastx.data import DataQualityReport, load_csv, quality_report, resample, split
from forecastx.diagnostics import distribution_drift, recommend_validation, stationarity_report
from forecastx.engine import ForecastEngine, build_mlf, fit, predict
from forecastx.features import add_temperature_features
from forecastx.intervals import add_conformal_intervals
from forecastx.metrics import evaluate_forecasts, interval_metrics, residual_diagnostics
from forecastx.models import list_models
from forecastx.outputs import ForecastArtifact
from forecastx.selection import compare_training_windows, recent_regime_split
from forecastx.visualization import forecast_dashboard

__all__ = [
    "DataQualityReport",
    "FAST_MODELS",
    "ForecastArtifact",
    "ForecastConfig",
    "ForecastEngine",
    "LeakageAuditReport",
    "backtest",
    "add_temperature_features",
    "add_conformal_intervals",
    "build_mlf",
    "compare_training_windows",
    "distribution_drift",
    "evaluate_forecasts",
    "fit",
    "forecast_dashboard",
    "future_target_invariance",
    "interval_metrics",
    "list_models",
    "load_csv",
    "predict",
    "quality_report",
    "recent_regime_split",
    "recommend_validation",
    "resample",
    "residual_diagnostics",
    "split",
    "stationarity_report",
    "statistical_backtest",
]
