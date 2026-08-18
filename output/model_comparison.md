# Model comparison

Rolling-origin evaluation over **6,192 forecasts**, with a **24-step horizon**, from **2025-01-09** to **2025-09-24 01:00**. Runtime: **41.88 minutes**.

The primary ranking uses MAE. **Ridge** has the lowest MAE.
The performance-weighted ensemble ranks **#3 by MAE**.

Color guide: 🟢 best MAE, 🟡 within 10% of the best MAE, 🔴 more than 10% above it. A green value marks the winner for that metric.

| Rank | MAE band | Model | MAE ↓ | RMSE ↓ | sMAPE ↓ | MASE ↓ | Bias ≈ 0 | Ensemble weight |
|---:|:---|:---|---:|---:|---:|---:|---:|---:|
| 1 | 🟢 Best | Ridge | 🟢 **0.3112** | 🟢 **0.4836** | 20.81% | 🟢 **0.3820** | 0.0208 | 12.17% |
| 2 | 🟡 Within 10% | Lasso | 0.3189 | 0.5089 | 19.72% | 0.3915 | 0.0424 | 9.80% |
| 3 | 🟡 Within 10% | ensemble | 0.3203 | 0.7042 | 🟢 **17.59%** | 0.3932 | -0.0138 | — |
| 4 | 🟡 Within 10% | ElasticNet | 0.3238 | 0.5209 | 19.82% | 0.3975 | 0.0542 | 9.47% |
| 5 | 🟡 Within 10% | XGBRegressor | 0.3309 | 0.5263 | 18.95% | 0.4062 | 🟢 **-0.0023** | 10.65% |
| 6 | 🔴 Above 10% | LGBMRegressor | 0.3448 | 0.5639 | 18.75% | 0.4233 | 0.0305 | 9.57% |
| 7 | 🔴 Above 10% | HistGradientBoostingRegressor | 0.3516 | 0.5715 | 19.21% | 0.4316 | 0.0531 | 10.68% |
| 8 | 🔴 Above 10% | SeasonalNaive | 0.4043 | 0.6198 | 22.92% | 0.4962 | 0.0171 | 12.49% |
| 9 | 🔴 Above 10% | LSTM | 0.4155 | 0.6540 | 24.08% | 0.5100 | -0.0279 | 6.78% |
| 10 | 🔴 Above 10% | SARIMAX | 0.4388 | 1.5661 | 22.92% | 0.5386 | -0.0742 | 10.57% |
| 11 | 🔴 Above 10% | LinearRegression | 0.6604 | 5.2914 | 22.21% | 0.8107 | -0.3366 | 7.82% |

## How the ensemble works

The ensemble is not another fitted forecasting model. It combines the individual point forecasts using a weighted average:

`ensemble forecast = Σ(model weight × model forecast)`

1. Each model is evaluated on rolling windows immediately before the reported backtest period.
2. A model's initial score is the inverse of its calibration MAE, so a smaller error produces a larger weight.
3. The weights receive 20% shrinkage toward equal weighting. This limits the influence of one unusually strong calibration result.
4. The non-negative weights are normalized to sum to 100% and are then held fixed throughout the reported evaluation period.

The weights come from pre-test calibration data, whereas the ranking above comes from the held-out backtest. Their ordering can therefore differ.

| Component model | Weight |
|:---|---:|
| SeasonalNaive | 12.49% |
| Ridge | 12.17% |
| HistGradientBoostingRegressor | 10.68% |
| XGBRegressor | 10.65% |
| SARIMAX | 10.57% |
| Lasso | 9.80% |
| LGBMRegressor | 9.57% |
| ElasticNet | 9.47% |
| LinearRegression | 7.82% |
| LSTM | 6.78% |
| **Total** | **100.00%** |

## Metric guide

- **MAE:** average absolute forecast error; the main ranking metric.
- **RMSE:** penalizes large misses more strongly than MAE.
- **sMAPE:** scale-independent percentage error.
- **MASE:** error relative to a seasonal-naive baseline; below 1 is better than that baseline on the scaling sample.
- **Bias:** signed average error; the value closest to zero is best.
