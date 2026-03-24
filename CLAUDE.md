# INCOMO 3 — InCommodities Case Crunch 2026 (Real Competition)

## Project Overview
Forecast hourly day-ahead electricity spot prices for **France (fr_spot)** and **United Kingdom (uk_spot)** over an 8-month out-of-sample period (July 2024 — February 2025). Metric: **RMSE** averaged across both targets.

## Key Constraints
- **No external data**: only the provided CSVs + feature engineering + calendar features (holidays via Python `holidays` lib)
- **Submission format**: CSV with columns `id, fr_spot, uk_spot` (5,833 rows, IDs 17544–23376)
- **Live evaluation** at finals on unseen future data — generalization > Kaggle LB score
- **Kaggle public LB** uses only 60% of test data — do not overfit to it

## Data Structure
- `data/raw/x_train.csv`: 17,544 rows (2022-07-01 → 2024-06-30), 111 columns
- `data/raw/y_train.csv`: targets fr_spot, uk_spot
- `data/raw/x_test.csv`: 5,833 rows (2024-07-01 → 2025-02-28)
- Column suffixes: `_f` = forecast, `_la` = 24h lagged actual, no suffix = daily (00:00 CET only)
- Daily columns (gas/emissions) are NaN except at 00:00 CET → forward-fill before use
- DK1-UK interconnectors: 72-74% missing (launched Sept 2023). NL: ~61% missing.

## Coding Rules
1. **No standard MAPE**: electricity prices can be zero/negative → use RMSE or sMAPE
2. **Chronological splits only**: never shuffle time-series data
3. **No data leakage**: fit scalers/encoders on training data ONLY
4. **Configuration-driven**: hyperparameters in `config.yaml`, not hardcoded
5. **Package manager**: use `uv`, not pip
6. **Functional pipelines** for data transforms, **OOP for models**
7. **Separate models** for FR and UK (different market dynamics)

## Architecture
- **Primary model**: CatBoost (RMSE loss, handles NaN natively)
- **Secondary model**: LightGBM (MAE loss for ensemble diversity)
- **Ensemble**: Ridge regression stacking or weighted average
- **Post-processing**: prediction clipping + hourly bias correction

## Key Domain Knowledge
- **Residual load** = demand - renewables → drives marginal price (merit order)
- **Spark spread** = gas_price / efficiency + emission_price * 0.37 → marginal cost of gas generation
- FR: nuclear-dominated, SDAC coupled market (EPEX SPOT, clears 12:00 CET)
- UK: gas/wind-dominated, separate auction (Nord Pool N2EX, clears 09:10 UK time)
- 6 UK interconnectors: IFA1, IFA2, ElecLink (FR-UK), NEMO (BE-UK), BritNed (NL-UK), Viking Link (DK1-UK)
- River temperatures >25C → nuclear curtailment in FR → price spikes
- RMSE penalizes spikes quadratically → getting spikes right is critical

## File Structure
```
INCOMO 3/
├── data/raw/                      # x_train.csv, y_train.csv, x_test.csv
├── docs/Subject.md                # Competition brief
├── src/
│   ├── data_loading.py
│   ├── feature_engineering.py
│   ├── validation.py
│   └── models/
│       ├── catboost_model.py
│       ├── lightgbm_model.py
│       └── ensemble.py
├── notebooks/01_eda.ipynb
├── outputs/
│   ├── models/
│   ├── predictions/
│   └── submissions/
├── config.yaml
├── pyproject.toml
└── run_pipeline.py
```
