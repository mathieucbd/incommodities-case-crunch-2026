# Market & Competition Context

## Competition

**InCommodities Case Crunch 2026** — competition organized by InCommodities (electricity trading, Aarhus, Denmark).

**Objective**: Forecast hourly day-ahead electricity spot prices for **France** (`fr_spot`) and the **United Kingdom** (`uk_spot`) over an 8-month out-of-sample period (July 2024 — February 2025).

**Metric**: `RMSE(FR) + RMSE(UK) = SUM` — both markets weighted equally.

**Constraints**:
- Provided data only (no external data, APIs, etc.)
- Feature engineering allowed (rolling, calendar, holidays via Python `holidays`)
- Kaggle public leaderboard uses 60% of test — do not overfit to it
- Final evaluation live at InCommodities HQ on unseen future data

**Final Rankings (Private Leaderboard)**:
| Rank | Team | Private Score | Notes |
|------|------|---------------|-------|
| **1st** | **2sigmas** | **19.5475** | winner |
| **2nd** | **AQTC (us)** | **20.0781** | **submission_attack_averaged** |
| **3rd** | **Team KISS** | **21.4419** | — |
| — | InCommodities benchmark | 21.4333 | beaten |

**Gap to 1st**: 0.53 RMSE

**Best Kaggle Public**: blend v17_85 + v9_15 = 23.20 (former public leader before private reveal)

---

## Data

### Raw Files (`data/raw/`)

**Original (competition)**:
| File | Rows | Period | Columns |
|------|------|--------|---------|
| `x_train.csv` | 17,544 | 2022-07-01 → 2024-06-30 | 111 (3 id + 67 forecast + 34 lagged + 7 daily) |
| `y_train.csv` | 17,544 | same | id, datetime_CET, datetime_UTC, fr_spot, uk_spot |
| `x_test.csv` | 5,833 | 2024-07-01 → 2025-02-28 | 111 (same schema, no targets) |

**Fulldata (received 2026-03-26, post-competition)**:
| File | Rows | Period | Columns |
|------|------|--------|---------|
| `x_train_full.csv` | 23,377 | 2022-01-01 → 2025-02-28 | 111 |
| `y_train_full.csv` | 23,377 | same | id, datetime_CET, datetime_UTC, fr_spot, uk_spot |

The fulldata files include actuals for the test period (Jul 2024 → Feb 2025), enabling submission validation against true values.

### Column Naming Conventions

| Suffix | Meaning | Resolution |
|--------|---------|------------|
| `_f` | Forecast (D-1 forecast) | Hourly |
| `_la` | Lagged actual (D-1 realized value, 24h shift) | Hourly |
| (none) | Daily (gas, emissions) | 1 value at 00:00 CET, NaN otherwise |

### Key Columns

- **Load forecasts** (`*_load_f`): electricity demand for 10 countries/zones
- **Renewables** (`*_solar_f`, `*_wind_f`): forecast solar/wind generation
- **Nuclear/Gas capacity** (`*_avcap_f`): available capacity (UMM)
- **Interconnectors**: ATC/NTC (capacity), flows and costs (lagged) for 6 UK links
- **Lagged prices** (`*_spot_la`): D-1 spot price for 10 European zones
- **Gas/emissions** (`de_gas`, `uk_gas`, `eu_emission`, `uk_emission`): daily, sparse

### Temporal Split

**Original (competition)**:
```
Train:  Jul 2022 ───────────────────────────── Jan 2024  (17544 samples)
Val:    ·········································· Feb 2024 ── Jun 2024  (3623h)
Test:   ··················································· Jul 2024 ── Feb 2025  (5833h)
```

**Fulldata (post-competition)**:
```
Full:   Jan 2022 ────────────────────────────────────────────── Feb 2025  (23377 samples)
        ↑
        +6 additional months
```

**New holdouts (ACTION 3)**:
- **SPRING**: Train 2022-07-01→2024-01-31, Val 2024-02-01→2024-06-30
- **WINTER**: Train 2022-07-01→2024-01-31, Val 2023-07-01→2023-11-30 (anti-seasonal)
- **SPRING NEW (fulldata)**: Train 2022-01-01→2024-09-30, Val 2024-10-01→2024-12-31
- **WINTER NEW (fulldata)**: Train 2022-01-01→2024-06-30, Val 2024-11-01→2025-02-28

UK uses a 12-month training window (Feb 2023 → Jan 2024) — the post-gas-crisis regime is more representative of the test period.

---

## Domain

### FR Market (SDAC / EPEX SPOT)
- **Nuclear-dominated**: 70% of generation, price often set by nuclear marginal cost
- Day-ahead auction at 12:00 CET (EUPHEMIA algorithm, European SDAC coupling)
- FR prices are linked to neighbours (DE, BE, CH, ES) via market coupling
- **River temperature** >25°C = nuclear curtailment risk = price spikes
- Aggressive feature selection: 32 features sufficient (from 330+ candidates)

### UK Market (N2EX / Nord Pool)
- **Gas-dominated**: merit order cost = gas/efficiency + emission × 0.37
- Separate from the continent (post-Brexit), clearing at 09:10 UK time
- 6 subsea interconnectors: IFA1, IFA2, ElecLink (FR-UK), NEMO (BE-UK), BritNed (NL-UK), Viking Link (DK1-UK)
- Heavy tails: prices from -205 to +1444 EUR/MWh → MAE loss more robust
- 154 features needed (more sources of variability)

### Core Concepts
- **Merit order**: plants are ranked by increasing marginal cost; price = cost of the last plant needed
- **Residual load** = demand - renewables → determines which plant is marginal
- **Spark spread** = gas price / efficiency + emission × factor → marginal cost of gas generation
- RMSE penalizes spikes quadratically → accurately predicting extremes is critical

---

## Project Architecture

```
incommodities-case-crunch-2026/
├── data/raw/                          # Raw data (x_train, y_train, x_test)
├── docs/
│   ├── Subject.md                     # Competition brief
│   ├── context.md                     # This file
│   └── FINDINGS.md                    # Key findings and decisions
├── notebooks/
│   ├── 01_eda.ipynb                   # Initial exploration
│   ├── 02_eda_deep_dives.ipynb        # Deep-dive analyses
│   ├── 03_feature_selection.ipynb     # SHAP/Boruta feature selection
│   ├── 04_catboost_results.ipynb      # CatBoost experiments
│   ├── 05_fr_error_diagnostic.ipynb   # FR error diagnostics
│   ├── 06_basis_modeling.ipynb        # UK basis modeling (spot - MOC)
│   └── 07_results_recap.ipynb         # Results summary
├── scripts/
│   ├── train_v1_baseline.py           # v9 baseline (~800 lines)
│   ├── train_v2_stacking_residual.py          # + Stacking Residual
│   ├── train_v3_stl_target.py          # + STL target FR
│   └── train_v4_dual_holdout.py       # Winning pipeline
├── src/
│   ├── __init__.py
│   ├── data_loading.py                # CSV loading (33 lines)
│   ├── feature_engineering.py         # Feature construction (1800 lines, 290+ features)
│   └── models/
│       ├── __init__.py                # Re-exports
│       ├── metrics.py                 # RMSE, HBC utilities
│       ├── targets.py                 # Stationary target preparation
│       ├── tree_models.py             # CatBoost/LGB/XGB unified wrapper
│       ├── elastic_net.py             # Elastic Net + scaler
│       ├── dnn.py                     # DNN PyTorch
│       └── ensemble.py                # Regime weights optimization
├── outputs/
│   ├── feature_selection_v5_fr.json   # 28 FR features selected
│   └── uk_feature_research.json       # 150 UK features selected
├── config.yaml                        # Hyperparameters and thresholds
└── pyproject.toml                     # Python dependencies
```

---

## Pipeline v17 (BEST pre-attack)

### Overview

Pipeline v17 = v16 (STL target FR) + Fix 2 (coherent STL retrain). It chains:

1. **Loading**: `data_loading.load_data()` → x_train, y_train, x_test
2. **Feature engineering**: `feature_engineering.build_features()` (~290 features)
3. **Split**: train (Jul 2022 — Jan 2024) / val (Feb — Jun 2024) / test (Jul 2024 — Feb 2025)
4. **Stationary target**:
   - **FR**: `y = spot - STL_trend(spot_la, 168h)` — seasonal decomposition (v16+)
   - **UK**: `y = spot - merit_order_cost` — MOC captures ~70% of the price signal
5. **Training** of 7 FR models + 8 UK models (15 total) on the deviation target
6. **Ensemble**: weights per hourly regime (5 regimes × N models), grid search step 0.1
7. **HBC**: systematic hourly bias correction (24 parameters)
8. **Retrain** on train+val with coherent STL (Fix 2), test prediction, CSV generation

### Model Architecture v17 (7 FR + 8 UK)

**France (7 models)**:
| Model | Config | Role |
|-------|--------|------|
| **CatBoost** | depth=3, Quantile:0.6, 28 feat | Base tree, asymmetric |
| **LightGBM** | 15 leaves, MAE loss | Loss diversity |
| **XGBoost** | depth=4, PseudoHuber:20 | Loss diversity |
| **Elastic Net** | alpha=10, l1=0.9, 14 non-zero | Linear, corr 0.60-0.70 with trees |
| **DNN** | [192,96], Huber:5, dropout=0.2 | Non-linear, 356 feat, corr 0.77-0.82 |
| **RidgeF** | Fundamental features only | Highly decorrelated (corr=0.65 FR, 0.41 UK) |
| **SR** | Ridge meta-learner on v9 errors | Residual stacking, T2 models (55 members) |

**UK (8 models)**:
| Model | Config | Role |
|-------|--------|------|
| **CatBoost** | depth=8, MAE loss, 12m window, 154 feat | Base tree |
| **LightGBM** | 63 leaves, Huber:5, 12m | Diversity |
| **XGBoost** | depth=7, PseudoHuber:20, 12m | Diversity |
| **Elastic Net** | alpha=1, 32 non-zero, 12m | Linear |
| **DNN** | [768,384,192], MSE, dropout=0.3, 12m | Non-linear |
| **RidgeF** | Fundamental features only | Decorrelated |
| **SR** | Ridge meta-learner, T2 models (65 members with combos) | Residual stacking |
| **XGB_cluster** | XGB on 6h-shifted clusters (4 clusters) | Temporal segmentation (v13) |

### Regime Ensemble (5 regimes)

5 hourly regimes: night (0-5h), morning (6-9h), day (10-16h), peak (17-21h), late (22-23h).

**FR morning example (submission_attack_spring)**:
- CB=0.1, LGB=0.0, XGB=0.0, EN=0.1, DNN=0.3, RidgeF=0.0, SR=0.5

**FR morning example (submission_attack_winter)**:
- CB=0.1, LGB=0.0, XGB=0.0, EN=0.1, DNN=0.1, RidgeF=0.0, SR=0.7

Weights diverge radically by season (ACTION 3). DNN disappears in winter, SR ultra-dominant.

### Post-processing

- **HBC**: 24 hourly corrections (e.g. h=7 +3.45, h=18 -3.20 for FR)
- **Clipping**: 0.1% / 99.9% percentiles of training set (FR: [-40.6, 800], UK: [-22.5, 798.5])

---

## Current Scores

### Validation — BEST (attack_averaged, dual holdout)

| Submission | FR +HBC | UK +HBC | SUM | Notes |
|------------|---------|---------|-----|-------|
| **attack_averaged** | **14.57** | **8.80** | **23.37** | spring+winter average |
| attack_spring | 14.35 | 8.83 | 23.18 | v17 baseline |
| attack_winter | 14.79 | 8.77 | 23.56 | winter weights only |

### Kaggle (Private Leaderboard)

| Submission | Private Score | Rank |
|------------|---------------|------|
| **attack_averaged** | **20.0781** | **2nd/68** |
| attack_winter | 20.2397 | — |
| blend v17_85 + v9_15 | 20.4818 | former best public |

### Individual Model Scores v17 (+HBC, spring holdout)

| Model | FR | UK |
|-------|-----|-----|
| CatBoost | 16.08 | 9.78 |
| LightGBM | 17.58 | 10.15 |
| XGBoost | 18.67 | 10.11 |
| Elastic Net | 16.44 | 13.10 |
| DNN | 16.69 | 10.96 |
| RidgeF | ~17.5 | ~11.5 |
| SR (residual stacking) | 16.55 | 9.80 |

---

## Version History

| Version | Main Change | SUM val | Kaggle |
|---------|------------|---------|--------|
| v3a | CatBoost + LightGBM + HBC | ~27.3 | 25.28 |
| v4 | + EMA 240h FR anchor | 26.74 | — |
| v4b | + MAE loss UK | 26.68 | — |
| v5 | + XGBoost 3rd model | ~26.2 | — |
| v5b | + Regime weights (5 regimes) | 26.39 | 24.88 |
| v6 | + Elastic Net 4th model | 25.89 | — |
| v7 | + DNN 5th model (Huber loss) | 25.12 | — |
| v8 | + rolling_336h, stress_index, UK 12m | 25.15 | — |
| v9 | Audit fixes + loss diversity | 25.10 | — |
| v10 | + Nystroem kernel features | 25.07 | — |
| v11 | + RidgeF + Residual Stacking T2 | 23.83 | 24.23 (+0.40 overfit) |
| v11b | Anti-overfitting (3+3 combos, alpha=500) | 23.85 | 23.92 (+0.07 gap) |
| v13 | + XGB_cluster UK (8th member) | 23.73 | 23.94 (+0.21 gap) |
| v16 | + STL target FR (replaces EMA 240h) | 23.08 | 23.26 (+0.18 STL bias) |
| **v17** | **+ Fix 2 coherent STL retrain** | **23.08** | **23.20** (+0.12 gap) |
| **attack_averaged** | **Dual holdout spring+winter** | **23.37** | **20.0781** ⭐ |

**Blending**:
- blend v17_85 + v9_15: Kaggle 20.4818 (former best public)
- blend v13_85 + v9_15: Kaggle 21.2563

---

## What Worked / What Didn't

### Confirmed gains

| Improvement | RMSE delta |
|-------------|-----------|
| Basis target UK (spot - MOC) | -0.68 UK |
| DNN as 5th model | -0.77 SUM |
| Elastic Net 4th model | -0.50 SUM |
| XGBoost 3rd model | -0.48 SUM |
| EMA 240h FR (vs rolling 168h) | -0.37 FR |
| Optuna v2 FR (depth=3, lr=0.059) | -0.86 FR |
| UK 12m training window | -0.43 UK (standalone) |

### Failures

| Attempt | Result |
|---------|--------|
| GRU / GAT-GRU | +0.4 to +1.1 vs trees — sequences useless on tabular features |
| Stacking (Ridge) | +2.1 FR — meta-learner overfits |
| arcsinh transform | +0.3-0.5 — systematically degrades |
| Basis modeling FR (spot - MOC) | +4.5 FR — gas crisis distorts the MOC |
| Regime weight regularization | +0.20 val — shrinkage degrades v7 weights |

---

## Dependencies

```
Python >=3.10
pandas, numpy, scikit-learn
catboost, lightgbm, xgboost
torch (PyTorch — for the DNN)
holidays (FR/UK public holidays)
optuna (tuning — notebooks only)
matplotlib, seaborn (visualization)
```

Package manager: `uv` (not pip).
