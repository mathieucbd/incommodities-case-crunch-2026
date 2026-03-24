# INCOMO 3 — Scores

Competition: InCommodities Case Crunch 2026
Metric: RMSE (FR) + RMSE (UK) = SUM
Validation: Feb 2024 — Jun 2024 (3623 hourly observations)

---

## Best Scores

| Pipeline | FR +HBC | UK +HBC | SUM | Kaggle |
|----------|---------|---------|-----|--------|
| **v7** (5-model regime) | **15.68** | **9.43** | **25.12** | non soumis |
| **v8** (+ new feats + UK 12m) | 15.70 | 9.45 | 25.15 | non soumis |
| v5b (4-model regime) | 16.31 | 10.09 | 26.39 | **24.88** |
| v3a (CB + LGB + HBC) | — | — | ~27.3 | 25.28 |
| **1er place** | — | — | — | **23.14** |

---

## Pipeline Evolution

| Version | Changement | FR +HBC | UK +HBC | SUM | Delta |
|---------|-----------|---------|---------|-----|-------|
| v3a | CB + LGB + HBC | ~17.5 | ~9.8 | ~27.3 | baseline |
| v4 | + EMA 240h FR anchor | 16.91 | 9.83 | 26.74 | -0.56 |
| v4b | + MAE loss UK | 16.91 | 9.78 | 26.68 | -0.06 |
| v5 | + XGBoost 3e modele | ~16.5 | ~9.7 | ~26.2 | -0.48 |
| v5b | + Regime weights (5 regimes) | 16.31 | 10.09 | 26.39 | +0.19 |
| v6 | + Elastic Net 4e modele | ~16.0 | ~9.9 | 25.89 | -0.50 |
| v7 | + DNN 5e modele (Huber) | 15.68 | 9.43 | 25.12 | -0.77 |
| v8 | + rolling_336h, stress_index, UK 12m | 15.70 | 9.45 | 25.15 | +0.03 |

---

## Kaggle Submissions

| Submission | Score | Notes |
|-----------|-------|-------|
| v3a (std HBC) | 25.28 | 1er soumis |
| v5b (regime) | 24.88 | meilleur score |
| v7 (5-model) | non soumis | val=25.12 |
| v8 | non soumis | val=25.15 |
| blend v5b+v7 | non soumis | diversite ok (std=8.93 FR, 2.97 UK) |
| **1er place** | **23.14** | Team KISS |

---

## Scores Standalone par Modele (validation +HBC)

### France

| Modele | RMSE | +HBC | Notes |
|--------|------|------|-------|
| CatBoost (Optuna v2) | 17.21 | 17.02 | depth=3, lr=0.059, 28 feat |
| LightGBM | 17.73 | 17.58 | 15 leaves, reg_alpha=5 |
| XGBoost | 28.38 | 24.88 | depth=4, lr=0.05 (instable) |
| Elastic Net | 18.09 | 16.56 | alpha=10, l1=0.9, 14 nonzero |
| DNN [192,96] | 17.23 | 16.69 | Huber delta=5, dropout=0.2 |
| GRU seq12_h128 | 18.22 | 17.80 | pire que trees |
| GAT-GRU s24_gat64 | 18.65 | 18.16 | pire que GRU |
| **Regime Ensemble (5)** | **15.81** | **15.70** | v8 weights |

### United Kingdom

| Modele | RMSE | +HBC | Notes |
|--------|------|------|-------|
| CatBoost (12m, MAE) | 10.12 | 9.94 | depth=8, basis target, 154 feat |
| CatBoost (full) | 10.03 | 9.87 | v7 full window |
| LightGBM | 10.42 | 10.15 | 63 leaves |
| XGBoost | 10.38 | 10.11 | depth=7 |
| Elastic Net | 11.72 | 11.32 | alpha=1, 32 nonzero |
| DNN [768,384,192] | 11.30 | 11.03 | Huber, dropout=0.3 |
| GRU seq24_h256 | 11.34 | 11.12 | pire que trees |
| GAT-GRU s24_gat64 | 10.99 | 10.65 | pire que trees |
| **Regime Ensemble (5)** | **9.55** | **9.45** | v8 weights |

---

## Regime Weights (v8)

### FR
| Regime | Heures | CB | LGB | XGB | EN | DNN |
|--------|--------|-----|-----|-----|-----|-----|
| Night | 0-5 | 0.0 | 0.6 | 0.0 | 0.1 | 0.3 |
| Morning | 6-9 | 0.0 | 0.0 | 0.0 | 0.3 | 0.7 |
| Day | 10-16 | 0.1 | 0.2 | 0.0 | 0.4 | 0.3 |
| Peak | 17-21 | 0.1 | 0.0 | 0.0 | 0.5 | 0.4 |
| Late | 22-23 | 0.5 | 0.1 | 0.0 | 0.0 | 0.4 |

### UK
| Regime | Heures | CB | LGB | XGB | EN | DNN |
|--------|--------|-----|-----|-----|-----|-----|
| Night | 0-5 | 0.5 | 0.0 | 0.0 | 0.4 | 0.1 |
| Morning | 6-9 | 0.6 | 0.0 | 0.0 | 0.1 | 0.3 |
| Day | 10-16 | 0.0 | 0.4 | 0.4 | 0.0 | 0.2 |
| Peak | 17-21 | 0.3 | 0.4 | 0.1 | 0.0 | 0.2 |
| Late | 22-23 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |

---

## HBC Variants (v7 ensemble)

| Methode | FR | UK | SUM |
|---------|-----|-----|-----|
| Standard HBC (24 params) | 15.68 | 9.43 | **25.12** |
| Monthly x Hour (120 params) | 15.41 | 8.95 | **24.36** |
| Dampened Monthly (alpha=0.7) | 15.45 | 9.00 | 24.45 |

---

## A/B Tests — Resultats Cles

### Target Engineering
| Config | FR +HBC | UK +HBC | Verdict |
|--------|---------|---------|---------|
| EMA 240h (FR) | **16.85** | — | gagnant FR |
| Rolling mean 240h (FR) | 17.22 | — | -0.37 vs EMA |
| Rolling mean 168h (FR) | 17.34 | — | baseline ancien |
| Basis merit_order_cost (UK) | — | **9.97** | gagnant UK |
| EMA 240h (UK) | — | 10.65 | -0.68 vs basis |

### Loss Functions (CatBoost)
| Loss | FR +HBC | UK +HBC |
|------|---------|---------|
| RMSE | **16.52** | 9.97 |
| MAE | 16.87 | **9.87** |
| Huber d=80 | 16.61 | — |
| Huber d=30 | 16.78 | — |

### Multi-Window (UK)
| Window | UK +HBC |
|--------|---------|
| Full (Jul 2022 — Jan 2024) | 10.29 |
| 18m | 10.11 |
| **12m (Feb 2023 — Jan 2024)** | **9.86** |
| 9m | 10.24 |
| 6m | 10.45 |

### New Features (v8, CatBoost only)
| Feature Group | FR Delta | UK Delta |
|---------------|----------|----------|
| rolling_336h (14d mean/std) | -0.12 | -0.16 |
| stress_index | 0.00 | -0.12 |
| load_surprise | -0.02 | -0.04 |
| All combined | **-0.23** | **-0.17** |

### Per-Hour Models (24 CB vs 1 CB + HBC)
| Config | FR | UK | SUM |
|--------|-----|-----|-----|
| 1 CB + HBC | 17.05 | 10.46 | 27.52 |
| 24 CB+LGB +HBC | 16.99 | 10.23 | **27.22** |
| Delta | -0.06 | **-0.23** | -0.30 |
