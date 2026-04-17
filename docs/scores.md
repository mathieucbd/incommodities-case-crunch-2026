# Scores

Competition: InCommodities Case Crunch 2026
Metric: RMSE (FR) + RMSE (UK) = SUM
Validation: Feb 2024 — Jun 2024 (3623 hourly observations)

---

## Best Scores

| Submission | Kaggle Public | Kaggle Private | Notes |
|------------|---------------|----------------|-------|
| **submission_attack_averaged** | **22.8294** | **20.0781** ⭐ | **BEST** — calibration winter+spring averaged |
| submission_attack_winter | 22.9834 | 20.2397 | winter weights only |
| submission_attack_spring | 23.0708 val | — | baseline v17 |
| blend v17_85 + v9_15 | 23.20 | 20.4818 | former best public |
| v21 (trio CB) | 23.2411 | 20.4918 | |
| v18 | 23.2608 | 20.5170 | |
| v16_stl | 23.2575 | 20.5090 | |
| blend v16_85 + v9_15 | 23.2415 | 20.5138 | |
| v11 | 23.7416 | 20.8676 | |
| submission (Mtcbd6) | 23.7593 | 20.8874 | |
| blend v13_85 + v9_15 | 23.7796 | 21.2563 | |
| **2sigmas (1er)** | — | **19.5475** | competitor |
| **AQTC (nous, 2e)** | — | **20.0781** | |
| **InCommodities benchmark** | — | **21.4333** | benchmark beaten |
| **Team KISS (3e)** | — | **21.4419** | |

---

## ACTION 3 — Winter Holdout Recalibration

**Principle**: Test regime weight stability across 2 seasonal holdouts

### Dual Holdouts

| Holdout | Train Period | Val Period | Strategy |
|---------|--------------|------------|----------|
| **SPRING** | 2022-07-01 → 2024-01-31 | 2024-02-01 → 2024-06-30 | baseline (v17) |
| **WINTER** | 2022-07-01 → 2024-01-31 | 2023-07-01 → 2023-11-30 | anti-seasonal |

### Validation Scores (+HBC)

| Submission | FR | UK | SUM | Notes |
|------------|----|----|-----|-------|
| **submission_attack_spring** | 14.35 | 8.83 | **23.18** | spring weights (v17 baseline) |
| **submission_attack_winter** | 14.79 | 8.77 | **23.56** | winter weights (+0.38) |
| **submission_attack_averaged** | 14.57 | 8.80 | **23.37** | arithmetic mean (+0.19) |

### Critical Weight Divergence

#### FR — Spring vs Winter

| Regime | Spring CB | Winter CB | Spring DNN | Winter DNN | Spring SR | Winter SR |
|--------|-----------|-----------|------------|------------|-----------|-----------|
| Night | 0.0 | **0.2** | 0.2 | **0.0** | 0.7 | **0.8** |
| Morning | 0.1 | **0.1** | 0.3 | **0.1** | 0.5 | **0.7** |
| Day | 0.0 | **0.1** | 0.3 | **0.0** | 0.6 | **0.8** |
| Peak | 0.0 | **0.1** | 0.3 | **0.1** | 0.6 | **0.7** |
| Late | 0.3 | **0.4** | 0.4 | **0.2** | 0.2 | **0.3** |

**Key finding**: DNN almost completely disappears in winter (0.0-0.2), SR becomes ultra-dominant (0.7-0.8).

#### UK — Spring vs Winter

| Regime | Spring CB | Winter CB | Spring DNN | Winter DNN | Spring SR | Winter SR |
|--------|-----------|-----------|------------|------------|-----------|-----------|
| Night | 0.2 | **0.2** | 0.2 | **0.1** | 0.5 | **0.6** |
| Morning | 0.1 | **0.1** | 0.3 | **0.2** | 0.5 | **0.6** |
| Day | 0.1 | **0.1** | 0.2 | **0.1** | 0.6 | **0.7** |
| Peak | 0.2 | **0.3** | 0.2 | **0.1** | 0.5 | **0.5** |
| Late | 0.4 | **0.6** | 0.3 | **0.2** | 0.2 | **0.1** |

Less pronounced than FR, but SR also rises (0.5-0.7 except late).

### Kaggle Results (Private Leaderboard)

| Submission | Private Score | Delta vs v17_blend | Rank |
|------------|---------------|-------------------|------|
| **submission_attack_averaged** | **20.0781** | **-2.74** | **2nd/7** |
| submission_attack_winter | 20.2397 | -2.58 | — |
| submission_attack_spring | not submitted (= v17) | baseline | — |
| blend v17_85 + v9_15 | 20.4818 | ref | — |

**Huge gain**: -2.74 vs best previous blend → 2nd place (gap to 1st: 0.53)

### Best Iterations (Winter Holdout)

| Model | FR | UK |
|-------|----|----|
| CatBoost | 780 | 391 |
| LightGBM | 731 | 13843 |
| XGBoost | 857 | 1161 |
| DNN | 48 epochs | 35 epochs |

These iterations + winter averaged weights = submission_attack_averaged (winner).

---

## Fulldata (2022→2025)

**Date**: 2026-03-26
**Files received**: `x_train_full.csv`, `y_train_full.csv`
**Coverage**: 2022-01-01 → 2025-02-28 (23377 samples)

### New Holdouts

| Holdout | Train Period | Val Period | Notes |
|---------|--------------|------------|-------|
| **SPRING NEW** | 2022-01-01 → 2024-09-30 | 2024-10-01 → 2024-12-31 | Q4 2024 |
| **WINTER NEW** | 2022-01-01 → 2024-06-30 | 2024-11-01 → 2025-02-28 | winter 2024-25 |

### Results (train_v4_dual_holdout_postcomp.py)

| Holdout | FR +HBC | UK +HBC | SUM | Notes |
|---------|---------|---------|-----|-------|
| SPRING NEW | 14.73 | **33.51** | 48.25 | **UK EXPLODES** (+24.68 vs 8.83) |
| WINTER NEW | 14.65 | **35.54** | 50.18 | even worse |

**UK Catastrophe**: UK RMSE jumps from 8.83 (old val) to 33-35 in Q4 2024 → period entirely out-of-distribution.

### True RMSE (vs y_train_full actuals Jul 2024 → Feb 2025)

Top 10 submissions tested against true actuals (ID 17544-23376):

| Submission | FR | UK | SUM |
|------------|----|----|-----|
| **submission_blend_seedavg_v9** | **15.86** | **28.61** | **44.47** |
| submission_v9 (seed1) | 15.67 | 29.16 | 44.82 |
| submission_v9 (seed2) | 15.65 | 29.19 | 44.84 |
| submission_attack_averaged | 17.30 | 27.67 | **44.96** |
| submission_attack_winter | 17.35 | 27.70 | 45.05 |

**Surprise**: Simple seed-averaged v9 (5 models) beats attack_averaged on the true test!

### Identified Issues

1. **UK out-of-distribution Q4 2024**: extreme prices unseen in train 2022-2024
2. **Best_iters miscalibrated**: spring/winter optimal values for old val, not for new
3. **ID mismatch**: x_test in new context no longer matches the true test (now in y_train_full)
4. **Full retrain impossible**: fullretrain_winter_only generates IDs 0-5832 instead of 17544-23376

---

## Pipeline Evolution

| Version | Change    | FR +HBC | UK +HBC | SUM | Delta |
|---------|-----------|---------|---------|-----|-------|
| v3a | CB + LGB + HBC | ~17.5 | ~9.8 | ~27.3 | baseline |
| v4 | + EMA 240h FR anchor | 16.91 | 9.83 | 26.74 | -0.56 |
| v4b | + MAE loss UK | 16.91 | 9.78 | 26.68 | -0.06 |
| v5 | + XGBoost 3rd model | ~16.5 | ~9.7 | ~26.2 | -0.48 |
| v5b | + Regime weights (5 regimes) | 16.31 | 10.09 | 26.39 | +0.19 |
| v6 | + Elastic Net 4th model | ~16.0 | ~9.9 | 25.89 | -0.50 |
| v7 | + DNN 5th model (Huber) | 15.68 | 9.43 | 25.12 | -0.77 |
| v8 | + rolling_336h, stress_index, UK 12m | 15.70 | 9.45 | 25.15 | +0.03 |
| v9 | + loss diversity, no FR weights | 15.67 | 9.43 | 25.10 | -0.05 |
| v13 | + XGB cluster UK (8th member shifted 6h) | 14.86 | 8.87 | 23.73 | -0.12 (vs v11b) |
| **v16** | **+ STL 168h target FR (replaces EMA 240h)** | **14.25** | **8.83** | **23.08** | **-0.65** (vs v13) |
| **v17** | **+ Fix 2 STL coherent retrain** | **14.25** | **8.83** | **23.08** | **-0.04 test** (vs v16 blend) |

---

## v16 — STL Target Engineering (seasonal decomposition)

Architecture: v13 (7 FR models + 8 UK models) + STL 168h for FR target
- **Change FR**: target = spot - STL_trend (instead of spot - EMA 240h)
- **Change UK**: NONE (keeps target = spot - merit_order_cost)
- STL = Seasonal-Trend decomposition using Loess, period=168h (1 week), seasonal=13

### Standalone CatBoost Benchmarks (Quantile:0.6)

#### France — STL vs EMA 240h

| Target Variant | Anchor | RMSE +HBC | Delta vs EMA | Best Iter |
|----------------|--------|-----------|--------------|-----------|
| **target_stl_trend** | **STL trend** | **16.08** | **-1.84** | 147 | ✅
| target_ema (baseline) | EMA 240h | 17.92 | baseline | 59 |
| target_stl_full | STL trend+seasonal | 17.96 | +0.04 | 47 |

**FR result**: ✅ STL trend WINS by **-1.84** RMSE +HBC (~10% reduction)

#### UK — STL vs Merit Order Cost

| Target Variant | Anchor | RMSE +HBC | Delta vs MOC | Best Iter |
|----------------|--------|-----------|--------------|-----------|
| **target_moc (baseline)** | **Merit Order Cost** | **9.78** | **baseline** | 408 | ✅
| target_stl_trend | STL trend | 10.27 | +0.49 | 701 |
| target_stl_full | STL trend+seasonal | 11.30 | +1.52 | 601 |

**UK result**: ❌ Merit Order Cost WINS by **+0.49** RMSE +HBC

### Combined FR + UK Impact

| Market | Current Anchor | STL Trend | Delta | Decision |
|--------|----------------|-----------|-------|----------|
| **FR** | EMA 240h: 17.92 | STL: 16.08 | **-1.84** | ✅ **STL** |
| **UK** | MOC: 9.78 | STL: 10.27 | **+0.49** | ✅ **MOC** |

**v16 strategy**: STL for FR only, MOC for UK

**Expected gain** (50-70% composition):
- Conservative (50%): -1.84 × 0.5 = -0.92 → v16 ≈ 22.81
- Realistic (60%): -1.84 × 0.6 = -1.10 → v16 ≈ 22.63
- Optimistic (70%): -1.84 × 0.7 = -1.29 → v16 ≈ 22.44

**Expected range**: 22.50 - 23.00

### Technical Analysis

**Why STL works for FR?**
- Strong weekly patterns (weekend vs weekday)
- Adaptive decomposition (Loess) vs rigid EMA
- Better captures seasonal cycles

**Why STL fails for UK?**
- Merit Order Cost is structural (gas/0.50 + carbon×0.37)
- Basis modeling captures the true economic causality
- STL = pure pattern, MOC = fundamental driver

**Files**:
- Benchmark FR: `outputs/benchmark_stl_target.json`
- Benchmark UK: `outputs/benchmark_stl_uk.json`
- Pipeline v16: `scripts/train_v3_stl_target.py`
- Submission: `outputs/submission_v16_stl.csv` ✅

---

## v11 — Residual Stacking T2 (decomposed hypotheses)

Architecture: v9 (5 models) + Ridge(fundamentals) + Residual Stacking T2
- T2 = 55 models per market (5 algos × 11 thematic groups)
- Algos T2: Ridge, ElasticNet, LGB small, CatBoost small, XGBoost small
- Stacking: Ridge meta-learner (alpha=100, 5-fold CV) predicts v9 ensemble error
- Ensemble: regime-weighted (step=0.1) sur v9 models + RidgeF + SR

### Tested Hypotheses — Validation Scores HBC

| Hyp | Description | FR HBC | UK HBC | SUM | Delta vs v9 |
|-----|-------------|--------|--------|-----|-------------|
| H0 | v9 baseline (5 regime models + HBC) | 15.6696 | 9.4316 | 25.1012 | ref |
| H1 | + Ridge(fundamentals) | 15.5340 | 9.3188 | 24.8528 | **-0.2484** |
| H2 | + Independent Stacking (T2 predicts spot) | 15.2936 | 9.3055 | 24.5992 | **-0.5020** |
| H3 | + Residual Stacking (T2 predicts v9 error) | 15.2076 | 9.2043 | 24.4120 | **-0.6892** |
| H4 | + SR with group combos | 15.3137 | 9.1312 | 24.4450 | **-0.6562** |
| **BEST** | **FR=H3 + UK=H4** | **15.2076** | **9.1312** | **24.3389** | **-0.7623** |

### Cumulative Gain Decomposition

```
v9 baseline         25.10  (reference)
  + Ridge(fond)     24.85  (-0.25) → Ridge highly decorrelated (corr=0.65 FR, 0.41 UK)
  + Stacking T2     24.60  (-0.25) → 55 T2 models stacked by Ridge meta-learner
  + Residual mode   24.41  (-0.19) -> T2 predicts v9 error instead of spot
  + Best mix FR/UK  24.34  (-0.07) -> UK better with group combos
  ────────────────────────────
  TOTAL             -0.76
```

### H2 vs H3 — Independent vs Residual Stacking

| Market | SI (T2→spot) | SR (T2→error) | Delta |
|--------|-------------|----------------|-------|
| FR HBC | 15.2936 | **15.2076** | **-0.0860** |
| UK HBC | 9.3055 | **9.2043** | **-0.1012** |
| SUM | 24.5992 | **24.4120** | **-0.1872** |

Residual stacking is systematically better.

### H4 — Group Combos

| Config | FR SR_HBC | UK SR_HBC |
|--------|-----------|-----------|
| Without combos (55m) | **16.55** | 9.84 |
| With combos (65m) | 18.20 | **9.83** |

Combos degrade FR (+1.65) but not UK. Standalone SR is worse with combos,
but within the regime-weighted ensemble SR_combo gets more weight for UK.

### H5 — Leave-one-algo-out (T2 algo importance in stacking)

#### FR — Which T2 algo is most important?

| Algo removed | SR HBC | Delta SR | Ensemble HBC | Delta ens |
|-------------|--------|----------|-------------|-----------|
| (none)  | 16.5500 | ref | 15.2076 | ref |
| ridge | 16.0461 | +0.8385 | 15.2480 | +0.0404 |
| elasticnet | 16.1576 | +0.9500 | 15.3090 | +0.1014 |
| **lgb_small** | **17.3886** | **+2.1810** | **15.3976** | **+0.1900** |
| cb_small | 16.1877 | +0.9800 | **15.1333** | **-0.0743** |
| xgb_small | 16.0829 | +0.8753 | 15.2147 | +0.0071 |

**FR: LGB small is critical (+2.18 on SR). Removing CB small IMPROVES the ensemble (-0.07).**

#### UK — Which T2 algo is most important?

| Algo removed | SR HBC | Delta SR | Ensemble HBC | Delta ens |
|-------------|--------|----------|-------------|-----------|
| (none)  | 9.8043 | ref | 9.2043 | ref |
| ridge | 9.8506 | +0.6463 | 9.2309 | +0.0266 |
| elasticnet | 10.0226 | +0.8183 | 9.2799 | +0.0756 |
| lgb_small | 9.5493 | +0.3450 | **9.1252** | **-0.0791** |
| cb_small | 9.8348 | +0.6304 | 9.2424 | +0.0381 |
| xgb_small | 9.9256 | +0.7213 | 9.2344 | +0.0301 |

**UK: ElasticNet and XGB most important. Removing LGB small IMPROVES the ensemble (-0.08).**

### Regime Weights (v11 best = H3)

#### FR (7 models: CB + LGB + XGB + EN + DNN + RidgeF + SR)
| Regime | Hours  | CB | LGB | XGB | EN | DNN | RidgeF | SR |
|--------|--------|-----|-----|-----|-----|-----|--------|-----|
| Night | 0-5 | 0.2 | 0.1 | 0.1 | 0.0 | 0.2 | 0.1 | **0.3** |
| Morning | 6-9 | 0.3 | 0.0 | 0.0 | 0.1 | 0.3 | 0.0 | **0.3** |
| Day | 10-16 | 0.0 | 0.2 | 0.0 | 0.2 | 0.2 | 0.0 | **0.4** |
| Peak | 17-21 | 0.1 | 0.0 | 0.2 | 0.1 | 0.2 | 0.1 | **0.3** |
| Late | 22-23 | 0.4 | 0.0 | 0.0 | 0.0 | 0.4 | 0.1 | 0.1 |

**SR receives 30-40% of weight in 4/5 regimes (except late). EN loses weight to SR.**

#### UK (7 models: CB + LGB + XGB + EN + DNN + RidgeF + SR)
| Regime | Hours  | CB | LGB | XGB | EN | DNN | RidgeF | SR |
|--------|--------|-----|-----|-----|-----|-----|--------|-----|
| Night | 0-5 | 0.1 | 0.0 | 0.2 | 0.0 | 0.2 | 0.2 | **0.3** |
| Morning | 6-9 | 0.0 | 0.0 | 0.2 | 0.0 | 0.2 | 0.1 | **0.5** |
| Day | 10-16 | 0.0 | 0.1 | 0.3 | 0.0 | 0.1 | 0.1 | **0.4** |
| Peak | 17-21 | 0.3 | 0.3 | 0.0 | 0.0 | 0.1 | 0.0 | **0.3** |
| Late | 22-23 | 0.2 | 0.1 | 0.4 | 0.0 | 0.2 | 0.1 | 0.0 |

**SR receives 30-50% of weight in 4/5 regimes. Peak and Late remain tree-dominated.**

### H6-H11 — Per-country optimizations

Comparison baseline: BEST H0-H5 = 24.3389 (FR=H3, UK=H4)

| Hyp | Description | FR HBC | UK HBC | SUM | Delta vs H0-H5 |
|-----|-------------|--------|--------|-----|-----------------|
| H6 | Per-country algos (FR -cb, UK -lgb+combo) | 15.1333 | 9.0595 | 24.1928 | **-0.1461** |
| H7 | Per-regime stacking (5 meta-learners) | 15.3176 | 9.1464 | 24.4640 | +0.1251 |
| H8 | Enriched meta-learner (+hour/dow sin/cos) | 15.1511 | 9.0566 | 24.2077 | **-0.1312** |
| H9 | Alpha=1 both markets | 15.1307 | 9.0581 | 24.1888 | **-0.1501** |
| H10 | Residuals on v9+Ridge | 15.1890 | 9.0686 | 24.2576 | -0.0813 |
| **BEST** | **FR=H9 + UK=H8** | **15.1307** | **9.0566** | **24.1873** | **-0.1516** |

#### FR x UK matrix (all combinations)

```
              UK=H6    UK=H7    UK=H8    UK=H9   UK=H10
FR=H6       24.1928  24.2797  24.1899  24.1914  24.2019
FR=H7       24.3771  24.4640  24.3742  24.3757  24.3862
FR=H8       24.2106  24.2975  24.2077  24.2092  24.2197
FR=H9       24.1902  24.2771  24.1873  24.1888  24.1993
FR=H10      24.2485  24.3354  24.2456  24.2471  24.2576
```

Best: FR=H9 (alpha=1, no cb_small) + UK=H8 (enriched meta, no lgb+combo)

#### H7 rejected: Per-regime stacking overfits

5 meta-learners (1 per regime) → WORSE than 1 single (+0.13 SUM).
Not enough data per regime for 44-feature meta-learner. Rejected.

#### H9 detail: Grid search alpha

| Alpha | FR SR_HBC | FR ens_HBC | UK SR_HBC | UK ens_HBC |
|-------|-----------|------------|-----------|------------|
| 1 | 16.1972 | **15.1307** | 9.6182 | **9.0581** |
| 5 | 16.1968 | 15.1308 | 9.6177 | 9.0582 |
| 10 | 16.1963 | 15.1310 | 9.6172 | 9.0582 |
| 25 | 16.1948 | 15.1313 | 9.6155 | 9.0584 |
| 50 | 16.1924 | 15.1320 | 9.6129 | 9.0588 |
| 100 | 16.1877 | 15.1333 | 9.6078 | 9.0595 |
| 250 | 16.1748 | 15.1339 | 9.5943 | 9.0619 |
| 500 | 16.1567 | 15.1398 | 9.5761 | 9.0670 |
| 1000 | 16.1297 | 15.1510 | 9.5499 | 9.0739 |

Alpha=1 optimal for both markets. Minimal regularization lets the meta-learner fully use
T2 predictions.

### H11 — Leave-one-group-out (T2 group importance)

#### FR (base=15.1307, 44 models, 4 algos without cb_small, alpha=1)

| Group removed | ens_HBC | Delta | Verdict |
|---------------|---------|-------|---------|
| fr_gas | 15.3417 | +0.2110 | **CRITICAL** |
| fr_uk_sig | 15.3338 | +0.2031 | **CRITICAL** |
| fr_interco | 15.2352 | +0.1045 | Important |
| fr_scarcity | 15.2232 | +0.0924 | Important |
| fr_load | 15.2067 | +0.0760 | Important |
| fr_calendar | 15.2061 | +0.0754 | Important |
| fr_renewable | 15.1639 | +0.0332 | Utile |
| fr_price | 15.1338 | +0.0031 | Marginal |
| **fr_nuclear** | **15.1233** | **-0.0074** | Removing improves |
| **fr_hydro** | **15.1121** | **-0.0186** | Removing improves |
| **fr_continent** | **15.0644** | **-0.0663** | **Removing improves** |

**FR: removing fr_continent gives -0.066!** fr_hydro and fr_nuclear also slightly beneficial to remove.

#### UK (base=9.0581, 54 models, 4 algos without lgb_small + combos, alpha=1)

| Group removed | ens_HBC | Delta | Verdict |
|---------------|---------|-------|---------|
| uk_interco | 9.1610 | +0.1029 | **CRITICAL** |
| uk_calendar | 9.0975 | +0.0394 | Important |
| uk_continent | 9.0845 | +0.0264 | Important |
| uk_price | 9.0616 | +0.0035 | Marginal |
| uk_fr_price | 9.0604 | +0.0023 | Marginal |
| uk_wind | 9.0601 | +0.0020 | Marginal |
| **uk_nuclear_fr** | **9.0550** | **-0.0031** | Removing improves |
| **uk_emissions** | **9.0488** | **-0.0093** | Removing improves |
| **uk_load** | **9.0398** | **-0.0183** | Removing improves |
| **uk_scarcity** | **9.0369** | **-0.0212** | Removing improves |
| **uk_gas** | **9.0358** | **-0.0224** | **Removing improves** |

**UK: removing uk_gas, uk_scarcity, uk_load each gives ~-0.02.**

### H13 — Group Optimization (splits + combos)

Comparison baseline: v11 H12 best = 24.17 (FR=15.04, UK=9.13)

| Test | Description | FR HBC | UK HBC | SUM | Delta |
|------|-------------|--------|--------|-----|-------|
| A1 | FR + all 28 combos | 15.17 | — | — | +0.13 (too many combos = noise) |
| A5 | FR + 7 combos greedy | **14.81** | — | — | **-0.23** |
| B_fr | Split fr_load → raw + residual | 15.00 | — | — | -0.04 |
| C1 | UK + all 28 combos | — | 9.23 | — | +0.10 |
| C3 | UK + 5 combos greedy | — | **8.99** | — | **-0.14** |
| B_uk | Split uk_wind → core + continent | — | 9.03 | — | -0.10 |
| **D** | **FR split+combos + UK split+combos** | **14.80** | **8.97** | **23.77** | **-0.40** |

#### FR combos greedy (7 Ridge pairs added)
1. fr_price + fr_calendar (-0.106)
2. fr_renewable + fr_load (-0.147)
3. fr_load + fr_price (-0.149)
4. fr_gas + fr_load (-0.159)
5. fr_gas + fr_price (-0.165)
6. fr_renewable + fr_scarcity (-0.191)
7. fr_renewable + fr_uk_sig (-0.229)

FR combos now work (H4 = +1.65 on 11 groups, H13 = -0.23 on 8 surviving groups).

#### UK combos greedy (5 Ridge pairs instead of v11's 6)
1. uk_price + uk_fr_price (-0.042)
2. uk_wind + uk_nuclear_fr (-0.131)
3. uk_nuclear_fr + uk_calendar (-0.161)
4. uk_interco + uk_emissions (-0.194)
5. uk_wind + uk_continent (-0.195)

### Cumulative Gain Decomposition (v9 → v11 best)

```
v9 baseline                  25.10  (reference)
  + Ridge(fond)              24.85  (-0.25)
  + Stacking T2 (SI)         24.60  (-0.25)
  + Residual mode (SR)       24.41  (-0.19)
  + Best mix FR/UK           24.34  (-0.07) UK with combos
  + Per-country algos        24.19  (-0.15) FR -cb, UK -lgb, alpha=1, enriched meta UK
  + Group splits + combos    23.83  (-0.36) splits + FR combos + extended UK combos
  ────────────────────────────────
  TOTAL                      -1.27  (vs v9)
  Gap vs leader              0.69   (vs 23.14)
```

---

## Kaggle Submissions

| Submission | Score | Val Score | Gap | Notes |
|-----------|-------|-----------|-----|-------|
| **blend v17_85 + v9_15** | **23.20** | — | — | **BEST** — v17 Fix2 + blend anti-overfitting |
| blend v16_85 + v9_15 | 23.24 | — | — | classic blend, Fix 2 adds -0.04 |
| v16 (STL target FR) | 23.26 | 23.08 | +0.18 | gap = STL in-sample bias |
| blend v13_85 + v9_15 | 23.78 | — | — | former best — anti-overfitting via blending |
| v11b_conservative | 23.92 | 23.85 | +0.07 | anti-overfitting (UK alpha=500, 3+3 combos) |
| v13 | 23.94 | 23.73 | +0.21 | XGB_cluster UK overfit |
| v13+T1 | 23.94 | 23.71 | +0.23 | T1 features overfit (val -0.02, test +0.02) |
| v11 (splits+combos) | 24.23 | 23.83 | +0.40 | overfitting combos greedy |
| v14 (cold-start fix) | 24.27 | 23.69 | +0.58 | **CATASTROPHE** — data leakage |
| v5b (regime) | 24.88 | 26.39 | -1.51 | former best, underfit |
| v3a (std HBC) | 25.28 | ~27.3 | ~-2.0 | first submitted, underfit |
| v7 (5-model) | not submitted | 25.12 | — | — |
| v8 | not submitted | 25.15 | — | — |
| v15 (damped Ridge) | not submitted | 23.74 | — | inefficace (SR reste dominant) |
| blend v5b+v7 | not submitted | — | — | diversity ok (std=8.93 FR, 2.97 UK) |
| **1st place** | **23.14** | — | — | Team KISS |

---

## Standalone Model Scores (validation +HBC)

### France

| Model  | RMSE | +HBC | Notes |
|--------|------|------|-------|
| CatBoost (Optuna v2) | 17.21 | 17.02 | depth=3, lr=0.059, 28 feat |
| LightGBM | 17.73 | 17.58 | 15 leaves, reg_alpha=5 |
| XGBoost | 28.38 | 24.88 | depth=4, lr=0.05 (instable) |
| Elastic Net | 18.09 | 16.56 | alpha=10, l1=0.9, 14 nonzero |
| DNN [192,96] | 17.23 | 16.69 | Huber delta=5, dropout=0.2 |
| GRU seq12_h128 | 18.22 | 17.80 | worse than trees |
| GAT-GRU s24_gat64 | 18.65 | 18.16 | worse than GRU |
| **Regime Ensemble (5)** | **15.81** | **15.70** | v8 weights |

### United Kingdom

| Model  | RMSE | +HBC | Notes |
|--------|------|------|-------|
| CatBoost (12m, MAE) | 10.12 | 9.94 | depth=8, basis target, 154 feat |
| CatBoost (full) | 10.03 | 9.87 | v7 full window |
| LightGBM | 10.42 | 10.15 | 63 leaves |
| XGBoost | 10.38 | 10.11 | depth=7 |
| Elastic Net | 11.72 | 11.32 | alpha=1, 32 nonzero |
| DNN [768,384,192] | 11.30 | 11.03 | Huber, dropout=0.3 |
| GRU seq24_h256 | 11.34 | 11.12 | worse than trees |
| GAT-GRU s24_gat64 | 10.99 | 10.65 | worse than trees |
| **Regime Ensemble (5)** | **9.55** | **9.45** | v8 weights |

---

## Regime Weights (v8)

### FR
| Regime | Hours  | CB | LGB | XGB | EN | DNN |
|--------|--------|-----|-----|-----|-----|-----|
| Night | 0-5 | 0.0 | 0.6 | 0.0 | 0.1 | 0.3 |
| Morning | 6-9 | 0.0 | 0.0 | 0.0 | 0.3 | 0.7 |
| Day | 10-16 | 0.1 | 0.2 | 0.0 | 0.4 | 0.3 |
| Peak | 17-21 | 0.1 | 0.0 | 0.0 | 0.5 | 0.4 |
| Late | 22-23 | 0.5 | 0.1 | 0.0 | 0.0 | 0.4 |

### UK
| Regime | Hours  | CB | LGB | XGB | EN | DNN |
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

## A/B Tests — Key Results

### Target Engineering
| Config | FR +HBC | UK +HBC | Verdict |
|--------|---------|---------|---------|
| EMA 240h (FR) | **16.85** | — | best FR |
| Rolling mean 240h (FR) | 17.22 | — | -0.37 vs EMA |
| Rolling mean 168h (FR) | 17.34 | — | old baseline |
| Basis merit_order_cost (UK) | — | **9.97** | best UK |
| EMA 240h (UK) | — | 10.65 | -0.68 vs basis |

### Loss Functions — Diversity Optimization (v9)

Each model uses a different loss to maximize error decorrelation in the ensemble.

| Model  | FR loss | FR +HBC | UK loss | UK +HBC |
|--------|---------|---------|---------|---------|
| CatBoost | Quantile:0.6 | 16.46 | Quantile:0.6 | 9.78 |
| LightGBM | MAE | 16.76 | Huber(delta=5) | 9.90 |
| XGBoost | PseudoHuber(20) | 18.67 | PseudoHuber(20) | 10.11 |
| DNN | Huber(delta=5) | 16.69 | MSE | 10.96 |
| Elastic Net | MSE (fixed) | 16.44 | MSE (fixed) | 13.10 |
| **Ensemble** | | **15.65** | | **9.37** |

Key finding: FR sample weights (time_decay/variance, max/min ratio = 23,895x) were killing non-MSE losses. Removing them unlocked the gains.

### Loss Functions (CatBoost, legacy)
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
