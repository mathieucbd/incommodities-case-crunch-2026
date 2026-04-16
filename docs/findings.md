# Findings & Decisions

---

## Key Decisions

### 1. Target Engineering

**FR: `y = spot - STL_trend(spot_la, 168h)`** ✅ v16 (replaces EMA 240h)
- **STL** (Seasonal-Trend decomposition using Loess) beats EMA 240h by **-1.84 RMSE +HBC**
- Period 168h = 1 week, seasonal=13 (odd integer > period/2)
- Decomposes spot_la = Trend + Seasonal + Residual
- **Advantages vs EMA**:
  - Better captures weekly patterns (weekend vs weekday)
  - Adaptive decomposition (Loess) vs rigid EMA
  - Trend + Seasonal separated → more precise prediction
- **Standalone benchmark** (CatBoost Quantile:0.6, holdout Feb-Jun 2024):
  - EMA 240h: 17.92 RMSE +HBC
  - STL trend: 16.08 RMSE +HBC
  - Delta: **-1.84** (>10% error reduction)
- **Expected v16 gain** (60% composition): ~-1.10 → score ≈ 22.60

**FR legacy (v9-v13): `y = spot - EMA(spot_la, 240h)`**
- EMA 240h beats rolling mean 168h by -0.37 RMSE
- EMA adapts to regime changes (gas crisis 2022 → normalization 2024)
- EMA is computed on lagged spot (not current spot) → no leakage

**UK: `y = spot - merit_order_cost`** ✅ Optimal (STL tested and rejected)
- merit_order_cost = gas/efficiency + emission × 0.37
- Beats EMA 240h by -0.68 RMSE (large)
- **Beats STL trend by +0.49** → MOC remains best for UK
- Reason: UK is a gas-dominated market, MOC captures ~70% of the price signal
- The residual (spot - MOC) is more stationary than raw spot
- **STL fails for UK** because MOC = fundamental driver (causality), STL = temporal pattern

### 2. Tree Depth

**FR: depth=3** (shallow)
- 83% of importance in Optuna optimization (300 trials)
- FR has few features (28) → shallow depth avoids overfitting
- lr=0.059 (fast), l2_reg=4.4 (light)

**UK: depth=8** (deep)
- UK has 150+ features with many interactions
- The basis target (spot - MOC) is simpler → model can go deeper
- MAE loss instead of RMSE (robust to UK's heavy tails: range [-205, +1444])

### 3. 5-Model Ensemble + Regime Weights + Loss Diversity (v9)

- **CatBoost**: Quantile:0.6 (FR+UK) — best standalone, asymmetric bias
- **LightGBM**: MAE (FR) / Huber_5 (UK) — robust to outliers
- **XGBoost**: PseudoHuber_20 (FR+UK) — MSE/MAE compromise
- **Elastic Net**: MSE (fixed) — linear model, correlation 0.60-0.70 with trees
- **DNN**: Huber_5 (FR) / MSE (UK) — 356 features, correlation 0.70-0.80 with trees

Each model uses a different loss → decorrelated errors → better ensemble.
FR sample weights removed (time_decay/variance ratio 23,895x was killing non-MSE losses).

Weights vary by hour:
- **Morning (6-9h)**: EN+DNN dominate (CB at 0.2) — trees miss morning ramps
- **Night**: CB+DNN dominate FR, CB+XGB+EN UK
- **Peak**: EN+DNN for FR, CB+LGB for UK

### 4. Residual Stacking T2 (v11)

**Architecture**: v9 ensemble + Ridge(fundamentals) + Residual Stacking T2
- T2 = lightweight models (Ridge, ElasticNet, LGB small, XGB small) trained by thematic group
- The Ridge meta-learner predicts the **v9 ensemble error** (not spot directly)
- Residual mode beats independent mode by -0.19 SUM

**Gain decomposition (v9 → v11)**:
```
v9 baseline                  25.10
  + Ridge(fundamentals)      24.85  (-0.25)
  + Stacking T2              24.60  (-0.25)
  + Residual mode            24.41  (-0.19)
  + Per-country optim        24.19  (-0.22) FR -cb_small, UK -lgb_small, enriched meta UK
  + Group splits + combos    23.85  (-0.34)
  TOTAL                      -1.25
```

**Per-country optimizations**:
- FR: alpha=1, 4 algos (no cb_small), 9 groups (fr_load split into raw+residual)
- UK: alpha=500 (anti-overfitting), 4 algos (no lgb_small), 9 groups (uk_wind split into core+continent), enriched meta-learner (+hour/dow sin/cos)

**Anti-overfitting (v11 → v11b)**:
- v11 aggressive (7+5 combos): val=23.83, Kaggle=24.23 → gap +0.40
- v11b conservative (3+3 combos, UK alpha 100→500): val=23.85, Kaggle=23.92 → gap +0.07
- Reducing combos and increasing UK regularization divided the gap by 6

### 5. UK Temporal Segmentation (v13)

**Principle**: Train the same algos on hour subsets to create decorrelated predictions.

**Split**: Shifted 6h → [3-8], [9-14], [15-20], [21-2] (~3480 samples/cluster)

**Screening (H2)**: Only XGB benefits from segmentation (error corr=0.838, RMSE<105% global).
CB (0.915), LGB (0.902), EN (0.915) → too correlated with global.

**Winning integration (H3)**: XGB_cluster as 8th ensemble member for UK regime.
- 7 models → 8 models (CB+LGB+XGB+EN+DNN+RidgeF+SR+XGB_cluster)
- UK: 8.99 → 8.87 (-0.12)
- SUM: 23.85 → 23.73 (-0.12)

**Rejected**:
- H4 (cluster preds as SR features): +0.02 — Ridge alpha=500 dilutes the signal
- H5 (pre-blend w=0.5): -0.08 — inferior to H3
- H6 (24 hourly LGB as SR feature): -0.05 — marginal

**UK only** — FR has error correlations >0.95 between strategies, segmentation does not help.

### 6. HBC (Hourly Bias Correction)

- Corrects systematic bias per hour (e.g. overestimate h=18, underestimate h=7)
- Average gain: 0.2-0.5 RMSE
- 3 variants tested: standard (24 params), monthly (120 params), dampened
- Monthly HBC gains ~0.8 RMSE on val but risks overfitting to test

---

## What Worked

| Improvement | RMSE delta | Impact |
|-------------|-----------|--------|
| Basis target UK (spot - MOC) | -0.68 UK | major |
| EMA 240h FR (vs rolling 168h) | -0.37 FR | significant |
| DNN 5th model (v7) | -0.77 SUM | major |
| Elastic Net 4th model (v6) | -0.50 SUM | significant |
| XGBoost 3rd model (v5) | -0.48 SUM | significant |
| Regime weights (v5b) | -0.13 FR | moderate |
| Optuna v2 FR (depth=3, lr=0.059) | -0.86 FR | significant |
| MAE loss UK | -0.06 UK | moderate |
| Loss diversity + no FR weights (v9) | -0.18 SUM | significant |
| Residual Stacking T2 (v11) | -1.25 SUM | major |
| Anti-overfitting combos (v11b) | gap /6 | major (0.40→0.07) |
| **STL target FR (v16)** | **-0.65 val, -0.68 Kaggle** | **major** (23.94→23.26) |
| **Fix 2 coherent STL retrain (v17)** | **-0.04 Kaggle** | moderate (23.24→23.20 via blend) |
| **Blend 85% v17 + 15% v9** | **-0.06 Kaggle** | **best score** (23.26→23.20) |
| Blend 85% v13 + 15% v9 | -0.14 Kaggle | major (23.92→23.78, gap 0.64) |
| XGB cluster 8th member UK (v13) | -0.12 val, **+0.02 Kaggle** | overfit (gap 0.21 vs 0.07) |
| UK 12m window (CatBoost only) | -0.43 UK | significant |
| New features (rolling_336h etc.) | -0.40 SUM | significant (CB only) |
| Per-hour models | -0.30 SUM | moderate (CB+LGB only) |

**Note:** UK 12m, new features, and per-hour models show gains in isolation (CB/LGB) but **do not compose** in the 5-model pipeline with regime weights. The regime ensemble already absorbs these gains.

---

## What Didn't Work

| Attempt | Result | Reason |
|---------|--------|--------|
| GRU (12 configs) | +0.8 FR, +0.4 UK vs CB | Trees > sequences on tabular features |
| GAT-GRU (9 configs) | +1.1 FR, +0.8 UK vs CB | Graph attention doesn't help here |
| GRU as 6th model | 0.00 SUM | Error correlation 0.88 with DNN (no diversity) |
| Independent stacking (Ridge, v11 H2) | +0.2 vs residual | Residual mode systematically better |
| arcsinh transform | +0.3-0.5 | Systematically degrades all models |
| Monthly x Hour HBC (V11) | 0.00 OOF | Massive in-sample overfit (-0.60), useless OOF. T2 already absorbs the bias. |
| Same-Hour Lookback features | +0.27 to +0.73 | Deep models (UK depth 8) already natively capture hour×lag interaction. |
| k-NN Analog Days (physical memory) | +0.02 (neutral) | Retrieving prices from physically similar days (L1, K=3) adds no exploitable info beyond V11. |
| Custom Asymmetric Loss (spikes/troughs) | +0.40 to +3.00 | Forcing the model (via α·Huber) to chase spikes destroys the continuous baseline. GBDT is already optimal with symmetric loss. |
| IVW sample weights (v14, 4 formulas) | -0.02 SUM pipeline | CB FR -0.18 standalone but CB UK +0.41. XGB degrades (+0.12). Only LGB gains (-0.03). T2 stacking absorbs the gains. |
| T1 Features in full pipeline (v13+T1) | val -0.02, test +0.02 | Isolated gain -0.29 +HBC on 5-model ensemble, but in the full v13 pipeline the optimized hyperparameters already capture the signal via correlated features. Causes overfitting (gap +0.23 vs +0.21). |
| Cold-Start Fix (v14, concat train+test) | test +0.58 (CATASTROPHE) | build_features(concat(train, test)) for full history → massive DATA LEAKAGE. Rolling/shift features see future test data. Val 23.69 looked good but test 24.27 reveals the disaster. |
| Ridge Damping (v15, FR alpha 1→10, UK 500→1000) | val +0.01 (ineffective) | Goal: reduce SR dominance (weights 0.4-0.8). Result: SR stays dominant, val 23.74 vs 23.73. Increasing alpha dilutes the signal without reducing overfitting. |

---

## Structural Revelations (Edge)

### 1. Inverse Variance Weighting — Debunked (v14)

**Initial finding (Holdout + LOMO 4 folds)**: The `benchmark_v12_weighting.py` showed spectacular gains (-2.51 FR XGB, -0.35 UK XGB). BUT the benchmark was **flawed**:
- STD baselines never trained (comparison against zeros)
- Only 4 months, 1000 iter (v11b=15000), ES=30 (v11b=200)

**Rigorous test (v14, 5 hypotheses)**: Using exact v11b params (15000 iter, ES=200) on standard holdout:

| Algo | FR delta | UK delta | Verdict |
|------|----------|----------|---------|
| CB (Quantile:0.6) | **-0.18** | **+0.41** | FR OK, UK destroyed |
| LGB (MAE/Huber) | -0.03 | -0.03 | Marginal |
| XGB (PseudoHuber) | +0.12 | +0.11 | Degrades (contrary to benchmark!) |

**Why CB UK is destroyed**: Quantile:0.6 already has asymmetric bias. Adding IVW (which downweights extremes) creates a double bias → model early-stops at 305 iter instead of 698.

**Full pipeline (H4)**: SUM 23.85 → **23.83** (delta=-0.02). T2 stacking absorbs almost all the standalone gain.

**4 formulas tested**: F1_base (1/(1+vol/mean)), F2_clip, F3_exp, F4_rank. F1_base is best but all are marginal in the full pipeline.

**Conclusion**: IVW does not compose with the v11b pipeline. Standalone gains (~-0.18 CB FR) are absorbed by the regime ensemble + T2 stacking.
| Basis modeling FR (spot - MOC) | +4.5 FR | basis_shift = +40 EUR (gas crisis shifts the MOC) |
| Multi-window averaging | -0.05 SUM | Gain too small, full window sufficient |
| Huber loss (CatBoost only) | +0.10 FR | Standard RMSE is optimal for the RMSE metric |
| Same loss everywhere (Quant:0.6) | +0.09 SUM | Models too correlated → ensemble loses diversity |
| Per-hour features (v8 in pipeline) | +0.03 SUM | Absorbed by regime ensemble |
| Regime simplification (3 regimes) | +0.03 SUM | 5 regimes step=0.1 is optimal |
| Regime step=0.2 | +0.08 SUM | Finer granularity always better |
| UK alpha sweep (1→1000) | ±0.01 SUM | Alpha=300 best (-0.001), not significant |
| UK greedy 8 combos (val only) | -0.19 SUM | Strong val gain but test overfitting risk |
| Cluster preds as SR features (v13 H4) | +0.02 SUM | Ridge alpha=500 dilutes cluster signal |
| Pre-blend global+cluster (v13 H5) | -0.08 SUM | Inferior to H3 (direct 8th member) |
| 24 hourly LGB as SR feature (v13 H6) | -0.05 SUM | Marginal, standalone RMSE 22.67 (2x worse) |
| CB/LGB/EN cluster models (v13 H2) | neutral | Error corr >0.90 with global → no diversity |

---

## Domain Insights

### FR Market (SDAC / EPEX SPOT)
- **Nuclear-dominated**: 70% of generation, price often set by nuclear marginal cost
- **EMA as anchor**: FR price follows a slow trend (EMA 240h = 10 days)
- **River temperature**: >25°C = nuclear curtailment risk (feature `fr_river_temp_risk`)
- **Continental coupling**: FR price is capped by neighbour imports (DE, BE, CH, ES)
- **28 features sufficient**: aggressive feature selection (from 330 → 28) improves stability

### UK Market (N2EX / Nord Pool)
- **Gas-dominated**: merit order cost = gas/0.50 + emission × 0.37
- **Heavy tails**: range [-205, +1444], MAE loss more robust
- **6 interconnectors**: IFA1/2, ElecLink, NEMO, BritNed, Viking Link
- **150 features needed**: more features than FR due to more sources of variability
- **12m window optimal**: post-crisis regime (2023+) is more representative of test

### Cross-Market Dynamics
- `euro_scarcity_ratio`: top feature for both markets
- `continental_residual_load`: net European demand = price driver
- 14-day z-scores (`fr_residual_zscore_14d`) capture anomalies

---

## Anti-Overfitting Lessons

1. **More combos ≠ better**: 7+5 greedy combos val=-0.40, but Kaggle gap +0.40. 3+3 conservative combos: gap +0.07
2. **High alpha = safety**: UK alpha=500 (vs 100) costs -0.004 val but protects test
3. **The 4 levers tested (v12) add nothing more**:
   - Regime simplification → worse
   - Alpha sweep → marginal (±0.01)
   - FR combos → already optimal
   - UK greedy combos → -0.19 val but same overfitting risk as aggressive v11
4. **The val→Kaggle gap IS the quality metric**, not the absolute val score
5. **Adding ensemble members = more combos = more overfitting**: v13 adds XGB_cluster as 8th UK member → 8 models × 5 regimes × grid search step=0.1. Val gain (-0.12) doesn't transfer (Kaggle +0.02, gap 0.07→0.21)
6. **Standalone gains don't compose**: IVW gives -0.18 CB FR standalone, but -0.02 SUM in the full pipeline. T2 stacking + regime ensemble already absorb what IVW would contribute. Similarly, T1 features give -0.29 +HBC in isolation but only -0.02 in v13 full pipeline.
7. **Flawed benchmarks mislead**: The LOMO IVW benchmark showed -2.51 FR XGB because the STD baselines were never trained (zeros). Always validate baselines.
8. **Blending reduces overfitting**: Mixing v13 (overfit gap +0.21) with v9 (more stable, less complex) at ratio 85/15 reduces the gap and improves test by -0.14 (23.92→23.78). The simple model tempers the excesses of the complex model.
9. **Leakage via concat is catastrophic**: build_features(concat(train, test)) to "fix cold-start" seemed logical (val -0.04) but destroys test (+0.58). Rolling/shift features see the future. NEVER concat before feature engineering.
10. **STL target engineering = large FR gain, ineffective UK**: STL (168h seasonal decomposition) beats EMA 240h by -1.84 standalone FR, but loses to Merit Order Cost for UK (+0.49). Temporal patterns (STL) work when they capture the real price structure (FR weekly cycles). Fundamental anchors (MOC UK) win when they reflect economic causality (gas+carbon). Benchmark before integrating = crucial to avoid false leads.
11. **The STL gap = in-sample bias, not model overfitting**: v16 gap +0.18 (val 23.08 → test 23.26). Fix 1 (walk-forward STL, fit on train-only) degrades val by exactly +0.18 → the gap is 100% due to bidirectional STL using future validation data. The model itself does NOT overfit. Implication: the gap cannot be reduced by regularizing the model.
12. **Fix 2 coherent STL = small real test gain**: Using a single STL fit on concat(train,test) for retrain (instead of mixing 2 different fits) gives -0.04 on Kaggle (23.24→23.20 via blend). Small but free.
13. **v16/v17+v9 blending less effective than with v13**: blend v13+v9 85/15 improved by -0.14. blend v16+v9 85/15 only improves by -0.02. v16 is already too good → v9 (val 25.02) dilutes more than it helps. Blending has diminishing returns as the main model improves.
14. **Dual holdout recalibration > single-season validation**: Testing on 2 seasonal holdouts (spring Feb-Jun 2024 + winter Jul-Nov 2023) reveals that regime weights diverge radically. DNN disappears in winter (0.0-0.2), SR ultra-dominant (0.7-0.8). Submission attack_averaged (arithmetic average of both configs) wins -2.74 Kaggle vs best previous blend → 2nd place (20.0781). Lesson: single-season validation is biased; averaging multiple seasonal holdouts = better generalization.
15. **Residual Stacking meta-learning > early stopping**: The SR (Ridge meta-learner) learns the error patterns of base learners and corrects them dynamically. It recovers -0.19 RMSE (H2→H3) even when early stopping is aggressive (CB_FR 147 iter, LGB_FR 731). Residual meta-learning captures what base learners miss → outperforms pure regularization via early stopping.
16. **Fulldata reveals UK non-stationarity Q4 2024**: New 2022→2025 dataset (received 2026-03-26) exposes UK RMSE explosion 8.83 → 33.51 on Oct-Dec 2024 holdout. The true test Jul 2024-Feb 2025 gives UK=27-29 (out-of-distribution but less extreme). Train 2022-2024 does not contain these price regimes. Lesson: electricity is non-stationary; major distribution shifts post-training-period are possible.

---

## Unexplored Directions

1. **Conformalized predictions**: prediction intervals for Winkler scoring (if metric changes)
2. **Online learning**: incrementally retrain on the test set (if permitted)
3. **Strict temporal cross-validation**: expanding window with re-optimized weights
4. **Submission blending**: v11b + v7 + v9 (test correlation ~0.96-0.99, limited diversity)
5. **Deep feature engineering**: automatic interactions (PolynomialFeatures + selection)
6. **Transformer models**: attention-based for time series
7. **Quantile regression ensemble**: predict median instead of mean
8. **External data**: weather forecasts, electricity futures, capacity auctions
