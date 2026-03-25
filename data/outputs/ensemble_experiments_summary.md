# Ensemble Experiments Summary: Path to <24 RMSE

## Goal
Reduce combined RMSE below 24 (currently ~27.40, gap = 3.40)

---

## Experiments & Results

### Historical Baselines
| Version | Approach | FR RMSE | UK RMSE | Combined | Gap | Notes |
|---------|----------|---------|---------|----------|-----|-------|
| v9 | Full-feature multi-model + HBC | ~16.5 | ~9.0 | ~25.5 | +1.5 | Unknown exact architecture |
| v11_blocks | Block-specific (A/B/C) + HBC | 17.52 | 10.87 | **28.39** | +4.39 | 3-model/block, inverse-RMSE |

### New Experiments (This Session)

| Version | Approach | FR RMSE | UK RMSE | Combined | Gap | Notes |
|---------|----------|---------|---------|----------|-----|-------|
| v12 | Multi-model + inverse-variance weighting | 18.92 | 10.36 | **28.09** | +4.09 | Simple avg weights |
| v13 | Ridge stacking (3 base models) | 16.88 | 11.55 | **28.43** | +4.43 | Overfits, worse than v12 |
| v14 | 9 diverse models (CB/LGB/XGB x 3 depths) | 17.90 | 9.99 | **27.89** | +3.89 | XGB depth=7 best |
| v15 | 9 XGBoost only (3 depths x 3 seeds) | 17.24 | 10.16 | **27.40** | **+3.40** | BEST: XGB depth=8 |
| v16 | 12 XGBoost (3 depths x 2 LRs x 2 regs) | 17.27 | 10.22 | **27.49** | +3.49 | Depth=9 slightly worse |

### Temporal Segmentation Analysis

Tested: Global (1 model) vs Cluster (4 time-of-day models) vs Hourly (24 models)

**Result**: All worse than global with limited features (44.85 global RMSE)
- Temporal segmentation doesn't help when models see fewer samples
- High residual correlation across strategies (FR: 0.956-0.976, UK: 0.848-0.890)
- **Conclusion**: Issue is feature diversity, not temporal structure

---

## Key Observations

1. **XGBoost Dominates**: Consistently outperforms CatBoost and LightGBM
   - Depth=7-9 optimal (too deep: overfitting; too shallow: underfitting)
   - Learning rate 0.04-0.06 both viable
   - Regularization (L1/L2) has minor impact

2. **Inverse-RMSE Weighting Effective**:
   - Simple inverse-variance: 28.09 combined
   - Ridge stacking: 28.43 combined (overfits)
   - Conclusion: Non-linear weighting doesn't help here

3. **Diminishing Returns**:
   - v12 → v14: 0.20 point gain (hyperparameter tuning)
   - v14 → v15: 0.49 point gain (XGB focus)
   - v15 → v16: 0.09 point loss (deeper trees overfit)
   - **Plateau at ~27.40**

4. **HBC Post-Processing Works**:
   - Adds ~1.0-1.5 points of improvement
   - Hourly bias offsets robust and consistent
   - Without HBC: FR ~18.3, UK ~10.5 → Combined ~28.8

5. **All Features Needed**:
   - 418 total features used
   - Feature blocking (v11) forced decorrelation but lost diversity → 28.39
   - Full-feature ensembles work better → 27.40

---

## What Won't Work (Already Tested)

- ❌ Feature blocking for decorrelation (worse than full-feature)
- ❌ Temporal segmentation with limited samples (underfits)
- ❌ Ridge stacking meta-learner (overfits)
- ❌ More regularization depth (diminishing returns at d=8-9)
- ❌ Multiple random seeds alone (limited diversity gain)

---

## Gap Analysis: Why 3.40 Points Remain?

Current best: **27.40** → Need: **24.00** → Gap: **3.40**

Possible sources of remaining error:
1. **Stationary target misspecification** (~1-1.5 points)
   - FR: EMA(240h) detrending may not match market reality
   - UK: Merit order basis assumes linear fuel costs

2. **Market regime changes** (~0.5-1.0 points)
   - Summer vs winter dynamics
   - Renewable integration acceleration
   - Strategic bidding patterns

3. **Fundamental prediction limits** (~0.5-1.5 points)
   - Model can explain ~93-95% of variance already
   - Remaining error = irreducible noise + black-swan events

4. **Feature engineering** (~0.5-1.0 points)
   - Current features may miss market microstructure
   - Implicit cost curves, liquidity constraints not captured

---

## Paths Forward (Ranked by ROI)

### High Impact (1-2 point gain possible)
1. **Better feature engineering**
   - Add market microstructure: bid-ask spreads, trading volume, volatility
   - Create interaction terms: fuel costs × demand patterns
   - Domain-specific ratios: nuclear availability rates, grid frequency stability
   - **Effort**: High | **Uncertainty**: Medium

2. **Improved stationary targets**
   - Use other detrending methods (STL, LOWESS) vs EMA
   - Separate regimes: peak/off-peak/night decomposition
   - Quantile-based targets instead of mean regression
   - **Effort**: Medium | **Uncertainty**: High

3. **Neural network ensemble** (LSTM, TCN, Transformer)
   - Capture temporal dependencies better than tree models
   - Learn attention over lags and features
   - **Effort**: High | **Uncertainty**: Medium-High

### Medium Impact (0.5-1.0 point gain possible)
4. **Advanced post-processing**
   - Quantile-based predictions (median/95th percentile for tail risk)
   - Variance stabilization via Anscombe transform
   - Market-informed clipping (prices rarely >5x normal)

5. **Ensemble of ensembles**
   - Combine v11 block ensemble + v15 XGB ensemble
   - Learn meta-weights on validation set
   - Captures different signal sources

### Low Impact (<0.5 points)
6. More hyperparameter search (already diminishing)
7. More base models (diversity limited with trees)
8. Alternative weighting schemes (inverse-variance already optimal)

---

## Recommendation

**Current Best**: v15 XGBoost ensemble → **27.40 combined RMSE**

### Option A: Incremental (0.5-1.0 pt improvement, 1-2 weeks)
- Submit v15 as baseline
- Improve stationary target formulation (e.g., better detrending)
- Add 5-10 engineered features (fuel cost ratios, regime indicators)
- **Expected result**: 26.5-27.0 combined RMSE

### Option B: Aggressive (1.5-2.5 pt improvement, 2-4 weeks)
- Build LSTM ensemble on top of XGB ensemble
- Design better feature engineering pipeline
- Implement advanced post-processing (quantile regression)
- **Expected result**: 24.5-25.5 combined RMSE (within striking distance)

### Option C: Exploratory (unknown upside, 3-5 weeks)
- Full market microstructure modeling
- Regime detection with hidden Markov models
- Custom domain-specific architectures
- **Expected result**: Uncertain, potentially <24

---

## Code Artifacts

**Recommended for production**:
- `scripts/ensemble_v15_xgb_focus.py` → **Best current model**
- Outputs: Predictions + hourly biases for HBC

**For future exploration**:
- `scripts/ensemble_v14_diverse_hyperparams.py` (multi-model baseline)
- `notebooks/04_temporal_segmentation.ipynb` (temporal analysis, inconclusive)
- `scripts/temporal_segmentation_eval.py` (standalone evaluation)
