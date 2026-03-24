# InCommodities Case Crunch 2026: AI Agent Master Blueprint

**Goal:** Turn European power market data into a competitive trading advantage by predicting Day-Ahead Market (DAM) spot prices for France (`fr_spot`) and the United Kingdom (`uk_spot`) in EUR/MWh.
**Scope:** Multi-output forecasting (2 targets) using 111 provided features.
**Official Metric:** Root Mean Squared Error (RMSE) averaged across both targets.

---

## ⚠️ FINALS FORMAT — CRITICAL CONSTRAINT (March 26, InCommodities HQ)

**The Kaggle leaderboard has zero influence on final standings.** Finals are run live.

### Live Inference Data Format
At the finals, the model receives:
- `x_validation`: **720 rows** (30 days × 24 hours) of features + targets leading up to the target day
- `y_validation`: **720 rows** of actual spot prices for those 30 days
- `x_test`: **24 rows** — only the target day's features
- **Output required:** 24 × (UK, FR) predictions for the target day

### Hard Rules Derived From This
1. **Max lag = 30 days (720 hours).** Any `.shift(n)` or `.rolling(n)` where `n > 720` will produce all-NaN in the finals. The current pipeline uses max 336h (14 days) → SAFE.
2. **Expanding windows BREAK in this format.** `df.expanding().max()` trained on 2 years of data returns a valid max. At inference with only 30 days of context, the same call returns a much lower (wrong) value. Affected functions in `src/features.py`:
   - `engineer_nuclear_shortfall()` — uses `expanding().max()` → **must be replaced with hardcoded constants at inference**
   - `engineer_regime_features()` — uses `expanding().max()` for `fr/uk_nuclear_avail_ratio` → **same issue**
3. **Model inference must be fast.** No long retraining loops at prediction time.
4. **The pipeline must accept the 30-day context window format** — train on full history, predict on 30-day rolling window.

### Hardcoded Max Nuclear Capacity (safe fallback for finals)
- FR: 61,400 MW (56-reactor fleet)
- UK: 5,800 MW

---

## AI Assistant Rules & Context
If you are an AI coding assistant reading this file, these are your absolute rules for this project:
1. **No External Data:** The competition strictly prohibits external datasets or APIs. Rely exclusively on feature engineering (e.g., temporal features, public holidays via `holidays` library, residual load).
2. **Handle Sparse Data:** Daily features (Gas and Emissions) are only populated at 00:00 CET and are extremely sparse (~3-5% coverage early on). Tree-based models that natively handle NaNs (like LightGBM/XGBoost) or strategic forward-filling are required.
3. **Loss Function Strategy:** While RMSE is the final leaderboard metric, power prices have heavy tails. Optimize training loops using robust loss functions like Huber or MAE to prevent outliers from destroying the model weights, then evaluate on RMSE.
4. **Beware Leaderboard Overfitting:** The public leaderboard is only 60% of the test data. Rely strictly on our own Chronological Walk-Forward Time-Series Cross-Validation. The Kaggle leaderboard score is irrelevant to finals ranking.
5. **Architecture:** Use a Functional approach for data pipelines (stateless transformations) and OOP exclusively for ML Models.
6. **Environment:** We use `uv` for package management. Do not generate `pip install` commands.
7. **Configuration-Driven:** Hardcoding data paths or model hyperparameters is strictly prohibited. Use `config.yaml`.
8. **Finals-Safe Features:** Every feature must be computable from ≤720 hours of context. Expanding windows must use hardcoded historical maxima (see above) rather than `expanding().max()` when running in inference mode.

---

## 1. Project Organization & Folder Structure

```text
incommodities-case-crunch/
├── literature/               # Foundational research papers the pipeline is based on
├── data/
│   ├── raw/                  # Unaltered Kaggle datasets (x_train.csv, y_train.csv, x_test.csv)
│   ├── processed/            # Cleaned data (sparse daily features ffilled, scaled tensors)
│   └── outputs/              # Final Kaggle submissions, catboost logs, and SHAP plots
├── notebooks/                # Jupyter notebooks for EDA and prototyping
├── src/                      
│   ├── constants.py          # Shared constants (target columns: fr_spot, uk_spot)
│   ├── data_ingestion.py     # Scripts to load Kaggle files and merge on 'id' / datetime_CET
│   ├── preprocessing.py      # Chronological split, StandardScaler (Train-fitted only)
│   ├── features.py           # 30-category, ~290-feature engineering pipeline (see docs/FEATURE_ENGINEERING.md)
│   ├── models/
│   │   ├── baselines.py      # Vectorized Naive model, LEAR (LassoLarsIC rolling window)
│   │   ├── tree_models.py    # Multi-output XGBoost, LightGBM, CatBoost (NaN-friendly)
│   │   ├── deep_learning.py  # Multivariate PyTorch DNN (Dual-target output: FR & UK)
│   │   ├── optimize.py       # Bayesian hyperparameter tuning (TPE)
│   │   └── ensembles.py      # Model blending/stacking for final Kaggle submission
│   ├── evaluation/
│   │   ├── metrics.py        # RMSE (Official Kaggle Metric), MAE, Huber (for robust training)
│   │   ├── probabilistic.py  # Pinball loss (optional for robust training experiments)
│   │   └── statistical.py    # Diebold-Mariano (DM) test
│   └── explainability.py     # SHAP value generators for tree models
├── pyproject.toml            # uv project metadata and dependencies
├── .gitignore                # Ignored files (catboost_info, data, config)
├── config.yaml               # Configuration controlling data paths and hyperparameters
├── PIPELINE.md               # End-to-end machine learning pipeline steps
└── AGENTS.md                 # This file (AI instructions)