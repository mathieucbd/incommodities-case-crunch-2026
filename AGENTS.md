# InCommodities Case Crunch 2026: AI Agent Master Blueprint

**Goal:** Turn European power market data into a competitive trading advantage by predicting Day-Ahead Market (DAM) spot prices for France (`fr_spot`) and the United Kingdom (`uk_spot`) in EUR/MWh.
**Scope:** Multi-output forecasting (2 targets) using 111 provided features.
**Official Metric:** Root Mean Squared Error (RMSE) averaged across both targets.

---

## AI Assistant Rules & Context
If you are an AI coding assistant reading this file, these are your absolute rules for this project:
1. **No External Data:** The competition strictly prohibits external datasets or APIs. Rely exclusively on feature engineering (e.g., temporal features, public holidays via `holidays` library, residual load).
2. **Handle Sparse Data:** Daily features (Gas and Emissions) are only populated at 00:00 CET and are extremely sparse (~3-5% coverage early on). Tree-based models that natively handle NaNs (like LightGBM/XGBoost) or strategic forward-filling are required.
3. **Loss Function Strategy:** While RMSE is the final leaderboard metric, power prices have heavy tails. Optimize training loops using robust loss functions like Huber or MAE to prevent outliers from destroying the model weights, then evaluate on RMSE.
4. **Beware Leaderboard Overfitting:** The public leaderboard is only 60% of the test data. Rely strictly on our own Chronological Walk-Forward Time-Series Cross-Validation.
5. **Architecture:** Use a Functional approach for data pipelines (stateless transformations) and OOP exclusively for ML Models. 
6. **Environment:** We use `uv` for package management. Do not generate `pip install` commands.
7. **Configuration-Driven:** Hardcoding data paths or model hyperparameters is strictly prohibited. Use `config.yaml`.

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
│   ├── features.py           # Residual load math, Interconnector ATC/NTC, sparse daily ffill
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