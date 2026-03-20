# InCommodities Case Crunch: AI Agent Master Blueprint

**Goal:** Turn European power market data into a competitive trading advantage by predicting Day-Ahead Market (DAM) spot prices and their prediction intervals.
**Scope:** Multi-zone forecasting across 12 zones (AT, BE, CH, CZ, DE, DK1, DK2, FR, NL, NO2, PL, SE4).
**Guiding Principle:** Maximize sharpness subject to reliability.

---

## AI Assistant Rules & Context
If you are an AI coding assistant reading this file, these are your absolute rules for this project:
1. **No standard MAPE:** Never use Mean Absolute Percentage Error. Power prices drop to zero or go negative, which breaks the denominator. Use sMAPE or rMAE instead.
2. **No CWC metric:** Do not use the Coverage Width-based Criterion for probabilistic evaluation. It is an improper scoring rule. Use Pinball Loss and Winkler Score.
3. **Architecture:** Use a Functional approach for data pipelines (stateless transformations) and OOP exclusively for ML Models. 
4. **Environment:** We use `uv` for package management. Do not generate `pip install` commands; assume dependencies are managed via `pyproject.toml`.
5. **Configuration-Driven:** Hardcoding data paths or model hyperparameters is strictly prohibited. Use `config.yaml`.

---

## 1. Project Organization & Folder Structure

```text
incommodities-case-crunch/
├── literature/               # Foundational research papers the pipeline is based on
├── data/
│   ├── raw/                  # Unaltered CSVs/Parquets for the European zones
│   ├── processed/            # Cleaned data (spikes handled, missing values imputed)
│   └── outputs/              # Final predictions, catboost logs, and SHAP plots
├── notebooks/                # Jupyter notebooks for EDA and prototyping
├── src/                      
│   ├── data_ingestion.py     # Scripts to load and merge the European zones
│   ├── preprocessing.py      # Chronological split, StandardScaler (Train-fitted only)
│   ├── features.py           # Lag generation, MAD filter, Greedy Feature Selector
│   ├── models/
│   │   ├── baselines.py      # Vectorized Naive model, LEAR (LassoLarsIC rolling window)
│   │   ├── tree_models.py    # XGBoost, LightGBM, Random Forest, CatBoost
│   │   ├── deep_learning.py  # Multivariate PyTorch DNN (24-hour tensor output)
│   │   └── ensembles.py      # Quantile Regression Averaging (QRA)
│   ├── evaluation/
│   │   ├── metrics.py        # sMAPE, rMAE, MAE (Standalone numpy implementations)
│   │   ├── probabilistic.py  # Pinball loss, Winkler score
│   │   └── statistical.py    # Diebold-Mariano (DM) test (per hour)
│   └── explainability.py     # SHAP value generators for tree models
├── pyproject.toml            # uv project metadata and dependencies
├── .gitignore                # Ignored files (catboost_info, data, config)
├── config.yaml               # Configuration controlling data paths and hyperparameters
├── PIPELINE.md               # End-to-end machine learning pipeline steps
└── AGENTS.md                 # This file (AI instructions)