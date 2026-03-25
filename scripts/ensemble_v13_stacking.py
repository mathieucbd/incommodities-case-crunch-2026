#!/usr/bin/env python
"""Ridge Stacking Ensemble: Learn optimal model weights from out-of-fold predictions.

Architecture:
  - Level 1: Train 3 base models (CatBoost, LightGBM, XGBoost) on full features
  - Level 2: Generate out-of-fold (OOF) predictions using TimeSeriesSplit
  - Level 3: Fit Ridge regression on OOF predictions to learn optimal weights
  - Level 4: Apply HBC post-processing

This approach often outperforms manual weighting by discovering complex relationships.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import load_data
from src.feature_engineering import build_features
from src.models.targets import prepare_stationary
from src.models.metrics import compute_rmse, compute_hbc

try:
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    HAS_MODELS = True
except ImportError:
    HAS_MODELS = False


def train_with_stacking(X, y, tss_splits=5, alpha=1.0):
    """Train base models and learn stacking weights via OOF predictions."""
    n_samples = len(X)

    # Initialize base estimators
    cb = CatBoostRegressor(depth=6, learning_rate=0.05, iterations=300,
                           loss_function="RMSE", verbose=0, random_state=42, thread_count=-1)
    lgb = LGBMRegressor(max_depth=7, num_leaves=63, learning_rate=0.05, n_estimators=300,
                        random_state=42, n_jobs=-1, verbose=-1)
    xgb = XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=300,
                       random_state=42, n_jobs=-1, verbosity=0)

    base_estimators = {"CB": cb, "LGB": lgb, "XGB": xgb}

    # Generate OOF predictions
    oof_preds = np.zeros((n_samples, 3))
    tss = TimeSeriesSplit(n_splits=tss_splits)

    for fold, (train_idx, test_idx) in enumerate(tss.split(X)):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold = y[train_idx]

        for i, (name, est) in enumerate(base_estimators.items()):
            model = clone(est)
            model.fit(X_train_fold, y_train_fold)
            oof_preds[test_idx, i] = model.predict(X_test_fold)

    # Train final models on ALL data
    final_models = {}
    for name, est in base_estimators.items():
        model = clone(est)
        model.fit(X, y)
        final_models[name] = model

    # Learn stacking weights (Ridge regression)
    stacker = Ridge(alpha=alpha, fit_intercept=False)
    stacker.fit(oof_preds, y)
    weights = np.abs(stacker.coef_) / np.sum(np.abs(stacker.coef_))  # Normalize

    return final_models, weights, oof_preds, stacker


def main():
    print("\n" + "="*80)
    print("  Ridge Stacking Ensemble (V13)")
    print("="*80)

    if not HAS_MODELS:
        print("ERROR: Missing catboost, lightgbm, xgboost")
        return

    with open(PROJECT_ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    print("\n  Loading data...")
    x_train, y_train, x_test = load_data(PROJECT_ROOT / "data" / "raw")
    train_fe = build_features(pd.concat([x_train], axis=0), config)
    train_fe = train_fe.join(y_train[["fr_spot", "uk_spot"]])

    holdout_start = config["validation"]["holdout_start"]
    mask_val = train_fe["datetime_CET"] >= holdout_start
    df_train = train_fe[~mask_val].copy()
    df_val = train_fe[mask_val].copy()

    exclude_cols = {"fr_spot", "uk_spot", "datetime_CET", "datetime_UTC", "date", "id"}
    numeric_cols = [c for c in df_train.columns
                    if c not in exclude_cols and df_train[c].dtype in (float, np.float64, np.float32, int, np.int64, np.int32)
                    and df_train[c].notna().sum() > len(df_train) * 0.5]

    X_train = df_train[numeric_cols].copy()
    X_val = df_val[numeric_cols].copy()
    hours_val = df_val["hour"].values

    print(f"  Features: {len(numeric_cols)}")
    print(f"  Train: {len(df_train)} rows  |  Val: {len(df_val)} rows")

    # ─────────────────────────────────────────────────────────────────────────
    # FRANCE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-"*80)
    print("  FRANCE")
    print("-"*80)

    fr_stat = prepare_stationary("fr_spot_la", "fr_spot", train_fe, df_train, df_val)
    y_train_fr_dev = fr_stat["y_dev_tr"][fr_stat["valid_tr"]]
    X_train_fr = X_train[fr_stat["valid_tr"]].copy()
    y_val_fr_actual = fr_stat["spot_va"]

    print(f"\n  Training stacking ensemble (3 base models + Ridge meta-learner)...")

    models_fr, weights_fr, oof_fr, stacker_fr = train_with_stacking(X_train_fr, y_train_fr_dev, tss_splits=5, alpha=1.0)

    # Validation predictions
    pred_cb_fr = models_fr["CB"].predict(X_val)
    pred_lgb_fr = models_fr["LGB"].predict(X_val)
    pred_xgb_fr = models_fr["XGB"].predict(X_val)

    # Stack predictions
    stacked_preds_fr = np.column_stack([pred_cb_fr, pred_lgb_fr, pred_xgb_fr])
    pred_ensemble_fr = stacker_fr.predict(stacked_preds_fr)

    preds_fr_stacked = fr_stat["rm_va"] + pred_ensemble_fr
    rmse_fr_stacked = compute_rmse(y_val_fr_actual, preds_fr_stacked)

    print(f"  Stacking RMSE: {rmse_fr_stacked:.2f}")
    print(f"  Meta weights (CB={weights_fr[0]:.3f}, LGB={weights_fr[1]:.3f}, XGB={weights_fr[2]:.3f})")

    # HBC
    hbc_dict_fr, rmse_fr_hbc = compute_hbc(preds_fr_stacked, y_val_fr_actual, hours_val)
    print(f"  After HBC: {rmse_fr_hbc:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # UK
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-"*80)
    print("  UNITED KINGDOM")
    print("-"*80)

    uk_spot_train = df_train["uk_spot"].values
    uk_moc_train = df_train["uk_merit_order_cost"].values
    uk_spot_val = df_val["uk_spot"].values
    uk_moc_val = df_val["uk_merit_order_cost"].values

    y_train_uk_basis = uk_spot_train - uk_moc_train
    valid_basis_tr = np.isfinite(y_train_uk_basis)
    X_train_uk = X_train[valid_basis_tr].copy()
    y_train_uk_basis = y_train_uk_basis[valid_basis_tr]

    print(f"\n  Training stacking ensemble (3 base models + Ridge meta-learner)...")

    models_uk, weights_uk, oof_uk, stacker_uk = train_with_stacking(X_train_uk, y_train_uk_basis, tss_splits=5, alpha=1.0)

    # Validation predictions
    pred_cb_uk = models_uk["CB"].predict(X_val)
    pred_lgb_uk = models_uk["LGB"].predict(X_val)
    pred_xgb_uk = models_uk["XGB"].predict(X_val)

    # Stack predictions
    stacked_preds_uk = np.column_stack([pred_cb_uk, pred_lgb_uk, pred_xgb_uk])
    pred_ensemble_uk = stacker_uk.predict(stacked_preds_uk)

    preds_uk_stacked = uk_moc_val + pred_ensemble_uk
    rmse_uk_stacked = compute_rmse(uk_spot_val, preds_uk_stacked)

    print(f"  Stacking RMSE: {rmse_uk_stacked:.2f}")
    print(f"  Meta weights (CB={weights_uk[0]:.3f}, LGB={weights_uk[1]:.3f}, XGB={weights_uk[2]:.3f})")

    # HBC
    hbc_dict_uk, rmse_uk_hbc = compute_hbc(preds_uk_stacked, uk_spot_val, hours_val)
    print(f"  After HBC: {rmse_uk_hbc:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  FINAL SUMMARY")
    print("="*80)
    print(f"\n  FR: {rmse_fr_hbc:.2f}  |  UK: {rmse_uk_hbc:.2f}  |  COMBINED: {rmse_fr_hbc + rmse_uk_hbc:.2f}")

    target = 24.0
    delta = (rmse_fr_hbc + rmse_uk_hbc) - target
    if delta < 0:
        print(f"  TARGET MET: {abs(delta):.2f} points under!")
    else:
        print(f"  Gap to target 24: +{delta:.2f}")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
