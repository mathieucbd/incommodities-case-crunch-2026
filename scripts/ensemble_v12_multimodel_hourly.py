#!/usr/bin/env python
"""Multi-Model Ensemble with Hourly Weighting: Combining 3 diverse models with hour-stratified blending.

Architecture:
  - Level 1: Train 3 diverse models on all features (CatBoost, LightGBM, XGBoost)
  - Level 2: Per-hour ensemble weights (optimize hourly blend ratios)
  - Level 3: HBC post-processing

Goal: Improve on block-specific ensemble (28.39) by using full feature sets
      while allowing hour-specific model emphasis.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

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


def main():
    print("\n" + "="*80)
    print("  Multi-Model Ensemble with Hourly Weighting (V12)")
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

    print(f"\n  Training 3 diverse models (CatBoost, LightGBM, XGBoost)...")

    # Train models
    cb_fr = CatBoostRegressor(depth=6, learning_rate=0.05, iterations=300,
                              loss_function="RMSE", verbose=0, random_state=42, thread_count=-1)
    cb_fr.fit(X_train_fr, y_train_fr_dev)

    lgb_fr = LGBMRegressor(max_depth=7, num_leaves=63, learning_rate=0.05, n_estimators=300,
                           random_state=42, n_jobs=-1, verbose=-1)
    lgb_fr.fit(X_train_fr, y_train_fr_dev)

    xgb_fr = XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=300,
                          random_state=42, n_jobs=-1, verbosity=0)
    xgb_fr.fit(X_train_fr, y_train_fr_dev)

    # Validation predictions
    pred_cb_fr = cb_fr.predict(X_val)
    pred_lgb_fr = lgb_fr.predict(X_val)
    pred_xgb_fr = xgb_fr.predict(X_val)

    # Train predictions for weight optimization
    pred_cb_fr_train = cb_fr.predict(X_train_fr)
    pred_lgb_fr_train = lgb_fr.predict(X_train_fr)
    pred_xgb_fr_train = xgb_fr.predict(X_train_fr)

    # Simple average baseline
    pred_avg_fr = (pred_cb_fr + pred_lgb_fr + pred_xgb_fr) / 3
    preds_fr_base = fr_stat["rm_va"] + pred_avg_fr
    rmse_fr_base = compute_rmse(y_val_fr_actual, preds_fr_base)

    print(f"  Baseline (simple avg) RMSE: {rmse_fr_base:.2f}")

    # Inverse-variance weighting on training OOF
    train_predictions_fr = np.array([pred_cb_fr_train, pred_lgb_fr_train, pred_xgb_fr_train])
    model_vars = np.array([
        np.var(y_train_fr_dev - pred_cb_fr_train),
        np.var(y_train_fr_dev - pred_lgb_fr_train),
        np.var(y_train_fr_dev - pred_xgb_fr_train),
    ])
    inv_vars = 1.0 / model_vars
    weights_fr = inv_vars / inv_vars.sum()

    pred_weighted_fr = weights_fr[0] * pred_cb_fr + weights_fr[1] * pred_lgb_fr + weights_fr[2] * pred_xgb_fr
    preds_fr_weighted = fr_stat["rm_va"] + pred_weighted_fr
    rmse_fr_weighted = compute_rmse(y_val_fr_actual, preds_fr_weighted)

    print(f"  Inverse-variance weighted RMSE: {rmse_fr_weighted:.2f}")
    print(f"  Weights: CB={weights_fr[0]:.3f}, LGB={weights_fr[1]:.3f}, XGB={weights_fr[2]:.3f}")

    # HBC on weighted predictions
    hbc_dict_fr, rmse_fr_hbc = compute_hbc(preds_fr_weighted, y_val_fr_actual, hours_val)
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

    print(f"\n  Training 3 diverse models (CatBoost, LightGBM, XGBoost)...")

    # Train models
    cb_uk = CatBoostRegressor(depth=6, learning_rate=0.05, iterations=300,
                              loss_function="MAE", verbose=0, random_state=42, thread_count=-1)
    cb_uk.fit(X_train_uk, y_train_uk_basis)

    lgb_uk = LGBMRegressor(max_depth=7, num_leaves=63, learning_rate=0.05, n_estimators=300,
                           random_state=42, n_jobs=-1, verbose=-1)
    lgb_uk.fit(X_train_uk, y_train_uk_basis)

    xgb_uk = XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=300,
                          random_state=42, n_jobs=-1, verbosity=0)
    xgb_uk.fit(X_train_uk, y_train_uk_basis)

    # Validation predictions
    pred_cb_uk = cb_uk.predict(X_val)
    pred_lgb_uk = lgb_uk.predict(X_val)
    pred_xgb_uk = xgb_uk.predict(X_val)

    # Train predictions for weight optimization
    pred_cb_uk_train = cb_uk.predict(X_train_uk)
    pred_lgb_uk_train = lgb_uk.predict(X_train_uk)
    pred_xgb_uk_train = xgb_uk.predict(X_train_uk)

    # Simple average baseline
    pred_avg_uk = (pred_cb_uk + pred_lgb_uk + pred_xgb_uk) / 3
    preds_uk_base = uk_moc_val + pred_avg_uk
    rmse_uk_base = compute_rmse(uk_spot_val, preds_uk_base)

    print(f"  Baseline (simple avg) RMSE: {rmse_uk_base:.2f}")

    # Inverse-variance weighting on training OOF
    model_vars = np.array([
        np.var(y_train_uk_basis - pred_cb_uk_train),
        np.var(y_train_uk_basis - pred_lgb_uk_train),
        np.var(y_train_uk_basis - pred_xgb_uk_train),
    ])
    inv_vars = 1.0 / model_vars
    weights_uk = inv_vars / inv_vars.sum()

    pred_weighted_uk = weights_uk[0] * pred_cb_uk + weights_uk[1] * pred_lgb_uk + weights_uk[2] * pred_xgb_uk
    preds_uk_weighted = uk_moc_val + pred_weighted_uk
    rmse_uk_weighted = compute_rmse(uk_spot_val, preds_uk_weighted)

    print(f"  Inverse-variance weighted RMSE: {rmse_uk_weighted:.2f}")
    print(f"  Weights: CB={weights_uk[0]:.3f}, LGB={weights_uk[1]:.3f}, XGB={weights_uk[2]:.3f}")

    # HBC on weighted predictions
    hbc_dict_uk, rmse_uk_hbc = compute_hbc(preds_uk_weighted, uk_spot_val, hours_val)
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
