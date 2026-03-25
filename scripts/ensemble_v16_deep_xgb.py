#!/usr/bin/env python
"""Deep XGBoost Ensemble: Focus on depths 7-9 with more hyperparameter variations.

Strategy: Depth=8 showed best results in V15. Now test deeper (9-10) and
introduce learning rate and regularization diversity.

12 models: (3 depths x 2 learning_rates x 2 regularization profiles)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import load_data
from src.feature_engineering import build_features
from src.models.targets import prepare_stationary
from src.models.metrics import compute_rmse, compute_hbc

try:
    from xgboost import XGBRegressor
    HAS_MODELS = True
except ImportError:
    HAS_MODELS = False


def main():
    print("\n" + "="*80)
    print("  Deep XGBoost Ensemble (V16): 12 Models with Varied Hyperparams")
    print("="*80)

    if not HAS_MODELS:
        print("ERROR: Missing xgboost")
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

    print(f"\n  Training 12 XGBoost models...")

    models_fr = {}
    depths = [7, 8, 9]
    lrs = [0.04, 0.06]
    regs = [
        {"l1": 0, "l2": 1},        # Light regularization
        {"l1": 1, "l2": 2},        # Stronger regularization
    ]

    model_idx = 0
    for depth in depths:
        for lr in lrs:
            for reg_idx, reg in enumerate(regs):
                xgb = XGBRegressor(
                    max_depth=depth,
                    learning_rate=lr,
                    n_estimators=400,
                    subsample=0.75,
                    colsample_bytree=0.75,
                    reg_alpha=reg["l1"],
                    reg_lambda=reg["l2"],
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
                xgb.fit(X_train_fr, y_train_fr_dev)
                models_fr[f"XGB_d{depth}_lr{lr:.2f}_reg{reg_idx}"] = xgb
                model_idx += 1

    print(f"  Trained {model_idx} models")

    # Generate validation predictions
    val_preds_fr = {}
    train_preds_fr = {}

    for name, model in models_fr.items():
        val_preds_fr[name] = model.predict(X_val)
        train_preds_fr[name] = model.predict(X_train_fr)

    # Compute inverse-RMSE weights
    rmses_fr = {}
    for name in models_fr.keys():
        rmse = compute_rmse(y_train_fr_dev, train_preds_fr[name])
        rmses_fr[name] = rmse

    inv_rmses_fr = np.array([1.0 / rmses_fr[name] for name in models_fr.keys()])
    weights_fr = inv_rmses_fr / inv_rmses_fr.sum()

    # Weighted ensemble
    pred_ensemble_fr = np.zeros(len(X_val))
    for i, name in enumerate(models_fr.keys()):
        pred_ensemble_fr += weights_fr[i] * val_preds_fr[name]

    preds_fr_ensemble = fr_stat["rm_va"] + pred_ensemble_fr
    rmse_fr_ensemble = compute_rmse(y_val_fr_actual, preds_fr_ensemble)

    print(f"  Ensemble RMSE (before HBC): {rmse_fr_ensemble:.2f}")

    # Print top models
    sorted_rmses = sorted(rmses_fr.items(), key=lambda x: x[1])
    print(f"  Top 3 models:")
    for name, rmse in sorted_rmses[:3]:
        print(f"    {name}: {rmse:.2f}")

    # HBC
    hbc_dict_fr, rmse_fr_hbc = compute_hbc(preds_fr_ensemble, y_val_fr_actual, hours_val)
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

    print(f"\n  Training 12 XGBoost models...")

    models_uk = {}

    model_idx = 0
    for depth in depths:
        for lr in lrs:
            for reg_idx, reg in enumerate(regs):
                xgb = XGBRegressor(
                    max_depth=depth,
                    learning_rate=lr,
                    n_estimators=400,
                    subsample=0.75,
                    colsample_bytree=0.75,
                    reg_alpha=reg["l1"],
                    reg_lambda=reg["l2"],
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
                xgb.fit(X_train_uk, y_train_uk_basis)
                models_uk[f"XGB_d{depth}_lr{lr:.2f}_reg{reg_idx}"] = xgb
                model_idx += 1

    print(f"  Trained {model_idx} models")

    # Generate validation predictions
    val_preds_uk = {}
    train_preds_uk = {}

    for name, model in models_uk.items():
        val_preds_uk[name] = model.predict(X_val)
        train_preds_uk[name] = model.predict(X_train_uk)

    # Compute inverse-RMSE weights
    rmses_uk = {}
    for name in models_uk.keys():
        rmse = compute_rmse(y_train_uk_basis, train_preds_uk[name])
        rmses_uk[name] = rmse

    inv_rmses_uk = np.array([1.0 / rmses_uk[name] for name in models_uk.keys()])
    weights_uk = inv_rmses_uk / inv_rmses_uk.sum()

    # Weighted ensemble
    pred_ensemble_uk = np.zeros(len(X_val))
    for i, name in enumerate(models_uk.keys()):
        pred_ensemble_uk += weights_uk[i] * val_preds_uk[name]

    preds_uk_ensemble = uk_moc_val + pred_ensemble_uk
    rmse_uk_ensemble = compute_rmse(uk_spot_val, preds_uk_ensemble)

    print(f"  Ensemble RMSE (before HBC): {rmse_uk_ensemble:.2f}")

    # Print top models
    sorted_rmses = sorted(rmses_uk.items(), key=lambda x: x[1])
    print(f"  Top 3 models:")
    for name, rmse in sorted_rmses[:3]:
        print(f"    {name}: {rmse:.2f}")

    # HBC
    hbc_dict_uk, rmse_uk_hbc = compute_hbc(preds_uk_ensemble, uk_spot_val, hours_val)
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
