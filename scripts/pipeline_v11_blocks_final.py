#!/usr/bin/env python
"""Block-Specific Ensemble FINAL: 3-Model Per Block + HBC (No Regime Weighting).

Optimized multi-model architecture:
  - Level 1: 3 diverse models per block (CatBoost, LightGBM, XGBoost)
  - Level 2: Simple average within each block
  - Level 3: Block-level inverse-RMSE weights
  - Level 4: Hourly bias correction (HBC)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import load_data
from src.feature_engineering import build_features
from src.models.targets import prepare_stationary
from src.models.metrics import compute_rmse, compute_hbc
from src.features import describe_blocks, get_feature_blocks

try:
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    HAS_MODELS = True
except ImportError:
    HAS_MODELS = False


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_huber(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 10.0) -> float:
    """Huber Loss."""
    r = np.abs(y_true - y_pred)
    loss = np.where(r <= delta, 0.5 * r**2, delta * (r - 0.5 * delta))
    return float(np.mean(loss))


def compute_pinball(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.5) -> float:
    """Pinball Loss."""
    r = y_true - y_pred
    loss = np.where(r >= 0, alpha * r, (alpha - 1) * r)
    return float(np.mean(loss))


def train_block_ensemble(X_block: pd.DataFrame, y: np.ndarray, loss_fn: str = "RMSE", seed: int = 42):
    """Train 3 diverse models on a block and return average predictions."""
    # Model 1: CatBoost
    cb = CatBoostRegressor(
        depth=6,
        learning_rate=0.05,
        iterations=300,
        loss_function=loss_fn,
        verbose=0,
        random_state=seed,
        thread_count=-1
    )
    cb.fit(X_block, y)

    # Model 2: LightGBM
    lgb = LGBMRegressor(
        max_depth=7,
        num_leaves=63,
        learning_rate=0.05,
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
        verbose=-1
    )
    lgb.fit(X_block, y)

    # Model 3: XGBoost
    xgb = XGBRegressor(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_block, y)

    return {"CB": cb, "LGB": lgb, "XGB": xgb}


def predict_ensemble(models: dict, X_block: pd.DataFrame) -> np.ndarray:
    """Average predictions from 3 models."""
    preds = np.array([m.predict(X_block) for m in models.values()])
    return np.mean(preds, axis=0)


def main():
    """Run optimized block ensemble."""
    print("\n" + "=" * 80)
    print("  Block-Specific Ensemble FINAL: 3-Model Per Block + HBC")
    print("=" * 80)

    if not HAS_MODELS:
        print("\nERROR: Required packages missing. Install: pip install catboost lightgbm xgboost")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────────────
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

    print(f"  Train: {len(df_train)} rows  |  Val: {len(df_val)} rows")

    # Features
    exclude_cols = {"fr_spot", "uk_spot", "datetime_CET", "datetime_UTC", "date", "id"}
    numeric_cols = [
        c for c in df_train.columns
        if c not in exclude_cols
        and df_train[c].dtype in (float, np.float64, np.float32, int, np.int64, np.int32)
        and df_train[c].notna().sum() > len(df_train) * 0.5
    ]

    X_train = df_train[numeric_cols].copy()
    X_val = df_val[numeric_cols].copy()
    hours_val = df_val["hour"].values

    blocks = get_feature_blocks(numeric_cols)
    block_a, block_b, block_c = blocks

    print(f"  Features: {len(numeric_cols)} total (A={len(block_a)}, B={len(block_b)}, C={len(block_c)})")

    # ─────────────────────────────────────────────────────────────────────────
    # FRANCE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("  FRANCE (fr_spot)")
    print("-" * 80)

    fr_stat = prepare_stationary("fr_spot_la", "fr_spot", train_fe, df_train, df_val)
    y_train_fr_dev = fr_stat["y_dev_tr"][fr_stat["valid_tr"]]
    X_train_fr = X_train[fr_stat["valid_tr"]].copy()
    y_val_fr_actual = fr_stat["spot_va"]

    print(f"\n  Training 3 models x 3 blocks (9 total)...")

    # Train block ensembles
    X_train_a = X_train_fr[[f for f in block_a if f in X_train_fr.columns]]
    X_train_b = X_train_fr[[f for f in block_b if f in X_train_fr.columns]]
    X_train_c = X_train_fr[[f for f in block_c if f in X_train_fr.columns]]

    models_a_fr = train_block_ensemble(X_train_a, y_train_fr_dev, loss_fn="RMSE")
    models_b_fr = train_block_ensemble(X_train_b, y_train_fr_dev, loss_fn="RMSE")
    models_c_fr = train_block_ensemble(X_train_c, y_train_fr_dev, loss_fn="RMSE")

    # Val predictions
    X_val_a = X_val[[f for f in block_a if f in X_val.columns]]
    X_val_b = X_val[[f for f in block_b if f in X_val.columns]]
    X_val_c = X_val[[f for f in block_c if f in X_val.columns]]

    pred_a_fr = predict_ensemble(models_a_fr, X_val_a)
    pred_b_fr = predict_ensemble(models_b_fr, X_val_b)
    pred_c_fr = predict_ensemble(models_c_fr, X_val_c)

    # Block weights (inverse RMSE on training)
    train_pred_a = predict_ensemble(models_a_fr, X_train_a)
    train_pred_b = predict_ensemble(models_b_fr, X_train_b)
    train_pred_c = predict_ensemble(models_c_fr, X_train_c)

    rmse_a_fr = compute_rmse(y_train_fr_dev, train_pred_a)
    rmse_b_fr = compute_rmse(y_train_fr_dev, train_pred_b)
    rmse_c_fr = compute_rmse(y_train_fr_dev, train_pred_c)

    inv_rmses = np.array([1.0 / rmse_a_fr, 1.0 / rmse_b_fr, 1.0 / rmse_c_fr])
    w_a, w_b, w_c = inv_rmses / inv_rmses.sum()

    print(f"  Block training RMSE: A={rmse_a_fr:.2f}  B={rmse_b_fr:.2f}  C={rmse_c_fr:.2f}")
    print(f"  Block weights:       A={w_a:.3f}  B={w_b:.3f}  C={w_c:.3f}")

    # Ensemble
    pred_fr_dev = w_a * pred_a_fr + w_b * pred_b_fr + w_c * pred_c_fr
    preds_fr = fr_stat["rm_va"] + pred_fr_dev

    # Metrics before HBC
    rmse_fr = compute_rmse(y_val_fr_actual, preds_fr)
    mae_fr = compute_mae(y_val_fr_actual, preds_fr)
    huber_fr = compute_huber(y_val_fr_actual, preds_fr)

    print(f"\n  Validation Metrics (before HBC)")
    print(f"    RMSE: {rmse_fr:.2f}  |  MAE: {mae_fr:.2f}  |  Huber: {huber_fr:.2f}")

    # HBC
    hbc_dict_fr, rmse_fr_hbc = compute_hbc(preds_fr, y_val_fr_actual, hours_val)
    mae_fr_hbc = compute_mae(y_val_fr_actual, preds_fr + np.array([hbc_dict_fr.get(h, 0) for h in hours_val]))

    print(f"  Validation Metrics (after HBC)")
    print(f"    RMSE: {rmse_fr_hbc:.2f}  |  MAE: {mae_fr_hbc:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # UNITED KINGDOM
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("  UNITED KINGDOM (uk_spot)")
    print("-" * 80)

    uk_spot_train = df_train["uk_spot"].values
    uk_moc_train = df_train["uk_merit_order_cost"].values
    uk_spot_val = df_val["uk_spot"].values
    uk_moc_val = df_val["uk_merit_order_cost"].values

    y_train_uk_basis = uk_spot_train - uk_moc_train
    valid_basis_tr = np.isfinite(y_train_uk_basis)
    X_train_uk = X_train[valid_basis_tr].copy()
    y_train_uk_basis = y_train_uk_basis[valid_basis_tr]

    print(f"\n  Training 3 models x 3 blocks (9 total)...")

    # Train block ensembles (UK uses MAE loss)
    X_train_a = X_train_uk[[f for f in block_a if f in X_train_uk.columns]]
    X_train_b = X_train_uk[[f for f in block_b if f in X_train_uk.columns]]
    X_train_c = X_train_uk[[f for f in block_c if f in X_train_uk.columns]]

    models_a_uk = train_block_ensemble(X_train_a, y_train_uk_basis, loss_fn="MAE")
    models_b_uk = train_block_ensemble(X_train_b, y_train_uk_basis, loss_fn="MAE")
    models_c_uk = train_block_ensemble(X_train_c, y_train_uk_basis, loss_fn="MAE")

    # Val predictions
    X_val_a = X_val[[f for f in block_a if f in X_val.columns]]
    X_val_b = X_val[[f for f in block_b if f in X_val.columns]]
    X_val_c = X_val[[f for f in block_c if f in X_val.columns]]

    pred_a_uk = predict_ensemble(models_a_uk, X_val_a)
    pred_b_uk = predict_ensemble(models_b_uk, X_val_b)
    pred_c_uk = predict_ensemble(models_c_uk, X_val_c)

    # Block weights
    train_pred_a = predict_ensemble(models_a_uk, X_train_a)
    train_pred_b = predict_ensemble(models_b_uk, X_train_b)
    train_pred_c = predict_ensemble(models_c_uk, X_train_c)

    rmse_a_uk = compute_rmse(y_train_uk_basis, train_pred_a)
    rmse_b_uk = compute_rmse(y_train_uk_basis, train_pred_b)
    rmse_c_uk = compute_rmse(y_train_uk_basis, train_pred_c)

    inv_rmses = np.array([1.0 / rmse_a_uk, 1.0 / rmse_b_uk, 1.0 / rmse_c_uk])
    w_a, w_b, w_c = inv_rmses / inv_rmses.sum()

    print(f"  Block training RMSE: A={rmse_a_uk:.2f}  B={rmse_b_uk:.2f}  C={rmse_c_uk:.2f}")
    print(f"  Block weights:       A={w_a:.3f}  B={w_b:.3f}  C={w_c:.3f}")

    # Ensemble
    pred_uk_basis = w_a * pred_a_uk + w_b * pred_b_uk + w_c * pred_c_uk
    preds_uk = uk_moc_val + pred_uk_basis

    # Metrics before HBC
    rmse_uk = compute_rmse(uk_spot_val, preds_uk)
    mae_uk = compute_mae(uk_spot_val, preds_uk)
    huber_uk = compute_huber(uk_spot_val, preds_uk)

    print(f"\n  Validation Metrics (before HBC)")
    print(f"    RMSE: {rmse_uk:.2f}  |  MAE: {mae_uk:.2f}  |  Huber: {huber_uk:.2f}")

    # HBC
    hbc_dict_uk, rmse_uk_hbc = compute_hbc(preds_uk, uk_spot_val, hours_val)
    mae_uk_hbc = compute_mae(uk_spot_val, preds_uk + np.array([hbc_dict_uk.get(h, 0) for h in hours_val]))

    print(f"  Validation Metrics (after HBC)")
    print(f"    RMSE: {rmse_uk_hbc:.2f}  |  MAE: {mae_uk_hbc:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FINAL SUMMARY")
    print("=" * 80)
    print(f"\n  France    : {rmse_fr_hbc:.2f}")
    print(f"  UK        : {rmse_uk_hbc:.2f}")
    print(f"  COMBINED  : {rmse_fr_hbc + rmse_uk_hbc:.2f}  ", end="")

    target = 24.0
    delta = (rmse_fr_hbc + rmse_uk_hbc) - target
    if delta < 0:
        print(f"(TARGET MET: -{abs(delta):.2f})")
    else:
        print(f"(Target gap: +{delta:.2f})")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
