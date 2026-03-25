#!/usr/bin/env python
"""Block-Specific Ensemble Pipeline V2: Multi-Model Per Block + Regime Weighting.

Nested ensemble architecture:
  - Level 1: 3 diverse models per block (CatBoost, LightGBM, XGBoost)
  - Level 2: Block-level ensemble (inverse-RMSE weights)
  - Level 3: Regime-based weights (hourly stratification)
  - Level 4: Hourly bias correction (HBC)
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
from src.models.ensemble import HOUR_TO_REGIME, optimize_regime_weights
from src.features import describe_blocks, get_feature_blocks

try:
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import StandardScaler
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


def train_block_models(X_block: pd.DataFrame, y: np.ndarray, block_name: str, loss_fn: str = "RMSE") -> dict:
    """Train 3 diverse models on a feature block."""
    models = {}

    # Model 1: CatBoost
    cb = CatBoostRegressor(
        depth=6,
        learning_rate=0.05,
        iterations=300,
        loss_function=loss_fn,
        verbose=0,
        random_state=42,
        thread_count=-1
    )
    models["CB"] = cb.fit(X_block, y)

    # Model 2: LightGBM
    lgb = LGBMRegressor(
        max_depth=7,
        num_leaves=63,
        learning_rate=0.05,
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    models["LGB"] = lgb.fit(X_block, y)

    # Model 3: XGBoost
    xgb = XGBRegressor(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    models["XGB"] = xgb.fit(X_block, y)

    return models


def predict_block_ensemble(models: dict, X_block: pd.DataFrame) -> np.ndarray:
    """Average predictions from 3 models."""
    preds = np.array([m.predict(X_block) for m in models.values()])
    return np.mean(preds, axis=0)


def main():
    """Run nested block ensemble with regime weighting."""
    print("\n" + "=" * 80)
    print("  Block-Specific Ensemble V2: Multi-Model + Regime Weighting")
    print("=" * 80)

    if not HAS_MODELS:
        print("\nERROR: Required packages missing. Install: pip install catboost lightgbm xgboost")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Load data
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

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Feature preparation
    # ─────────────────────────────────────────────────────────────────────────
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

    # Get feature blocks
    blocks = get_feature_blocks(numeric_cols)
    block_a, block_b, block_c = blocks

    print(f"  Features: {len(numeric_cols)} total")
    print(f"    Block A: {len(block_a)},  Block B: {len(block_b)},  Block C: {len(block_c)}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Train FR ensemble
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("  FRANCE (fr_spot) — Multi-Model Block Ensemble + Regime Weighting")
    print("-" * 80)

    fr_stat = prepare_stationary("fr_spot_la", "fr_spot", train_fe, df_train, df_val)
    y_train_fr_dev = fr_stat["y_dev_tr"][fr_stat["valid_tr"]]
    X_train_fr = X_train[fr_stat["valid_tr"]].copy()
    y_val_fr_actual = fr_stat["spot_va"]

    print(f"\n  Training 3 models × 3 blocks = 9 models total...")

    # Train models for each block
    X_train_a = X_train_fr[[f for f in block_a if f in X_train_fr.columns]]
    X_train_b = X_train_fr[[f for f in block_b if f in X_train_fr.columns]]
    X_train_c = X_train_fr[[f for f in block_c if f in X_train_fr.columns]]

    models_a_fr = train_block_models(X_train_a, y_train_fr_dev, "A", loss_fn="RMSE")
    models_b_fr = train_block_models(X_train_b, y_train_fr_dev, "B", loss_fn="RMSE")
    models_c_fr = train_block_models(X_train_c, y_train_fr_dev, "C", loss_fn="RMSE")

    # Block-level ensemble predictions
    X_val_a = X_val[[f for f in block_a if f in X_val.columns]]
    X_val_b = X_val[[f for f in block_b if f in X_val.columns]]
    X_val_c = X_val[[f for f in block_c if f in X_val.columns]]

    pred_a_fr = predict_block_ensemble(models_a_fr, X_val_a)
    pred_b_fr = predict_block_ensemble(models_b_fr, X_val_b)
    pred_c_fr = predict_block_ensemble(models_c_fr, X_val_c)

    # Compute block RMSEs for inverse weighting
    rmse_a_fr = compute_rmse(y_train_fr_dev, predict_block_ensemble(models_a_fr, X_train_a))
    rmse_b_fr = compute_rmse(y_train_fr_dev, predict_block_ensemble(models_b_fr, X_train_b))
    rmse_c_fr = compute_rmse(y_train_fr_dev, predict_block_ensemble(models_c_fr, X_train_c))

    inv_rmses_fr = np.array([1.0 / rmse_a_fr, 1.0 / rmse_b_fr, 1.0 / rmse_c_fr])
    block_weights_fr = inv_rmses_fr / inv_rmses_fr.sum()

    print(f"  Block RMSEs   |  A: {rmse_a_fr:.2f}   B: {rmse_b_fr:.2f}   C: {rmse_c_fr:.2f}")
    print(f"  Block Weights |  A: {block_weights_fr[0]:.3f}   B: {block_weights_fr[1]:.3f}   C: {block_weights_fr[2]:.3f}")

    # Combine blocks + add anchor
    pred_fr_dev_ensemble = (
        block_weights_fr[0] * pred_a_fr +
        block_weights_fr[1] * pred_b_fr +
        block_weights_fr[2] * pred_c_fr
    )
    preds_fr = fr_stat["rm_va"] + pred_fr_dev_ensemble

    # Regime-based weights (optimize per hour)
    models_dict_fr = {
        "A": pred_a_fr,
        "B": pred_b_fr,
        "C": pred_c_fr,
    }
    regime_weights_fr, preds_fr_regime = optimize_regime_weights(
        models_dict_fr, y_val_fr_actual, hours_val, "FR"
    )

    # Metrics
    rmse_fr_base = compute_rmse(y_val_fr_actual, preds_fr)
    rmse_fr_regime = compute_rmse(y_val_fr_actual, preds_fr_regime)
    mae_fr_regime = compute_mae(y_val_fr_actual, preds_fr_regime)

    # HBC
    hbc_dict_fr, rmse_fr_hbc = compute_hbc(preds_fr_regime, y_val_fr_actual, hours_val)

    print(f"\n  Validation RMSE (no regime): {rmse_fr_base:.2f}")
    print(f"  Validation RMSE (+regime):  {rmse_fr_regime:.2f}")
    print(f"  Validation RMSE (+HBC):     {rmse_fr_hbc:.2f}")
    print(f"  Validation MAE (+HBC):      {mae_fr_regime:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Train UK ensemble
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("  UNITED KINGDOM (uk_spot) — Multi-Model Block Ensemble + Regime Weighting")
    print("-" * 80)

    uk_spot_train = df_train["uk_spot"].values
    uk_moc_train = df_train["uk_merit_order_cost"].values
    uk_spot_val = df_val["uk_spot"].values
    uk_moc_val = df_val["uk_merit_order_cost"].values

    y_train_uk_basis = uk_spot_train - uk_moc_train
    valid_basis_tr = np.isfinite(y_train_uk_basis)
    X_train_uk = X_train[valid_basis_tr].copy()
    y_train_uk_basis = y_train_uk_basis[valid_basis_tr]

    print(f"\n  Training 3 models × 3 blocks = 9 models total...")

    # Train models for each block (UK uses MAE loss)
    X_train_a = X_train_uk[[f for f in block_a if f in X_train_uk.columns]]
    X_train_b = X_train_uk[[f for f in block_b if f in X_train_uk.columns]]
    X_train_c = X_train_uk[[f for f in block_c if f in X_train_uk.columns]]

    models_a_uk = train_block_models(X_train_a, y_train_uk_basis, "A", loss_fn="MAE")
    models_b_uk = train_block_models(X_train_b, y_train_uk_basis, "B", loss_fn="MAE")
    models_c_uk = train_block_models(X_train_c, y_train_uk_basis, "C", loss_fn="MAE")

    # Block-level ensemble
    X_val_a = X_val[[f for f in block_a if f in X_val.columns]]
    X_val_b = X_val[[f for f in block_b if f in X_val.columns]]
    X_val_c = X_val[[f for f in block_c if f in X_val.columns]]

    pred_a_uk = predict_block_ensemble(models_a_uk, X_val_a)
    pred_b_uk = predict_block_ensemble(models_b_uk, X_val_b)
    pred_c_uk = predict_block_ensemble(models_c_uk, X_val_c)

    # Compute block RMSEs
    rmse_a_uk = compute_rmse(y_train_uk_basis, predict_block_ensemble(models_a_uk, X_train_a))
    rmse_b_uk = compute_rmse(y_train_uk_basis, predict_block_ensemble(models_b_uk, X_train_b))
    rmse_c_uk = compute_rmse(y_train_uk_basis, predict_block_ensemble(models_c_uk, X_train_c))

    inv_rmses_uk = np.array([1.0 / rmse_a_uk, 1.0 / rmse_b_uk, 1.0 / rmse_c_uk])
    block_weights_uk = inv_rmses_uk / inv_rmses_uk.sum()

    print(f"  Block RMSEs   |  A: {rmse_a_uk:.2f}   B: {rmse_b_uk:.2f}   C: {rmse_c_uk:.2f}")
    print(f"  Block Weights |  A: {block_weights_uk[0]:.3f}   B: {block_weights_uk[1]:.3f}   C: {block_weights_uk[2]:.3f}")

    # Combine + add merit order
    pred_uk_basis_ensemble = (
        block_weights_uk[0] * pred_a_uk +
        block_weights_uk[1] * pred_b_uk +
        block_weights_uk[2] * pred_c_uk
    )
    preds_uk = uk_moc_val + pred_uk_basis_ensemble

    # Regime weights
    models_dict_uk = {
        "A": pred_a_uk,
        "B": pred_b_uk,
        "C": pred_c_uk,
    }
    regime_weights_uk, preds_uk_regime = optimize_regime_weights(
        models_dict_uk, uk_spot_val, hours_val, "UK"
    )

    # Metrics
    rmse_uk_base = compute_rmse(uk_spot_val, preds_uk)
    rmse_uk_regime = compute_rmse(uk_spot_val, preds_uk_regime)
    mae_uk_regime = compute_mae(uk_spot_val, preds_uk_regime)

    # HBC
    hbc_dict_uk, rmse_uk_hbc = compute_hbc(preds_uk_regime, uk_spot_val, hours_val)

    print(f"\n  Validation RMSE (no regime): {rmse_uk_base:.2f}")
    print(f"  Validation RMSE (+regime):  {rmse_uk_regime:.2f}")
    print(f"  Validation RMSE (+HBC):     {rmse_uk_hbc:.2f}")
    print(f"  Validation MAE (+HBC):      {mae_uk_regime:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FINAL SUMMARY")
    print("=" * 80)
    print(f"\n  France RMSE:")
    print(f"    Base ensemble (no regime) : {rmse_fr_base:.2f}")
    print(f"    + Regime weighting       : {rmse_fr_regime:.2f}")
    print(f"    + Hourly bias correction : {rmse_fr_hbc:.2f}")
    print(f"\n  United Kingdom RMSE:")
    print(f"    Base ensemble (no regime) : {rmse_uk_base:.2f}")
    print(f"    + Regime weighting       : {rmse_uk_regime:.2f}")
    print(f"    + Hourly bias correction : {rmse_uk_hbc:.2f}")
    print(f"\n  COMBINED SUM (with HBC)  : {rmse_fr_hbc + rmse_uk_hbc:.2f}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
