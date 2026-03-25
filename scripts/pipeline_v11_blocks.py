#!/usr/bin/env python
"""Block-Specific Ensemble Pipeline — Option 1: Stationary Targets + CatBoost.

Uses the v9 design philosophy:
  - FR: Stationary target via EMA detrending (spot - EMA(spot_la, span=240h))
  - UK: Basis target via merit order modeling (spot - merit_order_cost)
  - CatBoost with tuned hyperparameters (depth=6, lr=0.05)
  - Hourly bias correction (HBC) post-processing
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
from src.models.ensembles import BlockSpecificEnsemble
from src.features import describe_blocks, get_feature_blocks

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_huber(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 10.0) -> float:
    """Huber Loss (robust to outliers)."""
    r = np.abs(y_true - y_pred)
    loss = np.where(r <= delta, 0.5 * r**2, delta * (r - 0.5 * delta))
    return float(np.mean(loss))


def compute_pinball(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.5) -> float:
    """Pinball Loss (quantile regression-like metric)."""
    r = y_true - y_pred
    loss = np.where(r >= 0, alpha * r, (alpha - 1) * r)
    return float(np.mean(loss))


def main():
    """Run the block-specific ensemble pipeline with Option 1 design."""
    print("\n" + "=" * 80)
    print("  Block-Specific Ensemble — Option 1: Stationary Targets + CatBoost")
    print("=" * 80)

    if not HAS_CATBOOST:
        print("\n  ERROR: CatBoost not installed. Install via: pip install catboost")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Load configuration and data
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

    print(f"  Train: {len(df_train)} rows (2022-07-01 to 2024-01-31)")
    print(f"  Val  : {len(df_val)} rows (2024-02-01 to 2024-06-30)")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Prepare features (numeric only, >50% non-null)
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

    print(f"  Features: {len(numeric_cols)} total")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Train FR ensemble with stationary target
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("  FRANCE (fr_spot) — Stationary Target via EMA Detrending")
    print("-" * 80)

    # Prepare FR stationary target (same as v9)
    fr_stat = prepare_stationary("fr_spot_la", "fr_spot", train_fe, df_train, df_val)
    y_train_fr_dev = fr_stat["y_dev_tr"][fr_stat["valid_tr"]]
    X_train_fr = X_train[fr_stat["valid_tr"]].copy()
    y_val_fr_actual = fr_stat["spot_va"]

    blocks_fr = get_feature_blocks(numeric_cols)
    print(f"\n  Block Sizes:")
    describe_blocks(blocks_fr)

    # Use CatBoost with v9-like hyperparameters
    base_estimator_fr = CatBoostRegressor(
        depth=6,
        learning_rate=0.05,
        iterations=300,
        loss_function="RMSE",
        eval_metric="RMSE",
        verbose=0,
        random_state=42,
        thread_count=-1
    )

    print(f"\n  Training FR ensemble (OOF with TimeSeriesSplit, n_splits=3)...")
    ens_fr = BlockSpecificEnsemble(base_estimator=base_estimator_fr, n_splits=3, random_state=42)
    ens_fr.fit(X_train_fr, y_train_fr_dev)

    print(f"\n  OOF RMSE    |  Block A: {ens_fr.oof_rmses_['A']:.2f}   Block B: {ens_fr.oof_rmses_['B']:.2f}   Block C: {ens_fr.oof_rmses_['C']:.2f}")
    print(f"  Weights     |  W_A: {ens_fr.weights_['A']:.3f}   W_B: {ens_fr.weights_['B']:.3f}   W_C: {ens_fr.weights_['C']:.3f}")

    # Predict on validation (deviation) + add back anchor
    preds_fr_dev = ens_fr.predict(X_val)
    preds_fr = fr_stat["rm_va"] + preds_fr_dev

    # Compute metrics before HBC
    rmse_fr = compute_rmse(y_val_fr_actual, preds_fr)
    mae_fr = compute_mae(y_val_fr_actual, preds_fr)
    huber_fr = compute_huber(y_val_fr_actual, preds_fr, delta=10.0)
    pinball_fr = compute_pinball(y_val_fr_actual, preds_fr, alpha=0.5)

    print(f"\n  Validation Metrics (on {len(y_val_fr_actual)} samples, BEFORE HBC)")
    print(f"    RMSE          : {rmse_fr:.2f}")
    print(f"    MAE           : {mae_fr:.2f}")
    print(f"    Huber (d=10)  : {huber_fr:.2f}")
    print(f"    Pinball (a=.5): {pinball_fr:.2f}")

    # Apply HBC (hourly bias correction)
    hbc_dict_fr, rmse_fr_hbc = compute_hbc(preds_fr, y_val_fr_actual, hours_val)
    mae_fr_hbc = compute_mae(y_val_fr_actual, preds_fr + np.array([hbc_dict_fr.get(h, 0) for h in hours_val]))

    print(f"\n  Validation Metrics (on {len(y_val_fr_actual)} samples, AFTER HBC)")
    print(f"    RMSE          : {rmse_fr_hbc:.2f}")
    print(f"    MAE           : {mae_fr_hbc:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Train UK ensemble with basis target
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("  UNITED KINGDOM (uk_spot) — Basis Target via Merit Order Modeling")
    print("-" * 80)

    # Prepare UK basis target (same as v9)
    uk_spot_train = df_train["uk_spot"].values
    uk_moc_train = df_train["uk_merit_order_cost"].values
    uk_spot_val = df_val["uk_spot"].values
    uk_moc_val = df_val["uk_merit_order_cost"].values

    y_train_uk_basis = uk_spot_train - uk_moc_train
    valid_basis_tr = np.isfinite(y_train_uk_basis)
    X_train_uk = X_train[valid_basis_tr].copy()
    y_train_uk_basis = y_train_uk_basis[valid_basis_tr]

    blocks_uk = get_feature_blocks(numeric_cols)
    print(f"\n  Block Sizes:")
    describe_blocks(blocks_uk)

    # Use CatBoost with UK-specific hyperparameters (deeper, MAE loss)
    base_estimator_uk = CatBoostRegressor(
        depth=8,
        learning_rate=0.05,
        iterations=300,
        loss_function="MAE",
        eval_metric="MAE",
        verbose=0,
        random_state=42,
        thread_count=-1
    )

    print(f"\n  Training UK ensemble (OOF with TimeSeriesSplit, n_splits=3)...")
    ens_uk = BlockSpecificEnsemble(base_estimator=base_estimator_uk, n_splits=3, random_state=42)
    ens_uk.fit(X_train_uk, y_train_uk_basis)

    print(f"\n  OOF RMSE    |  Block A: {ens_uk.oof_rmses_['A']:.2f}   Block B: {ens_uk.oof_rmses_['B']:.2f}   Block C: {ens_uk.oof_rmses_['C']:.2f}")
    print(f"  Weights     |  W_A: {ens_uk.weights_['A']:.3f}   W_B: {ens_uk.weights_['B']:.3f}   W_C: {ens_uk.weights_['C']:.3f}")

    # Predict on validation (basis) + add back merit order cost
    preds_uk_basis = ens_uk.predict(X_val)
    preds_uk = uk_moc_val + preds_uk_basis

    # Compute metrics before HBC
    rmse_uk = compute_rmse(uk_spot_val, preds_uk)
    mae_uk = compute_mae(uk_spot_val, preds_uk)
    huber_uk = compute_huber(uk_spot_val, preds_uk, delta=10.0)
    pinball_uk = compute_pinball(uk_spot_val, preds_uk, alpha=0.5)

    print(f"\n  Validation Metrics (on {len(uk_spot_val)} samples, BEFORE HBC)")
    print(f"    RMSE          : {rmse_uk:.2f}")
    print(f"    MAE           : {mae_uk:.2f}")
    print(f"    Huber (d=10)  : {huber_uk:.2f}")
    print(f"    Pinball (a=.5): {pinball_uk:.2f}")

    # Apply HBC
    hbc_dict_uk, rmse_uk_hbc = compute_hbc(preds_uk, uk_spot_val, hours_val)
    mae_uk_hbc = compute_mae(uk_spot_val, preds_uk + np.array([hbc_dict_uk.get(h, 0) for h in hours_val]))

    print(f"\n  Validation Metrics (on {len(uk_spot_val)} samples, AFTER HBC)")
    print(f"    RMSE          : {rmse_uk_hbc:.2f}")
    print(f"    MAE           : {mae_uk_hbc:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Final summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FINAL SUMMARY")
    print("=" * 80)
    print(f"\n  France RMSE (before HBC) : {rmse_fr:.2f}")
    print(f"  France RMSE (after HBC)  : {rmse_fr_hbc:.2f}")
    print(f"\n  UK RMSE (before HBC)     : {rmse_uk:.2f}")
    print(f"  UK RMSE (after HBC)      : {rmse_uk_hbc:.2f}")
    print(f"\n  Combined SUM (before HBC): {rmse_fr + rmse_uk:.2f}")
    print(f"  Combined SUM (after HBC) : {rmse_fr_hbc + rmse_uk_hbc:.2f}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
