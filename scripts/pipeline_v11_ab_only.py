#!/usr/bin/env python
"""Block A+B Only (No Interconnectors): Simplified Ensemble.

Drop Block C (interconnectors) and use only:
  - Block A: Calendar/AR signals
  - Block B: Physical fundamentals
  - 3 models per block
  - Inverse-RMSE weighting
  - HBC post-processing
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
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    HAS_MODELS = True
except ImportError:
    HAS_MODELS = False


def train_block_ensemble(X_block, y, loss_fn="RMSE"):
    """Train 3 diverse models."""
    cb = CatBoostRegressor(depth=6, learning_rate=0.05, iterations=300,
                           loss_function=loss_fn, verbose=0, random_state=42, thread_count=-1)
    cb.fit(X_block, y)

    lgb = LGBMRegressor(max_depth=7, num_leaves=63, learning_rate=0.05, n_estimators=300,
                        random_state=42, n_jobs=-1, verbose=-1)
    lgb.fit(X_block, y)

    xgb = XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=300,
                       random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(X_block, y)

    return {"CB": cb, "LGB": lgb, "XGB": xgb}


def predict_ensemble(models, X_block):
    preds = np.array([m.predict(X_block) for m in models.values()])
    return np.mean(preds, axis=0)


def get_blocks_ab_only(all_features):
    """Get only blocks A and B, drop C."""
    block_a = []
    block_b = []

    # Block C patterns (to exclude)
    c_keywords = ("atc_", "ntc_", "_atc_", "_ntc_", "_flow_la", "_avg_cost_la",
                  "_cost_spread_la", "_uk_utilization", "_fr_utilization", "_uk_congested",
                  "any_direction_congested", "_unused_capacity", "flow_over_atc")

    # Block B patterns
    b_keywords = ("_f", "_residual_load", "_thermal_need", "_thermal_floor", "_nuclear_shortfall",
                  "_spark_spread", "_merit_order_cost", "_scarcity_ratio", "_baseload_gap",
                  "_supply_demand_ratio", "_security_margin", "_wind_pen", "_solar_pen",
                  "de_gas", "fr_gas", "uk_gas", "es_gas", "nl_gas", "eu_emission", "uk_emission")

    for feat in all_features:
        # Skip Block C
        if any(kw in feat for kw in c_keywords) or feat.startswith(("atc_", "ntc_")):
            continue

        # Check Block B
        is_b = any(kw in feat for kw in b_keywords)

        if is_b:
            block_b.append(feat)
        else:
            block_a.append(feat)

    return block_a, block_b


def main():
    print("\n" + "=" * 80)
    print("  Block A+B Only: Multi-Model Ensemble + HBC (No Interconnectors)")
    print("=" * 80)

    if not HAS_MODELS:
        print("\nERROR: Missing catboost, lightgbm, xgboost")
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

    block_a, block_b = get_blocks_ab_only(numeric_cols)
    print(f"  Features: {len(numeric_cols)} total (A={len(block_a)}, B={len(block_b)}, C=dropped)")

    # ─── FRANCE ───
    print("\n" + "-" * 80)
    print("  FRANCE")
    print("-" * 80)

    fr_stat = prepare_stationary("fr_spot_la", "fr_spot", train_fe, df_train, df_val)
    y_train_fr_dev = fr_stat["y_dev_tr"][fr_stat["valid_tr"]]
    X_train_fr = X_train[fr_stat["valid_tr"]].copy()
    y_val_fr_actual = fr_stat["spot_va"]

    print(f"  Training 2 x 3 models (6 total)...")

    X_train_a = X_train_fr[[f for f in block_a if f in X_train_fr.columns]]
    X_train_b = X_train_fr[[f for f in block_b if f in X_train_fr.columns]]

    models_a_fr = train_block_ensemble(X_train_a, y_train_fr_dev, "RMSE")
    models_b_fr = train_block_ensemble(X_train_b, y_train_fr_dev, "RMSE")

    X_val_a = X_val[[f for f in block_a if f in X_val.columns]]
    X_val_b = X_val[[f for f in block_b if f in X_val.columns]]

    pred_a_fr = predict_ensemble(models_a_fr, X_val_a)
    pred_b_fr = predict_ensemble(models_b_fr, X_val_b)

    train_a = predict_ensemble(models_a_fr, X_train_a)
    train_b = predict_ensemble(models_b_fr, X_train_b)

    rmse_a_fr = compute_rmse(y_train_fr_dev, train_a)
    rmse_b_fr = compute_rmse(y_train_fr_dev, train_b)

    inv_rmses = np.array([1.0 / rmse_a_fr, 1.0 / rmse_b_fr])
    w_a, w_b = inv_rmses / inv_rmses.sum()

    print(f"  Block RMSE: A={rmse_a_fr:.2f}  B={rmse_b_fr:.2f}")
    print(f"  Weights:    A={w_a:.3f}  B={w_b:.3f}")

    pred_fr_dev = w_a * pred_a_fr + w_b * pred_b_fr
    preds_fr = fr_stat["rm_va"] + pred_fr_dev

    rmse_fr = compute_rmse(y_val_fr_actual, preds_fr)
    print(f"  Val RMSE (before HBC): {rmse_fr:.2f}")

    hbc_dict_fr, rmse_fr_hbc = compute_hbc(preds_fr, y_val_fr_actual, hours_val)
    print(f"  Val RMSE (after HBC):  {rmse_fr_hbc:.2f}")

    # ─── UK ───
    print("\n" + "-" * 80)
    print("  UNITED KINGDOM")
    print("-" * 80)

    uk_spot_train = df_train["uk_spot"].values
    uk_moc_train = df_train["uk_merit_order_cost"].values
    uk_spot_val = df_val["uk_spot"].values
    uk_moc_val = df_val["uk_merit_order_cost"].values

    y_train_uk_basis = uk_spot_train - uk_moc_train
    valid_basis_tr = np.isfinite(y_train_uk_basis)
    X_train_uk = X_train[valid_basis_tr].copy()
    y_train_uk_basis = y_train_uk_basis[valid_basis_tr]

    print(f"  Training 2 x 3 models (6 total)...")

    X_train_a = X_train_uk[[f for f in block_a if f in X_train_uk.columns]]
    X_train_b = X_train_uk[[f for f in block_b if f in X_train_uk.columns]]

    models_a_uk = train_block_ensemble(X_train_a, y_train_uk_basis, "MAE")
    models_b_uk = train_block_ensemble(X_train_b, y_train_uk_basis, "MAE")

    X_val_a = X_val[[f for f in block_a if f in X_val.columns]]
    X_val_b = X_val[[f for f in block_b if f in X_val.columns]]

    pred_a_uk = predict_ensemble(models_a_uk, X_val_a)
    pred_b_uk = predict_ensemble(models_b_uk, X_val_b)

    train_a = predict_ensemble(models_a_uk, X_train_a)
    train_b = predict_ensemble(models_b_uk, X_train_b)

    rmse_a_uk = compute_rmse(y_train_uk_basis, train_a)
    rmse_b_uk = compute_rmse(y_train_uk_basis, train_b)

    inv_rmses = np.array([1.0 / rmse_a_uk, 1.0 / rmse_b_uk])
    w_a, w_b = inv_rmses / inv_rmses.sum()

    print(f"  Block RMSE: A={rmse_a_uk:.2f}  B={rmse_b_uk:.2f}")
    print(f"  Weights:    A={w_a:.3f}  B={w_b:.3f}")

    pred_uk_basis = w_a * pred_a_uk + w_b * pred_b_uk
    preds_uk = uk_moc_val + pred_uk_basis

    rmse_uk = compute_rmse(uk_spot_val, preds_uk)
    print(f"  Val RMSE (before HBC): {rmse_uk:.2f}")

    hbc_dict_uk, rmse_uk_hbc = compute_hbc(preds_uk, uk_spot_val, hours_val)
    print(f"  Val RMSE (after HBC):  {rmse_uk_hbc:.2f}")

    # Summary
    print("\n" + "=" * 80)
    total = rmse_fr_hbc + rmse_uk_hbc
    print(f"  FR: {rmse_fr_hbc:.2f}  |  UK: {rmse_uk_hbc:.2f}  |  COMBINED: {total:.2f}")
    if total < 24:
        print(f"  ** TARGET HIT: {24 - total:.2f} points under! **")
    else:
        print(f"  Gap to target 24: +{total - 24:.2f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
