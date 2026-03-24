"""Test stationary targets — make the series stationary before modeling.

Instead of predicting spot (non-stationary, regime-dependent),
predict the DEVIATION from an anchor that captures the current level:

  A) spot - spot_la                    (daily diff, D-1 same hour)
  B) spot - spot_la_roll_24h_mean      (deviation from 24h mean)
  C) spot - spot_la_roll_168h_mean     (deviation from 7d mean)
  D) A + arcsinh
  E) B + arcsinh
  F) C + arcsinh

At prediction: pred_spot = anchor + model.predict(X)

Usage: python scripts/test_stationary_target.py
"""

import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features
from catboost import CatBoostRegressor, Pool
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

print("Loading data...")
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
train_fe = build_features(train, config)

holdout_start = config["validation"]["holdout_start"]
mask_val = train_fe["datetime_CET"] >= holdout_start
df_train = train_fe[~mask_val].copy()
df_val = train_fe[mask_val].copy()

with open("outputs/shap_ranking_v3_clean.json") as f:
    clean_ranking = json.load(f)

print(f"Train: {df_train.shape}, Val: {df_val.shape}")

CB_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 5000, "learning_rate": 0.03, "depth": 8,
    "l2_leaf_reg": 5, "subsample": 0.8, "random_seed": 42,
    "verbose": 0, "allow_writing_files": False, "use_best_model": True,
}

CAT32_FR = [
    "fr_opportunity_cost", "fr_dynamic_marginal", "fr_import_price",
    "fr_scarcity_barrier", "fr_load_price_signal_7d",
    "fr_load_price_signal_load", "fr_hydro_opp_cost",
    "fr_basis_v2", "fr_basis_v2_lag_48h", "fr_basis_v2_roll_24h_mean",
    "fr_price_per_mw_7d",
]

base_feat = [f for f in clean_ranking["fr_spot"][:20] if f in df_train.columns]
extras = [f for f in CAT32_FR if f in df_train.columns and f not in base_feat]
features = base_feat + extras


def compute_weights(df, decay=2.0):
    dt = pd.to_datetime(df["datetime_CET"])
    days_ago = (dt.max() - dt).dt.total_seconds() / 86400
    return np.exp(-decay * days_ago / 365).values


def eval_all(actual, preds, label):
    rmse = np.sqrt(np.mean((actual - preds) ** 2))
    bias = np.mean(actual - preds)
    mae = np.mean(np.abs(actual - preds))
    print(f"  {label:50s}  RMSE={rmse:7.3f}  Bias={bias:+6.1f}  MAE={mae:.2f}")
    return rmse


weights = compute_weights(df_train, 2.0)
y_va_spot = df_val["fr_spot"].values

# ── Check stationarity of each target ─────────────────────────────────────
print("\n" + "=" * 70)
print("  STATIONARITY CHECK (train mean by semester)")
print("=" * 70)

dt_tr = pd.to_datetime(df_train["datetime_CET"])
semesters = dt_tr.dt.to_period("Q")

targets_to_check = {
    "spot (raw)": df_train["fr_spot"],
    "spot - spot_la": df_train["fr_spot"] - df_train["fr_spot_la"],
    "spot - roll_24h": df_train["fr_spot"] - df_train["fr_spot_la_roll_24h_mean"],
    "spot - roll_168h": df_train["fr_spot"] - df_train["fr_spot_la_roll_168h_mean"],
}

for name, series in targets_to_check.items():
    means = series.groupby(semesters).mean()
    stds = series.groupby(semesters).std()
    print(f"\n  {name}:")
    for q in means.index:
        print(f"    {q}: mean={means[q]:+8.2f}  std={stds[q]:6.2f}")

# ── Define anchor configs ─────────────────────────────────────────────────
anchor_configs = [
    ("A) spot - spot_la",               "fr_spot_la",                False),
    ("B) spot - roll_24h_mean",          "fr_spot_la_roll_24h_mean",  False),
    ("C) spot - roll_168h_mean",         "fr_spot_la_roll_168h_mean", False),
    ("D) arcsinh(spot - spot_la)",       "fr_spot_la",                True),
    ("E) arcsinh(spot - roll_24h_mean)", "fr_spot_la_roll_24h_mean",  True),
    ("F) arcsinh(spot - roll_168h)",     "fr_spot_la_roll_168h_mean", True),
]

# ── Run experiments ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  EXPERIMENTS")
print("=" * 70)

# Baseline first
print("\n--- Baseline (arcsinh spot, config N) ---")
y_tr_bl = np.arcsinh(df_train["fr_spot"])
y_va_bl = np.arcsinh(df_val["fr_spot"])
model_bl = CatBoostRegressor(**CB_PARAMS)
model_bl.fit(Pool(df_train[features], y_tr_bl, weight=weights),
             eval_set=Pool(df_val[features], y_va_bl),
             early_stopping_rounds=100, verbose=0)
preds_bl = np.sinh(model_bl.predict(df_val[features]))
rmse_bl = eval_all(y_va_spot, preds_bl, "Baseline arcsinh(spot)")

results = [("Baseline arcsinh(spot)", rmse_bl)]

for label, anchor_col, use_arcsinh in anchor_configs:
    print(f"\n--- {label} ---")

    anchor_tr = df_train[anchor_col].values
    anchor_va = df_val[anchor_col].values

    # Target = spot - anchor
    y_tr_diff = df_train["fr_spot"].values - anchor_tr
    y_va_diff = df_val["fr_spot"].values - anchor_va

    # Drop anchor from features to avoid leakage (model shouldn't see anchor directly)
    feat_here = [f for f in features if f != anchor_col]

    if use_arcsinh:
        y_tr_t = np.arcsinh(y_tr_diff)
        y_va_t = np.arcsinh(y_va_diff)
    else:
        y_tr_t = y_tr_diff
        y_va_t = y_va_diff

    # Check for NaN
    valid_tr = ~np.isnan(y_tr_t) & ~np.isnan(anchor_tr)
    valid_va = ~np.isnan(y_va_t) & ~np.isnan(anchor_va)
    if valid_tr.sum() < len(y_tr_t):
        print(f"  Dropping {(~valid_tr).sum()} NaN rows from train")

    model = CatBoostRegressor(**CB_PARAMS)
    model.fit(
        Pool(df_train.loc[valid_tr, feat_here] if valid_tr.sum() < len(y_tr_t)
             else df_train[feat_here],
             y_tr_t[valid_tr] if valid_tr.sum() < len(y_tr_t) else y_tr_t,
             weight=weights[valid_tr] if valid_tr.sum() < len(y_tr_t) else weights),
        eval_set=Pool(df_val.loc[valid_va, feat_here] if valid_va.sum() < len(y_va_t)
                      else df_val[feat_here],
                      y_va_t[valid_va] if valid_va.sum() < len(y_va_t) else y_va_t),
        early_stopping_rounds=100, verbose=0
    )

    preds_diff = model.predict(df_val[feat_here])
    if use_arcsinh:
        preds_diff = np.sinh(preds_diff)

    # Reconstruct spot prediction
    preds_spot = anchor_va + preds_diff

    rmse = eval_all(y_va_spot, preds_spot, label)
    results.append((label, rmse))

    # Bias by price bin
    errors = y_va_spot - preds_spot
    bins = [-500, 0, 20, 40, 60, 100, 5000]
    bin_labels = ["<0", "0-20", "20-40", "40-60", "60-100", ">100"]
    price_bins = pd.cut(y_va_spot, bins=bins, labels=bin_labels)
    for b in bin_labels:
        mask = price_bins == b
        if mask.sum() > 0:
            r = np.sqrt(np.mean(errors[mask] ** 2))
            bi = errors[mask].mean()
            print(f"    {b:>8s}: RMSE={r:6.2f}  Bias={bi:+6.1f}  N={mask.sum()}")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  SUMMARY — STATIONARY TARGET TESTS")
print("=" * 70)

best_rmse = min(r[1] for r in results)
for label, rmse in results:
    delta = rmse - rmse_bl
    marker = " ***" if rmse == best_rmse else ""
    print(f"  {label:50s}  RMSE={rmse:7.3f}  Δ={delta:+6.2f}{marker}")
