"""Feature sweep v4 — with stationary target + SHAP v4 ranking.

Tests: 10, 15, 20, 30, 50, 75, 100, 150, 200, all features
Also tests: hourly bias correction as post-processing
Also tests: UK with basis v1 target

Usage: python scripts/sweep_v4_stationary.py
"""

import sys, json, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features
from catboost import CatBoostRegressor, Pool
import yaml

warnings.filterwarnings("ignore")

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

with open("outputs/shap_ranking_v4_stationary.json") as f:
    v4_ranking = json.load(f)

print(f"Train: {df_train.shape}, Val: {df_val.shape}")

CB_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 5000, "learning_rate": 0.03, "depth": 8,
    "l2_leaf_reg": 5, "subsample": 0.8, "random_seed": 42,
    "verbose": 0, "allow_writing_files": False, "use_best_model": True,
}

# ── Rolling stats ─────────────────────────────────────────────────────────
fr_la = train_fe["fr_spot_la"]
roll_mean = fr_la.rolling(168, min_periods=24).mean()
roll_std = fr_la.rolling(168, min_periods=24).std()

n_tr = len(df_train)
roll_mean_tr = roll_mean.iloc[:n_tr].values
roll_std_tr = roll_std.iloc[:n_tr].values
roll_mean_va = roll_mean.iloc[n_tr:n_tr+len(df_val)].values

spot_tr = df_train["fr_spot"].values
spot_va = df_val["fr_spot"].values
hours_va = df_val["hour"].values
months_va = pd.to_datetime(df_val["datetime_CET"]).dt.to_period("M")

# FR target
y_dev_tr = spot_tr - roll_mean_tr
y_dev_va = spot_va - roll_mean_va
valid_tr = np.isfinite(y_dev_tr)
valid_va = np.isfinite(y_dev_va)

# FR weights
dt = pd.to_datetime(df_train["datetime_CET"])
days_ago = (dt.max() - dt).dt.total_seconds() / 86400
time_decay = np.exp(-2.0 * days_ago.values / 365)
var_168h = np.clip(roll_std_tr ** 2, 1.0, None)
var_168h = np.where(np.isnan(var_168h), 1.0, var_168h)
weights = time_decay / var_168h

# ══════════════════════════════════════════════════════════════════════════
# FR SWEEP
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  FR FEATURE SWEEP — stationary target (spot - roll_168h_mean)")
print("=" * 90)

fr_ranking = v4_ranking["fr_spot"]
sweep_sizes = [10, 15, 20, 30, 50, 75, 100, 150, 200, len(fr_ranking)]

results_fr = []

for n_feat in sweep_sizes:
    feat = [f for f in fr_ranking[:n_feat] if f in df_train.columns]
    actual_n = len(feat)

    X_tr = df_train.loc[df_train.index[valid_tr], feat]
    X_va = df_val.loc[df_val.index[valid_va], feat]

    model = CatBoostRegressor(**CB_PARAMS)
    model.fit(Pool(X_tr, y_dev_tr[valid_tr], weight=weights[valid_tr]),
              eval_set=Pool(X_va, y_dev_va[valid_va]),
              early_stopping_rounds=100, verbose=0)

    preds_dev = model.predict(df_val[feat])
    preds_spot = roll_mean_va + preds_dev
    rmse = np.sqrt(np.mean((spot_va - preds_spot) ** 2))
    bias = np.mean(spot_va - preds_spot)
    best_iter = model.get_best_iteration()

    # Hourly bias correction
    errors = spot_va - preds_spot
    hourly_bias = pd.Series(errors).groupby(hours_va).mean()
    hb = {h: hourly_bias.get(h, 0) for h in range(24)}
    corrected = preds_spot + np.array([hb[h] for h in hours_va])
    rmse_hbc = np.sqrt(np.mean((spot_va - corrected) ** 2))

    # Hour x month bias correction
    ym_va = months_va.values
    hm_bias = {}
    for h in range(24):
        for m in sorted(set(ym_va)):
            mask = (hours_va == h) & (ym_va == m)
            if mask.sum() > 0:
                hm_bias[(h, m)] = errors[mask].mean()
    corrected_hm = preds_spot + np.array([hm_bias.get((hours_va[i], ym_va[i]), 0) for i in range(len(preds_spot))])
    rmse_hm = np.sqrt(np.mean((spot_va - corrected_hm) ** 2))

    results_fr.append({
        "n": actual_n, "rmse": rmse, "bias": bias, "iters": best_iter,
        "rmse_hbc": rmse_hbc, "rmse_hm": rmse_hm,
    })
    print(f"  n={actual_n:3d}  RMSE={rmse:6.2f}  Bias={bias:+5.1f}  "
          f"iters={best_iter:4d}  +HBC={rmse_hbc:6.2f}  +HxM={rmse_hm:6.2f}")

# ══════════════════════════════════════════════════════════════════════════
# UK SWEEP (with basis v1 target)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  UK FEATURE SWEEP — basis v1 (spot - merit_order_cost)")
print("=" * 90)

uk_ranking = v4_ranking["uk_spot"]
uk_spot_tr = df_train["uk_spot"].values
uk_spot_va = df_val["uk_spot"].values
uk_merit_tr = df_train["uk_merit_order_cost"].values
uk_merit_va = df_val["uk_merit_order_cost"].values

# UK basis target
uk_basis_tr = uk_spot_tr - uk_merit_tr
uk_basis_va = uk_spot_va - uk_merit_va

results_uk = []

for n_feat in sweep_sizes:
    feat = [f for f in uk_ranking[:n_feat] if f in df_train.columns]
    actual_n = len(feat)

    # Test both raw and basis
    for target_label, y_tr_uk, y_va_uk, reconstruct in [
        ("raw", uk_spot_tr, uk_spot_va, lambda p: p),
        ("basis", uk_basis_tr, uk_basis_va, lambda p: uk_merit_va + p),
    ]:
        model = CatBoostRegressor(**CB_PARAMS)
        model.fit(Pool(df_train[feat], y_tr_uk),
                  eval_set=Pool(df_val[feat], y_va_uk),
                  early_stopping_rounds=100, verbose=0)

        preds = reconstruct(model.predict(df_val[feat]))
        rmse = np.sqrt(np.mean((uk_spot_va - preds) ** 2))
        bias = np.mean(uk_spot_va - preds)

        results_uk.append({
            "n": actual_n, "target": target_label, "rmse": rmse,
            "bias": bias, "iters": model.get_best_iteration(),
        })
        print(f"  n={actual_n:3d}  {target_label:5s}  RMSE={rmse:6.2f}  "
              f"Bias={bias:+5.1f}  iters={model.get_best_iteration():4d}")

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  SUMMARY — BEST CONFIGS")
print("=" * 90)

best_fr = min(results_fr, key=lambda r: r["rmse"])
best_fr_hm = min(results_fr, key=lambda r: r["rmse_hm"])
best_uk_raw = min([r for r in results_uk if r["target"] == "raw"], key=lambda r: r["rmse"])
best_uk_basis = min([r for r in results_uk if r["target"] == "basis"], key=lambda r: r["rmse"])

print(f"\n  FR best (no post):  n={best_fr['n']:3d}  RMSE={best_fr['rmse']:.3f}")
print(f"  FR best (+HxM):    n={best_fr_hm['n']:3d}  RMSE={best_fr_hm['rmse_hm']:.3f}")
print(f"  UK best (raw):     n={best_uk_raw['n']:3d}  RMSE={best_uk_raw['rmse']:.3f}")
print(f"  UK best (basis):   n={best_uk_basis['n']:3d}  RMSE={best_uk_basis['rmse']:.3f}")

# Cherry-pick combined
combo_raw = (best_fr['rmse'] + best_uk_raw['rmse']) / 2
combo_basis = (best_fr['rmse'] + best_uk_basis['rmse']) / 2
combo_hm = (best_fr_hm['rmse_hm'] + best_uk_basis['rmse']) / 2

print(f"\n  Combined (FR + UK raw):    {combo_raw:.3f}")
print(f"  Combined (FR + UK basis):  {combo_basis:.3f}")
print(f"  Combined (FR+HxM + UK):    {combo_hm:.3f}")
