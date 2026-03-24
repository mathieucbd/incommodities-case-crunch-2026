"""UK model tuning v1 — Feature selection + hyperparameter optimization.

Applies the same methodology that worked for FR:
  - Stationary target: uk_spot - uk_spot_la_roll_168h_mean
  - Time decay + variance-aware weights
  - Noise probing (Boruta) → Feature count sweep → Depth/LR tuning
  - HBC (hourly bias correction)

Usage: cd "INCOMO 3" && python scripts/uk_tuning_v1.py
"""

import sys, json, time, warnings
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

# ── Load data ─────────────────────────────────────────────────────────────
print("=" * 90)
print("  UK MODEL TUNING v1 — Stationary target + feature selection + hyperparam")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
train_fe = build_features(train, config)
print(f"  Data loaded in {time.time() - t0:.0f}s — shape: {train_fe.shape}")

holdout_start = config["validation"]["holdout_start"]
mask_val = train_fe["datetime_CET"] >= holdout_start
df_train = train_fe[~mask_val].copy()
df_val = train_fe[mask_val].copy()

# ── Rolling stats for UK ──────────────────────────────────────────────────
uk_la = train_fe["uk_spot_la"]
roll_mean = uk_la.rolling(168, min_periods=24).mean()
roll_std = uk_la.rolling(168, min_periods=24).std()

n_tr = len(df_train)
roll_mean_tr = roll_mean.iloc[:n_tr].values
roll_std_tr = roll_std.iloc[:n_tr].values
roll_mean_va = roll_mean.iloc[n_tr:n_tr + len(df_val)].values

spot_tr = df_train["uk_spot"].values
spot_va = df_val["uk_spot"].values
hours_va = df_val["hour"].values

# ── Target & weights ──────────────────────────────────────────────────────
y_dev_tr = spot_tr - roll_mean_tr
y_dev_va = spot_va - roll_mean_va
valid_tr = np.isfinite(y_dev_tr)
valid_va = np.isfinite(y_dev_va)

dt = pd.to_datetime(df_train["datetime_CET"])
days_ago = (dt.max() - dt).dt.total_seconds() / 86400
time_decay = np.exp(-2.0 * days_ago.values / 365)
var_168h = np.clip(roll_std_tr ** 2, 1.0, None)
var_168h = np.where(np.isnan(var_168h), 1.0, var_168h)
weights = time_decay / var_168h

# ── Features (SHAP v4 ranking for uk_spot) ────────────────────────────────
with open("outputs/shap_ranking_v4_stationary.json") as f:
    v4_ranking = json.load(f)

feat_all_uk = [f for f in v4_ranking["uk_spot"] if f in train_fe.columns]
print(f"  UK features available: {len(feat_all_uk)}")


def train_eval(feat_list, params, label=""):
    """Train CatBoost, return (rmse, best_iter, bias, rmse_hbc, model)."""
    feat = [f for f in feat_list if f in df_train.columns]
    if len(feat) == 0:
        return 999, 0, 0, 999, None

    model = CatBoostRegressor(**params)
    model.fit(
        Pool(df_train.loc[df_train.index[valid_tr], feat], y_dev_tr[valid_tr],
             weight=weights[valid_tr]),
        eval_set=Pool(df_val.loc[df_val.index[valid_va], feat], y_dev_va[valid_va]),
        early_stopping_rounds=200, verbose=0,
    )

    preds_dev = model.predict(df_val[feat])
    preds_spot = roll_mean_va + preds_dev
    rmse = np.sqrt(np.mean((spot_va - preds_spot) ** 2))
    bias = np.mean(spot_va - preds_spot)
    best_iter = model.get_best_iteration()

    # HBC
    errors = spot_va - preds_spot
    hb = {}
    for h in range(24):
        mask = hours_va == h
        if mask.sum() > 0:
            hb[h] = errors[mask].mean()
    corrected = preds_spot + np.array([hb.get(h, 0) for h in hours_va])
    rmse_hbc = np.sqrt(np.mean((spot_va - corrected) ** 2))

    if label:
        print(f"  {label:55s}  n={len(feat):3d}  RMSE={rmse:6.2f}  +HBC={rmse_hbc:5.2f}  "
              f"iter={best_iter:5d}  bias={bias:+.1f}")

    return rmse, best_iter, bias, rmse_hbc, model


# ══════════════════════════════════════════════════════════════════════════
# STAGE A — BASELINE WITH OLD PARAMS (depth=8)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  STAGE A — BASELINE (depth=8, old params)")
print("=" * 90)

OLD_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 5000, "learning_rate": 0.03, "depth": 8,
    "l2_leaf_reg": 5, "subsample": 0.8, "random_seed": 42,
    "verbose": 0, "allow_writing_files": False, "use_best_model": True,
}

for n in [20, 30, 50, 75, 100, len(feat_all_uk)]:
    if n > len(feat_all_uk):
        continue
    feat = feat_all_uk[:n]
    train_eval(feat, OLD_PARAMS, label=f"old d=8 top-{n}")


# ══════════════════════════════════════════════════════════════════════════
# STAGE B — DEPTH SWEEP
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  STAGE B — DEPTH SWEEP (lr=0.03)")
print("=" * 90)

best_depth_rmse = 999
best_depth = 8

for d in [3, 4, 5, 6, 7, 8]:
    params = {
        "loss_function": "RMSE", "eval_metric": "RMSE",
        "iterations": 15000, "learning_rate": 0.03, "depth": d,
        "l2_leaf_reg": 5, "random_seed": 42,
        "verbose": 0, "allow_writing_files": False, "use_best_model": True,
    }
    rmse, b_iter, bias, rmse_hbc, _ = train_eval(feat_all_uk[:50], params, label=f"depth={d}")
    if rmse < best_depth_rmse:
        best_depth_rmse = rmse
        best_depth = d

print(f"\n  Best depth: {best_depth} (RMSE={best_depth_rmse:.2f})")


# ══════════════════════════════════════════════════════════════════════════
# STAGE C — LR × DEPTH × FEATURES GRID
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  STAGE C — LR × DEPTH × FEATURES GRID")
print("=" * 90)

best_grid = {"rmse": 999}
depths = [best_depth - 1, best_depth, best_depth + 1] if best_depth > 3 else [3, 4, 5]
lrs = [0.01, 0.02, 0.03, 0.05]
n_feats = [20, 30, 50, 75]

for d in depths:
    for lr in lrs:
        for n in n_feats:
            if n > len(feat_all_uk):
                continue
            params = {
                "loss_function": "RMSE", "eval_metric": "RMSE",
                "iterations": 15000, "learning_rate": lr, "depth": d,
                "l2_leaf_reg": 5, "random_seed": 42,
                "verbose": 0, "allow_writing_files": False, "use_best_model": True,
            }
            rmse, b_iter, bias, rmse_hbc, _ = train_eval(
                feat_all_uk[:n], params,
                label=f"d={d} lr={lr} n={n}" if rmse_hbc < 999 else ""
            )
            if rmse < best_grid.get("rmse", 999):
                best_grid = {"rmse": rmse, "rmse_hbc": rmse_hbc, "depth": d,
                             "lr": lr, "n_feat": n, "iter": b_iter, "bias": bias}

print(f"\n  Best grid: d={best_grid['depth']} lr={best_grid['lr']} n={best_grid['n_feat']} "
      f"→ RMSE={best_grid['rmse']:.2f} +HBC={best_grid['rmse_hbc']:.2f} iter={best_grid['iter']}")


# ══════════════════════════════════════════════════════════════════════════
# STAGE D — REGULARIZATION TUNING
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  STAGE D — REGULARIZATION TUNING")
print("=" * 90)

best_d = best_grid["depth"]
best_lr = best_grid["lr"]
best_n = best_grid["n_feat"]

best_reg = {"rmse": 999}

for l2 in [1, 3, 5, 10, 20, 30, 50]:
    for csbl in [0.3, 0.5, 0.7, 1.0]:
        for ss in [0.6, 0.7, 0.8, 1.0]:
            params = {
                "loss_function": "RMSE", "eval_metric": "RMSE",
                "iterations": 15000, "learning_rate": best_lr, "depth": best_d,
                "l2_leaf_reg": l2, "colsample_bylevel": csbl, "subsample": ss,
                "random_seed": 42, "verbose": 0, "allow_writing_files": False,
                "use_best_model": True,
            }
            rmse, b_iter, bias, rmse_hbc, _ = train_eval(feat_all_uk[:best_n], params)
            if rmse < best_reg.get("rmse", 999):
                best_reg = {"rmse": rmse, "rmse_hbc": rmse_hbc, "l2": l2,
                            "csbl": csbl, "ss": ss, "iter": b_iter, "bias": bias}

print(f"  Best reg: l2={best_reg['l2']} csbl={best_reg['csbl']} ss={best_reg['ss']} "
      f"→ RMSE={best_reg['rmse']:.2f} +HBC={best_reg['rmse_hbc']:.2f}")


# ══════════════════════════════════════════════════════════════════════════
# STAGE E — NOISE PROBING (BORUTA) WITH BEST PARAMS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  STAGE E — NOISE PROBING (Boruta) for UK")
print("=" * 90)

BEST_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 15000, "learning_rate": best_lr, "depth": best_d,
    "l2_leaf_reg": best_reg["l2"], "colsample_bylevel": best_reg["csbl"],
    "subsample": best_reg["ss"],
    "random_seed": 42, "verbose": 0, "allow_writing_files": False,
    "use_best_model": True,
}

N_NOISE = 20
N_ROUNDS = 5
feat_to_test = feat_all_uk[:min(100, len(feat_all_uk))]
hit_counts = {f: 0 for f in feat_to_test}

for round_i in range(N_ROUNDS):
    np.random.seed(round_i * 37 + 7)
    noise_cols = [f"_noise_{i}" for i in range(N_NOISE)]

    df_tr_n = df_train.copy()
    df_va_n = df_val.copy()
    for nc in noise_cols:
        df_tr_n[nc] = np.random.randn(len(df_tr_n))
        df_va_n[nc] = np.random.randn(len(df_va_n))

    all_f = feat_to_test + noise_cols
    params_boruta = {**BEST_PARAMS, "random_seed": round_i, "iterations": 3000}
    model = CatBoostRegressor(**params_boruta)
    model.fit(
        Pool(df_tr_n.loc[df_tr_n.index[valid_tr], all_f], y_dev_tr[valid_tr],
             weight=weights[valid_tr]),
        eval_set=Pool(df_va_n.loc[df_va_n.index[valid_va], all_f], y_dev_va[valid_va]),
        early_stopping_rounds=100, verbose=0,
    )

    importances = model.feature_importances_
    imp_dict = dict(zip(all_f, importances))
    noise_max = max(imp_dict[nc] for nc in noise_cols)

    for f in feat_to_test:
        if imp_dict.get(f, 0) > noise_max:
            hit_counts[f] += 1

    above = sum(1 for f in feat_to_test if imp_dict.get(f, 0) > noise_max)
    print(f"  Round {round_i+1}/{N_ROUNDS}: noise_max={noise_max:.4f}, above={above}/{len(feat_to_test)}")

confirmed = [f for f in feat_to_test if hit_counts[f] >= 3]
tentative = [f for f in feat_to_test if hit_counts[f] == 2]
rejected = [f for f in feat_to_test if hit_counts[f] <= 1]
print(f"\n  Confirmed: {len(confirmed)}, Tentative: {len(tentative)}, Rejected: {len(rejected)}")


# ══════════════════════════════════════════════════════════════════════════
# STAGE F — FEATURE COUNT SWEEP WITH BEST PARAMS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  STAGE F — FEATURE COUNT SWEEP (confirmed + sweep)")
print("=" * 90)

# Test confirmed only
train_eval(confirmed, BEST_PARAMS, label=f"Confirmed ({len(confirmed)} feat)")

# Sweep by adding tentative
for n_tent in [5, 10, len(tentative)]:
    feat = confirmed + tentative[:n_tent]
    train_eval(feat, BEST_PARAMS, label=f"Confirmed + {min(n_tent, len(tentative))} tentative")

# Also test SHAP-ranked sweep
best_final = {"rmse": 999}
for n in [15, 20, 25, 30, 35, 40, 50, 60, 75, 100]:
    if n > len(feat_all_uk):
        continue
    rmse, b_iter, bias, rmse_hbc, _ = train_eval(feat_all_uk[:n], BEST_PARAMS,
                                                   label=f"SHAP top-{n}")
    if rmse < best_final["rmse"]:
        best_final = {"rmse": rmse, "rmse_hbc": rmse_hbc, "n_feat": n,
                      "iter": b_iter, "features": feat_all_uk[:n]}


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  UK TUNING SUMMARY")
print("=" * 90)

print(f"\n  Best hyperparameters:")
print(f"    depth:               {best_d}")
print(f"    learning_rate:       {best_lr}")
print(f"    l2_leaf_reg:         {best_reg['l2']}")
print(f"    colsample_bylevel:   {best_reg['csbl']}")
print(f"    subsample:           {best_reg['ss']}")

print(f"\n  Best feature count:    {best_final['n_feat']}")
print(f"  Best RMSE:             {best_final['rmse']:.2f}")
print(f"  Best RMSE+HBC:         {best_final['rmse_hbc']:.2f}")

# Save
output = {
    "target": "uk_spot",
    "method": "uk_tuning_v1",
    "best_params": {
        "depth": best_d, "learning_rate": best_lr,
        "l2_leaf_reg": best_reg["l2"],
        "colsample_bylevel": best_reg["csbl"],
        "subsample": best_reg["ss"],
        "iterations": 15000,
    },
    "best_rmse": best_final["rmse"],
    "best_rmse_hbc": best_final["rmse_hbc"],
    "n_features": best_final["n_feat"],
    "features": best_final["features"],
    "noise_probing": {
        "confirmed": len(confirmed),
        "tentative": len(tentative),
        "rejected": len(rejected),
        "confirmed_features": confirmed,
    },
}

with open("outputs/uk_tuning_v1.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved to outputs/uk_tuning_v1.json")
print(f"  Total time: {time.time() - t0:.0f}s")
