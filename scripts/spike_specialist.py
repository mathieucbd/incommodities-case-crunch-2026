"""Spike specialist experiment -- FR negative-price-risk regime.

Tests three approaches to handle low/negative price hours:

  A. Upweighted training  -- base CB with spike hours weighted xW
  B. Specialist blend     -- separate CB trained only on spike hours,
                            prediction replaced for flagged val hours
  C. Alpha blend sweep    -- mix (base * (1-a) + specialist * a) for flagged hours

Spike hours defined two ways:
  - Price-based flag  : fr_spot < P15 of training prices (~low/negative price regime)
  - Feature-based flag: fr_daily_residual_load_min < P25 of training (available at pred time)

Reports RMSE split: overall / spike-hours / normal-hours for each variant.

Usage: python scripts/spike_specialist.py
"""

import sys, json, warnings
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, ".")
from src.data_loading import load_data
from src.feature_engineering import build_features
from src.models import compute_rmse, compute_hbc
from src.models.targets import prepare_stationary
from src.models.tree_models import train_tree, predict_tree

warnings.filterwarnings("ignore")

# -- Config & data -----------------------------------------------------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)

x_train, y_train, x_test = load_data("data/raw")
train_fe = build_features(x_train.copy(), config)
train_fe = train_fe.join(y_train[["fr_spot", "uk_spot"]])

holdout_start = config["validation"]["holdout_start"]
mask_val = train_fe["datetime_CET"] >= holdout_start
df_train = train_fe[~mask_val].copy()
df_val   = train_fe[mask_val].copy()

for df in [df_train, df_val]:
    if "fr_spot_la_roll_168h_mean" in df.columns and "uk_price_per_mw_7d" in df.columns:
        df["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
            df["fr_spot_la_roll_168h_mean"] * df["uk_price_per_mw_7d"]
        )

# -- FR stationary target ----------------------------------------------------
fr_stat = prepare_stationary("fr_spot_la", "fr_spot", train_fe, df_train, df_val)
hours_va = df_val["hour"].values

# -- Feature list ------------------------------------------------------------
with open("data/outputs/feature_selection_v5_fr.json") as f:
    fs_v5 = json.load(f)
feat_fr_27 = fs_v5["features"]
feat_fr_28 = feat_fr_27 + ["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"]
feat_fr = [f for f in feat_fr_28 if f in df_train.columns]

# -- CB params (same as pipeline stage 1) ------------------------------------
FR_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 15000,
    "learning_rate": 0.059,
    "depth": 3,
    "l2_leaf_reg": 4.42,
    "subsample": 0.533,
    "colsample_bylevel": 0.228,
    "min_child_samples": 14,
    "random_strength": 0.9,
    "random_seed": 42,
    "verbose": 0,
    "allow_writing_files": False,
    "use_best_model": True,
}

valid_tr = fr_stat["valid_tr"]
valid_va = fr_stat["valid_va"]
X_tr = df_train.loc[df_train.index[valid_tr], feat_fr]
y_tr = fr_stat["y_dev_tr"][valid_tr]
w_tr = fr_stat["weights"][valid_tr]
X_va = df_val.loc[df_val.index[valid_va], feat_fr]
y_va = fr_stat["y_dev_va"][valid_va]
spot_va = fr_stat["spot_va"][valid_va]
rm_va   = fr_stat["rm_va"][valid_va]
spot_tr = fr_stat["spot_tr"][valid_tr]

# -- Diagnostics: feature distributions ------------------------------------
resid_min_tr = df_train["fr_daily_residual_load_min"].values[valid_tr]
resid_min_va = df_val["fr_daily_residual_load_min"].values[valid_va]

print("=" * 70)
print("  SPIKE SPECIALIST EXPERIMENT -- FR")
print("=" * 70)
print(f"\n  fr_spot distribution in train ({len(spot_tr)} hours):")
for p in [1, 5, 10, 15, 25]:
    print(f"    P{p:2d}: {np.percentile(spot_tr, p):8.1f} EUR/MWh")
print(f"    min: {spot_tr.min():.1f}  max: {spot_tr.max():.1f}")

print(f"\n  fr_daily_residual_load_min distribution in train:")
for p in [5, 10, 15, 20, 25, 50]:
    print(f"    P{p:2d}: {np.percentile(resid_min_tr, p):8.0f} MW")
print(f"    min: {resid_min_tr.min():.0f}  max: {resid_min_tr.max():.0f}")

# -- Spike flags (data-driven thresholds) ------------------------------------
# Price-based: low price hours (fr_spot < P15 of training) -- used for train labeling
PRICE_THRESHOLD = np.percentile(spot_tr, 15)
price_flag_tr = spot_tr < PRICE_THRESHOLD
price_flag_va = spot_va < PRICE_THRESHOLD  # how many val hours actually had low prices

# Feature-based: low residual load day (fr_daily_residual_load_min < P25 of train)
# This is what we can use at inference time (no future info needed)
RESID_THRESHOLD = np.percentile(resid_min_tr, 25)
resid_flag_tr = resid_min_tr < RESID_THRESHOLD
resid_flag_va = resid_min_va < RESID_THRESHOLD

print(f"\n  Spike flags:")
print(f"    Price < P15 ({PRICE_THRESHOLD:.1f} EUR/MWh)   train: {price_flag_tr.sum()} ({100*price_flag_tr.mean():.1f}%)   val: {price_flag_va.sum()} ({100*price_flag_va.mean():.1f}%)")
print(f"    Resid < P25 ({RESID_THRESHOLD:.0f} MW) train: {resid_flag_tr.sum()} ({100*resid_flag_tr.mean():.1f}%)   val: {resid_flag_va.sum()} ({100*resid_flag_va.mean():.1f}%)")


def rmse_split(actual, pred, flag, label):
    overall = compute_rmse(actual, pred)
    spike  = compute_rmse(actual[flag], pred[flag]) if flag.sum() > 0 else float("nan")
    normal = compute_rmse(actual[~flag], pred[~flag]) if (~flag).sum() > 0 else float("nan")
    print(f"  {label:<50} overall={overall:.2f}  spike={spike:.2f}  normal={normal:.2f}")
    return overall


# =========================================================================
# BASELINE
# =========================================================================
print("\n-- BASELINE " + "-" * 58)
cb_base = train_tree("catboost", FR_PARAMS, X_tr, y_tr, X_va, y_va,
                     sample_weight=w_tr)
pred_base = rm_va + predict_tree(cb_base.model, X_va)
_, rmse_base_hbc = compute_hbc(pred_base, spot_va, hours_va[valid_va])
base_overall = rmse_split(spot_va, pred_base, resid_flag_va,
                          f"Baseline (iter={cb_base.best_iteration})")
print(f"  {'+ HBC':50} {rmse_base_hbc:.2f}")

# =========================================================================
# EXPERIMENT A -- upweighted training (price-based flag for training)
# =========================================================================
print("\n-- EXPERIMENT A: upweighted training (price flag) " + "-" * 20)
for spike_w in [3, 5, 10]:
    w_spike = w_tr.copy()
    w_spike[price_flag_tr] *= spike_w
    cb_w = train_tree("catboost", FR_PARAMS, X_tr, y_tr, X_va, y_va,
                      sample_weight=w_spike)
    pred_w = rm_va + predict_tree(cb_w.model, X_va)
    overall = rmse_split(spot_va, pred_w, resid_flag_va,
                         f"Upweight x{spike_w} (iter={cb_w.best_iteration})")
    print(f"  {'delta vs baseline':50} {overall - base_overall:+.2f}")


# =========================================================================
# EXPERIMENT B -- specialist model
# =========================================================================
print("\n-- EXPERIMENT B: specialist model " + "-" * 36)

# Train specialist on low-price hours, apply to low-resid-load val hours
for flag_name, flag_tr, flag_va in [
    ("price-trained / resid-applied", price_flag_tr, resid_flag_va),
    ("price-trained / price-applied (oracle)", price_flag_tr, price_flag_va),
    ("resid-trained / resid-applied", resid_flag_tr, resid_flag_va),
]:
    n_tr_sp = flag_tr.sum()
    n_va_sp = flag_va.sum()
    if n_tr_sp < 200 or n_va_sp < 10:
        print(f"  Skipping {flag_name}: too few samples (tr={n_tr_sp}, va={n_va_sp})")
        continue

    sp_params = {**FR_PARAMS, "iterations": 5000}
    cb_sp = train_tree("catboost", sp_params,
                       X_tr[flag_tr], y_tr[flag_tr],
                       X_va[flag_va], y_va[flag_va],
                       sample_weight=w_tr[flag_tr])

    pred_blend = pred_base.copy()
    pred_blend[flag_va] = rm_va[flag_va] + predict_tree(cb_sp.model, X_va[flag_va])

    overall = rmse_split(spot_va, pred_blend, flag_va,
                         f"Specialist/{flag_name}")
    print(f"  {'delta vs baseline':50} {overall - base_overall:+.2f}")


# =========================================================================
# EXPERIMENT C -- alpha blend sweep (price-trained, resid-applied)
# =========================================================================
print("\n-- EXPERIMENT C: alpha blend sweep " + "-" * 35)
if price_flag_tr.sum() >= 200 and resid_flag_va.sum() >= 10:
    sp_params = {**FR_PARAMS, "iterations": 5000}
    cb_sp = train_tree("catboost", sp_params,
                       X_tr[price_flag_tr], y_tr[price_flag_tr],
                       X_va[resid_flag_va], y_va[resid_flag_va],
                       sample_weight=w_tr[price_flag_tr])
    pred_sp_va_flag = rm_va[resid_flag_va] + predict_tree(cb_sp.model, X_va[resid_flag_va])
    pred_base_flag  = pred_base[resid_flag_va]

    best_alpha, best_rmse = 0.0, base_overall
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        pred_blend = pred_base.copy()
        pred_blend[resid_flag_va] = (1 - alpha) * pred_base_flag + alpha * pred_sp_va_flag
        overall = compute_rmse(spot_va, pred_blend)
        marker = " <-- best" if overall < best_rmse else ""
        print(f"  alpha={alpha:.1f}  overall={overall:.2f}  delta={overall - base_overall:+.2f}{marker}")
        if overall < best_rmse:
            best_rmse, best_alpha = overall, alpha
    print(f"\n  Best alpha={best_alpha}  RMSE={best_rmse:.2f}  (baseline={base_overall:.2f}  delta={best_rmse - base_overall:+.2f})")
else:
    print("  Skipping: too few spike hours")

print()
