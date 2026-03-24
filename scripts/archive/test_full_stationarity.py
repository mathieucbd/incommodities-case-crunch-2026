"""Test methods to achieve full stationarity (mean + variance).

Problem: spot - roll_168h makes the mean stationary but variance is still 4.2x.
Z-score fixes variance but reconstruction (pred * std) amplifies errors.

Solutions tested:
  A) Variance-aware weights: weight = exp_decay / rolling_std
  B) Power normalization: (spot - mean) / std^alpha, alpha in [0, 0.25, 0.5, 0.75, 1.0]
  C) Truncate train (remove 2022 crisis) + differencing
  D) Combined: truncate + variance-aware weights

Usage: python scripts/test_full_stationarity.py
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
df_train_full = train_fe[~mask_val].copy()
df_val = train_fe[mask_val].copy()

with open("outputs/shap_ranking_v3_clean.json") as f:
    clean_ranking = json.load(f)

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

base_feat = [f for f in clean_ranking["fr_spot"][:20] if f in df_train_full.columns]
extras = [f for f in CAT32_FR if f in df_train_full.columns and f not in base_feat]
features = base_feat + extras

spot_va = df_val["fr_spot"].values
months_va = pd.to_datetime(df_val["datetime_CET"]).dt.to_period("M")

# Rolling stats on full dataset
full_la = train_fe["fr_spot_la"]
n_full_tr = len(df_train_full)

roll_168_mean_full = full_la.rolling(168, min_periods=24).mean().values
roll_168_std_full = full_la.rolling(168, min_periods=24).std().values

roll_168_mean_va = roll_168_mean_full[n_full_tr:n_full_tr + len(df_val)]
roll_168_std_va = np.clip(roll_168_std_full[n_full_tr:n_full_tr + len(df_val)], 1.0, None)


def run_model(label, df_train, y_tr, y_va_t, w, reconstruct_fn, feat=None):
    """Train and evaluate."""
    if feat is None:
        feat = features

    valid_tr = np.isfinite(y_tr)
    valid_va = np.isfinite(y_va_t)

    if valid_tr.sum() < 100:
        print(f"  {label:60s}  SKIPPED")
        return None

    X_tr = df_train.loc[valid_tr, feat] if valid_tr.sum() < len(y_tr) else df_train[feat]
    X_va = df_val.loc[valid_va, feat] if valid_va.sum() < len(y_va_t) else df_val[feat]
    w_clean = w[valid_tr] if valid_tr.sum() < len(y_tr) else w

    model = CatBoostRegressor(**CB_PARAMS)
    model.fit(Pool(X_tr, y_tr[valid_tr], weight=w_clean),
              eval_set=Pool(X_va, y_va_t[valid_va]),
              early_stopping_rounds=100, verbose=0)

    preds_t = model.predict(df_val[feat])
    preds_spot = reconstruct_fn(preds_t)

    rmse = np.sqrt(np.mean((spot_va - preds_spot) ** 2))
    bias = np.mean(spot_va - preds_spot)
    mae = np.mean(np.abs(spot_va - preds_spot))

    # RMSE by month
    rmse_months = []
    for m in sorted(months_va.unique()):
        mask = months_va == m
        rmse_months.append(np.sqrt(np.mean((spot_va[mask] - preds_spot[mask]) ** 2)))

    month_str = "  ".join(f"{r:5.1f}" for r in rmse_months)
    print(f"  {label:60s}  RMSE={rmse:6.2f}  Bias={bias:+5.1f}  [{month_str}]")
    return rmse


results = []

# Month header
month_names = [str(m) for m in sorted(months_va.unique())]
print(f"\n{'':62s} {'':6s} {'':7s}  [{' '.join(f'{m:>5s}' for m in month_names)}]")

# ══════════════════════════════════════════════════════════════════════════
# BASELINES
# ══════════════════════════════════════════════════════════════════════════
print("\n--- BASELINES ---")

# Baseline: arcsinh(spot)
dt = pd.to_datetime(df_train_full["datetime_CET"])
days_ago = (dt.max() - dt).dt.total_seconds() / 86400
w_decay = np.exp(-2.0 * days_ago / 365).values
spot_tr = df_train_full["fr_spot"].values

r = run_model("BL1) arcsinh(spot) + decay(2.0)",
              df_train_full, np.arcsinh(spot_tr), np.arcsinh(spot_va), w_decay,
              lambda p: np.sinh(p))
if r: results.append(("BL1) arcsinh(spot)", r))

# Baseline: spot - roll_168h
roll_168_mean_tr = roll_168_mean_full[:n_full_tr]
roll_168_std_tr = np.clip(roll_168_std_full[:n_full_tr], 1.0, None)
y_dev_tr = spot_tr - roll_168_mean_tr
y_dev_va = spot_va - roll_168_mean_va

r = run_model("BL2) spot - roll_168h + decay(2.0)",
              df_train_full, y_dev_tr, y_dev_va, w_decay,
              lambda p: roll_168_mean_va + p)
if r: results.append(("BL2) spot - roll_168h", r))

# ══════════════════════════════════════════════════════════════════════════
# A) VARIANCE-AWARE WEIGHTS
# ══════════════════════════════════════════════════════════════════════════
print("\n--- A) VARIANCE-AWARE WEIGHTS (target = spot - roll_168h) ---")

for decay in [2.0, 3.0, 4.0]:
    w_base = np.exp(-decay * days_ago / 365).values

    # A1) weight = decay / rolling_std
    w_var = w_base / roll_168_std_tr
    w_var = np.where(np.isfinite(w_var), w_var, 0)

    r = run_model(f"A1) dev168 + decay({decay}) / std",
                  df_train_full, y_dev_tr, y_dev_va, w_var,
                  lambda p: roll_168_mean_va + p)
    if r: results.append((f"A1) dev168 + decay({decay})/std", r))

    # A2) weight = decay / rolling_std^2
    w_var2 = w_base / (roll_168_std_tr ** 2)
    w_var2 = np.where(np.isfinite(w_var2), w_var2, 0)

    r = run_model(f"A2) dev168 + decay({decay}) / std²",
                  df_train_full, y_dev_tr, y_dev_va, w_var2,
                  lambda p: roll_168_mean_va + p)
    if r: results.append((f"A2) dev168 + decay({decay})/std²", r))

# ══════════════════════════════════════════════════════════════════════════
# B) POWER NORMALIZATION: (spot - mean) / std^alpha
# ══════════════════════════════════════════════════════════════════════════
print("\n--- B) POWER NORMALIZATION: (spot-mean)/std^alpha ---")

for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    denom_tr = roll_168_std_tr ** alpha
    denom_va = roll_168_std_va ** alpha

    y_pn_tr = (spot_tr - roll_168_mean_tr) / denom_tr
    y_pn_va = (spot_va - roll_168_mean_va) / denom_va

    r = run_model(f"B) alpha={alpha:.1f}",
                  df_train_full, y_pn_tr, y_pn_va, w_decay,
                  lambda p, d=denom_va: roll_168_mean_va + p * d)
    if r: results.append((f"B) alpha={alpha:.1f}", r))

# ══════════════════════════════════════════════════════════════════════════
# C) TRUNCATE TRAIN
# ══════════════════════════════════════════════════════════════════════════
print("\n--- C) TRUNCATE TRAIN (remove crisis) ---")

for cutoff in ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01"]:
    mask_cut = df_train_full["datetime_CET"] >= cutoff
    df_tr_cut = df_train_full[mask_cut].copy()

    if len(df_tr_cut) < 500:
        continue

    spot_cut = df_tr_cut["fr_spot"].values
    roll_mean_cut = roll_168_mean_full[:n_full_tr][mask_cut.values]
    y_cut = spot_cut - roll_mean_cut

    dt_cut = pd.to_datetime(df_tr_cut["datetime_CET"])
    days_ago_cut = (dt_cut.max() - dt_cut).dt.total_seconds() / 86400

    for decay in [1.0, 2.0, 3.0]:
        w_cut = np.exp(-decay * days_ago_cut / 365).values

        r = run_model(f"C) cut={cutoff} + dev168 + decay({decay})",
                      df_tr_cut, y_cut, y_dev_va, w_cut,
                      lambda p: roll_168_mean_va + p)
        if r: results.append((f"C) cut={cutoff}+d{decay}", r))

# ══════════════════════════════════════════════════════════════════════════
# D) COMBINED: truncate + variance weights
# ══════════════════════════════════════════════════════════════════════════
print("\n--- D) TRUNCATE + VARIANCE WEIGHTS ---")

for cutoff in ["2023-01-01", "2023-07-01"]:
    mask_cut = df_train_full["datetime_CET"] >= cutoff
    df_tr_cut = df_train_full[mask_cut].copy()

    spot_cut = df_tr_cut["fr_spot"].values
    roll_mean_cut = roll_168_mean_full[:n_full_tr][mask_cut.values]
    roll_std_cut = np.clip(roll_168_std_full[:n_full_tr][mask_cut.values], 1.0, None)
    y_cut = spot_cut - roll_mean_cut

    dt_cut = pd.to_datetime(df_tr_cut["datetime_CET"])
    days_ago_cut = (dt_cut.max() - dt_cut).dt.total_seconds() / 86400

    for decay in [2.0, 3.0]:
        w_base = np.exp(-decay * days_ago_cut / 365).values
        w_var = w_base / roll_std_cut
        w_var = np.where(np.isfinite(w_var), w_var, 0)

        r = run_model(f"D) cut={cutoff} + dev168 + decay({decay})/std",
                      df_tr_cut, y_cut, y_dev_va, w_var,
                      lambda p: roll_168_mean_va + p)
        if r: results.append((f"D) {cutoff}+d{decay}/std", r))

# ══════════════════════════════════════════════════════════════════════════
# E) HIGHER DECAY on full data
# ══════════════════════════════════════════════════════════════════════════
print("\n--- E) HIGHER DECAY (full data, dev168) ---")

for decay in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
    w_hi = np.exp(-decay * days_ago / 365).values
    r = run_model(f"E) dev168 + decay({decay})",
                  df_train_full, y_dev_tr, y_dev_va, w_hi,
                  lambda p: roll_168_mean_va + p)
    if r: results.append((f"E) dev168+d{decay}", r))


# ══════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("  SUMMARY — sorted by RMSE")
print("=" * 80)

results.sort(key=lambda x: x[1])
bl_rmse = next(r[1] for r in results if "BL2" in r[0])

for i, (label, rmse) in enumerate(results[:25]):
    delta = rmse - bl_rmse
    marker = " <<<" if i == 0 else ""
    print(f"  {i+1:2d}. {label:45s}  RMSE={rmse:7.3f}  Δ={delta:+6.2f}{marker}")

print(f"\n  Baseline (dev168+decay2): {bl_rmse:.3f}")
print(f"  Best: {results[0][0]} → {results[0][1]:.3f} (Δ={results[0][1]-bl_rmse:+.2f})")
