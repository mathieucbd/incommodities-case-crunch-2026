"""compare_detrending.py — Compare EMA(240h) vs STL detrending for FR and UK.

Tests 5 detrending strategies on the same CatBoost model (same features/params):

  1. EMA(240h)       — current pipeline baseline
  2. EMA(720h)       — 30-day half-life, slower trend
  3. STL-24 trend    — LOESS trend, period=24h (removes daily seasonality)
  4. STL-168 trend   — LOESS trend, period=168h (removes weekly seasonality)
  5. STL-24 full     — Removes trend + seasonal24 from target
                       (residual target; seasonal re-added at inference)

Key question: does a cleaner trend reduce RMSE by giving the model
a more stationary target to predict?

Usage: python -X utf8 scripts/compare_detrending.py
"""

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from statsmodels.tsa.seasonal import STL
from src.data_loading import load_data
from src.feature_engineering import build_features
from src.models.metrics import compute_rmse, compute_hbc
from src.models import train_tree, predict_tree

warnings.filterwarnings("ignore")

# ── Load data + features ───────────────────────────────────────────────────
print("=" * 80)
print("  Detrending Comparison: EMA vs STL")
print("=" * 80)

with open(PROJECT_ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)

x_train, y_train, x_test = load_data(PROJECT_ROOT / "data" / "raw")
train_fe = build_features(pd.concat([x_train], axis=0), config)
train_fe = train_fe.join(y_train[["fr_spot", "uk_spot"]])
print(f"  Train shape: {train_fe.shape}")

holdout_start = config["validation"]["holdout_start"]
mask_val = train_fe["datetime_CET"] >= holdout_start
df_train = train_fe[~mask_val].copy()
df_val   = train_fe[mask_val].copy()
n_tr     = len(df_train)

hours_va = df_val["hour"].values

# Add the V8 interaction feature needed by feat_fr
for df in [df_train, df_val]:
    if "fr_spot_la_roll_168h_mean" in df.columns and "uk_price_per_mw_7d" in df.columns:
        df["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
            df["fr_spot_la_roll_168h_mean"] * df["uk_price_per_mw_7d"]
        )

# ── Feature list (same as pipeline v11) ───────────────────────────────────
import json
with open(PROJECT_ROOT / "outputs" / "feature_selection_v5_fr.json") as f:
    fs_v5 = json.load(f)
feat_fr = [f for f in fs_v5["features"] if f in df_train.columns]
feat_fr += [f for f in [
    "X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d",
    "fr_spot_la_roll_336h_mean", "fr_spot_la_roll_336h_std",
    "fr_stress_index", "fr_load_surprise",
] if f in df_train.columns and f not in feat_fr]
print(f"  FR features: {len(feat_fr)}")

# ── CatBoost params (same as pipeline v11) ────────────────────────────────
FR_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 15000, "learning_rate": 0.059, "depth": 3,
    "l2_leaf_reg": 4.42, "subsample": 0.533, "colsample_bylevel": 0.228,
    "min_child_samples": 14, "random_strength": 0.9, "random_seed": 42,
    "verbose": 0, "allow_writing_files": False, "use_best_model": True,
}

# ── Helper: build time-decay weights ──────────────────────────────────────
def time_decay_weights(dt_series, rs_tr, half_life_days=182):
    dt = pd.to_datetime(dt_series)
    days_ago = (dt.max() - dt).dt.total_seconds() / 86400
    td = np.exp(-2.0 * days_ago.values / half_life_days)
    var = np.clip(rs_tr ** 2, 1.0, None)
    var = np.where(np.isnan(var), 1.0, var)
    return td / var

# ── Helper: train + eval one detrending strategy ──────────────────────────
def run_variant(name, rm_tr, rm_va, la_series_for_std=None):
    """Train CatBoost with given trend removal, return (rmse_raw, rmse_hbc, rmse_hbc_str)."""
    spot_tr = df_train["fr_spot"].values
    spot_va = df_val["fr_spot"].values

    y_dev_tr = spot_tr - rm_tr
    y_dev_va = spot_va - rm_va
    valid_tr = np.isfinite(y_dev_tr)
    valid_va = np.isfinite(y_dev_va)

    # Sample weights
    if la_series_for_std is not None:
        rs_tr = la_series_for_std.rolling(168, min_periods=24).std().iloc[:n_tr].values
    else:
        rs_tr = train_fe["fr_spot_la"].rolling(168, min_periods=24).std().iloc[:n_tr].values
    w = time_decay_weights(df_train["datetime_CET"], rs_tr)

    cb = train_tree(
        "catboost", FR_PARAMS,
        df_train.loc[df_train.index[valid_tr], feat_fr], y_dev_tr[valid_tr],
        df_val.loc[df_val.index[valid_va], feat_fr],   y_dev_va[valid_va],
        sample_weight=w[valid_tr],
    )
    preds = rm_va + predict_tree(cb.model, df_val[feat_fr])
    rmse_raw = compute_rmse(spot_va, preds)
    _, rmse_hbc = compute_hbc(preds, spot_va, hours_va)

    print(f"  {name:<30}  target_std={np.nanstd(y_dev_tr):.1f}"
          f"  RMSE={rmse_raw:.2f}  +HBC={rmse_hbc:.2f}  iter={cb.best_iteration}")
    return rmse_raw, rmse_hbc

# ── Compute all trend variants ─────────────────────────────────────────────
la = train_fe["fr_spot_la"]  # full series (train + val rows)
la_np = la.values
n_full = len(la_np)

# Variant 1: EMA(240h) — current baseline
ema240 = la.ewm(span=240).mean().values
rm_ema240_tr = ema240[:n_tr]
rm_ema240_va = ema240[n_tr:]

# Variant 2: EMA(720h) — slower 30-day trend
ema720 = la.ewm(span=720).mean().values
rm_ema720_tr = ema720[:n_tr]
rm_ema720_va = ema720[n_tr:]

# Variant 3: EMA(8760h) — very slow 1-year trend
ema8760 = la.ewm(span=8760, min_periods=168).mean().values
rm_ema8760_tr = ema8760[:n_tr]
rm_ema8760_va = ema8760[n_tr:]

# Variant 4: STL period=24 — trend only
print("\n  Fitting STL period=24 (robust)...")
stl24 = STL(la_np, period=24, robust=True, seasonal=13)
res24 = stl24.fit()
rm_stl24_tr = res24.trend[:n_tr]
rm_stl24_va = res24.trend[n_tr:]

# Variant 5: STL period=168 — trend only (removes weekly seasonality from trend estimate)
print("  Fitting STL period=168 (robust)...")
stl168 = STL(la_np, period=168, robust=True, seasonal=25)
res168 = stl168.fit()
rm_stl168_tr = res168.trend[:n_tr]
rm_stl168_va = res168.trend[n_tr:]

# Variant 6: STL-24 full — remove trend + seasonal from target
# At inference: add back seasonal (last 4 weeks mean per hour)
seasonal24 = res24.seasonal  # shape (n_full,)
rm_stl24full_tr = res24.trend[:n_tr] + seasonal24[:n_tr]
rm_stl24full_va = res24.trend[n_tr:] + seasonal24[n_tr:]

# Variant 7: STL-168 full — remove trend + seasonal (weekly)
seasonal168 = res168.seasonal
rm_stl168full_tr = res168.trend[:n_tr] + seasonal168[:n_tr]
rm_stl168full_va = res168.trend[n_tr:] + seasonal168[n_tr:]

# ── Diagnostics: target std for each variant ──────────────────────────────
spot_tr_np = df_train["fr_spot"].values
print("\n  Target std comparison (lower = simpler prediction task):")
print(f"    Raw spot_tr std:             {np.nanstd(spot_tr_np):.2f}")
for nm, rm in [
    ("EMA-240h",         rm_ema240_tr),
    ("EMA-720h",         rm_ema720_tr),
    ("EMA-8760h",        rm_ema8760_tr),
    ("STL-24 trend",     rm_stl24_tr),
    ("STL-168 trend",    rm_stl168_tr),
    ("STL-24 full",      rm_stl24full_tr),
    ("STL-168 full",     rm_stl168full_tr),
]:
    dev = spot_tr_np - rm
    print(f"    {nm:<20}: {np.nanstd(dev):.2f}")

# ── Alignment check: how much do trend estimates diverge? ─────────────────
print("\n  Trend estimate comparison vs EMA-240 at validation start (first 5 hours):")
print(f"    {'Method':<20}  {'First 5 trend values'}")
for nm, rm in [
    ("EMA-240h",         rm_ema240_va[:5]),
    ("EMA-720h",         rm_ema720_va[:5]),
    ("EMA-8760h",        rm_ema8760_va[:5]),
    ("STL-24 trend",     rm_stl24_va[:5]),
    ("STL-168 trend",    rm_stl168_va[:5]),
]:
    print(f"    {nm:<20}  {np.round(rm, 1)}")

# ── Run all variants ───────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  RESULTS — CatBoost FR (same params, same features)")
print("=" * 80)
print(f"  {'Variant':<30}  {'target_std':>10}  {'RMSE':>7}  {'+HBC':>7}  {'iters':>6}")
print(f"  {'-'*68}")

results = {}
variants = [
    ("EMA-240h (current)",         rm_ema240_tr,    rm_ema240_va),
    ("EMA-720h",                   rm_ema720_tr,    rm_ema720_va),
    ("EMA-8760h",                  rm_ema8760_tr,   rm_ema8760_va),
    ("STL-24 trend only",          rm_stl24_tr,     rm_stl24_va),
    ("STL-168 trend only",         rm_stl168_tr,    rm_stl168_va),
    ("STL-24 trend+seasonal",      rm_stl24full_tr, rm_stl24full_va),
    ("STL-168 trend+seasonal",     rm_stl168full_tr,rm_stl168full_va),
]

for name, rm_tr, rm_va in variants:
    raw, hbc = run_variant(name, rm_tr, rm_va)
    results[name] = {"rmse": raw, "hbc": hbc}

# ── Final summary table ────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  SUMMARY")
print("=" * 80)
print(f"  {'Variant':<30}  {'RMSE':>7}  {'+HBC':>7}  {'Delta HBC vs EMA-240':>22}")
baseline_hbc = results["EMA-240h (current)"]["hbc"]
for name, v in results.items():
    delta = v["hbc"] - baseline_hbc
    sign = "+" if delta >= 0 else ""
    print(f"  {name:<30}  {v['rmse']:>7.2f}  {v['hbc']:>7.2f}  {sign}{delta:>+.2f}")

best = min(results, key=lambda k: results[k]["hbc"])
print(f"\n  Best variant: {best}  (HBC RMSE={results[best]['hbc']:.2f})")
if results[best]["hbc"] < baseline_hbc:
    print(f"  Improvement vs EMA-240h: {baseline_hbc - results[best]['hbc']:.2f} RMSE points")
else:
    print("  EMA-240h is still the best — STL detrending does not help here.")
print("=" * 80)
