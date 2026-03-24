"""Per-hour model test — 24 specialized CatBoost models vs 1 model + HBC.

Instead of training one model on all hours then applying 24 HBC corrections,
train 24 separate CatBoost models, each specialized for its hour's dynamics.

Hour 3 (night) has very different feature interactions than hour 8 (morning ramp).
A single model can't capture these hour-specific patterns as well as dedicated models.

Tests:
  1. 24 per-hour CatBoost (no HBC needed)
  2. 24 per-hour CatBoost + per-hour LGB ensemble
  3. Compare with current 1-model + HBC baseline
"""

import sys, yaml, warnings, time, json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

print("=" * 90)
print("  PER-HOUR MODEL TEST — 24 specialized models")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
df = build_features(train, config)

holdout_start = config["validation"]["holdout_start"]
mask_val = (df["datetime_CET"] >= holdout_start).values
df_tr = df[~mask_val].copy()
df_va = df[mask_val].copy()

# Features
with open("outputs/feature_selection_v5_fr.json") as f:
    FR_FEAT = [f for f in json.load(f)["features"] if f in df_tr.columns]
with open("outputs/uk_feature_research.json") as f:
    UK_FEAT = [f for f in json.load(f)["confirmed_features"] if f in df_tr.columns]

# Targets
fr_la = df["fr_spot_la"].values
ema_fr = pd.Series(fr_la).ewm(span=240).mean().values
fr_anchor_tr = ema_fr[~mask_val]
fr_anchor_va = ema_fr[mask_val]
fr_spot_va = df_va["fr_spot"].values
fr_y_tr = df_tr["fr_spot"].values - fr_anchor_tr
fr_y_va = fr_spot_va - fr_anchor_va

uk_moc_va = df_va["uk_merit_order_cost"].values
uk_spot_va = df_va["uk_spot"].values
uk_y_tr = df_tr["uk_spot"].values - df_tr["uk_merit_order_cost"].values
uk_y_va = uk_spot_va - uk_moc_va

hours_tr = df_tr["hour"].values
hours_va = df_va["hour"].values

fr_valid_tr = np.isfinite(fr_y_tr) & np.isfinite(fr_anchor_tr)
uk_valid_tr = np.isfinite(uk_y_tr)

# Sample weights
days_ago = (df_tr["datetime_CET"].max() - df_tr["datetime_CET"]).dt.total_seconds() / 86400
roll_std = df_tr["fr_spot_la"].rolling(168, min_periods=24).std().fillna(df_tr["fr_spot_la"].std())
fr_sw = np.exp(-2 * days_ago.values / 365) / np.clip(roll_std.values ** 2, 1, None)
fr_sw[~fr_valid_tr] = 0

cb_params_fr = config.get("catboost_params_fr_optuna_v2", config.get("catboost_params_fr", {}))
cb_params_uk = config.get("catboost_params_uk", {})
lgb_params_fr = config.get("lightgbm_params_fr", {})
lgb_params_fr_clean = {k: v for k, v in lgb_params_fr.items() if k != "n_estimators"}
lgb_params_uk = config.get("lightgbm_params_uk", {})
lgb_params_uk_clean = {k: v for k, v in lgb_params_uk.items() if k != "n_estimators"}


def compute_rmse(actual, preds):
    return np.sqrt(np.mean((actual - preds) ** 2))

def compute_hbc(preds, actual, hours):
    errors = actual - preds
    hbc = {h: float(errors[hours == h].mean()) for h in range(24) if (hours == h).sum() > 0}
    corrected = preds + np.array([hbc.get(h, 0) for h in hours])
    return np.sqrt(np.mean((actual - corrected) ** 2))


# ══════════════════════════════════════════════════════════════════════
#  BASELINE: 1 model + HBC
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("  BASELINE: 1 model on all hours + HBC")
print(f"{'='*90}")

# FR CatBoost baseline
cb_fr_base = CatBoostRegressor(**{**cb_params_fr, "verbose": 0})
cb_fr_base.fit(df_tr[FR_FEAT].values[fr_valid_tr], fr_y_tr[fr_valid_tr],
               sample_weight=fr_sw[fr_valid_tr],
               eval_set=(df_va[FR_FEAT].values, fr_y_va))
p_fr_base = fr_anchor_va + cb_fr_base.predict(df_va[FR_FEAT].values)
fr_base_rmse = compute_rmse(fr_spot_va, p_fr_base)
fr_base_hbc = compute_hbc(p_fr_base, fr_spot_va, hours_va)
print(f"  FR CB baseline:  RMSE={fr_base_rmse:.2f}  +HBC={fr_base_hbc:.2f}")

# UK CatBoost baseline
cb_uk_base = CatBoostRegressor(**{**cb_params_uk, "verbose": 0})
cb_uk_base.fit(df_tr[UK_FEAT].values[uk_valid_tr], uk_y_tr[uk_valid_tr],
               eval_set=(df_va[UK_FEAT].values, uk_y_va))
p_uk_base = uk_moc_va + cb_uk_base.predict(df_va[UK_FEAT].values)
uk_base_rmse = compute_rmse(uk_spot_va, p_uk_base)
uk_base_hbc = compute_hbc(p_uk_base, uk_spot_va, hours_va)
print(f"  UK CB baseline:  RMSE={uk_base_rmse:.2f}  +HBC={uk_base_hbc:.2f}")


# ══════════════════════════════════════════════════════════════════════
#  PER-HOUR MODELS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("  PER-HOUR MODELS: 24 CatBoost")
print(f"{'='*90}")

preds_fr_perhour_cb = np.full(len(df_va), np.nan)
preds_uk_perhour_cb = np.full(len(df_va), np.nan)
preds_fr_perhour_lgb = np.full(len(df_va), np.nan)
preds_uk_perhour_lgb = np.full(len(df_va), np.nan)

for h in range(24):
    # Train masks
    h_tr = hours_tr == h
    h_va = hours_va == h

    n_tr_fr = (h_tr & fr_valid_tr).sum()
    n_va = h_va.sum()

    if n_tr_fr < 50 or n_va < 10:
        print(f"    h={h:2d}  SKIP (n_tr={n_tr_fr}, n_va={n_va})")
        continue

    # FR CatBoost per-hour
    mask_tr_fr = h_tr & fr_valid_tr
    cb_h_fr = CatBoostRegressor(**{**cb_params_fr, "verbose": 0})
    cb_h_fr.fit(df_tr[FR_FEAT].values[mask_tr_fr], fr_y_tr[mask_tr_fr],
                sample_weight=fr_sw[mask_tr_fr],
                eval_set=(df_va[FR_FEAT].values[h_va], fr_y_va[h_va]))
    preds_fr_perhour_cb[h_va] = fr_anchor_va[h_va] + cb_h_fr.predict(df_va[FR_FEAT].values[h_va])

    # FR LGB per-hour
    ds_h_tr = lgb.Dataset(df_tr[FR_FEAT].values[mask_tr_fr], fr_y_tr[mask_tr_fr],
                          weight=fr_sw[mask_tr_fr])
    ds_h_va = lgb.Dataset(df_va[FR_FEAT].values[h_va], fr_y_va[h_va], reference=ds_h_tr)
    lgb_h_fr = lgb.train(lgb_params_fr_clean, ds_h_tr,
                         num_boost_round=lgb_params_fr.get("n_estimators", 5000),
                         valid_sets=[ds_h_va], callbacks=[lgb.early_stopping(50, verbose=False)])
    preds_fr_perhour_lgb[h_va] = fr_anchor_va[h_va] + lgb_h_fr.predict(df_va[FR_FEAT].values[h_va])

    # UK CatBoost per-hour
    mask_tr_uk = h_tr & uk_valid_tr
    cb_h_uk = CatBoostRegressor(**{**cb_params_uk, "verbose": 0})
    cb_h_uk.fit(df_tr[UK_FEAT].values[mask_tr_uk], uk_y_tr[mask_tr_uk],
                eval_set=(df_va[UK_FEAT].values[h_va], uk_y_va[h_va]))
    preds_uk_perhour_cb[h_va] = uk_moc_va[h_va] + cb_h_uk.predict(df_va[UK_FEAT].values[h_va])

    # UK LGB per-hour
    ds_h_uk_tr = lgb.Dataset(df_tr[UK_FEAT].values[mask_tr_uk], uk_y_tr[mask_tr_uk])
    ds_h_uk_va = lgb.Dataset(df_va[UK_FEAT].values[h_va], uk_y_va[h_va], reference=ds_h_uk_tr)
    lgb_h_uk = lgb.train(lgb_params_uk_clean, ds_h_uk_tr,
                         num_boost_round=lgb_params_uk.get("n_estimators", 5000),
                         valid_sets=[ds_h_uk_va], callbacks=[lgb.early_stopping(50, verbose=False)])
    preds_uk_perhour_lgb[h_va] = uk_moc_va[h_va] + lgb_h_uk.predict(df_va[UK_FEAT].values[h_va])

    # Per-hour stats
    fr_h_rmse = compute_rmse(fr_spot_va[h_va], preds_fr_perhour_cb[h_va])
    uk_h_rmse = compute_rmse(uk_spot_va[h_va], preds_uk_perhour_cb[h_va])
    fr_h_base = compute_rmse(fr_spot_va[h_va], p_fr_base[h_va])
    uk_h_base = compute_rmse(uk_spot_va[h_va], p_uk_base[h_va])
    fr_delta = fr_h_rmse - fr_h_base
    uk_delta = uk_h_rmse - uk_h_base
    print(f"    h={h:2d}  n={n_tr_fr:4d}/{n_va:3d}  "
          f"FR: base={fr_h_base:.1f} perH={fr_h_rmse:.1f} Δ={fr_delta:+.1f}  "
          f"UK: base={uk_h_base:.1f} perH={uk_h_rmse:.1f} Δ={uk_delta:+.1f}")


# ══════════════════════════════════════════════════════════════════════
#  COMPARISON
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("  RESULTS COMPARISON")
print(f"{'='*90}")

# Per-hour CB (no HBC needed)
fr_ph_cb_rmse = compute_rmse(fr_spot_va, preds_fr_perhour_cb)
uk_ph_cb_rmse = compute_rmse(uk_spot_va, preds_uk_perhour_cb)

# Per-hour CB+LGB average
preds_fr_ph_avg = (preds_fr_perhour_cb + preds_fr_perhour_lgb) / 2
preds_uk_ph_avg = (preds_uk_perhour_cb + preds_uk_perhour_lgb) / 2
fr_ph_avg_rmse = compute_rmse(fr_spot_va, preds_fr_ph_avg)
uk_ph_avg_rmse = compute_rmse(uk_spot_va, preds_uk_ph_avg)

# Also compute with HBC on per-hour (should be minimal since already per-hour)
fr_ph_cb_hbc = compute_hbc(preds_fr_perhour_cb, fr_spot_va, hours_va)
uk_ph_cb_hbc = compute_hbc(preds_uk_perhour_cb, uk_spot_va, hours_va)
fr_ph_avg_hbc = compute_hbc(preds_fr_ph_avg, fr_spot_va, hours_va)
uk_ph_avg_hbc = compute_hbc(preds_uk_ph_avg, uk_spot_va, hours_va)

print(f"\n  {'Config':35s}  {'FR RMSE':>8s}  {'FR+HBC':>7s}  {'UK RMSE':>8s}  {'UK+HBC':>7s}  {'SUM+HBC':>8s}")
print(f"  {'-'*85}")

configs = [
    ("1-model CB + HBC (baseline)", fr_base_rmse, fr_base_hbc, uk_base_rmse, uk_base_hbc),
    ("24 per-hour CB (no HBC)", fr_ph_cb_rmse, fr_ph_cb_rmse, uk_ph_cb_rmse, uk_ph_cb_rmse),
    ("24 per-hour CB (+HBC)", fr_ph_cb_rmse, fr_ph_cb_hbc, uk_ph_cb_rmse, uk_ph_cb_hbc),
    ("24 per-hour CB+LGB avg", fr_ph_avg_rmse, fr_ph_avg_rmse, uk_ph_avg_rmse, uk_ph_avg_rmse),
    ("24 per-hour CB+LGB avg (+HBC)", fr_ph_avg_rmse, fr_ph_avg_hbc, uk_ph_avg_rmse, uk_ph_avg_hbc),
]

for label, fr_r, fr_h, uk_r, uk_h in configs:
    s = fr_h + uk_h
    print(f"  {label:35s}  {fr_r:8.2f}  {fr_h:7.2f}  {uk_r:8.2f}  {uk_h:7.2f}  {s:8.2f}")

print(f"\n  Note: Pipeline v7 uses 5-model ensemble + regime weights + HBC → SUM=25.12")
print(f"  This test compares per-hour vs single-model for CB only")

print(f"\n  Total time: {time.time() - t0:.0f}s")
