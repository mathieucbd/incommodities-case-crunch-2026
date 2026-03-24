"""Residual improvement tests — rigorous A/B testing.

Tests:
  A. Spike/crash features (wind excess, solar crash, price collapse)
  B. Loss function alternatives (Huber, Quantile, MAE)
  C. HBC variants (monthly×hourly, dampened, no HBC)
  D. Clipping strategies

Each test: train model, measure val RMSE, compare to baseline.
Baseline: current best FR (17.18) and UK (10.19).

Usage: cd "INCOMO 3" && python scripts/residual_improvements.py
"""

import sys, json, warnings, time
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features
from catboost import CatBoostRegressor, Pool
import yaml

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
train_fe = build_features(train, config)
print(f"Data loaded in {time.time()-t0:.0f}s")

holdout_start = config["validation"]["holdout_start"]
mask_val = train_fe["datetime_CET"] >= holdout_start
df_train = train_fe[~mask_val].copy()
df_val = train_fe[mask_val].copy()

# ── Load features ──
with open("outputs/feature_selection_v5_fr.json") as f:
    feat_fr_27 = json.load(f)["features"]
with open("outputs/uk_feature_research.json") as f:
    feat_uk_confirmed = json.load(f)["confirmed_features"]

for df in [df_train, df_val]:
    df["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
        df["fr_spot_la_roll_168h_mean"] * df["uk_price_per_mw_7d"])
feat_fr_base = [f for f in feat_fr_27 + ["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"]
                if f in df_train.columns]
feat_uk_base = [f for f in feat_uk_confirmed if f in df_train.columns]

# ── Prepare targets ──
# FR stationary
la_fr = train_fe["fr_spot_la"]
rm_fr = la_fr.rolling(168, min_periods=24).mean()
rs_fr = la_fr.rolling(168, min_periods=24).std()
n_tr = len(df_train)
rm_tr_fr = rm_fr.iloc[:n_tr].values
rm_va_fr = rm_fr.iloc[n_tr:n_tr+len(df_val)].values
y_dev_tr_fr = df_train["fr_spot"].values - rm_tr_fr
y_dev_va_fr = df_val["fr_spot"].values - rm_va_fr
valid_tr_fr = np.isfinite(y_dev_tr_fr)
valid_va_fr = np.isfinite(y_dev_va_fr)

dt = pd.to_datetime(df_train["datetime_CET"])
days_ago = (dt.max()-dt).dt.total_seconds()/86400
td = np.exp(-2.0*days_ago.values/365)
var = np.clip(rs_fr.iloc[:n_tr].values**2, 1.0, None)
var = np.where(np.isnan(var), 1.0, var)
w_fr = td / var

# UK basis
uk_moc_tr = df_train["uk_merit_order_cost"].values
uk_moc_va = df_val["uk_merit_order_cost"].values
y_basis_tr = df_train["uk_spot"].values - uk_moc_tr
y_basis_va = df_val["uk_spot"].values - uk_moc_va
valid_tr_uk = np.isfinite(y_basis_tr)
valid_va_uk = np.isfinite(y_basis_va)

spot_fr_va = df_val["fr_spot"].values
spot_uk_va = df_val["uk_spot"].values
hours_va = df_val["hour"].values
months_va = df_val["datetime_CET"].dt.month.values

def rmse(y, p): return np.sqrt(np.mean((y-p)**2))
def hbc(preds, actual, hours):
    hb = {h: (actual[hours==h]-preds[hours==h]).mean() for h in range(24)}
    return preds + np.array([hb.get(h,0) for h in hours])
def hbc_rmse(preds, actual, hours):
    return rmse(actual, hbc(preds, actual, hours))

# ── Baseline models ──
print("\n" + "="*80)
print("  BASELINE MODELS")
print("="*80)

FR_P = {"loss_function":"RMSE","eval_metric":"RMSE","iterations":15000,
        "learning_rate":0.059,"depth":3,"l2_leaf_reg":4.42,"subsample":0.533,
        "colsample_bylevel":0.228,"min_child_samples":14,"random_strength":0.9,
        "random_seed":42,"verbose":0,"allow_writing_files":False,"use_best_model":True}

UK_P = {"loss_function":"RMSE","eval_metric":"RMSE","iterations":15000,
        "learning_rate":0.03,"depth":8,"l2_leaf_reg":5,"colsample_bylevel":0.8,
        "subsample":0.8,"random_seed":42,"verbose":0,"allow_writing_files":False,
        "use_best_model":True}

def train_fr(params, feats, label=""):
    model = CatBoostRegressor(**params)
    model.fit(
        Pool(df_train.loc[df_train.index[valid_tr_fr], feats], y_dev_tr_fr[valid_tr_fr],
             weight=w_fr[valid_tr_fr]),
        eval_set=Pool(df_val.loc[df_val.index[valid_va_fr], feats], y_dev_va_fr[valid_va_fr]),
        early_stopping_rounds=200, verbose=0)
    preds = rm_va_fr + model.predict(df_val[feats])
    r = rmse(spot_fr_va, preds)
    rh = hbc_rmse(preds, spot_fr_va, hours_va)
    return r, rh, model, preds

def train_uk(params, feats, label=""):
    model = CatBoostRegressor(**params)
    model.fit(
        Pool(df_train.loc[df_train.index[valid_tr_uk], feats], y_basis_tr[valid_tr_uk]),
        eval_set=Pool(df_val.loc[df_val.index[valid_va_uk], feats], y_basis_va[valid_va_uk]),
        early_stopping_rounds=200, verbose=0)
    preds = uk_moc_va + model.predict(df_val[feats])
    r = rmse(spot_uk_va, preds)
    rh = hbc_rmse(preds, spot_uk_va, hours_va)
    return r, rh, model, preds

r_fr_base, rh_fr_base, _, p_fr_base = train_fr(FR_P, feat_fr_base)
r_uk_base, rh_uk_base, _, p_uk_base = train_uk(UK_P, feat_uk_base)
print(f"  FR baseline: RMSE={r_fr_base:.2f}  +HBC={rh_fr_base:.2f}")
print(f"  UK baseline: RMSE={r_uk_base:.2f}  +HBC={rh_uk_base:.2f}")
print(f"  SUM baseline: {r_fr_base+r_uk_base:.2f}  +HBC={rh_fr_base+rh_uk_base:.2f}")


# ══════════════════════════════════════════════════════════════════════════
# TEST A: SPIKE/CRASH FEATURES
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  TEST A: SPIKE/CRASH FEATURES")
print("="*80)

# Create new features on both train and val
for df in [df_train, df_val]:
    # UK crash indicators
    if "uk_wind_f" in df.columns and "uk_load_f" in df.columns:
        wind_pen = df["uk_wind_f"] / df["uk_load_f"].clip(1)
        df["spike_uk_wind_excess"] = (wind_pen > wind_pen.quantile(0.9)).astype(float)
        df["spike_uk_wind_pen_gt50"] = (wind_pen > 0.5).astype(float)
        df["spike_uk_wind_x_low_load"] = wind_pen * (df["uk_load_f"] < df["uk_load_f"].quantile(0.25)).astype(float)

    # FR negative/spike indicators
    if "fr_spot_la" in df.columns:
        df["spike_fr_la_negative"] = (df["fr_spot_la"] < 0).astype(float)
        df["spike_fr_la_very_high"] = (df["fr_spot_la"] > df["fr_spot_la"].quantile(0.95)).astype(float)
        df["spike_fr_la_crash_24h"] = df["fr_spot_la"] - df["fr_spot_la"].shift(24)
        df["spike_fr_la_abs_change_24h"] = (df["fr_spot_la"] - df["fr_spot_la"].shift(24)).abs()

    if "uk_spot_la" in df.columns:
        df["spike_uk_la_negative"] = (df["uk_spot_la"] < 0).astype(float)
        df["spike_uk_la_very_low"] = (df["uk_spot_la"] < df["uk_spot_la"].quantile(0.1)).astype(float)
        df["spike_uk_la_crash_24h"] = df["uk_spot_la"] - df["uk_spot_la"].shift(24)
        df["spike_uk_la_abs_change_24h"] = (df["uk_spot_la"] - df["uk_spot_la"].shift(24)).abs()

    # Weekend × hour interaction (UK weekend is worst)
    if "day_of_week" in df.columns:
        df["spike_is_weekend"] = (df["day_of_week"] >= 5).astype(float)
        df["spike_weekend_x_hour"] = df["spike_is_weekend"] * df["hour"]
        df["spike_weekend_x_low_load_uk"] = df["spike_is_weekend"] * (
            df.get("uk_load_f", pd.Series(0, index=df.index)) < df.get("uk_load_f", pd.Series(0, index=df.index)).quantile(0.25)
        ).astype(float)

spike_features = [c for c in df_train.columns if c.startswith("spike_")]
print(f"  Created {len(spike_features)} spike features: {spike_features}")

# Test FR with spike features
feat_fr_spike = feat_fr_base + [f for f in spike_features if f in df_train.columns]
r_fr_spike, rh_fr_spike, _, _ = train_fr(FR_P, feat_fr_spike)
delta_fr = r_fr_spike - r_fr_base
print(f"\n  FR + spike features: RMSE={r_fr_spike:.2f} (Δ={delta_fr:+.2f})  +HBC={rh_fr_spike:.2f}")

# Test UK with spike features
feat_uk_spike = feat_uk_base + [f for f in spike_features if f in df_train.columns and f not in feat_uk_base]
r_uk_spike, rh_uk_spike, _, _ = train_uk(UK_P, feat_uk_spike)
delta_uk = r_uk_spike - r_uk_base
print(f"  UK + spike features: RMSE={r_uk_spike:.2f} (Δ={delta_uk:+.2f})  +HBC={rh_uk_spike:.2f}")


# ══════════════════════════════════════════════════════════════════════════
# TEST B: LOSS FUNCTION ALTERNATIVES
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  TEST B: LOSS FUNCTION ALTERNATIVES")
print("="*80)

# B1: MAE (less sensitive to outliers)
for loss_name, loss_fn in [("MAE", "MAE"), ("Huber:delta=10", "Huber:delta=10"),
                            ("Huber:delta=20", "Huber:delta=20"), ("Quantile:alpha=0.5", "Quantile:alpha=0.5")]:
    fr_p_alt = {**FR_P, "loss_function": loss_fn, "eval_metric": "RMSE"}
    uk_p_alt = {**UK_P, "loss_function": loss_fn, "eval_metric": "RMSE"}

    try:
        r_fr_alt, rh_fr_alt, _, _ = train_fr(fr_p_alt, feat_fr_base)
        delta_fr = r_fr_alt - r_fr_base
    except Exception as e:
        r_fr_alt, rh_fr_alt, delta_fr = 999, 999, 999

    try:
        r_uk_alt, rh_uk_alt, _, _ = train_uk(uk_p_alt, feat_uk_base)
        delta_uk = r_uk_alt - r_uk_base
    except Exception as e:
        r_uk_alt, rh_uk_alt, delta_uk = 999, 999, 999

    print(f"  {loss_name:25s}: FR={r_fr_alt:.2f}(Δ={delta_fr:+.2f})  UK={r_uk_alt:.2f}(Δ={delta_uk:+.2f})  "
          f"SUM={r_fr_alt+r_uk_alt:.2f}(Δ={r_fr_alt+r_uk_alt-r_fr_base-r_uk_base:+.2f})")


# ══════════════════════════════════════════════════════════════════════════
# TEST C: HBC VARIANTS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  TEST C: HBC VARIANTS")
print("="*80)

def hbc_monthly(preds, actual, hours, months):
    """HBC by month × hour."""
    corrected = preds.copy()
    for m in sorted(set(months)):
        for h in range(24):
            mask = (hours == h) & (months == m)
            if mask.sum() > 5:
                corrected[mask] += (actual[mask] - preds[mask]).mean()
    return corrected

def hbc_dampened(preds, actual, hours, alpha=0.5):
    """HBC with dampening factor."""
    hb = {h: (actual[hours==h]-preds[hours==h]).mean() * alpha for h in range(24)}
    return preds + np.array([hb.get(h,0) for h in hours])

def hbc_dow_hour(preds, actual, hours, dow):
    """HBC by weekday/weekend × hour."""
    corrected = preds.copy()
    for is_we in [0, 1]:
        for h in range(24):
            mask = (hours == h) & ((dow >= 5) == is_we)
            if mask.sum() > 5:
                corrected[mask] += (actual[mask] - preds[mask]).mean()
    return corrected

dow_va = df_val["datetime_CET"].dt.dayofweek.values

for hbc_name, hbc_fn_fr, hbc_fn_uk in [
    ("No HBC",        lambda p: p,        lambda p: p),
    ("HBC standard",  lambda p: hbc(p, spot_fr_va, hours_va), lambda p: hbc(p, spot_uk_va, hours_va)),
    ("HBC dampened 0.5", lambda p: hbc_dampened(p, spot_fr_va, hours_va, 0.5),
                         lambda p: hbc_dampened(p, spot_uk_va, hours_va, 0.5)),
    ("HBC dampened 0.3", lambda p: hbc_dampened(p, spot_fr_va, hours_va, 0.3),
                         lambda p: hbc_dampened(p, spot_uk_va, hours_va, 0.3)),
    ("HBC monthly",   lambda p: hbc_monthly(p, spot_fr_va, hours_va, months_va),
                       lambda p: hbc_monthly(p, spot_uk_va, hours_va, months_va)),
    ("HBC dow×hour",  lambda p: hbc_dow_hour(p, spot_fr_va, hours_va, dow_va),
                       lambda p: hbc_dow_hour(p, spot_uk_va, hours_va, dow_va)),
]:
    fr_corrected = hbc_fn_fr(p_fr_base)
    uk_corrected = hbc_fn_uk(p_uk_base)
    r_fr_h = rmse(spot_fr_va, fr_corrected)
    r_uk_h = rmse(spot_uk_va, uk_corrected)
    total = r_fr_h + r_uk_h
    delta = total - (r_fr_base + r_uk_base)
    print(f"  {hbc_name:20s}: FR={r_fr_h:.2f}  UK={r_uk_h:.2f}  SUM={total:.2f}  (Δ vs no HBC={delta:+.2f})")


# ══════════════════════════════════════════════════════════════════════════
# TEST D: RETRAIN ITERATIONS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  TEST D: RETRAIN ITERATIONS SENSITIVITY")
print("="*80)

# Simulate retrain effect: train on 80% of train, validate on last 20%
# to see if more iterations helps or hurts
n_80 = int(n_tr * 0.8)
df_tr80 = df_train.iloc[:n_80].copy()
df_va20 = df_train.iloc[n_80:].copy()

# Recalc rolling for this split
rm_80 = rm_fr.iloc[:n_80].values
rm_20 = rm_fr.iloc[n_80:n_tr].values
y_80 = df_tr80["fr_spot"].values - rm_80
y_20 = df_va20["fr_spot"].values - rm_20
v_80 = np.isfinite(y_80)
v_20 = np.isfinite(y_20)
w_80 = w_fr[:n_80]

for n_iter in [50, 100, 150, 200, 300, 500]:
    p_test = {**FR_P, "iterations": n_iter, "use_best_model": False}
    m = CatBoostRegressor(**p_test)
    m.fit(Pool(df_tr80.loc[df_tr80.index[v_80], feat_fr_base], y_80[v_80], weight=w_80[v_80]),
          verbose=0)
    preds_20 = rm_20 + m.predict(df_va20[feat_fr_base])
    r20 = rmse(df_va20["fr_spot"].values, preds_20)
    print(f"  FR {n_iter:3d} iters (no early stop): RMSE={r20:.2f}")


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  SUMMARY — Best improvements")
print("="*80)
print(f"  Baseline SUM: {r_fr_base + r_uk_base:.2f} (FR={r_fr_base:.2f} + UK={r_uk_base:.2f})")
print(f"  Kaggle score:  25.87")
print(f"  Target:        24.45 (1st place)")
print(f"  Gap to close:  {25.87 - 24.45:.2f}")

print(f"\n  Total time: {time.time()-t0:.0f}s")
