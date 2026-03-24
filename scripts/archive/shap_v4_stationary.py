"""SHAP v4 — Feature ranking with stationary target for FR.

Target transforms:
  FR: fr_spot - fr_spot_la_roll_168h_mean  (deviation from 7-day rolling mean)
      Weights: exp(-2.0 * days_ago / 365) / clip(rolling_168h_std^2, 1)
  UK: raw uk_spot, no weights

Usage: cd INCOMO\ 3 && python scripts/shap_v4_stationary.py
"""

import sys, json, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features
from catboost import CatBoostRegressor, Pool
import shap
import yaml

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

HOLDOUT_START = config["validation"]["holdout_start"]
CORR_THRESHOLD = 0.98

CB_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 3000, "learning_rate": 0.03, "depth": 8,
    "l2_leaf_reg": 5, "subsample": 0.8, "random_seed": 42,
    "verbose": 0, "allow_writing_files": False, "use_best_model": True,
}

# ── Load data ─────────────────────────────────────────────────────────────
print("=" * 70)
print("  SHAP v4 — Stationary Target Feature Ranking")
print("=" * 70)

print("\n[1/7] Loading data...")
t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
train_fe = build_features(train, config)
print(f"  Done in {time.time() - t0:.1f}s — shape: {train_fe.shape}")

# ── Split ─────────────────────────────────────────────────────────────────
print(f"\n[2/7] Splitting at {HOLDOUT_START}")
mask_val = train_fe["datetime_CET"] >= HOLDOUT_START
df_train = train_fe[~mask_val].copy()
df_val = train_fe[mask_val].copy()
print(f"  Train: {df_train.shape}, Val: {df_val.shape}")

# ── Feature columns ──────────────────────────────────────────────────────
EXCLUDE = {"datetime_CET", "datetime_UTC", "fr_spot", "uk_spot"}
features = sorted([c for c in train_fe.columns
                   if c not in EXCLUDE and train_fe[c].dtype.kind in ("f", "i", "u", "b")])
print(f"  Total numeric features: {len(features)}")

# ── Stationary target FR ─────────────────────────────────────────────────
print("\n[3/7] Building stationary target for FR...")
fr_la = train_fe["fr_spot_la"]
roll_mean = fr_la.rolling(168, min_periods=24).mean()
roll_std = fr_la.rolling(168, min_periods=24).std()

fr_target_tr = df_train["fr_spot"].values - roll_mean.loc[df_train.index].values
fr_target_va = df_val["fr_spot"].values - roll_mean.loc[df_val.index].values

dt = pd.to_datetime(df_train["datetime_CET"])
days_ago = (dt.max() - dt).dt.total_seconds() / 86400
time_decay = np.exp(-2.0 * days_ago.values / 365)
var_168h = np.clip(roll_std.loc[df_train.index].values ** 2, 1.0, None)
var_168h = np.where(np.isnan(var_168h), 1.0, var_168h)
fr_weights = time_decay / var_168h

valid_tr = ~np.isnan(fr_target_tr)
valid_va = ~np.isnan(fr_target_va)
print(f"  FR: {valid_tr.sum()} train, {valid_va.sum()} val (dropped {(~valid_tr).sum()} NaN)")

# ── Train models ──────────────────────────────────────────────────────────
print("\n[4/7] Training CatBoost models...")

# FR
print("\n  --- FR (stationary target) ---")
t1 = time.time()
X_tr_fr = df_train.loc[df_train.index[valid_tr], features]
X_va_fr = df_val.loc[df_val.index[valid_va], features]

model_fr = CatBoostRegressor(**CB_PARAMS)
model_fr.fit(Pool(X_tr_fr, fr_target_tr[valid_tr], weight=fr_weights[valid_tr]),
             eval_set=Pool(X_va_fr, fr_target_va[valid_va]),
             early_stopping_rounds=100, verbose=0)

anchor_va = roll_mean.loc[df_val.index].values[valid_va]
preds_fr = anchor_va + model_fr.predict(X_va_fr)
actual_fr = df_val["fr_spot"].values[valid_va]
rmse_fr = np.sqrt(np.mean((actual_fr - preds_fr) ** 2))
print(f"  FR RMSE: {rmse_fr:.3f}, best_iter: {model_fr.get_best_iteration()}, time: {time.time()-t1:.0f}s")

# UK
print("\n  --- UK (raw target) ---")
t2 = time.time()
model_uk = CatBoostRegressor(**CB_PARAMS)
model_uk.fit(Pool(df_train[features], df_train["uk_spot"]),
             eval_set=Pool(df_val[features], df_val["uk_spot"]),
             early_stopping_rounds=100, verbose=0)
preds_uk = model_uk.predict(df_val[features])
rmse_uk = np.sqrt(np.mean((df_val["uk_spot"].values - preds_uk) ** 2))
print(f"  UK RMSE: {rmse_uk:.3f}, best_iter: {model_uk.get_best_iteration()}, time: {time.time()-t2:.0f}s")

# ── SHAP ──────────────────────────────────────────────────────────────────
print("\n[5/7] Computing SHAP values...")

print("  FR SHAP...")
t3 = time.time()
shap_fr = np.abs(shap.TreeExplainer(model_fr).shap_values(X_va_fr)).mean(axis=0)
shap_df_fr = pd.DataFrame({"feature": features, "shap": shap_fr}).sort_values("shap", ascending=False).reset_index(drop=True)
print(f"  Done in {time.time()-t3:.0f}s")

print("  UK SHAP...")
t4 = time.time()
shap_uk = np.abs(shap.TreeExplainer(model_uk).shap_values(df_val[features])).mean(axis=0)
shap_df_uk = pd.DataFrame({"feature": features, "shap": shap_uk}).sort_values("shap", ascending=False).reset_index(drop=True)
print(f"  Done in {time.time()-t4:.0f}s")

# ── Correlation dedup ────────────────────────────────────────────────────
print(f"\n[6/7] Correlation dedup (threshold={CORR_THRESHOLD})...")

def corr_dedup(df_feat, shap_df, threshold, label):
    ranked = shap_df["feature"].tolist()
    corr = df_feat[ranked].corr()
    dropped, kept, removed = {}, set(), set()

    for feat in ranked:
        if feat in removed:
            continue
        kept.add(feat)
        for other in ranked:
            if other == feat or other in removed or other in kept:
                continue
            if abs(corr.loc[feat, other]) > threshold:
                removed.add(other)
                dropped[other] = (feat, corr.loc[feat, other])

    surviving = [f for f in ranked if f in kept]
    print(f"\n  {label}: {len(ranked)} -> {len(surviving)} (dropped {len(dropped)})")
    if dropped:
        shap_lookup = dict(zip(shap_df["feature"], shap_df["shap"]))
        top_dropped = sorted(dropped.items(), key=lambda x: shap_lookup.get(x[0], 0), reverse=True)[:15]
        for d, (k, c) in top_dropped:
            print(f"    {d:45s} -> {k:45s} (r={c:.3f})")
    return surviving, dropped

surviving_fr, dropped_fr = corr_dedup(X_va_fr, shap_df_fr, CORR_THRESHOLD, "FR")
surviving_uk, dropped_uk = corr_dedup(df_val[features], shap_df_uk, CORR_THRESHOLD, "UK")

# ── Save ──────────────────────────────────────────────────────────────────
print("\n[7/7] Saving...")
output = {
    "threshold": CORR_THRESHOLD,
    "target_transform": "stationary (spot - roll_168h_mean)",
    "fr_spot": surviving_fr,
    "uk_spot": surviving_uk,
}
with open("outputs/shap_ranking_v4_stationary.json", "w") as f:
    json.dump(output, f, indent=2)
print("  Saved to outputs/shap_ranking_v4_stationary.json")

# ── Top-30 ────────────────────────────────────────────────────────────────
for label, shap_df, dropped in [("FR (stationary)", shap_df_fr, dropped_fr),
                                 ("UK (raw)", shap_df_uk, dropped_uk)]:
    print(f"\n{'='*70}")
    print(f"  TOP-30 — {label}")
    print(f"{'='*70}")
    for i, row in shap_df.head(30).iterrows():
        d = " (dropped)" if row["feature"] in dropped else ""
        print(f"  {i+1:3d}. {row['feature']:50s}  {row['shap']:.4f}{d}")

# ── Comparison v3 vs v4 ──────────────────────────────────────────────────
try:
    with open("outputs/shap_ranking_v3_clean.json") as f:
        v3 = json.load(f)
    print(f"\n{'='*70}")
    print("  v3 vs v4 — FR top-10 rank changes")
    print(f"{'='*70}")
    for i, feat in enumerate(v3["fr_spot"][:10]):
        if feat in surviving_fr:
            v4_rank = surviving_fr.index(feat) + 1
            print(f"  v3#{i+1:2d} -> v4#{v4_rank:2d}  {feat}")
        else:
            print(f"  v3#{i+1:2d} -> DROPPED  {feat}")
except FileNotFoundError:
    pass

print(f"\nTotal time: {time.time()-t0:.0f}s")
