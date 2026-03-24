"""Quick check: RMSE by month for the top methods to see WHERE we lose."""

import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features
from catboost import CatBoostRegressor, Pool
import yaml, warnings
warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
train_fe = build_features(train, config)

holdout_start = config["validation"]["holdout_start"]
mask_val = train_fe["datetime_CET"] >= holdout_start
df_train = train_fe[~mask_val].copy()
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

base_feat = [f for f in clean_ranking["fr_spot"][:20] if f in df_train.columns]
extras = [f for f in CAT32_FR if f in df_train.columns and f not in base_feat]
features = base_feat + extras

dt = pd.to_datetime(df_train["datetime_CET"])
days_ago = (dt.max() - dt).dt.total_seconds() / 86400
weights = np.exp(-2.0 * days_ago / 365).values

spot_va = df_val["fr_spot"].values
months_va = pd.to_datetime(df_val["datetime_CET"]).dt.to_period("M")

# Rolling stats for val
full_la = train_fe["fr_spot_la"]
roll_168_mean = full_la.rolling(168, min_periods=24).mean().values
roll_168_std = full_la.rolling(168, min_periods=24).std().values
n_tr = len(df_train)

roll_168_mean_tr = roll_168_mean[:n_tr]
roll_168_std_tr = roll_168_std[:n_tr]
roll_168_mean_va = roll_168_mean[n_tr:n_tr+len(df_val)]
roll_168_std_va = roll_168_std[n_tr:n_tr+len(df_val)]

spot_la_va = df_val["fr_spot_la"].values
spot_la_tr = df_train["fr_spot_la"].values
spot_tr = df_train["fr_spot"].values

# ── Train 3 models ────────────────────────────────────────────────────────
models = {}

# A) arcsinh(spot) — baseline
m = CatBoostRegressor(**CB_PARAMS)
m.fit(Pool(df_train[features], np.arcsinh(spot_tr), weight=weights),
      eval_set=Pool(df_val[features], np.arcsinh(spot_va)),
      early_stopping_rounds=100, verbose=0)
models["arcsinh(spot)"] = np.sinh(m.predict(df_val[features]))

# B) spot - spot_la
y_tr = spot_tr - spot_la_tr
m = CatBoostRegressor(**CB_PARAMS)
m.fit(Pool(df_train[features], y_tr, weight=weights),
      eval_set=Pool(df_val[features], spot_va - spot_la_va),
      early_stopping_rounds=100, verbose=0)
models["spot - spot_la"] = spot_la_va + m.predict(df_val[features])

# C) spot - roll_168h_mean
y_tr_c = spot_tr - roll_168_mean_tr
y_va_c = spot_va - roll_168_mean_va
valid_tr_c = np.isfinite(y_tr_c)
valid_va_c = np.isfinite(y_va_c)
m = CatBoostRegressor(**CB_PARAMS)
m.fit(Pool(df_train.loc[valid_tr_c, features], y_tr_c[valid_tr_c], weight=weights[valid_tr_c]),
      eval_set=Pool(df_val.loc[valid_va_c, features], y_va_c[valid_va_c]),
      early_stopping_rounds=100, verbose=0)
models["spot - roll_168h"] = roll_168_mean_va + m.predict(df_val[features])

# D) z-score 168h
std_tr = np.clip(roll_168_std_tr, 1, None)
std_va = np.clip(roll_168_std_va, 1, None)
y_tr_d = (spot_tr - roll_168_mean_tr) / std_tr
y_va_d = (spot_va - roll_168_mean_va) / std_va
valid_tr_d = np.isfinite(y_tr_d)
valid_va_d = np.isfinite(y_va_d)
m = CatBoostRegressor(**CB_PARAMS)
m.fit(Pool(df_train.loc[valid_tr_d, features], y_tr_d[valid_tr_d], weight=weights[valid_tr_d]),
      eval_set=Pool(df_val.loc[valid_va_d, features], y_va_d[valid_va_d]),
      early_stopping_rounds=100, verbose=0)
models["z-score 168h"] = roll_168_mean_va + m.predict(df_val[features]) * std_va

# ── RMSE by month ─────────────────────────────────────────────────────────
print("=" * 90)
print("  RMSE PAR MOIS — comparaison des methodes")
print("=" * 90)

header = f"{'Mois':>10s}  {'N':>5s}  {'MeanPrice':>9s}  {'StdPrice':>8s}"
for name in models:
    header += f"  {name:>16s}"
print(header)
print("-" * 90)

for month in sorted(months_va.unique()):
    mask = months_va == month
    n = mask.sum()
    mean_p = spot_va[mask].mean()
    std_p = spot_va[mask].std()
    row = f"{str(month):>10s}  {n:5d}  {mean_p:9.1f}  {std_p:8.1f}"
    for name, preds in models.items():
        rmse = np.sqrt(np.mean((spot_va[mask] - preds[mask]) ** 2))
        row += f"  {rmse:16.2f}"
    print(row)

# Total
row = f"{'TOTAL':>10s}  {len(spot_va):5d}  {spot_va.mean():9.1f}  {spot_va.std():8.1f}"
for name, preds in models.items():
    rmse = np.sqrt(np.mean((spot_va - preds) ** 2))
    row += f"  {rmse:16.2f}"
print("-" * 90)
print(row)

# ── Bias by month ─────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  BIAS PAR MOIS")
print("=" * 90)

header = f"{'Mois':>10s}"
for name in models:
    header += f"  {name:>16s}"
print(header)
print("-" * 90)

for month in sorted(months_va.unique()):
    mask = months_va == month
    row = f"{str(month):>10s}"
    for name, preds in models.items():
        bias = np.mean(spot_va[mask] - preds[mask])
        row += f"  {bias:+16.1f}"
    print(row)

# ── Stationarity of TARGET in train ───────────────────────────────────────
print("\n" + "=" * 90)
print("  STATIONARITY: variance du TARGET dans le train (std par trimestre)")
print("=" * 90)

quarters_tr = pd.to_datetime(df_train["datetime_CET"]).dt.to_period("Q")

targets_tr = {
    "arcsinh(spot)": np.arcsinh(spot_tr),
    "spot - spot_la": spot_tr - spot_la_tr,
    "spot - roll_168h": spot_tr - roll_168_mean_tr,
    "z-score 168h": (spot_tr - roll_168_mean_tr) / std_tr,
}

for name, target in targets_tr.items():
    valid = np.isfinite(target)
    s = pd.Series(target[valid], index=quarters_tr[valid])
    stds = s.groupby(level=0).std()
    ratio = stds.max() / stds.min()
    print(f"\n  {name} (max/min std ratio: {ratio:.1f}x):")
    for q in stds.index:
        print(f"    {q}: std={stds[q]:8.2f}")
