"""New feature interactions test — Targeted physics-motivated features.

Tests adding a few new features to the existing CatBoost models:
  - stress_index: thermal_need × (1 - renewable_pen)
  - wind_ramp × congestion interaction
  - 14-day rolling means (bi-weekly cycle)
  - arbitrage signal: spread × unused_capacity

Each feature is tested individually and in combination to measure marginal gain.
"""

import sys, yaml, warnings, time, json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

print("=" * 90)
print("  NEW FEATURES TEST — Targeted interactions")
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
fr_anchor_va = ema_fr[mask_val]
fr_spot_va = df_va["fr_spot"].values
fr_y_tr = df_tr["fr_spot"].values - ema_fr[~mask_val]
fr_y_va = fr_spot_va - fr_anchor_va

uk_moc_va = df_va["uk_merit_order_cost"].values
uk_spot_va = df_va["uk_spot"].values
uk_y_tr = df_tr["uk_spot"].values - df_tr["uk_merit_order_cost"].values
uk_y_va = uk_spot_va - uk_moc_va

hours_va = df_va["hour"].values

fr_valid_tr = np.isfinite(fr_y_tr) & np.isfinite(ema_fr[~mask_val])
uk_valid_tr = np.isfinite(uk_y_tr)

# Sample weights
days_ago = (df_tr["datetime_CET"].max() - df_tr["datetime_CET"]).dt.total_seconds() / 86400
roll_std = df_tr["fr_spot_la"].rolling(168, min_periods=24).std().fillna(df_tr["fr_spot_la"].std())
fr_sw = np.exp(-2 * days_ago.values / 365) / np.clip(roll_std.values ** 2, 1, None)
fr_sw[~fr_valid_tr] = 0

cb_params_fr = config.get("catboost_params_fr_optuna_v2", config.get("catboost_params_fr", {}))
cb_params_uk = config.get("catboost_params_uk", {})


def compute_hbc(preds, actual, hours):
    errors = actual - preds
    hbc = {h: float(errors[hours == h].mean()) for h in range(24) if (hours == h).sum() > 0}
    corrected = preds + np.array([hbc.get(h, 0) for h in hours])
    return np.sqrt(np.mean((actual - corrected) ** 2))


# ══════════════════════════════════════════════════════════════════════
#  ADD NEW FEATURES TO DATAFRAMES
# ══════════════════════════════════════════════════════════════════════
print("\n  Adding new features...")

def add_new_features(d):
    """Add candidate features to a dataframe."""
    # 1. Stress index: thermal need × (1 - renewable penetration)
    d["fr_stress_index"] = d["fr_thermal_need_pos"] * (1 - d["fr_renewable_pen"].clip(0, 1))
    d["uk_stress_index"] = d["uk_thermal_need_pos"] * (1 - d["uk_renewable_pen"].clip(0, 1))

    # 2. Wind ramp × congestion (wind drops when imports constrained → price spike)
    if "fr_wind_ramp_6h" in d.columns and "fr_uk_max_utilization" in d.columns:
        d["fr_wind_ramp_x_congestion"] = d["fr_wind_ramp_6h"] * d["fr_uk_max_utilization"].clip(0, 1)
        d["uk_wind_ramp_x_congestion"] = d["fr_wind_ramp_6h"] * d["fr_uk_max_utilization"].clip(0, 1)
    else:
        d["fr_wind_ramp_x_congestion"] = 0
        d["uk_wind_ramp_x_congestion"] = 0

    # 3. 14-day rolling means (bi-weekly cycle)
    for col_base in ["fr_spot_la", "uk_spot_la"]:
        if col_base in d.columns:
            prefix = col_base.replace("_spot_la", "")
            d[f"{prefix}_spot_la_roll_336h_mean"] = d[col_base].rolling(336, min_periods=48).mean()
            d[f"{prefix}_spot_la_roll_336h_std"] = d[col_base].rolling(336, min_periods=48).std()

    # 4. Load deviation from recent average (demand surprise)
    for prefix in ["fr", "uk"]:
        load_col = f"{prefix}_load_f"
        if load_col in d.columns:
            load_7d = d[load_col].rolling(168, min_periods=24).mean()
            d[f"{prefix}_load_surprise"] = (d[load_col] - load_7d) / load_7d.clip(lower=1)

    # 5. Spread × unused capacity (arbitrage signal)
    if "fr_spot_la" in d.columns and "uk_spot_la" in d.columns:
        spread = d["fr_spot_la"] - d["uk_spot_la"]
        if "fr_uk_total_unused_capacity" in d.columns:
            d["fr_uk_arbitrage_signal"] = spread * d["fr_uk_total_unused_capacity"] / 10000
        else:
            d["fr_uk_arbitrage_signal"] = spread

    # 6. Nuclear × gas price (marginal cost signal)
    if "fr_nuclear_avcap_f" in d.columns and "fr_gas" in d.columns:
        nuke_low = (d["fr_nuclear_avcap_f"] < d["fr_nuclear_avcap_f"].quantile(0.3)).astype(float)
        d["fr_nuke_low_x_gas"] = nuke_low * d["fr_gas"]

    # 7. Rolling price acceleration (2nd derivative)
    if "fr_spot_la" in d.columns:
        d["fr_price_accel_24h"] = d["fr_spot_la"].diff(24).diff(24)
        d["uk_price_accel_24h"] = d["uk_spot_la"].diff(24).diff(24)

    return d


df_tr = add_new_features(df_tr)
df_va = add_new_features(df_va)

# List all new features
NEW_FEATURES = [
    "fr_stress_index", "uk_stress_index",
    "fr_wind_ramp_x_congestion", "uk_wind_ramp_x_congestion",
    "fr_spot_la_roll_336h_mean", "fr_spot_la_roll_336h_std",
    "uk_spot_la_roll_336h_mean", "uk_spot_la_roll_336h_std",
    "fr_load_surprise", "uk_load_surprise",
    "fr_uk_arbitrage_signal",
    "fr_nuke_low_x_gas",
    "fr_price_accel_24h", "uk_price_accel_24h",
]
NEW_FEATURES = [f for f in NEW_FEATURES if f in df_tr.columns]
print(f"  New features added: {len(NEW_FEATURES)}")
for f in NEW_FEATURES:
    nn = df_tr[f].notna().sum()
    print(f"    {f:40s}  non-null={nn}/{len(df_tr)}  mean={df_tr[f].mean():.2f}  std={df_tr[f].std():.2f}")


# ══════════════════════════════════════════════════════════════════════
#  A/B TESTS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("  A/B TESTS — Baseline vs + new features")
print(f"{'='*90}")

def train_and_eval(feat_list, label, market):
    """Train CB and return +HBC RMSE."""
    feat = [f for f in feat_list if f in df_tr.columns]
    if market == "FR":
        cb = CatBoostRegressor(**{**cb_params_fr, "verbose": 0})
        cb.fit(df_tr[feat].values[fr_valid_tr], fr_y_tr[fr_valid_tr],
               sample_weight=fr_sw[fr_valid_tr],
               eval_set=(df_va[feat].values, fr_y_va))
        preds_spot = fr_anchor_va + cb.predict(df_va[feat].values)
        rmse_hbc = compute_hbc(preds_spot, fr_spot_va, hours_va)
    else:
        cb = CatBoostRegressor(**{**cb_params_uk, "verbose": 0})
        cb.fit(df_tr[feat].values[uk_valid_tr], uk_y_tr[uk_valid_tr],
               eval_set=(df_va[feat].values, uk_y_va))
        preds_spot = uk_moc_va + cb.predict(df_va[feat].values)
        rmse_hbc = compute_hbc(preds_spot, uk_spot_va, hours_va)
    return rmse_hbc


# FR tests
print("\n  FR CatBoost:")
fr_base = train_and_eval(FR_FEAT, "baseline", "FR")
print(f"    Baseline ({len(FR_FEAT)} feat):           +HBC={fr_base:.2f}")

# Add all new FR features
fr_new = [f for f in NEW_FEATURES if "uk_" not in f or "fr_uk" in f]
fr_all_new = train_and_eval(FR_FEAT + fr_new, "all_new", "FR")
print(f"    + all new ({len(FR_FEAT) + len(fr_new)} feat):     +HBC={fr_all_new:.2f}  Δ={fr_all_new - fr_base:+.2f}")

# Test each group individually
feature_groups = {
    "stress_index": ["fr_stress_index"],
    "wind_ramp_x_cong": ["fr_wind_ramp_x_congestion"],
    "rolling_336h": ["fr_spot_la_roll_336h_mean", "fr_spot_la_roll_336h_std"],
    "load_surprise": ["fr_load_surprise"],
    "arbitrage": ["fr_uk_arbitrage_signal"],
    "nuke_low_x_gas": ["fr_nuke_low_x_gas"],
    "price_accel": ["fr_price_accel_24h"],
}

for group_name, feats in feature_groups.items():
    feats_avail = [f for f in feats if f in df_tr.columns]
    if not feats_avail:
        continue
    result = train_and_eval(FR_FEAT + feats_avail, group_name, "FR")
    delta = result - fr_base
    flag = " ***" if delta < -0.05 else ""
    print(f"    + {group_name:20s} ({len(feats_avail)} feat):  +HBC={result:.2f}  Δ={delta:+.2f}{flag}")


# UK tests
print("\n  UK CatBoost:")
uk_base = train_and_eval(UK_FEAT, "baseline", "UK")
print(f"    Baseline ({len(UK_FEAT)} feat):           +HBC={uk_base:.2f}")

uk_new = [f for f in NEW_FEATURES if "fr_" not in f or "fr_uk" in f]
uk_all_new = train_and_eval(UK_FEAT + uk_new, "all_new", "UK")
print(f"    + all new ({len(UK_FEAT) + len(uk_new)} feat):     +HBC={uk_all_new:.2f}  Δ={uk_all_new - uk_base:+.2f}")

uk_feature_groups = {
    "stress_index": ["uk_stress_index"],
    "wind_ramp_x_cong": ["uk_wind_ramp_x_congestion"],
    "rolling_336h": ["uk_spot_la_roll_336h_mean", "uk_spot_la_roll_336h_std"],
    "load_surprise": ["uk_load_surprise"],
    "arbitrage": ["fr_uk_arbitrage_signal"],
    "price_accel": ["uk_price_accel_24h"],
}

for group_name, feats in uk_feature_groups.items():
    feats_avail = [f for f in feats if f in df_tr.columns]
    if not feats_avail:
        continue
    result = train_and_eval(UK_FEAT + feats_avail, group_name, "UK")
    delta = result - uk_base
    flag = " ***" if delta < -0.05 else ""
    print(f"    + {group_name:20s} ({len(feats_avail)} feat):  +HBC={result:.2f}  Δ={delta:+.2f}{flag}")


# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("  SUMMARY")
print(f"{'='*90}")
print(f"  FR baseline: {fr_base:.2f}  → + all new: {fr_all_new:.2f}  Δ={fr_all_new - fr_base:+.2f}")
print(f"  UK baseline: {uk_base:.2f}  → + all new: {uk_all_new:.2f}  Δ={uk_all_new - uk_base:+.2f}")
print(f"  Combined baseline SUM: {fr_base + uk_base:.2f}")
print(f"  Combined + new SUM:    {fr_all_new + uk_all_new:.2f}")

print(f"\n  Total time: {time.time() - t0:.0f}s")
