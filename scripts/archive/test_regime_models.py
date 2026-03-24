"""Test two approaches to fix FR root cause: gas features dominating when gas doesn't set price.

Approach 1 — Two separate models:
  Model A: trained ONLY on gas-on-margin hours
  Model B: trained ONLY on nuclear/RE-on-margin hours
  At prediction: route each hour to the appropriate model via fr_gas_on_margin

Approach 2 — Masked features:
  Replace gas-based features with masked versions:
    spark_if_marginal = spark_spread * gas_on_margin (=0 when gas off)
  Single model but gas features are "silenced" when irrelevant

Baseline: config N (arcsinh + Cat32 + weights 2.0) → FR RMSE = 24.45

Usage: python scripts/test_regime_models.py
"""

import sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features
from catboost import CatBoostRegressor, Pool
import yaml

# ── Load data ─────────────────────────────────────────────────────────────
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

with open("outputs/shap_ranking_v3_clean.json") as f:
    clean_ranking = json.load(f)

print(f"Train: {df_train.shape}, Val: {df_val.shape}")
print(f"  Train gas-on-margin: {df_train['fr_gas_on_margin'].sum()} / {len(df_train)} "
      f"({df_train['fr_gas_on_margin'].mean()*100:.1f}%)")
print(f"  Val   gas-on-margin: {df_val['fr_gas_on_margin'].sum()} / {len(df_val)} "
      f"({df_val['fr_gas_on_margin'].mean()*100:.1f}%)")

# ── Common params ─────────────────────────────────────────────────────────
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

# Gas-related features that mislead the model when gas is NOT marginal
GAS_FEATURES = [
    "de_spark_spread", "fr_spark_spread",
    "fr_gas_price_if_marginal", "fr_thermal_need_x_gas",
    "fr_merit_order_cost", "es_thermal_floor",
    "uk_merit_order_cost", "fr_residual_x_spark",
    "fr_scarcity_barrier", "fr_hydro_opp_cost",
    "fr_spark_spread_log", "fr_asinh_spark",
]


def compute_weights(df, decay=2.0):
    dt = pd.to_datetime(df["datetime_CET"])
    max_dt = dt.max()
    days_ago = (max_dt - dt).dt.total_seconds() / 86400
    return np.exp(-decay * days_ago / 365)


def eval_rmse(actual, preds):
    return float(np.sqrt(np.mean((actual - preds) ** 2)))


def eval_bias(actual, preds):
    return float(np.mean(actual - preds))


def regime_report(actual, preds, gas_mask, label):
    """Print RMSE breakdown by regime."""
    rmse_all = eval_rmse(actual, preds)
    bias_all = eval_bias(actual, preds)

    gas = gas_mask.astype(bool)
    nuc = ~gas

    print(f"\n  {label}")
    print(f"  {'─' * 65}")
    print(f"  Overall:         RMSE={rmse_all:.2f}  Bias={bias_all:+.1f}  N={len(actual)}")

    if gas.sum() > 0:
        rmse_gas = eval_rmse(actual[gas], preds[gas])
        bias_gas = eval_bias(actual[gas], preds[gas])
        print(f"  Gas on margin:   RMSE={rmse_gas:.2f}  Bias={bias_gas:+.1f}  "
              f"N={gas.sum()} ({gas.mean()*100:.1f}%)")

    if nuc.sum() > 0:
        rmse_nuc = eval_rmse(actual[nuc], preds[nuc])
        bias_nuc = eval_bias(actual[nuc], preds[nuc])
        print(f"  Nuclear/RE:      RMSE={rmse_nuc:.2f}  Bias={bias_nuc:+.1f}  "
              f"N={nuc.sum()} ({(~gas).mean()*100:.1f}%)")

    # By price bin
    bins = [-500, 0, 20, 40, 60, 100, 5000]
    labels_b = ["<0", "0-20", "20-40", "40-60", "60-100", ">100"]
    price_bins = pd.cut(actual, bins=bins, labels=labels_b)
    for b in labels_b:
        mask = price_bins == b
        if mask.sum() > 0:
            r = eval_rmse(actual[mask], preds[mask])
            bi = eval_bias(actual[mask], preds[mask])
            pct = mask.sum() / len(actual) * 100
            print(f"    {b:>8s}: RMSE={r:6.2f}  Bias={bi:+6.1f}  "
                  f"N={mask.sum():4d} ({pct:5.1f}%)")

    return rmse_all


# ══════════════════════════════════════════════════════════════════════════
# BASELINE — Config N (arcsinh + Cat32 + weights 2.0)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  BASELINE — Config N (arcsinh + Cat32 + weights 2.0)")
print("=" * 70)

base_feat = [f for f in clean_ranking["fr_spot"][:20] if f in df_train.columns]
extras = [f for f in CAT32_FR if f in df_train.columns and f not in base_feat]
features_baseline = base_feat + extras
print(f"  Features: {len(features_baseline)}")

X_tr = df_train[features_baseline]
X_va = df_val[features_baseline]
y_tr = np.arcsinh(df_train["fr_spot"])
y_va_spot = df_val["fr_spot"].values
weights = compute_weights(df_train, 2.0).values

model_bl = CatBoostRegressor(**CB_PARAMS)
model_bl.fit(Pool(X_tr, y_tr, weight=weights), eval_set=Pool(X_va, np.arcsinh(y_va_spot)),
             early_stopping_rounds=100, verbose=0)
preds_bl = np.sinh(model_bl.predict(X_va))

gas_val = df_val["fr_gas_on_margin"].values
rmse_baseline = regime_report(y_va_spot, preds_bl, gas_val, "BASELINE")


# ══════════════════════════════════════════════════════════════════════════
# APPROACH 1 — Two separate models
# ══════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  APPROACH 1 — TWO SEPARATE MODELS (gas vs nuclear/RE)")
print("=" * 70)

# --- Model A: Gas on margin ---
mask_gas_tr = df_train["fr_gas_on_margin"] == 1
mask_gas_va = df_val["fr_gas_on_margin"] == 1

# For gas model: use full feature set (gas features are relevant)
features_gas = features_baseline.copy()
print(f"\n  Model A (gas): Train={mask_gas_tr.sum()}, Val={mask_gas_va.sum()}")
print(f"  Features: {len(features_gas)}")

if mask_gas_tr.sum() > 50 and mask_gas_va.sum() > 0:
    df_tr_gas = df_train[mask_gas_tr]
    df_va_gas = df_val[mask_gas_va]

    X_tr_gas = df_tr_gas[features_gas]
    X_va_gas = df_va_gas[features_gas]
    y_tr_gas = np.arcsinh(df_tr_gas["fr_spot"])
    w_gas = compute_weights(df_tr_gas, 2.0).values

    model_gas = CatBoostRegressor(**CB_PARAMS)
    model_gas.fit(Pool(X_tr_gas, y_tr_gas, weight=w_gas),
                  eval_set=Pool(X_va_gas, np.arcsinh(df_va_gas["fr_spot"])),
                  early_stopping_rounds=100, verbose=0)
    preds_gas = np.sinh(model_gas.predict(X_va_gas))
    print(f"  Gas model RMSE: {eval_rmse(df_va_gas['fr_spot'].values, preds_gas):.2f}")
else:
    print("  WARNING: Not enough gas-on-margin samples")
    preds_gas = np.array([])

# --- Model B: Nuclear/RE on margin ---
mask_nuc_tr = df_train["fr_gas_on_margin"] == 0
mask_nuc_va = df_val["fr_gas_on_margin"] == 0

# For nuclear model: REMOVE gas-dependent features, keep nuclear/RE/import features
features_nuc = [f for f in features_baseline if f not in GAS_FEATURES]
# Add nuclear/RE-specific features that might be lower ranked
nuclear_extras = [
    "fr_oversupply_mw", "fr_renewable_pen", "fr_import_price",
    "fr_baseload_surplus", "fr_implied_re_surplus",
    "fr_nuclear_avcap_f", "fr_opportunity_cost", "fr_dynamic_marginal",
    "fr_wind_f", "fr_solar_f", "fr_nuclear_change_48h",
    "fr_nuclear_rolling_7d_mean", "fr_nuclear_trend_3d",
    "fr_hydro_ror_f", "fr_basis_v2",
]
for f in nuclear_extras:
    if f in df_train.columns and f not in features_nuc:
        features_nuc.append(f)

print(f"\n  Model B (nuclear/RE): Train={mask_nuc_tr.sum()}, Val={mask_nuc_va.sum()}")
print(f"  Features: {len(features_nuc)}")
print(f"  Removed gas features: {[f for f in features_baseline if f in GAS_FEATURES]}")
print(f"  Added nuclear extras: {[f for f in nuclear_extras if f in df_train.columns and f not in features_baseline]}")

df_tr_nuc = df_train[mask_nuc_tr]
df_va_nuc = df_val[mask_nuc_va]

X_tr_nuc = df_tr_nuc[features_nuc]
X_va_nuc = df_va_nuc[features_nuc]
y_tr_nuc = np.arcsinh(df_tr_nuc["fr_spot"])
w_nuc = compute_weights(df_tr_nuc, 2.0).values

model_nuc = CatBoostRegressor(**CB_PARAMS)
model_nuc.fit(Pool(X_tr_nuc, y_tr_nuc, weight=w_nuc),
              eval_set=Pool(X_va_nuc, np.arcsinh(df_va_nuc["fr_spot"])),
              early_stopping_rounds=100, verbose=0)
preds_nuc = np.sinh(model_nuc.predict(X_va_nuc))
print(f"  Nuclear model RMSE: {eval_rmse(df_va_nuc['fr_spot'].values, preds_nuc):.2f}")

# --- Combine predictions ---
preds_two_model = np.zeros(len(df_val))
preds_two_model[mask_gas_va.values] = preds_gas
preds_two_model[mask_nuc_va.values] = preds_nuc

rmse_two = regime_report(y_va_spot, preds_two_model, gas_val, "APPROACH 1 — Two models")


# ══════════════════════════════════════════════════════════════════════════
# APPROACH 2 — Masked features (single model)
# ══════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  APPROACH 2 — MASKED FEATURES")
print("=" * 70)

# Create masked versions: gas features × gas_on_margin
# When gas is off, these features become 0 → model can't anchor on them
df_train_masked = df_train.copy()
df_val_masked = df_val.copy()

gas_cols_in_features = [f for f in GAS_FEATURES if f in features_baseline]
print(f"\n  Gas features to mask: {gas_cols_in_features}")

masked_feature_map = {}  # old_name → new_name
for col in gas_cols_in_features:
    new_col = col + "_masked"
    df_train_masked[new_col] = df_train_masked[col] * df_train_masked["fr_gas_on_margin"]
    df_val_masked[new_col] = df_val_masked[col] * df_val_masked["fr_gas_on_margin"]
    masked_feature_map[col] = new_col

# Build feature list: replace gas features with masked versions
features_masked = []
for f in features_baseline:
    if f in masked_feature_map:
        features_masked.append(masked_feature_map[f])
    else:
        features_masked.append(f)

# Also add fr_gas_on_margin itself as a feature so model knows the regime
if "fr_gas_on_margin" not in features_masked and "fr_gas_on_margin" in df_train.columns:
    features_masked.append("fr_gas_on_margin")

print(f"  Total features: {len(features_masked)}")

X_tr_m = df_train_masked[features_masked]
X_va_m = df_val_masked[features_masked]
y_tr_m = np.arcsinh(df_train_masked["fr_spot"])
weights_m = compute_weights(df_train_masked, 2.0).values

model_masked = CatBoostRegressor(**CB_PARAMS)
model_masked.fit(Pool(X_tr_m, y_tr_m, weight=weights_m),
                 eval_set=Pool(X_va_m, np.arcsinh(y_va_spot)),
                 early_stopping_rounds=100, verbose=0)
preds_masked = np.sinh(model_masked.predict(X_va_m))

rmse_masked = regime_report(y_va_spot, preds_masked, gas_val, "APPROACH 2 — Masked features")


# ══════════════════════════════════════════════════════════════════════════
# APPROACH 2B — Masked + keep originals (model sees both)
# ══════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  APPROACH 2B — MASKED + ORIGINAL (model sees both)")
print("=" * 70)

# Keep both original and masked versions → model can learn to use masked when useful
features_2b = features_baseline.copy()
for col in gas_cols_in_features:
    new_col = col + "_masked"
    if new_col not in features_2b:
        features_2b.append(new_col)
if "fr_gas_on_margin" not in features_2b:
    features_2b.append("fr_gas_on_margin")

print(f"  Features: {len(features_2b)} (original + masked)")

X_tr_2b = df_train_masked[features_2b]
X_va_2b = df_val_masked[features_2b]

model_2b = CatBoostRegressor(**CB_PARAMS)
model_2b.fit(Pool(X_tr_2b, np.arcsinh(df_train["fr_spot"]), weight=weights_m),
             eval_set=Pool(X_va_2b, np.arcsinh(y_va_spot)),
             early_stopping_rounds=100, verbose=0)
preds_2b = np.sinh(model_2b.predict(X_va_2b))

rmse_2b = regime_report(y_va_spot, preds_2b, gas_val, "APPROACH 2B — Masked + original")


# ══════════════════════════════════════════════════════════════════════════
# APPROACH 3 — Nuclear-regime features ONLY (drop ALL gas features)
# ══════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  APPROACH 3 — DROP ALL GAS FEATURES (single model)")
print("=" * 70)

# Simply remove gas features, keep everything else + add nuclear extras
features_no_gas = [f for f in features_baseline if f not in GAS_FEATURES]
for f in nuclear_extras:
    if f in df_train.columns and f not in features_no_gas:
        features_no_gas.append(f)

print(f"  Features: {len(features_no_gas)} (no gas features)")
print(f"  Dropped: {[f for f in features_baseline if f in GAS_FEATURES]}")

X_tr_ng = df_train[features_no_gas]
X_va_ng = df_val[features_no_gas]

model_ng = CatBoostRegressor(**CB_PARAMS)
model_ng.fit(Pool(X_tr_ng, np.arcsinh(df_train["fr_spot"]), weight=weights),
             eval_set=Pool(X_va_ng, np.arcsinh(y_va_spot)),
             early_stopping_rounds=100, verbose=0)
preds_ng = np.sinh(model_ng.predict(X_va_ng))

rmse_ng = regime_report(y_va_spot, preds_ng, gas_val, "APPROACH 3 — No gas features")


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  SUMMARY — FR RMSE COMPARISON")
print("=" * 70)

results = [
    ("Baseline (config N)", rmse_baseline),
    ("Approach 1: Two models (gas/nuc)", rmse_two),
    ("Approach 2: Masked features", rmse_masked),
    ("Approach 2B: Masked + original", rmse_2b),
    ("Approach 3: Drop gas features", rmse_ng),
]

best_rmse = min(r[1] for r in results)
for label, rmse in results:
    delta = rmse - rmse_baseline
    marker = " ***" if rmse == best_rmse else ""
    print(f"  {label:40s}  RMSE={rmse:7.3f}  Δ={delta:+6.2f}{marker}")

print("\n  Best approach: " + min(results, key=lambda r: r[1])[0])
