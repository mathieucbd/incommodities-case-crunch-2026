"""A/B test — Interaction features + volatility target + cross-market momentum.

Tests:
  A. Baseline (28 features, EMA 240h target)
  B. + peak interactions (is_peak * scarcity, is_morning * thermal_need)
  C. + UK momentum (uk_spot_la.diff(24))
  D. + volatility-weighted target (residual / rolling_std_24h)
  E. All combined (B + C)
"""

import sys, yaml, warnings, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# ── Load data ─────────────────────────────────────────────────────────────
print("=" * 90)
print("  INTERACTION FEATURES + MOMENTUM A/B TEST")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
df = build_features(train, config)

holdout_start = config["validation"]["holdout_start"]
mask_val = (df["datetime_CET"] >= holdout_start).values

# Interaction feature (existing)
if "fr_spot_la_roll_168h_mean" in df.columns and "uk_price_per_mw_7d" in df.columns:
    df["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
        df["fr_spot_la_roll_168h_mean"] * df["uk_price_per_mw_7d"]
    )

# ── NEW FEATURES ─────────────────────────────────────────────────────────
hours = df["hour"].values

# B. Peak interactions
df["is_evening_peak"] = ((hours >= 18) & (hours <= 21)).astype(float)
df["is_morning_ramp"] = ((hours >= 7) & (hours <= 9)).astype(float)

# is_evening_peak * scarcity_ratio
if "euro_scarcity_ratio" in df.columns:
    df["X_peak_x_scarcity"] = df["is_evening_peak"] * df["euro_scarcity_ratio"]

# is_morning_ramp * thermal need (continental_residual_load as proxy)
if "continental_residual_load" in df.columns:
    df["X_morning_x_thermal"] = df["is_morning_ramp"] * df["continental_residual_load"]

# is_evening_peak * fr deviation
if "fr_spot_la_deviation_168h" in df.columns:
    df["X_peak_x_fr_dev"] = df["is_evening_peak"] * df["fr_spot_la_deviation_168h"]

# C. UK momentum (diff of uk_spot_la = change over 24h, already known at prediction time)
if "uk_spot_la" in df.columns:
    df["uk_spot_la_diff24"] = df["uk_spot_la"].diff(24)  # 24h price change
    df["uk_spot_la_diff48"] = df["uk_spot_la"].diff(48)  # 48h price change

# FR momentum too
if "fr_spot_la" in df.columns:
    df["fr_spot_la_diff24"] = df["fr_spot_la"].diff(24)
    df["fr_spot_la_diff48"] = df["fr_spot_la"].diff(48)

df_tr = df[~mask_val].copy()
df_va = df[mask_val].copy()

print(f"  Data loaded in {time.time() - t0:.0f}s  |  Train: {len(df_tr)}, Val: {len(df_va)}")

# ── Base config ──────────────────────────────────────────────────────────
BASE_FEATURES = [
    "fr_spot_la_roll_168h_mean", "fr_residual_zscore_14d", "uk_residual_zscore_14d",
    "fr_spot_la_deviation_168h", "continental_residual_load", "euro_scarcity_ratio",
    "wind_nuke_deviation_gap", "fr_residual_change_24h", "uk_spot_la_deviation_168h",
    "uk_price_per_mw_7d", "uk_spot_la_roll_168h_std", "fr_spot_la_deviation_24h",
    "fr_residual_ramp_3h", "de_residual_load", "fr_load_roll_168h_mean",
    "carbon_to_gas_ratio", "fr_mean_reversion_strength", "uk_load_change_24h",
    "uk_spot_la_roll_168h_mean", "doy_cos", "fr_spot_la_roll_168h_std",
    "fr_dynamic_marginal", "de_gas", "uk_nuclear_avail_ratio",
    "uk_load_roll_168h_mean", "ntc_dk1-uk_f", "fr_gas_roll_168h_mean",
    "X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d",
]

PEAK_FEATURES = ["X_peak_x_scarcity", "X_morning_x_thermal", "X_peak_x_fr_dev"]
MOMENTUM_FEATURES = ["uk_spot_la_diff24", "uk_spot_la_diff48",
                      "fr_spot_la_diff24", "fr_spot_la_diff48"]

FR_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 15000, "learning_rate": 0.059, "depth": 3,
    "l2_leaf_reg": 4.42, "subsample": 0.533, "colsample_bylevel": 0.228,
    "min_child_samples": 14, "random_strength": 0.9,
    "random_seed": 42, "verbose": 0, "use_best_model": True,
    "allow_writing_files": False,
}

# ── Target: EMA 240h ────────────────────────────────────────────────────
fr_spot_la_full = df["fr_spot_la"].values
ema_full = pd.Series(fr_spot_la_full).ewm(span=240).mean().values
anchor_tr = ema_full[~mask_val]
anchor_va = ema_full[mask_val]

fr_spot_tr = df_tr["fr_spot"].values
fr_spot_va = df_va["fr_spot"].values
hours_va = df_va["hour"].values

y_tr = fr_spot_tr - anchor_tr
y_va = fr_spot_va - anchor_va

# Base weights
dates_tr = pd.to_datetime(df_tr["datetime_CET"])
days_ago = (dates_tr.max() - dates_tr).dt.total_seconds() / 86400
w_recency = np.exp(-2.0 * days_ago.values / 365)
roll_std = df_tr["fr_spot_la_roll_168h_std"].values
w_stability = 1.0 / np.clip(roll_std ** 2, 1, None)
w_stability = np.where(np.isnan(w_stability), 1.0, w_stability)
weights = w_recency * w_stability

valid_tr = np.isfinite(y_tr) & np.isfinite(anchor_tr) & np.isfinite(weights) & (weights > 0)
valid_va = np.isfinite(y_va) & np.isfinite(anchor_va)

# Volatility-weighted target variant
roll_std_24h_full = pd.Series(fr_spot_la_full).rolling(24, min_periods=6).std().values
roll_std_24h_tr = roll_std_24h_full[~mask_val]
roll_std_24h_va = roll_std_24h_full[mask_val]
y_vol_tr = y_tr / np.clip(roll_std_24h_tr, 1, None)
y_vol_va = y_va / np.clip(roll_std_24h_va, 1, None)
valid_vol_tr = valid_tr & np.isfinite(y_vol_tr)
valid_vol_va = valid_va & np.isfinite(y_vol_va)


# ── Helper ────────────────────────────────────────────────────────────────
def evaluate(features, y_train, y_valid, v_tr, v_va, anchor_v, spot_v,
             w, hours_v, label, vol_rescale=None):
    """Train and evaluate with given features and target."""
    feat = [f for f in features if f in df_tr.columns]

    pool_tr = Pool(df_tr.loc[df_tr.index[v_tr], feat], y_train[v_tr], weight=w[v_tr])
    pool_va = Pool(df_va.loc[df_va.index[v_va], feat], y_valid[v_va])

    model = CatBoostRegressor(**FR_PARAMS)
    model.fit(pool_tr, eval_set=pool_va, early_stopping_rounds=200, verbose=0)

    preds_dev = model.predict(df_va.loc[df_va.index[v_va], feat])

    # If volatility-rescaled, un-rescale
    if vol_rescale is not None:
        preds_dev = preds_dev * vol_rescale[v_va]

    preds_spot = anchor_v[v_va] + preds_dev
    actual = spot_v[v_va]
    hrs = hours_v[v_va]

    rmse = np.sqrt(np.mean((actual - preds_spot) ** 2))
    bias = float(np.mean(actual - preds_spot))

    # HBC
    errors = actual - preds_spot
    hbc = {h: float(errors[hrs == h].mean()) for h in range(24) if (hrs == h).sum() > 0}
    corrected = preds_spot + np.array([hbc.get(h, 0) for h in hrs])
    rmse_hbc = np.sqrt(np.mean((actual - corrected) ** 2))

    iters = model.get_best_iteration()

    print(f"  {label:45s}  RMSE={rmse:7.4f}  +HBC={rmse_hbc:7.4f}  "
          f"bias={bias:+6.2f}  feat={len(feat):2d}  iters={iters:4d}")
    sys.stdout.flush()

    return {"label": label, "rmse": round(rmse, 4), "rmse_hbc": round(rmse_hbc, 4),
            "bias": round(bias, 2), "n_features": len(feat), "iters": iters}


# ══════════════════════════════════════════════════════════════════════════
# RUN TESTS
# ══════════════════════════════════════════════════════════════════════════
print(f"\n  {'Method':45s}  {'RMSE':>7s}  {'+HBC':>7s}  {'Bias':>6s}  {'#F':>4s}  {'Iter':>5s}")
print("  " + "-" * 85)

results = []

# A. Baseline
results.append(evaluate(
    BASE_FEATURES, y_tr, y_va, valid_tr, valid_va,
    anchor_va, fr_spot_va, weights, hours_va,
    "A. Baseline (28 feat, EMA 240h)"))

# B. + Peak interactions
results.append(evaluate(
    BASE_FEATURES + PEAK_FEATURES, y_tr, y_va, valid_tr, valid_va,
    anchor_va, fr_spot_va, weights, hours_va,
    "B. + Peak interactions (3 feat)"))

# C. + Momentum features
results.append(evaluate(
    BASE_FEATURES + MOMENTUM_FEATURES, y_tr, y_va, valid_tr, valid_va,
    anchor_va, fr_spot_va, weights, hours_va,
    "C. + Momentum FR/UK diff24/48 (4 feat)"))

# D. Volatility-weighted target
results.append(evaluate(
    BASE_FEATURES, y_vol_tr, y_vol_va, valid_vol_tr, valid_vol_va,
    anchor_va, fr_spot_va, weights, hours_va,
    "D. Volatility-weighted target (y/std24h)",
    vol_rescale=np.clip(roll_std_24h_va, 1, None)))

# E. Peak + Momentum combined
results.append(evaluate(
    BASE_FEATURES + PEAK_FEATURES + MOMENTUM_FEATURES, y_tr, y_va, valid_tr, valid_va,
    anchor_va, fr_spot_va, weights, hours_va,
    "E. All combined (B+C, 7 new feat)"))

# F. Just momentum (no peak)
results.append(evaluate(
    BASE_FEATURES + ["uk_spot_la_diff24"], y_tr, y_va, valid_tr, valid_va,
    anchor_va, fr_spot_va, weights, hours_va,
    "F. + UK momentum only (1 feat)"))

# ══════════════════════════════════════════════════════════════════════════
# RANKING
# ══════════════════════════════════════════════════════════════════════════
print(f"\n  RANKING (by RMSE+HBC):")
print(f"  {'#':>3s}  {'Method':45s}  {'+HBC':>7s}  {'vs base':>8s}")
print("  " + "-" * 68)
baseline = results[0]["rmse_hbc"]
for i, r in enumerate(sorted(results, key=lambda x: x["rmse_hbc"]), 1):
    delta = f"{r['rmse_hbc'] - baseline:+.4f}"
    best = " <<<" if i == 1 else ""
    print(f"  {i:3d}  {r['label']:45s}  {r['rmse_hbc']:7.4f}  {delta:>8s}{best}")

with open("outputs/interaction_features_ab_test.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  Total time: {time.time() - t0:.0f}s")
