"""A/B test — Peak hour weighting for FR.

Tests: multiply sample weights by a factor for evening peak hours (18-21)
where HBC shows -4 to -6 EUR systematic under-prediction.

Variants:
  - Baseline: current weights (recency * stability)
  - Peak 1.5x: hours 18-21 get w *= 1.5
  - Peak 2.0x: hours 18-21 get w *= 2.0
  - Peak 3.0x: hours 18-21 get w *= 3.0
  - Extended peak 1.5x: hours 17-22 get w *= 1.5
  - Morning+evening 1.5x: hours 7-8 + 18-21 get w *= 1.5 (both HBC problem zones)
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
print("  PEAK WEIGHTING A/B TEST — FR")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
df = build_features(train, config)

holdout_start = config["validation"]["holdout_start"]
mask_val = (df["datetime_CET"] >= holdout_start).values

# Create interaction feature
if "fr_spot_la_roll_168h_mean" in df.columns and "uk_price_per_mw_7d" in df.columns:
    df["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
        df["fr_spot_la_roll_168h_mean"] * df["uk_price_per_mw_7d"]
    )

df_tr = df[~mask_val].copy()
df_va = df[mask_val].copy()

print(f"  Data loaded in {time.time() - t0:.0f}s  |  Train: {len(df_tr)}, Val: {len(df_va)}")

# ── Features & params ────────────────────────────────────────────────────
FR_FEATURES = [
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

FR_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 15000, "learning_rate": 0.059, "depth": 3,
    "l2_leaf_reg": 4.42, "subsample": 0.533, "colsample_bylevel": 0.228,
    "min_child_samples": 14, "random_strength": 0.9,
    "random_seed": 42, "verbose": 0, "use_best_model": True,
    "allow_writing_files": False,
}

# ── Target: EMA 240h (current best) ─────────────────────────────────────
fr_spot_la_full = df["fr_spot_la"].values
ema_full = pd.Series(fr_spot_la_full).ewm(span=240).mean().values
anchor_tr = ema_full[~mask_val]
anchor_va = ema_full[mask_val]

fr_spot_tr = df_tr["fr_spot"].values
fr_spot_va = df_va["fr_spot"].values
hours_tr = df_tr["hour"].values
hours_va = df_va["hour"].values

y_tr = fr_spot_tr - anchor_tr
y_va = fr_spot_va - anchor_va
valid_tr = np.isfinite(y_tr) & np.isfinite(anchor_tr)
valid_va = np.isfinite(y_va) & np.isfinite(anchor_va)

# Base weights (recency + stability)
dates_tr = pd.to_datetime(df_tr["datetime_CET"])
days_ago = (dates_tr.max() - dates_tr).dt.total_seconds() / 86400
w_recency = np.exp(-2.0 * days_ago.values / 365)
roll_std = df_tr["fr_spot_la_roll_168h_std"].values
w_stability = 1.0 / np.clip(roll_std ** 2, 1, None)
w_stability = np.where(np.isnan(w_stability), 1.0, w_stability)
base_weights = w_recency * w_stability

# Filter valid for weights
valid_tr = valid_tr & np.isfinite(base_weights) & (base_weights > 0)


# ── Helper ────────────────────────────────────────────────────────────────
def evaluate(weights, label):
    """Train CatBoost with given weights, return RMSE + hourly breakdown."""
    pool_tr = Pool(df_tr.loc[df_tr.index[valid_tr], FR_FEATURES],
                   y_tr[valid_tr], weight=weights[valid_tr])
    pool_va = Pool(df_va.loc[df_va.index[valid_va], FR_FEATURES],
                   y_va[valid_va])

    model = CatBoostRegressor(**FR_PARAMS)
    model.fit(pool_tr, eval_set=pool_va, early_stopping_rounds=200, verbose=0)

    preds_dev = model.predict(df_va.loc[df_va.index[valid_va], FR_FEATURES])
    preds_spot = anchor_va[valid_va] + preds_dev
    actual = fr_spot_va[valid_va]
    hrs = hours_va[valid_va]

    rmse = np.sqrt(np.mean((actual - preds_spot) ** 2))
    bias = float(np.mean(actual - preds_spot))

    # HBC
    errors = actual - preds_spot
    hbc = {h: float(errors[hrs == h].mean()) for h in range(24) if (hrs == h).sum() > 0}
    corrected = preds_spot + np.array([hbc.get(h, 0) for h in hrs])
    rmse_hbc = np.sqrt(np.mean((actual - corrected) ** 2))

    # Hourly RMSE
    hourly_rmse = {}
    for h in range(24):
        mask = hrs == h
        if mask.sum() > 0:
            hourly_rmse[h] = float(np.sqrt(np.mean((actual[mask] - preds_spot[mask]) ** 2)))

    # Peak hour RMSE (18-21)
    peak_mask = np.isin(hrs, [18, 19, 20, 21])
    rmse_peak = np.sqrt(np.mean((actual[peak_mask] - preds_spot[peak_mask]) ** 2))

    # Non-peak RMSE
    non_peak_mask = ~peak_mask
    rmse_non_peak = np.sqrt(np.mean((actual[non_peak_mask] - preds_spot[non_peak_mask]) ** 2))

    iters = model.get_best_iteration()

    print(f"  {label:35s}  RMSE={rmse:7.4f}  +HBC={rmse_hbc:7.4f}  "
          f"peak={rmse_peak:7.2f}  rest={rmse_non_peak:7.2f}  "
          f"bias={bias:+6.2f}  iters={iters:4d}")
    sys.stdout.flush()

    return {
        "label": label, "rmse": round(rmse, 4), "rmse_hbc": round(rmse_hbc, 4),
        "bias": round(bias, 2), "iters": iters,
        "rmse_peak_18_21": round(rmse_peak, 4), "rmse_non_peak": round(rmse_non_peak, 4),
        "hourly_rmse": hourly_rmse, "hbc": hbc,
    }


# ══════════════════════════════════════════════════════════════════════════
# RUN TESTS
# ══════════════════════════════════════════════════════════════════════════
print(f"\n  {'Method':35s}  {'RMSE':>7s}  {'+HBC':>7s}  {'Peak':>7s}  {'Rest':>7s}  {'Bias':>6s}  {'Iter':>5s}")
print("  " + "-" * 90)

results = []

# 1. Baseline (no peak weighting)
results.append(evaluate(base_weights, "Baseline (no peak weight)"))

# 2. Peak evening (18-21) x1.5
w = base_weights.copy()
w[np.isin(hours_tr, [18, 19, 20, 21])] *= 1.5
results.append(evaluate(w, "Peak 18-21 x1.5"))

# 3. Peak evening (18-21) x2.0
w = base_weights.copy()
w[np.isin(hours_tr, [18, 19, 20, 21])] *= 2.0
results.append(evaluate(w, "Peak 18-21 x2.0"))

# 4. Peak evening (18-21) x3.0
w = base_weights.copy()
w[np.isin(hours_tr, [18, 19, 20, 21])] *= 3.0
results.append(evaluate(w, "Peak 18-21 x3.0"))

# 5. Extended peak (17-22) x1.5
w = base_weights.copy()
w[np.isin(hours_tr, [17, 18, 19, 20, 21, 22])] *= 1.5
results.append(evaluate(w, "Extended peak 17-22 x1.5"))

# 6. Morning + evening (7-8 + 18-21) x1.5
w = base_weights.copy()
w[np.isin(hours_tr, [7, 8, 18, 19, 20, 21])] *= 1.5
results.append(evaluate(w, "Morning+Evening 7-8,18-21 x1.5"))

# 7. All problem hours from HBC (|bias| > 2) x1.5
w = base_weights.copy()
w[np.isin(hours_tr, [7, 8, 9, 10, 11, 17, 18, 19, 20])] *= 1.5
results.append(evaluate(w, "All HBC problem hours x1.5"))

# 8. Inverse-HBC weighting: weight proportional to |HBC correction|
# Use baseline HBC as reference
baseline_hbc = results[0]["hbc"]
w = base_weights.copy()
for h in range(24):
    h_mask = hours_tr == h
    boost = 1.0 + abs(baseline_hbc.get(h, 0)) / 5.0  # scale: |5 EUR bias| → 2x weight
    w[h_mask] *= boost
results.append(evaluate(w, "Inverse-HBC proportional"))


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print(f"\n  RANKING (by RMSE+HBC):")
print(f"  {'#':>3s}  {'Method':35s}  {'+HBC':>7s}  {'vs base':>8s}  {'Peak':>7s}  {'Rest':>7s}")
print("  " + "-" * 70)
baseline_hbc_val = results[0]["rmse_hbc"]
for i, r in enumerate(sorted(results, key=lambda x: x["rmse_hbc"]), 1):
    delta = f"{r['rmse_hbc'] - baseline_hbc_val:+.4f}"
    best = " <<<" if i == 1 else ""
    print(f"  {i:3d}  {r['label']:35s}  {r['rmse_hbc']:7.4f}  {delta:>8s}  "
          f"{r['rmse_peak_18_21']:7.2f}  {r['rmse_non_peak']:7.2f}{best}")

# Save
with open("outputs/peak_weighting_ab_test.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  Total time: {time.time() - t0:.0f}s")
