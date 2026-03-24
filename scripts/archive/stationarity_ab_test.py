"""Stationarity A/B test — sweep target anchor methods for FR and UK.

Tests:
  FR: rolling mean (48h-336h), EMA (72h-240h), rolling median 168h, basis (merit_order_cost)
  UK: basis (current), rolling mean 168h, EMA 168h

Same features + params for each market. Only the TARGET changes.
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
print("  STATIONARITY A/B TEST — Target Anchor Sweep")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
df = build_features(train, config)

holdout_start = config["validation"]["holdout_start"]
mask_val = (df["datetime_CET"] >= holdout_start).values
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

with open("outputs/uk_feature_research.json") as f:
    uk_research = json.load(f)
UK_FEATURES = [f for f in uk_research["confirmed_features"] if f in df_tr.columns]

UK_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 15000, "learning_rate": 0.03, "depth": 8,
    "l2_leaf_reg": 5, "colsample_bylevel": 0.8, "subsample": 0.8,
    "random_seed": 42, "verbose": 0, "use_best_model": True,
    "allow_writing_files": False,
}

# FR weights (recency + stability)
dates_tr = pd.to_datetime(df_tr["datetime_CET"])
days_ago = (dates_tr.max() - dates_tr).dt.total_seconds() / 86400
w_recency = np.exp(-2.0 * days_ago.values / 365)
roll_std = df_tr["fr_spot_la_roll_168h_std"].values
w_stability = 1.0 / np.clip(roll_std ** 2, 1, None)
fr_weights = w_recency * w_stability

# Create interaction feature (computed in pipeline, not in build_features)
if "fr_spot_la_roll_168h_mean" in df.columns and "uk_price_per_mw_7d" in df.columns:
    df["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
        df["fr_spot_la_roll_168h_mean"] * df["uk_price_per_mw_7d"]
    )
    df_tr = df[~mask_val].copy()
    df_va = df[mask_val].copy()

print(f"  FR features: {len(FR_FEATURES)}, UK features: {len(UK_FEATURES)}")


# ── Helper ────────────────────────────────────────────────────────────────

def evaluate(market, features, params, anchor_tr, anchor_va, spot_tr, spot_va,
             hours_va, weights=None, label=""):
    """Train CatBoost on (spot - anchor), evaluate RMSE on spot reconstruction."""
    y_tr = spot_tr - anchor_tr
    y_va = spot_va - anchor_va

    valid_tr = np.isfinite(y_tr) & np.isfinite(anchor_tr)
    valid_va = np.isfinite(y_va) & np.isfinite(anchor_va)
    if weights is not None:
        valid_tr = valid_tr & np.isfinite(weights) & (weights > 0)

    if valid_tr.sum() < 500 or valid_va.sum() < 100:
        print(f"  {label:40s}  SKIPPED (valid: tr={valid_tr.sum()}, va={valid_va.sum()})")
        return None

    src = df_tr if market == "fr" else df_tr
    src_va = df_va

    w = weights[valid_tr] if weights is not None else None

    pool_tr = Pool(src.loc[src.index[valid_tr], features], y_tr[valid_tr], weight=w)
    pool_va = Pool(src_va.loc[src_va.index[valid_va], features], y_va[valid_va])

    model = CatBoostRegressor(**params)
    model.fit(pool_tr, eval_set=pool_va, early_stopping_rounds=200, verbose=0)

    preds_dev = model.predict(src_va.loc[src_va.index[valid_va], features])
    preds_spot = anchor_va[valid_va] + preds_dev
    actual = spot_va[valid_va]

    rmse = np.sqrt(np.mean((actual - preds_spot) ** 2))
    bias = float(np.mean(actual - preds_spot))

    # HBC
    hrs = hours_va[valid_va]
    errors = actual - preds_spot
    hbc = {h: float(errors[hrs == h].mean()) for h in range(24) if (hrs == h).sum() > 0}
    corrected = preds_spot + np.array([hbc.get(h, 0) for h in hrs])
    rmse_hbc = np.sqrt(np.mean((actual - corrected) ** 2))

    best_iter = model.get_best_iteration()

    # Target stats
    y_tr_valid = y_tr[valid_tr]
    tgt_mean = float(np.mean(y_tr_valid))
    tgt_std = float(np.std(y_tr_valid))

    print(f"  {label:40s}  RMSE={rmse:7.4f}  +HBC={rmse_hbc:7.4f}  "
          f"bias={bias:+6.2f}  iters={best_iter:4d}  "
          f"tgt: m={tgt_mean:+6.1f} s={tgt_std:5.1f}")
    sys.stdout.flush()

    return {
        "label": label, "rmse": round(rmse, 4), "rmse_hbc": round(rmse_hbc, 4),
        "bias": round(bias, 2), "iters": best_iter,
        "tgt_mean": round(tgt_mean, 2), "tgt_std": round(tgt_std, 2),
    }


# ══════════════════════════════════════════════════════════════════════════
# FR SWEEP
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  FR — TARGET ANCHOR SWEEP (28 features, Optuna v2 params, variance weights)")
print("=" * 90)
print(f"  {'Method':40s}  {'RMSE':>7s}  {'+HBC':>7s}  {'Bias':>6s}  {'Iter':>5s}  {'Target stats':>16s}")
print("  " + "-" * 88)

fr_spot = df["fr_spot"].values
fr_spot_la = df["fr_spot_la"].values
fr_moc = df["fr_merit_order_cost"].values

fr_spot_tr = fr_spot[~mask_val]
fr_spot_va = fr_spot[mask_val]
fr_hours_va = df_va["hour"].values

fr_results = []

# A. Rolling mean — window sweep
for W in [48, 72, 120, 168, 240, 336]:
    anchor = pd.Series(fr_spot_la).rolling(W, min_periods=W).mean().values
    tag = " << CURRENT" if W == 168 else ""
    r = evaluate("fr", FR_FEATURES, FR_PARAMS,
                 anchor[~mask_val], anchor[mask_val],
                 fr_spot_tr, fr_spot_va, fr_hours_va,
                 weights=fr_weights, label=f"Rolling mean {W}h{tag}")
    if r:
        fr_results.append(r)

# B. EMA sweep
for span in [72, 120, 168, 240]:
    anchor = pd.Series(fr_spot_la).ewm(span=span).mean().values
    r = evaluate("fr", FR_FEATURES, FR_PARAMS,
                 anchor[~mask_val], anchor[mask_val],
                 fr_spot_tr, fr_spot_va, fr_hours_va,
                 weights=fr_weights, label=f"EMA span={span}h")
    if r:
        fr_results.append(r)

# C. Rolling median 168h
anchor = pd.Series(fr_spot_la).rolling(168, min_periods=168).median().values
r = evaluate("fr", FR_FEATURES, FR_PARAMS,
             anchor[~mask_val], anchor[mask_val],
             fr_spot_tr, fr_spot_va, fr_hours_va,
             weights=fr_weights, label="Rolling median 168h")
if r:
    fr_results.append(r)

# D. Basis (merit_order_cost)
r = evaluate("fr", FR_FEATURES, FR_PARAMS,
             fr_moc[~mask_val], fr_moc[mask_val],
             fr_spot_tr, fr_spot_va, fr_hours_va,
             weights=fr_weights, label="Basis (merit_order_cost)")
if r:
    fr_results.append(r)

# FR ranking
print(f"\n  FR RANKING (by RMSE+HBC):")
print(f"  {'#':>3s}  {'Method':40s}  {'+HBC':>7s}  {'vs 168h':>8s}")
print("  " + "-" * 62)
baseline_hbc = next((r["rmse_hbc"] for r in fr_results if "168h" in r["label"] and "Rolling mean" in r["label"]), None)
for i, r in enumerate(sorted(fr_results, key=lambda x: x["rmse_hbc"]), 1):
    delta = f"{r['rmse_hbc'] - baseline_hbc:+.4f}" if baseline_hbc else ""
    best = " <<<" if i == 1 else ""
    print(f"  {i:3d}  {r['label']:40s}  {r['rmse_hbc']:7.4f}  {delta:>8s}{best}")


# ══════════════════════════════════════════════════════════════════════════
# UK SWEEP
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  UK — TARGET ANCHOR SWEEP (150 features, current params, NO weights)")
print("=" * 90)
print(f"  {'Method':40s}  {'RMSE':>7s}  {'+HBC':>7s}  {'Bias':>6s}  {'Iter':>5s}  {'Target stats':>16s}")
print("  " + "-" * 88)

uk_spot = df["uk_spot"].values
uk_spot_la = df["uk_spot_la"].values
uk_moc = df["uk_merit_order_cost"].values

uk_spot_tr = uk_spot[~mask_val]
uk_spot_va = uk_spot[mask_val]
uk_hours_va = df_va["hour"].values

uk_results = []

# A. Basis (current)
r = evaluate("uk", UK_FEATURES, UK_PARAMS,
             uk_moc[~mask_val], uk_moc[mask_val],
             uk_spot_tr, uk_spot_va, uk_hours_va,
             label="Basis (merit_order_cost) << CURRENT")
if r:
    uk_results.append(r)

# B. Rolling mean sweep
for W in [48, 72, 120, 168, 240, 336]:
    anchor = pd.Series(uk_spot_la).rolling(W, min_periods=W).mean().values
    r = evaluate("uk", UK_FEATURES, UK_PARAMS,
                 anchor[~mask_val], anchor[mask_val],
                 uk_spot_tr, uk_spot_va, uk_hours_va,
                 label=f"Rolling mean {W}h")
    if r:
        uk_results.append(r)

# C. EMA sweep
for span in [72, 120, 168, 240]:
    anchor = pd.Series(uk_spot_la).ewm(span=span).mean().values
    r = evaluate("uk", UK_FEATURES, UK_PARAMS,
                 anchor[~mask_val], anchor[mask_val],
                 uk_spot_tr, uk_spot_va, uk_hours_va,
                 label=f"EMA span={span}h")
    if r:
        uk_results.append(r)

# D. Rolling median 168h
anchor = pd.Series(uk_spot_la).rolling(168, min_periods=168).median().values
r = evaluate("uk", UK_FEATURES, UK_PARAMS,
             anchor[~mask_val], anchor[mask_val],
             uk_spot_tr, uk_spot_va, uk_hours_va,
             label="Rolling median 168h")
if r:
    uk_results.append(r)

# UK ranking
print(f"\n  UK RANKING (by RMSE+HBC):")
print(f"  {'#':>3s}  {'Method':40s}  {'+HBC':>7s}  {'vs basis':>8s}")
print("  " + "-" * 62)
baseline_uk = next((r["rmse_hbc"] for r in uk_results if "CURRENT" in r["label"]), None)
for i, r in enumerate(sorted(uk_results, key=lambda x: x["rmse_hbc"]), 1):
    delta = f"{r['rmse_hbc'] - baseline_uk:+.4f}" if baseline_uk else ""
    best = " <<<" if i == 1 else ""
    print(f"  {i:3d}  {r['label']:40s}  {r['rmse_hbc']:7.4f}  {delta:>8s}{best}")


# ══════════════════════════════════════════════════════════════════════════
# COMBINED SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  COMBINED SUMMARY — SUM(FR + UK) RMSE+HBC")
print("=" * 90)

# Current combination
fr_current = next((r for r in fr_results if "168h" in r["label"] and "Rolling mean" in r["label"]), None)
uk_current = next((r for r in uk_results if "CURRENT" in r["label"]), None)

if fr_current and uk_current:
    current_sum = fr_current["rmse_hbc"] + uk_current["rmse_hbc"]
    print(f"\n  Current: FR={fr_current['rmse_hbc']:.4f} + UK={uk_current['rmse_hbc']:.4f} = {current_sum:.4f}")

    # Best FR + current UK
    best_fr = min(fr_results, key=lambda x: x["rmse_hbc"])
    print(f"  Best FR ({best_fr['label']}): {best_fr['rmse_hbc']:.4f} + UK={uk_current['rmse_hbc']:.4f} "
          f"= {best_fr['rmse_hbc'] + uk_current['rmse_hbc']:.4f}  "
          f"(delta={best_fr['rmse_hbc'] + uk_current['rmse_hbc'] - current_sum:+.4f})")

    # Current FR + best UK
    best_uk = min(uk_results, key=lambda x: x["rmse_hbc"])
    print(f"  Best UK ({best_uk['label']}): FR={fr_current['rmse_hbc']:.4f} + {best_uk['rmse_hbc']:.4f} "
          f"= {fr_current['rmse_hbc'] + best_uk['rmse_hbc']:.4f}  "
          f"(delta={fr_current['rmse_hbc'] + best_uk['rmse_hbc'] - current_sum:+.4f})")

    # Best FR + best UK
    print(f"  Best combo: FR={best_fr['rmse_hbc']:.4f} + UK={best_uk['rmse_hbc']:.4f} "
          f"= {best_fr['rmse_hbc'] + best_uk['rmse_hbc']:.4f}  "
          f"(delta={best_fr['rmse_hbc'] + best_uk['rmse_hbc'] - current_sum:+.4f})")

# Save results
output = {"fr": fr_results, "uk": uk_results, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
with open("outputs/stationarity_ab_test.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Total time: {time.time() - t0:.0f}s")
print(f"  Saved to outputs/stationarity_ab_test.json")
