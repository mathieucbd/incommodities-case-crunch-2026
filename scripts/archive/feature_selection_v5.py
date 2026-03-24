"""Feature selection v5 — Multi-stage rigorous selection for FR.

Pipeline:
  Stage 0: Baseline sweep (SHAP-ranked top-N)
  Stage 1: Noise probing (Boruta-style) — eliminate features below noise floor
  Stage 2: Forward bucket selection — greedily add semantic feature groups
  Stage 3: Backward bucket ablation — remove useless buckets
  Stage 4: RFE within remaining features — fine-tune count

Key diagnostic: best_iteration.
  Currently 53 with all 379 features → severe overfitting.
  Goal: best_iter > 200 AND RMSE ≈ 18-19.

Usage: cd "INCOMO 3" && python scripts/feature_selection_v5.py
"""

import sys, json, time, warnings
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features
from catboost import CatBoostRegressor, Pool
import yaml

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# ── Load data ─────────────────────────────────────────────────────────────
print("=" * 90)
print("  FEATURE SELECTION v5 — Multi-stage rigorous (FR)")
print("=" * 90)

print("\nLoading data...")
t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
train_fe = build_features(train, config)
print(f"  Done in {time.time() - t0:.0f}s — shape: {train_fe.shape}")

holdout_start = config["validation"]["holdout_start"]
mask_val = train_fe["datetime_CET"] >= holdout_start
df_train = train_fe[~mask_val].copy()
df_val = train_fe[mask_val].copy()

# ── Rolling stats ─────────────────────────────────────────────────────────
fr_la = train_fe["fr_spot_la"]
roll_mean = fr_la.rolling(168, min_periods=24).mean()
roll_std = fr_la.rolling(168, min_periods=24).std()

n_tr = len(df_train)
roll_mean_tr = roll_mean.iloc[:n_tr].values
roll_std_tr = roll_std.iloc[:n_tr].values
roll_mean_va = roll_mean.iloc[n_tr:n_tr + len(df_val)].values

spot_tr = df_train["fr_spot"].values
spot_va = df_val["fr_spot"].values

# ── Target & weights ──────────────────────────────────────────────────────
y_dev_tr = spot_tr - roll_mean_tr
y_dev_va = spot_va - roll_mean_va
valid_tr = np.isfinite(y_dev_tr)
valid_va = np.isfinite(y_dev_va)

dt = pd.to_datetime(df_train["datetime_CET"])
days_ago = (dt.max() - dt).dt.total_seconds() / 86400
time_decay = np.exp(-2.0 * days_ago.values / 365)
var_168h = np.clip(roll_std_tr ** 2, 1.0, None)
var_168h = np.where(np.isnan(var_168h), 1.0, var_168h)
weights = time_decay / var_168h

# ── Feature list (SHAP v4 ranking) ────────────────────────────────────────
with open("outputs/shap_ranking_v4_stationary.json") as f:
    v4_ranking = json.load(f)
fr_ranking = v4_ranking["fr_spot"]
all_features = [f for f in fr_ranking if f in df_train.columns]
print(f"  Features: {len(all_features)}, Train: {df_train.shape[0]}, Val: {df_val.shape[0]}")

CB_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 5000, "learning_rate": 0.03, "depth": 8,
    "l2_leaf_reg": 5, "subsample": 0.8, "random_seed": 42,
    "verbose": 0, "allow_writing_files": False, "use_best_model": True,
}


def train_eval(feat_list, label="", seed=42, verbose=True):
    """Train CatBoost on FR stationary target, return (rmse, best_iter, bias, model)."""
    params = {**CB_PARAMS, "random_seed": seed}
    model = CatBoostRegressor(**params)
    X_tr = df_train.loc[df_train.index[valid_tr], feat_list]
    X_va = df_val.loc[df_val.index[valid_va], feat_list]
    model.fit(
        Pool(X_tr, y_dev_tr[valid_tr], weight=weights[valid_tr]),
        eval_set=Pool(X_va, y_dev_va[valid_va]),
        early_stopping_rounds=100, verbose=0,
    )
    preds_dev = model.predict(df_val[feat_list])
    preds_spot = roll_mean_va + preds_dev
    rmse = np.sqrt(np.mean((spot_va - preds_spot) ** 2))
    bias = np.mean(spot_va - preds_spot)
    best_iter = model.get_best_iteration()
    if label and verbose:
        print(
            f"  {label:55s}  n={len(feat_list):3d}  "
            f"RMSE={rmse:6.2f}  iter={best_iter:4d}  bias={bias:+5.1f}"
        )
    return rmse, best_iter, bias, model


# ══════════════════════════════════════════════════════════════════════════
# STAGE 0 — BASELINE SWEEP
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  STAGE 0 — BASELINE SWEEP (top-N by SHAP v4)")
print("=" * 90)

sweep_ns = [10, 15, 20, 30, 50, 75, 100, 150, 200, len(all_features)]
baseline_results = []

for n in sweep_ns:
    feat = all_features[:n]
    rmse, b_iter, bias, _ = train_eval(feat, f"Top-{n}")
    baseline_results.append({"n": n, "rmse": rmse, "iter": b_iter, "bias": bias})

best_baseline = min(baseline_results, key=lambda x: x["rmse"])
print(f"\n  Best baseline: top-{best_baseline['n']}, RMSE={best_baseline['rmse']:.3f}, "
      f"iter={best_baseline['iter']}")


# ══════════════════════════════════════════════════════════════════════════
# STAGE 1 — NOISE PROBING (Boruta-style)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  STAGE 1 — NOISE PROBING (5 rounds, 20 noise features)")
print("=" * 90)

N_NOISE = 20
N_ROUNDS = 5
hit_counts = {f: 0 for f in all_features}

for round_i in range(N_ROUNDS):
    t1 = time.time()
    np.random.seed(round_i * 42 + 7)

    # Create noise features
    noise_cols = [f"_noise_{i}" for i in range(N_NOISE)]
    noise_tr = pd.DataFrame(
        np.random.randn(len(df_train), N_NOISE),
        columns=noise_cols, index=df_train.index,
    )
    noise_va = pd.DataFrame(
        np.random.randn(len(df_val), N_NOISE),
        columns=noise_cols, index=df_val.index,
    )

    combined_feat = all_features + noise_cols
    X_tr_c = pd.concat([df_train[all_features], noise_tr], axis=1)
    X_va_c = pd.concat([df_val[all_features], noise_va], axis=1)

    model = CatBoostRegressor(
        **{**CB_PARAMS, "random_seed": round_i, "iterations": 2000}
    )
    model.fit(
        Pool(X_tr_c.loc[valid_tr], y_dev_tr[valid_tr], weight=weights[valid_tr]),
        eval_set=Pool(X_va_c.loc[valid_va], y_dev_va[valid_va]),
        early_stopping_rounds=50, verbose=0,
    )

    importances = model.feature_importances_
    imp_dict = dict(zip(combined_feat, importances))

    # Noise floor = max importance among noise features
    noise_imps = [imp_dict[c] for c in noise_cols]
    noise_max = max(noise_imps)
    noise_mean = np.mean(noise_imps)

    n_above = 0
    for f in all_features:
        if imp_dict[f] > noise_max:
            hit_counts[f] += 1
            n_above += 1

    print(
        f"  Round {round_i + 1}/{N_ROUNDS}:  noise_max={noise_max:.4f}  "
        f"noise_mean={noise_mean:.4f}  above_noise={n_above}/{len(all_features)}  "
        f"({time.time() - t1:.0f}s)"
    )

# Classify features
confirmed = [f for f in all_features if hit_counts[f] >= 3]
tentative = [f for f in all_features if 1 <= hit_counts[f] < 3]
rejected = [f for f in all_features if hit_counts[f] == 0]

print(f"\n  RESULTS:")
print(f"    Confirmed:  {len(confirmed)} features (beat noise ≥3/5 rounds)")
print(f"    Tentative:  {len(tentative)} features (beat noise 1-2/5 rounds)")
print(f"    Rejected:   {len(rejected)} features (never beat noise)")

# Show some rejected features
if rejected:
    print(f"\n  Sample rejected features (first 20):")
    for f in rejected[:20]:
        print(f"    {f}")

# Evaluate confirmed features
rmse_conf, iter_conf, _, _ = train_eval(confirmed, "Confirmed features only")

# Also try confirmed + tentative
all_surviving = confirmed + tentative
rmse_surv, iter_surv, _, _ = train_eval(all_surviving, "Confirmed + tentative")


# ══════════════════════════════════════════════════════════════════════════
# STAGE 2 — BUCKET DEFINITION & FORWARD SELECTION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  STAGE 2 — FORWARD BUCKET SELECTION")
print("=" * 90)


def assign_bucket(f):
    """Assign feature to a semantic bucket based on name."""
    # Calendar
    if f in ("hour", "month", "dow", "quarter") or f.startswith(
        ("doy_", "is_", "days_")
    ):
        return "calendar"
    # Gas & Carbon & Emissions
    if (
        any(x in f for x in ["ttf_", "nbp_"])
        or f == "de_gas"
        or f.startswith(("carbon_", "eu_emission"))
        or "gas_momentum" in f
        or "gas_roll" in f
    ):
        return "gas_carbon"
    # Neighbor spot prices (DE, BE, NL, CH, etc.)
    if any(
        f.startswith(p)
        for p in [
            "de_spot", "be_spot", "nl_spot", "ch_spot",
            "no2_spot", "dk1_spot", "es_spot",
        ]
    ):
        return "neighbor_prices"
    # FR price lags & rolling stats
    if f.startswith(("fr_spot", "fr_asinh_spot", "fr_mean_reversion")):
        return "fr_price"
    # UK price lags & rolling stats
    if f.startswith(("uk_spot", "uk_asinh_spot", "uk_mean_reversion")):
        return "uk_price"
    # Nuclear (all countries)
    if "nuclear" in f or "nuke" in f:
        return "nuclear"
    # Wind / Solar / Renewable
    if any(x in f for x in ["wind", "solar", "renewable"]):
        return "renewables"
    # Load & Residual load (all countries)
    if "residual" in f or "_load" in f:
        return "load_residual"
    # Interconnectors & flows
    if any(
        f.startswith(p)
        for p in [
            "fr_uk_", "atc_", "ntc_", "flow_", "total_import",
            "uk_cheapest", "uk_import",
        ]
    ):
        return "interconnectors"
    # Spark spread / Merit order / Basis
    if any(x in f for x in ["spark", "merit", "basis"]):
        return "spark_merit"
    # Supply-demand ratios (scarcity, capacity, reserve, security)
    if any(
        x in f
        for x in [
            "scarcity", "capacity_margin", "reserve_margin",
            "security", "self_suf",
        ]
    ):
        return "supply_demand"
    # Thermal need / Baseload gap
    if any(x in f for x in ["thermal", "baseload"]):
        return "thermal"
    # Hydro / River temperatures
    if any(x in f for x in ["hydro", "river", "alpine"]):
        return "hydro_river"
    # Advanced proxies (Cat 32)
    if any(
        x in f
        for x in [
            "dynamic_marginal", "opportunity_cost", "price_per_mw",
            "gas_cost_per", "gas_price_if",
        ]
    ):
        return "advanced_proxies"
    # Z-scores
    if "zscore" in f:
        return "zscores"
    # Momentum / Changes / Ramps
    if any(x in f for x in ["change_", "momentum", "_ramp_"]):
        return "momentum"
    # Load-price signals
    if "signal" in f:
        return "signals"
    # Continental / Regional aggregates
    if any(
        f.startswith(p)
        for p in ["continental", "euro_", "iberian_", "nordic_", "itn_"]
    ):
        return "continental"
    # Spreads (cross-border)
    if f.startswith("spread_"):
        return "spreads"

    return "other"


# Group confirmed features into buckets
buckets = defaultdict(list)
for f in confirmed:
    buckets[assign_bucket(f)].append(f)

# Also compute SHAP-based bucket importance for ordering
shap_rank = {f: i for i, f in enumerate(all_features)}

print("\n  Buckets (from confirmed features):")
bucket_shap = {}
for bname in sorted(buckets.keys(), key=lambda b: -len(buckets[b])):
    bfeat = buckets[bname]
    # Average SHAP rank (lower = more important)
    avg_rank = np.mean([shap_rank.get(f, 999) for f in bfeat])
    bucket_shap[bname] = avg_rank
    print(f"    {bname:25s}  {len(bfeat):3d} features  (avg SHAP rank: {avg_rank:.0f})")

# Forward selection: add buckets ordered by avg SHAP rank
bucket_order = sorted(buckets.keys(), key=lambda b: bucket_shap[b])

print("\n  Forward selection (SHAP-ordered):")
selected_features = []
forward_history = []

for bucket in bucket_order:
    candidate = selected_features + buckets[bucket]
    rmse, b_iter, bias, _ = train_eval(candidate, seed=42, verbose=False)
    delta = rmse - forward_history[-1]["rmse"] if forward_history else 0
    forward_history.append({
        "bucket": bucket, "n": len(candidate), "n_added": len(buckets[bucket]),
        "rmse": rmse, "iter": b_iter, "delta": delta,
    })
    selected_features = candidate
    marker = " <<<" if rmse <= min(h["rmse"] for h in forward_history) else ""
    print(
        f"  + {bucket:25s} (+{len(buckets[bucket]):3d})  "
        f"total={len(candidate):3d}  RMSE={rmse:6.2f}  "
        f"iter={b_iter:4d}  Δ={delta:+.2f}{marker}"
    )

best_forward = min(forward_history, key=lambda x: x["rmse"])
print(
    f"\n  Best forward point: after '{best_forward['bucket']}', "
    f"n={best_forward['n']}, RMSE={best_forward['rmse']:.3f}"
)


# ══════════════════════════════════════════════════════════════════════════
# STAGE 3 — BACKWARD BUCKET ABLATION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  STAGE 3 — BACKWARD BUCKET ABLATION")
print("=" * 90)

# Start with all confirmed features, try removing each bucket
all_confirmed_feat = list(confirmed)  # preserve SHAP order
rmse_full, iter_full, _, _ = train_eval(all_confirmed_feat, "All confirmed")

print("\n  Ablation (remove one bucket at a time):")
ablation_results = []

for bucket in sorted(buckets.keys()):
    reduced = [f for f in all_confirmed_feat if f not in buckets[bucket]]
    if len(reduced) == len(all_confirmed_feat):
        continue  # bucket not in confirmed
    rmse_abl, iter_abl, _, _ = train_eval(reduced, seed=42, verbose=False)
    delta = rmse_abl - rmse_full
    impact = "HURTS" if delta > 0.1 else ("HELPS" if delta < -0.1 else "neutral")
    ablation_results.append({
        "bucket": bucket, "n_removed": len(buckets[bucket]),
        "rmse": rmse_abl, "iter": iter_abl, "delta": delta, "impact": impact,
    })
    print(
        f"    Remove {bucket:25s} (-{len(buckets[bucket]):3d})  "
        f"RMSE={rmse_abl:6.2f}  iter={iter_abl:4d}  "
        f"Δ={delta:+.2f}  [{impact}]"
    )

# Remove all neutral/helping buckets
harmful_buckets = [r["bucket"] for r in ablation_results if r["impact"] == "HELPS"]
if harmful_buckets:
    print(f"\n  Removing harmful buckets: {harmful_buckets}")
    pruned_feat = [f for f in all_confirmed_feat
                   if assign_bucket(f) not in harmful_buckets]
    rmse_pruned, iter_pruned, _, _ = train_eval(
        pruned_feat, "After removing harmful buckets"
    )
else:
    pruned_feat = all_confirmed_feat
    print("\n  No harmful buckets found — keeping all.")


# ══════════════════════════════════════════════════════════════════════════
# STAGE 4 — RFE WITH BEST_ITER MONITORING
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  STAGE 4 — RFE (iterative 10% removal)")
print("=" * 90)

current_feat = list(pruned_feat)
rfe_history = []
rfe_features_at_step = {}

while len(current_feat) >= 10:
    rmse, b_iter, bias, model = train_eval(
        current_feat, f"n={len(current_feat)}", seed=42
    )
    rfe_history.append({
        "n": len(current_feat), "rmse": rmse, "iter": b_iter, "bias": bias,
    })
    rfe_features_at_step[len(current_feat)] = list(current_feat)

    # Remove bottom 10%
    n_remove = max(1, int(len(current_feat) * 0.10))
    if len(current_feat) - n_remove < 10:
        break

    importances = model.feature_importances_
    imp_sorted = sorted(zip(current_feat, importances), key=lambda x: x[1])
    remove = {f for f, _ in imp_sorted[:n_remove]}
    current_feat = [f for f in current_feat if f not in remove]

# Also test very small sets
for n in [30, 25, 20, 15, 10]:
    top_n = all_features[:n]
    rmse, b_iter, bias, _ = train_eval(top_n, f"SHAP top-{n}", seed=42)
    rfe_history.append({"n": n, "rmse": rmse, "iter": b_iter, "bias": bias})

# Find optimal
best_rfe = min(rfe_history, key=lambda x: x["rmse"])


# ══════════════════════════════════════════════════════════════════════════
# STAGE 5 — ADD TENTATIVE FEATURES BACK
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  STAGE 5 — TRY ADDING TENTATIVE FEATURES")
print("=" * 90)

# Take the best RFE feature set
optimal_n = best_rfe["n"]
if optimal_n in rfe_features_at_step:
    optimal_feat = rfe_features_at_step[optimal_n]
else:
    optimal_feat = all_features[:optimal_n]

print(f"  Starting from optimal RFE set: n={len(optimal_feat)}, RMSE={best_rfe['rmse']:.3f}")

# Try adding each tentative feature one by one
if tentative:
    improvements = []
    for f in tentative:
        if f in optimal_feat:
            continue
        candidate = optimal_feat + [f]
        rmse, b_iter, _, _ = train_eval(candidate, seed=42, verbose=False)
        delta = rmse - best_rfe["rmse"]
        if delta < -0.02:
            improvements.append({"feature": f, "rmse": rmse, "delta": delta})

    if improvements:
        improvements.sort(key=lambda x: x["delta"])
        print(f"\n  Tentative features that help ({len(improvements)}):")
        for imp in improvements[:15]:
            print(f"    {imp['feature']:50s}  Δ={imp['delta']:+.3f}")

        # Try adding all helpful tentative features together
        helpful = [imp["feature"] for imp in improvements]
        combined = optimal_feat + helpful
        rmse_combined, iter_combined, _, _ = train_eval(
            combined, "Optimal + helpful tentative"
        )
    else:
        print("  No tentative feature improves RMSE.")
else:
    print("  No tentative features to test.")


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  SUMMARY — FULL RESULTS")
print("=" * 90)

print("\n  RFE CURVE:")
rfe_sorted = sorted(rfe_history, key=lambda x: x["n"])
for h in rfe_sorted:
    marker = " <<<" if h["rmse"] == best_rfe["rmse"] else ""
    print(
        f"    n={h['n']:3d}  RMSE={h['rmse']:6.3f}  "
        f"iter={h['iter']:4d}  bias={h['bias']:+5.1f}{marker}"
    )

print(f"\n  OPTIMAL: n={best_rfe['n']}, RMSE={best_rfe['rmse']:.3f}, "
      f"iter={best_rfe['iter']}")

# Compare with baselines
print("\n  COMPARISON:")
print(f"    All {len(all_features)} features:  {baseline_results[-1]['rmse']:.3f}  "
      f"(iter={baseline_results[-1]['iter']})")
print(f"    Best SHAP sweep:      {best_baseline['rmse']:.3f}  "
      f"(top-{best_baseline['n']}, iter={best_baseline['iter']})")
print(f"    Noise-probed:         {rmse_conf:.3f}  (iter={iter_conf})")
print(f"    Best RFE:             {best_rfe['rmse']:.3f}  (n={best_rfe['n']}, "
      f"iter={best_rfe['iter']})")

# Save optimal feature list
if optimal_n in rfe_features_at_step:
    final_features = rfe_features_at_step[optimal_n]
else:
    final_features = all_features[:optimal_n]

output = {
    "method": "feature_selection_v5",
    "n_features": len(final_features),
    "rmse": best_rfe["rmse"],
    "best_iteration": best_rfe["iter"],
    "features": final_features,
}
with open("outputs/feature_selection_v5_fr.json", "w") as fout:
    json.dump(output, fout, indent=2)
print(f"\n  Saved optimal features to outputs/feature_selection_v5_fr.json")

# Top-30 final features
print(f"\n  TOP-30 SELECTED FEATURES:")
for i, f in enumerate(final_features[:30]):
    shap_pos = shap_rank.get(f, "?")
    print(f"    {i + 1:3d}. {f:50s}  (SHAP rank: {shap_pos})")

print(f"\n  Total time: {time.time() - t0:.0f}s")
