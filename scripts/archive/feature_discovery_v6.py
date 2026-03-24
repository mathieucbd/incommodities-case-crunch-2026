"""Feature discovery v6 — Quantitative residual-driven feature search.

Approach: 100% data-driven, zero intuition.
  Phase 1: Train v7, extract residuals, analyze systematically
  Phase 2: Correlate ALL 471 features with residual → find what model misses
  Phase 3: Generate candidate features (interactions, thresholds, temporal variants)
           driven by Phase 2 findings
  Phase 4: Score candidates by |corr(candidate, residual)|
  Phase 5: Noise probing on expanded feature set
  Phase 6: Re-train with v7 params, measure improvement

Usage: cd "INCOMO 3" && python scripts/feature_discovery_v6.py
"""

import sys, json, time, warnings
import numpy as np
import pandas as pd
from itertools import combinations

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features
from catboost import CatBoostRegressor, Pool
import yaml

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# ── Load ──────────────────────────────────────────────────────────────────
print("=" * 90)
print("  FEATURE DISCOVERY v6 — Residual-driven quantitative search")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
train_fe = build_features(train, config)

holdout_start = config["validation"]["holdout_start"]
mask_val = train_fe["datetime_CET"] >= holdout_start
df_train = train_fe[~mask_val].copy()
df_val = train_fe[mask_val].copy()

# Rolling stats
fr_la = train_fe["fr_spot_la"]
roll_mean = fr_la.rolling(168, min_periods=24).mean()
roll_std = fr_la.rolling(168, min_periods=24).std()

n_tr = len(df_train)
roll_mean_tr = roll_mean.iloc[:n_tr].values
roll_std_tr = roll_std.iloc[:n_tr].values
roll_mean_va = roll_mean.iloc[n_tr:n_tr + len(df_val)].values

spot_tr = df_train["fr_spot"].values
spot_va = df_val["fr_spot"].values
hours_va = df_val["hour"].values
months_va = pd.to_datetime(df_val["datetime_CET"]).dt.month.values

# Target & weights
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

# Features
with open("outputs/shap_ranking_v4_stationary.json") as f:
    v4_ranking = json.load(f)
feat_all = [f for f in v4_ranking["fr_spot"] if f in df_train.columns]

with open("outputs/feature_selection_v5_fr.json") as f:
    fs_v5 = json.load(f)
feat_27 = fs_v5["features"]

# All numeric features (for residual correlation)
EXCLUDE = {"datetime_CET", "datetime_UTC", "fr_spot", "uk_spot"}
all_numeric = [c for c in train_fe.columns
               if c not in EXCLUDE and train_fe[c].dtype.kind in ("f", "i", "u", "b")]

V7_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 15000, "learning_rate": 0.03, "depth": 3,
    "l2_leaf_reg": 30, "subsample": 0.7, "colsample_bylevel": 0.5,
    "random_seed": 42, "verbose": 0, "allow_writing_files": False,
    "use_best_model": True,
}

print(f"  Data loaded in {time.time() - t0:.0f}s")
print(f"  Train: {df_train.shape[0]}, Val: {df_val.shape[0]}")
print(f"  v5 features: {len(feat_27)}, All numeric: {len(all_numeric)}")


def train_eval(feat_list, df_tr=None, df_va=None, y_tr=None, y_va=None,
               w=None, label="", extra_params=None):
    """Train CatBoost, return (rmse, best_iter, bias, model, preds_spot)."""
    if df_tr is None:
        df_tr = df_train
    if df_va is None:
        df_va = df_val
    if y_tr is None:
        y_tr = y_dev_tr
    if y_va is None:
        y_va = y_dev_va
    if w is None:
        w = weights

    vt = np.isfinite(y_tr)
    vv = np.isfinite(y_va)

    params = {**V7_PARAMS, **(extra_params or {})}
    model = CatBoostRegressor(**params)
    model.fit(
        Pool(df_tr.loc[df_tr.index[vt], feat_list], y_tr[vt], weight=w[vt]),
        eval_set=Pool(df_va.loc[df_va.index[vv], feat_list], y_va[vv]),
        early_stopping_rounds=200, verbose=0,
    )
    preds_dev = model.predict(df_va[feat_list])
    preds_spot = roll_mean_va + preds_dev
    rmse = np.sqrt(np.mean((spot_va - preds_spot) ** 2))
    bias = np.mean(spot_va - preds_spot)
    best_iter = model.get_best_iteration()
    if label:
        print(f"  {label:55s}  n={len(feat_list):3d}  "
              f"RMSE={rmse:6.2f}  iter={best_iter:4d}  bias={bias:+5.1f}")
    return rmse, best_iter, bias, model, preds_spot


# ══════════════════════════════════════════════════════════════════════════
# PHASE 1 — TRAIN v7 & EXTRACT RESIDUALS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  PHASE 1 — RESIDUAL EXTRACTION")
print("=" * 90)

rmse_v7, iter_v7, bias_v7, model_v7, preds_v7 = train_eval(feat_27, label="v7 baseline (27 feat)")
residuals = spot_va - preds_v7  # positive = under-prediction

# Also train with all features for comparison
rmse_all, _, _, _, preds_all = train_eval(feat_all, label="v7 all features")
residuals_all = spot_va - preds_all


# ══════════════════════════════════════════════════════════════════════════
# PHASE 2 — SYSTEMATIC RESIDUAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  PHASE 2 — RESIDUAL ANALYSIS")
print("=" * 90)

# 2a) Error by hour
print("\n  Error by hour (mean residual = bias by hour):")
for h in range(24):
    mask = hours_va == h
    mean_err = residuals[mask].mean()
    rmse_h = np.sqrt(np.mean(residuals[mask] ** 2))
    bar = "+" * max(0, int(mean_err)) + "-" * max(0, int(-mean_err))
    print(f"    h={h:2d}  bias={mean_err:+6.1f}  RMSE={rmse_h:5.1f}  {bar}")

# 2b) Error by month
print("\n  Error by month:")
for m in sorted(set(months_va)):
    mask = months_va == m
    mean_err = residuals[mask].mean()
    rmse_m = np.sqrt(np.mean(residuals[mask] ** 2))
    print(f"    month={m:2d}  bias={mean_err:+6.1f}  RMSE={rmse_m:5.1f}  n={mask.sum()}")

# 2c) Correlation of ALL features with residual
print("\n  Correlating ALL features with residual...")
corr_with_residual = {}
for feat in all_numeric:
    vals = df_val[feat].values
    finite = np.isfinite(vals) & np.isfinite(residuals)
    if finite.sum() < 100:
        continue
    r = np.corrcoef(vals[finite], residuals[finite])[0, 1]
    if np.isfinite(r):
        corr_with_residual[feat] = r

# Sort by absolute correlation
sorted_corr = sorted(corr_with_residual.items(), key=lambda x: abs(x[1]), reverse=True)

print(f"\n  TOP-30 features correlated with RESIDUAL (what model misses):")
print(f"  {'Feature':50s}  {'r':>7s}  {'In v5?':>6s}  {'In SHAP top-50?':>15s}")
for i, (feat, r) in enumerate(sorted_corr[:30]):
    in_v5 = "YES" if feat in feat_27 else "no"
    shap_pos = feat_all.index(feat) if feat in feat_all else 999
    in_top50 = f"#{shap_pos+1}" if shap_pos < 50 else "no"
    print(f"  {i+1:2d}. {feat:48s}  {r:+7.4f}  {in_v5:>6s}  {in_top50:>15s}")

# Features NOT in v5-27 that correlate with residual
missed_features = [(f, r) for f, r in sorted_corr if f not in feat_27 and abs(r) > 0.03]
print(f"\n  Features NOT in v5-27 with |r_residual| > 0.03: {len(missed_features)}")
for f, r in missed_features[:20]:
    print(f"    {f:50s}  r={r:+.4f}")

# 2d) Error by feature quantile (find non-linearities)
print("\n  Non-linearity check: RMSE by quantile of top features")
for feat in feat_27[:10]:
    vals = df_val[feat].values
    if not np.all(np.isfinite(vals)):
        continue
    q25, q50, q75, q90 = np.percentile(vals, [25, 50, 75, 90])
    bins = [
        ("Q1", vals <= q25),
        ("Q2", (vals > q25) & (vals <= q50)),
        ("Q3", (vals > q50) & (vals <= q75)),
        ("Q4", (vals > q75) & (vals <= q90)),
        ("TOP", vals > q90),
    ]
    parts = []
    for label, mask in bins:
        if mask.sum() > 0:
            rmse_bin = np.sqrt(np.mean(residuals[mask] ** 2))
            bias_bin = residuals[mask].mean()
            parts.append(f"{label}:{rmse_bin:5.1f}({bias_bin:+4.1f})")
    print(f"    {feat:45s}  {' | '.join(parts)}")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 3 — CANDIDATE FEATURE GENERATION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  PHASE 3 — CANDIDATE GENERATION (data-driven)")
print("=" * 90)

top10 = feat_27[:10]
top15 = feat_27[:15]

# Store candidates as (name, train_values, val_values)
candidates = {}

# 3a) Pairwise PRODUCTS of top-10 (45 candidates)
print("  Generating pairwise products of top-10...")
for f1, f2 in combinations(top10, 2):
    name = f"X_{f1}_x_{f2}"
    candidates[name] = (
        df_train[f1].values * df_train[f2].values,
        df_val[f1].values * df_val[f2].values,
    )

# 3b) Pairwise RATIOS of top-10 (90 candidates)
print("  Generating pairwise ratios of top-10...")
for f1 in top10:
    for f2 in top10:
        if f1 == f2:
            continue
        denom = df_val[f2].values
        denom_tr = df_train[f2].values
        # Avoid division by zero
        safe_denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        safe_denom_tr = np.where(np.abs(denom_tr) < 1e-6, 1e-6, denom_tr)
        name = f"R_{f1}_div_{f2}"
        candidates[name] = (
            df_train[f1].values / safe_denom_tr,
            df_val[f1].values / safe_denom,
        )

# 3c) Squared and abs transforms of top-10 (20 candidates)
print("  Generating squared/abs transforms...")
for f in top10:
    candidates[f"SQ_{f}"] = (df_train[f].values ** 2, df_val[f].values ** 2)
    candidates[f"ABS_{f}"] = (np.abs(df_train[f].values), np.abs(df_val[f].values))

# 3d) Threshold indicators at p75, p90, p95 of top-15 (45 candidates)
print("  Generating threshold indicators...")
for f in top15:
    vals_tr = df_train[f].values
    vals_va = df_val[f].values
    for pct, label in [(75, "p75"), (90, "p90"), (95, "p95")]:
        threshold = np.nanpercentile(vals_tr, pct)
        candidates[f"THR_{f}_{label}"] = (
            (vals_tr > threshold).astype(float),
            (vals_va > threshold).astype(float),
        )
        # Also below threshold
        low_threshold = np.nanpercentile(vals_tr, 100 - pct)
        candidates[f"THR_{f}_low{100-pct}"] = (
            (vals_tr < low_threshold).astype(float),
            (vals_va < low_threshold).astype(float),
        )

# 3e) Rate-of-change (24h diff) of top-15 (15 candidates)
print("  Generating rate-of-change features...")
for f in top15:
    vals_tr = pd.Series(df_train[f].values)
    vals_va = pd.Series(df_val[f].values)
    candidates[f"DELTA24_{f}"] = (
        (vals_tr - vals_tr.shift(24)).values,
        (vals_va - vals_va.shift(24)).values,
    )

# 3f) Alternative rolling windows for key price/load features
print("  Generating alternative rolling windows...")
key_raw = ["fr_spot_la", "uk_spot_la", "fr_load_f", "uk_load_f",
           "fr_residual_load", "uk_residual_load"]
key_raw = [f for f in key_raw if f in df_train.columns]

for f in key_raw:
    full_series = train_fe[f]
    for window in [48, 72, 96, 120, 144, 240, 336]:
        rm = full_series.rolling(window, min_periods=24).mean()
        rs = full_series.rolling(window, min_periods=24).std()
        dev = full_series - rm
        candidates[f"ROLL_{f}_mean_{window}h"] = (
            rm.iloc[:n_tr].values, rm.iloc[n_tr:n_tr + len(df_val)].values,
        )
        candidates[f"ROLL_{f}_std_{window}h"] = (
            rs.iloc[:n_tr].values, rs.iloc[n_tr:n_tr + len(df_val)].values,
        )
        candidates[f"ROLL_{f}_dev_{window}h"] = (
            dev.iloc[:n_tr].values, dev.iloc[n_tr:n_tr + len(df_val)].values,
        )

# 3g) Products of "missed features" (top corr with residual, not in v5) with top-5
print("  Generating interactions with missed features...")
missed_top10 = [f for f, r in missed_features[:10] if f in df_train.columns]
for mf in missed_top10:
    for tf in feat_27[:5]:
        candidates[f"MX_{mf}_x_{tf}"] = (
            df_train[mf].values * df_train[tf].values,
            df_val[mf].values * df_val[tf].values,
        )

print(f"\n  Total candidates generated: {len(candidates)}")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 4 — SCORE CANDIDATES BY CORRELATION WITH RESIDUAL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  PHASE 4 — CANDIDATE SCORING (|corr with residual|)")
print("=" * 90)

candidate_scores = {}
for name, (tr_vals, va_vals) in candidates.items():
    finite = np.isfinite(va_vals) & np.isfinite(residuals)
    if finite.sum() < 100:
        continue
    r = np.corrcoef(va_vals[finite], residuals[finite])[0, 1]
    if np.isfinite(r):
        candidate_scores[name] = abs(r)

# Sort by score
sorted_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])

print(f"\n  TOP-40 candidates by |r_residual|:")
for i, (name, score) in enumerate(sorted_candidates[:40]):
    print(f"    {i+1:3d}. {name:60s}  |r|={score:.4f}")

# Keep candidates with |r| > 0.03
significant = [(n, s) for n, s in sorted_candidates if s > 0.03]
print(f"\n  Candidates with |r| > 0.03: {len(significant)}")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 5 — ADD BEST CANDIDATES & TEST
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  PHASE 5 — TESTING CANDIDATE FEATURES")
print("=" * 90)

# Add top candidates to dataframes
top_candidates = [n for n, s in sorted_candidates[:50]]  # test top 50

# Create expanded dataframes
df_train_exp = df_train.copy()
df_val_exp = df_val.copy()

added = []
for name in top_candidates:
    tr_vals, va_vals = candidates[name]
    # Check no NaN issues
    if np.isnan(tr_vals[valid_tr]).sum() / valid_tr.sum() > 0.1:
        continue
    if np.isnan(va_vals).sum() / len(va_vals) > 0.1:
        continue
    df_train_exp[name] = tr_vals
    df_val_exp[name] = va_vals
    added.append(name)

print(f"  Added {len(added)} candidate features to dataset")

# 5a) Test: v5-27 + each candidate individually
print("\n  Individual candidate impact (added to v5-27):")
individual_results = []

for cand in added[:30]:  # test top 30 individually
    feat = feat_27 + [cand]
    rmse, b_iter, bias, _, _ = train_eval(
        feat, df_tr=df_train_exp, df_va=df_val_exp
    )
    delta = rmse - rmse_v7
    individual_results.append({"name": cand, "rmse": rmse, "delta": delta, "iter": b_iter})
    if delta < -0.02:
        print(f"    + {cand:55s}  RMSE={rmse:6.2f}  Δ={delta:+.3f}  iter={b_iter}")

helpful_individuals = [r for r in individual_results if r["delta"] < -0.02]
print(f"\n  Candidates that improve RMSE by >0.02: {len(helpful_individuals)}")

# 5b) Add ALL helpful candidates together
if helpful_individuals:
    helpful_names = [r["name"] for r in helpful_individuals]
    feat_expanded = feat_27 + helpful_names
    rmse_exp, iter_exp, bias_exp, _, preds_exp = train_eval(
        feat_expanded, df_tr=df_train_exp, df_va=df_val_exp,
        label=f"v5-27 + {len(helpful_names)} candidates"
    )

# 5c) Also test: add all significant candidates
if significant:
    sig_names = [n for n, s in significant if n in added]
    feat_sig = feat_27 + sig_names
    rmse_sig, iter_sig, _, _, _ = train_eval(
        feat_sig, df_tr=df_train_exp, df_va=df_val_exp,
        label=f"v5-27 + {len(sig_names)} significant (|r|>0.03)"
    )

# 5d) Test: add top missed features directly (not as interactions)
print("\n  Direct addition of missed features (not in v5-27, corr with residual):")
missed_direct = [f for f, r in missed_features[:20] if f in df_train.columns]
for n_add in [5, 10, 15, 20]:
    feat_missed = feat_27 + missed_direct[:n_add]
    rmse_m, iter_m, _, _, _ = train_eval(feat_missed, label=f"v5-27 + {n_add} missed")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 6 — NOISE PROBING ON EXPANDED SET
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  PHASE 6 — NOISE PROBING ON BEST EXPANDED SET")
print("=" * 90)

# Combine: v5-27 + helpful individuals + top missed direct
best_additions = []
if helpful_individuals:
    best_additions += [r["name"] for r in helpful_individuals]
best_additions += missed_direct[:10]
best_additions = list(dict.fromkeys(best_additions))  # deduplicate

expanded_feat = feat_27 + best_additions
expanded_feat = [f for f in expanded_feat if f in df_train_exp.columns]

print(f"  Expanded feature set: {len(expanded_feat)} features")

# Noise probing
N_NOISE = 15
N_ROUNDS = 5
hit_counts = {f: 0 for f in expanded_feat}

for round_i in range(N_ROUNDS):
    np.random.seed(round_i * 53 + 11)
    noise_cols = [f"_noise_{i}" for i in range(N_NOISE)]

    df_tr_n = df_train_exp.copy()
    df_va_n = df_val_exp.copy()
    for nc in noise_cols:
        df_tr_n[nc] = np.random.randn(len(df_tr_n))
        df_va_n[nc] = np.random.randn(len(df_va_n))

    all_f = expanded_feat + noise_cols
    model = CatBoostRegressor(**{**V7_PARAMS, "random_seed": round_i, "iterations": 3000})
    model.fit(
        Pool(df_tr_n.loc[valid_tr, all_f], y_dev_tr[valid_tr], weight=weights[valid_tr]),
        eval_set=Pool(df_va_n.loc[valid_va, all_f], y_dev_va[valid_va]),
        early_stopping_rounds=100, verbose=0,
    )

    importances = model.feature_importances_
    imp_dict = dict(zip(all_f, importances))
    noise_max = max(imp_dict[nc] for nc in noise_cols)

    for f in expanded_feat:
        if imp_dict[f] > noise_max:
            hit_counts[f] += 1

    above = sum(1 for f in expanded_feat if imp_dict[f] > noise_max)
    print(f"  Round {round_i+1}/{N_ROUNDS}: noise_max={noise_max:.4f}, above={above}/{len(expanded_feat)}")

confirmed_exp = [f for f in expanded_feat if hit_counts[f] >= 3]
new_confirmed = [f for f in confirmed_exp if f not in feat_27]
print(f"\n  Confirmed after noise probing: {len(confirmed_exp)} "
      f"(of which {len(new_confirmed)} NEW)")

if new_confirmed:
    print("  New confirmed features:")
    for f in new_confirmed:
        score = candidate_scores.get(f, corr_with_residual.get(f, 0))
        print(f"    {f:55s}  hits={hit_counts[f]}/5  |r_resid|={score:.4f}")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 7 — FINAL EVALUATION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  PHASE 7 — FINAL COMPARISON")
print("=" * 90)

# Test confirmed expanded set
if len(confirmed_exp) > len(feat_27):
    rmse_final, iter_final, bias_final, _, preds_final = train_eval(
        confirmed_exp, df_tr=df_train_exp, df_va=df_val_exp,
        label="Confirmed expanded"
    )

    # With HBC
    errors = spot_va - preds_final
    hb = {}
    for h in range(24):
        mask = hours_va == h
        if mask.sum() > 0:
            hb[h] = errors[mask].mean()
    corrected = preds_final + np.array([hb.get(h, 0) for h in hours_va])
    rmse_hbc = np.sqrt(np.mean((spot_va - corrected) ** 2))
    print(f"  Confirmed expanded + HBC:                                n={len(confirmed_exp):3d}  "
          f"RMSE={rmse_hbc:6.2f}")

# Also test all feature counts with new features
print("\n  Feature count sweep with expanded set:")
for n in [20, 27, 35, 40, 50, 75, 100, len(confirmed_exp)]:
    if n > len(confirmed_exp):
        continue
    feat = confirmed_exp[:n]
    rmse_n, iter_n, _, _, _ = train_eval(
        feat, df_tr=df_train_exp, df_va=df_val_exp,
        label=f"Expanded top-{n}"
    )

# Summary
print("\n  SUMMARY:")
print(f"    v7 baseline (27 feat):       RMSE={rmse_v7:.3f}  iter={iter_v7}")
print(f"    v7 all features (379):       RMSE={rmse_all:.3f}")
if len(confirmed_exp) > len(feat_27):
    print(f"    Expanded confirmed ({len(confirmed_exp)}):    RMSE={rmse_final:.3f}  iter={iter_final}")
    print(f"    Expanded + HBC:              RMSE={rmse_hbc:.3f}")
    delta = rmse_v7 - rmse_final
    print(f"    Improvement: {delta:+.3f} RMSE")

# Save results
output = {
    "method": "feature_discovery_v6",
    "baseline_rmse": rmse_v7,
    "expanded_features": confirmed_exp,
    "new_features": new_confirmed,
    "candidate_scores": {n: s for n, s in sorted_candidates[:50]},
    "residual_correlations": {f: r for f, r in sorted_corr[:50]},
}
with open("outputs/feature_discovery_v6.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved to outputs/feature_discovery_v6.json")
print(f"  Total time: {time.time() - t0:.0f}s")
