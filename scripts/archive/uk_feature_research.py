"""UK Feature Research — SHAP + correlation + selection with BASIS target.

The previous UK work used SHAP rankings computed on FR's stationary target,
which is wrong for UK. UK uses basis target (spot - merit_order_cost).

This script does proper UK-specific feature work:
  Phase 1: Train baseline (replicate best ever: d=8, 200f, no weights → 9.84)
  Phase 2: SHAP on basis target → UK-specific ranking
  Phase 3: Correlation of ALL features with UK residual
  Phase 4: Feature selection (Boruta noise probing)
  Phase 5: Feature count sweep with best params
  Phase 6: Residual analysis (by hour, month)
  Phase 7: Summary + save

Usage: cd "INCOMO 3" && python scripts/uk_feature_research.py
"""

import sys, json, time, warnings
import numpy as np
import pandas as pd

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
print("  UK FEATURE RESEARCH — Basis target (spot - merit_order_cost)")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
train_fe = build_features(train, config)
print(f"  Data loaded in {time.time() - t0:.0f}s — shape: {train_fe.shape}")

holdout_start = config["validation"]["holdout_start"]
mask_val = train_fe["datetime_CET"] >= holdout_start
df_train = train_fe[~mask_val].copy()
df_val = train_fe[mask_val].copy()

# ── Basis target ──────────────────────────────────────────────────────────
spot_tr = df_train["uk_spot"].values
spot_va = df_val["uk_spot"].values
moc_tr = df_train["uk_merit_order_cost"].values
moc_va = df_val["uk_merit_order_cost"].values
hours_va = df_val["hour"].values

y_basis_tr = spot_tr - moc_tr
y_basis_va = spot_va - moc_va
valid_tr = np.isfinite(y_basis_tr)
valid_va = np.isfinite(y_basis_va)

print(f"  Train: {len(df_train)}, Val: {len(df_val)}")
print(f"  UK basis target: mean={np.nanmean(y_basis_tr):.1f}, std={np.nanstd(y_basis_tr):.1f}")

# All numeric features
exclude = {"fr_spot", "uk_spot", "datetime_CET", "datetime_UTC"}
all_numeric = [c for c in df_train.columns
               if c not in exclude and df_train[c].dtype in ["float64", "float32", "int64"]]
print(f"  All numeric features: {len(all_numeric)}")

# Old SHAP ranking (from v2, used for the 9.84 result)
with open("outputs/shap_ranking_v4_stationary.json") as f:
    v4_ranking = json.load(f)
feat_v4_uk = [f for f in v4_ranking["uk_spot"] if f in df_train.columns]


def compute_hbc(preds_spot, spot_actual, hours):
    """Return (hbc_dict, rmse_with_hbc)."""
    errors = spot_actual - preds_spot
    hb = {}
    for h in range(24):
        mask = hours == h
        if mask.sum() > 0:
            hb[h] = errors[mask].mean()
    corrected = preds_spot + np.array([hb.get(h, 0) for h in hours])
    rmse_hbc = np.sqrt(np.mean((spot_actual - corrected) ** 2))
    return hb, rmse_hbc


def train_eval(feat_list, params, label="", w=None):
    """Train CatBoost with basis target, return (rmse, best_iter, bias, rmse_hbc, model)."""
    feat = [f for f in feat_list if f in df_train.columns]
    if len(feat) == 0:
        return 999, 0, 0, 999, None

    kw = {}
    if w is not None:
        kw["weight"] = w[valid_tr]

    model = CatBoostRegressor(**params)
    model.fit(
        Pool(df_train.loc[df_train.index[valid_tr], feat], y_basis_tr[valid_tr], **kw),
        eval_set=Pool(df_val.loc[df_val.index[valid_va], feat], y_basis_va[valid_va]),
        early_stopping_rounds=200, verbose=0,
    )

    preds_basis = model.predict(df_val[feat])
    preds_spot = moc_va + preds_basis
    rmse = np.sqrt(np.mean((spot_va - preds_spot) ** 2))
    bias = np.mean(spot_va - preds_spot)
    best_iter = model.get_best_iteration()

    _, rmse_hbc = compute_hbc(preds_spot, spot_va, hours_va)

    if label:
        print(f"  {label:55s}  n={len(feat):3d}  RMSE={rmse:6.2f}  +HBC={rmse_hbc:5.2f}  "
              f"iter={best_iter:5d}  bias={bias:+.1f}")

    return rmse, best_iter, bias, rmse_hbc, model


# ══════════════════════════════════════════════════════════════════════════
# PHASE 1 — REPLICATE BEST EVER (9.84)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  PHASE 1 — REPLICATE BEST KNOWN UK RESULT")
print("=" * 90)

BASELINE_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 5000, "learning_rate": 0.03, "depth": 8,
    "l2_leaf_reg": 5, "subsample": 0.8, "random_seed": 42,
    "verbose": 0, "allow_writing_files": False, "use_best_model": True,
}

# Test the configs that gave 9.84
for n in [75, 100, 150, 200, len(feat_v4_uk)]:
    if n > len(feat_v4_uk):
        continue
    train_eval(feat_v4_uk[:n], BASELINE_PARAMS, label=f"Baseline d=8, SHAP v4 top-{n}")

# Also without any feature selection
train_eval(all_numeric, BASELINE_PARAMS, label=f"Baseline d=8, ALL numeric")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 2 — SHAP ON BASIS TARGET
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  PHASE 2 — SHAP RANKING (basis target, top-200 features)")
print("=" * 90)

# Train model with top-200 features for SHAP
shap_feat = feat_v4_uk[:200]
shap_model = CatBoostRegressor(**BASELINE_PARAMS)
shap_model.fit(
    Pool(df_train.loc[df_train.index[valid_tr], shap_feat], y_basis_tr[valid_tr]),
    eval_set=Pool(df_val.loc[df_val.index[valid_va], shap_feat], y_basis_va[valid_va]),
    early_stopping_rounds=200, verbose=0,
)

# CatBoost feature importances (fast proxy for SHAP)
importances = shap_model.feature_importances_
imp_pairs = sorted(zip(shap_feat, importances), key=lambda x: -x[1])

print(f"\n  TOP-30 features by CatBoost importance (basis target):")
print(f"  {'#':>3s}  {'Feature':50s}  {'Importance':>10s}  {'In v4 top-30?':>14s}")
v4_top30 = set(feat_v4_uk[:30])
for i, (f, imp) in enumerate(imp_pairs[:30]):
    in_v4 = "YES" if f in v4_top30 else "no"
    print(f"  {i+1:3d}  {f:50s}  {imp:10.2f}  {in_v4:>14s}")

# New ranking
feat_basis_ranked = [f for f, _ in imp_pairs if _ > 0]
print(f"\n  Features with importance > 0: {len(feat_basis_ranked)}")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 3 — CORRELATION WITH RESIDUAL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  PHASE 3 — RESIDUAL CORRELATION ANALYSIS")
print("=" * 90)

# Get residuals from baseline model
preds_baseline = shap_model.predict(df_val[shap_feat])
residual = y_basis_va - preds_baseline  # what the model misses
residual_valid = np.isfinite(residual)

print(f"  Residual: mean={np.mean(residual[residual_valid]):.2f}, "
      f"std={np.std(residual[residual_valid]):.2f}")

# Correlate ALL features with residual
corr_results = []
for f in all_numeric:
    vals = df_val[f].values
    mask = np.isfinite(vals) & residual_valid
    if mask.sum() < 100:
        continue
    r = np.corrcoef(vals[mask], residual[mask])[0, 1]
    if np.isfinite(r):
        corr_results.append((f, r))

corr_results.sort(key=lambda x: -abs(x[1]))

print(f"\n  TOP-30 features correlated with UK RESIDUAL:")
print(f"  {'#':>3s}  {'Feature':50s}  {'r':>8s}  {'In model?':>10s}")
model_feats = set(shap_feat)
for i, (f, r) in enumerate(corr_results[:30]):
    in_model = "YES" if f in model_feats else "no"
    print(f"  {i+1:3d}  {f:50s}  {r:+8.4f}  {in_model:>10s}")

missed_features = [(f, r) for f, r in corr_results if f not in model_feats and abs(r) > 0.05]
print(f"\n  Features NOT in model with |r_residual| > 0.05: {len(missed_features)}")
for f, r in missed_features[:15]:
    print(f"    {f:50s}  r={r:+.4f}")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 4 — NOISE PROBING (Boruta)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  PHASE 4 — NOISE PROBING (Boruta) for UK")
print("=" * 90)

# Test top-150 features from basis importance ranking
feat_to_test = feat_basis_ranked[:150]
N_NOISE = 20
N_ROUNDS = 5
hit_counts = {f: 0 for f in feat_to_test}

for round_i in range(N_ROUNDS):
    np.random.seed(round_i * 41 + 13)
    noise_cols = [f"_noise_{i}" for i in range(N_NOISE)]

    df_tr_n = df_train.copy()
    df_va_n = df_val.copy()
    for nc in noise_cols:
        df_tr_n[nc] = np.random.randn(len(df_tr_n))
        df_va_n[nc] = np.random.randn(len(df_va_n))

    all_f = feat_to_test + noise_cols
    params_boruta = {**BASELINE_PARAMS, "random_seed": round_i, "iterations": 3000}
    model = CatBoostRegressor(**params_boruta)
    model.fit(
        Pool(df_tr_n.loc[df_tr_n.index[valid_tr], all_f], y_basis_tr[valid_tr]),
        eval_set=Pool(df_va_n.loc[df_va_n.index[valid_va], all_f], y_basis_va[valid_va]),
        early_stopping_rounds=100, verbose=0,
    )

    importances = model.feature_importances_
    imp_dict = dict(zip(all_f, importances))
    noise_max = max(imp_dict[nc] for nc in noise_cols)

    for f in feat_to_test:
        if imp_dict.get(f, 0) > noise_max:
            hit_counts[f] += 1

    above = sum(1 for f in feat_to_test if imp_dict.get(f, 0) > noise_max)
    print(f"  Round {round_i+1}/{N_ROUNDS}: noise_max={noise_max:.4f}, "
          f"above={above}/{len(feat_to_test)}")

confirmed = [f for f in feat_to_test if hit_counts[f] >= 3]
tentative = [f for f in feat_to_test if hit_counts[f] == 2]
rejected = [f for f in feat_to_test if hit_counts[f] <= 1]
print(f"\n  Confirmed: {len(confirmed)}, Tentative: {len(tentative)}, Rejected: {len(rejected)}")

print(f"\n  Confirmed features (top-30):")
for i, f in enumerate(confirmed[:30]):
    imp_rank = next((j for j, (ff, _) in enumerate(imp_pairs) if ff == f), 999)
    print(f"    {i+1:3d}. {f:50s}  hits={hit_counts[f]}/5  imp_rank={imp_rank+1}")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 5 — FEATURE COUNT SWEEP
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  PHASE 5 — FEATURE COUNT SWEEP (basis ranking)")
print("=" * 90)

# Sweep with basis-ranked features
print("\n  A) Basis importance ranking:")
for n in [20, 30, 40, 50, 75, 100, 125, 150, 200, len(feat_basis_ranked)]:
    if n > len(feat_basis_ranked):
        n = len(feat_basis_ranked)
    train_eval(feat_basis_ranked[:n], BASELINE_PARAMS, label=f"Basis top-{n}")

# Sweep with confirmed features + adding tentative
print("\n  B) Confirmed features + tentative:")
train_eval(confirmed, BASELINE_PARAMS, label=f"Confirmed ({len(confirmed)})")
if tentative:
    train_eval(confirmed + tentative, BASELINE_PARAMS,
               label=f"Confirmed + tentative ({len(confirmed)+len(tentative)})")

# Sweep with v4 ranking (for comparison)
print("\n  C) SHAP v4 ranking (stationary target — comparison):")
for n in [50, 100, 200]:
    if n <= len(feat_v4_uk):
        train_eval(feat_v4_uk[:n], BASELINE_PARAMS, label=f"SHAP v4 top-{n}")

# Test missed features
if missed_features:
    print("\n  D) Adding missed features (correlated with residual):")
    missed_names = [f for f, _ in missed_features[:20]]
    base_feat = feat_basis_ranked[:100]
    for n_add in [5, 10, 15, 20]:
        feat = base_feat + missed_names[:n_add]
        feat = list(dict.fromkeys(feat))  # dedup
        train_eval(feat, BASELINE_PARAMS, label=f"Basis-100 + {n_add} missed")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 6 — RESIDUAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  PHASE 6 — RESIDUAL ANALYSIS (best config)")
print("=" * 90)

# Use the best feature count from phase 5 — retrain
best_feat_count = 200  # from original best
rmse_best, iter_best, _, _, model_best = train_eval(
    feat_basis_ranked[:best_feat_count], BASELINE_PARAMS,
    label=f"Best config for residual analysis"
)

preds_best = model_best.predict(df_val[feat_basis_ranked[:best_feat_count]])
preds_spot_best = moc_va + preds_best
errors = spot_va - preds_spot_best

print(f"\n  Error by hour:")
for h in range(24):
    mask = hours_va == h
    if mask.sum() > 0:
        h_bias = errors[mask].mean()
        h_rmse = np.sqrt(np.mean(errors[mask] ** 2))
        bar = "+" * int(abs(h_bias)) if h_bias > 0 else "-" * int(abs(h_bias))
        print(f"    h={h:2d}  bias={h_bias:+6.1f}  RMSE={h_rmse:5.1f}  {bar}")

print(f"\n  Error by month:")
months_va = pd.to_datetime(df_val["datetime_CET"]).dt.month.values
for m in sorted(set(months_va)):
    mask = months_va == m
    m_bias = errors[mask].mean()
    m_rmse = np.sqrt(np.mean(errors[mask] ** 2))
    print(f"    month={m:2d}  bias={m_bias:+6.1f}  RMSE={m_rmse:5.1f}  n={mask.sum()}")

# Weekend vs weekday
dow_va = pd.to_datetime(df_val["datetime_CET"]).dt.dayofweek.values
for label, mask in [("Weekday", dow_va < 5), ("Weekend", dow_va >= 5)]:
    m_rmse = np.sqrt(np.mean(errors[mask] ** 2))
    m_bias = errors[mask].mean()
    print(f"\n    {label:10s}  bias={m_bias:+6.1f}  RMSE={m_rmse:5.1f}  n={mask.sum()}")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 7 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  UK FEATURE RESEARCH — SUMMARY")
print("=" * 90)

print(f"\n  Basis importance ranking: {len(feat_basis_ranked)} features with imp > 0")
print(f"  Boruta: {len(confirmed)} confirmed, {len(tentative)} tentative, {len(rejected)} rejected")
print(f"  Missed features (corr w/ residual > 0.05): {len(missed_features)}")

# Save
output = {
    "target": "uk_spot",
    "target_transform": "basis (spot - merit_order_cost)",
    "basis_importance_ranking": feat_basis_ranked,
    "confirmed_features": confirmed,
    "tentative_features": tentative,
    "n_confirmed": len(confirmed),
    "n_tentative": len(tentative),
    "n_rejected": len(rejected),
    "top30_importance": [{"feature": f, "importance": float(imp)} for f, imp in imp_pairs[:30]],
    "residual_correlations": {f: float(r) for f, r in corr_results[:50]},
    "missed_features": [{"feature": f, "r": float(r)} for f, r in missed_features[:30]],
}

with open("outputs/uk_feature_research.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved to outputs/uk_feature_research.json")
print(f"  Total time: {time.time() - t0:.0f}s")
