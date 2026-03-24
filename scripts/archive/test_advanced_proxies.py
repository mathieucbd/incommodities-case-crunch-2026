"""A/B test: advanced price proxies + basis v2 + sample weights.

Tests all combinations:
  - Target: raw spot / arcsinh(spot) / basis v1 / basis v2 / arcsinh(basis v2)
  - Sample weights: uniform / exponential decay
  - Feature sets: SHAP v3 rankings + new Cat 32 features

Usage: python scripts/test_advanced_proxies.py
"""

import sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features
from catboost import CatBoostRegressor, Pool
import yaml

# ── Load data ────────────────────────────────────────────────────────────
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

# ── CatBoost params ──────────────────────────────────────────────────────
CB_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 5000, "learning_rate": 0.03, "depth": 8,
    "l2_leaf_reg": 5, "subsample": 0.8, "random_seed": 42,
    "verbose": 0, "allow_writing_files": False, "use_best_model": True,
}

# ── New Cat 32 features to inject into rankings ─────────────────────────
CAT32_FR = [
    "fr_opportunity_cost", "fr_dynamic_marginal", "fr_import_price",
    "fr_scarcity_barrier", "fr_load_price_signal_7d",
    "fr_load_price_signal_load", "fr_hydro_opp_cost",
    "fr_basis_v2", "fr_basis_v2_lag_48h", "fr_basis_v2_roll_24h_mean",
    "fr_price_per_mw_7d",
]

CAT32_UK = [
    "uk_opportunity_cost", "uk_dynamic_marginal", "uk_import_floor",
    "uk_scarcity_barrier", "uk_load_price_signal_7d",
    "uk_hydro_opp_cost",
    "uk_basis_v2", "uk_basis_v2_lag_48h", "uk_basis_v2_roll_24h_mean",
    "uk_price_per_mw_7d",
]


def get_features(target_name, n_base, inject_cat32=False):
    """Get feature list: SHAP v3 top-N + optionally Cat 32."""
    base = [f for f in clean_ranking[target_name][:n_base] if f in df_train.columns]
    if inject_cat32:
        cat32 = CAT32_FR if "fr" in target_name else CAT32_UK
        extras = [f for f in cat32 if f in df_train.columns and f not in base]
        base = base + extras
    return base


def compute_sample_weights(df, decay=1.0):
    """Exponential decay weights: recent data weighted more."""
    dt = pd.to_datetime(df["datetime_CET"])
    max_dt = dt.max()
    days_ago = (max_dt - dt).dt.total_seconds() / 86400
    weights = np.exp(-decay * days_ago / 365)
    return weights.values


def train_eval(X_tr, y_tr, X_va, y_va, merit_va, transform, weights=None):
    """Train CatBoost, return (preds_on_spot_scale, best_iter)."""
    if transform == "arcsinh":
        y_tr_t = np.arcsinh(y_tr)
        y_va_t = np.arcsinh(y_va)
    else:
        y_tr_t = y_tr
        y_va_t = y_va

    pool_tr = Pool(X_tr, y_tr_t, weight=weights)
    pool_va = Pool(X_va, y_va_t)

    model = CatBoostRegressor(**CB_PARAMS)
    model.fit(pool_tr, eval_set=pool_va, early_stopping_rounds=100, verbose=0)

    preds_t = model.predict(X_va)
    preds = np.sinh(preds_t) if transform == "arcsinh" else preds_t

    if merit_va is not None:
        preds_spot = merit_va + preds
    else:
        preds_spot = preds

    return preds_spot, model.get_best_iteration(), model


# ── Run all experiments ──────────────────────────────────────────────────
results = []

configs = [
    # (label, target_key, n_feat_fr, n_feat_uk, transform, basis_col, inject_cat32, decay)
    ("A) raw spot (baseline)",          "spot",    20, 75,  "raw",     None,             False, None),
    ("B) arcsinh(spot)",                "spot",    20, 75,  "arcsinh", None,             False, None),
    ("C) basis v1 (merit_order)",       "basis1",  20, 75,  "raw",     "merit_order_cost", False, None),
    ("D) basis v2 (opportunity_cost)",  "basis2",  20, 75,  "raw",     "opportunity_cost", False, None),
    ("E) arcsinh(basis v2)",            "basis2",  20, 75,  "arcsinh", "opportunity_cost", False, None),
    ("F) raw spot + Cat32",             "spot",    20, 75,  "raw",     None,             True,  None),
    ("G) arcsinh(spot) + Cat32",        "spot",    20, 75,  "arcsinh", None,             True,  None),
    ("H) basis v2 + Cat32",             "basis2",  20, 75,  "raw",     "opportunity_cost", True,  None),
    ("I) raw spot + weights(1.0)",      "spot",    20, 75,  "raw",     None,             False, 1.0),
    ("J) arcsinh(spot) + weights(1.0)", "spot",    20, 75,  "arcsinh", None,             False, 1.0),
    ("K) raw spot + weights(2.0)",      "spot",    20, 75,  "raw",     None,             False, 2.0),
    ("L) arcsinh(spot) + weights(2.0)", "spot",    20, 75,  "arcsinh", None,             False, 2.0),
    ("M) arcsinh + Cat32 + w(1.0)",     "spot",    20, 75,  "arcsinh", None,             True,  1.0),
    ("N) arcsinh + Cat32 + w(2.0)",     "spot",    20, 75,  "arcsinh", None,             True,  2.0),
    ("O) basis v2 + Cat32 + w(1.0)",    "basis2",  20, 75,  "raw",     "opportunity_cost", True,  1.0),
]

t0 = time.time()

for label, tgt_key, n_fr, n_uk, transform, basis_col, inject_cat32, decay in configs:
    row = {"config": label}

    for target_name, n_feat in [("fr_spot", n_fr), ("uk_spot", n_uk)]:
        prefix = target_name.split("_")[0]
        features = get_features(target_name, n_feat, inject_cat32)

        X_tr = df_train[features]
        X_va = df_val[features]
        y_va_spot = df_val[target_name].values

        # Target selection
        if basis_col is not None:
            y_tr = df_train[target_name] - df_train[f"{prefix}_{basis_col}"]
            y_va = df_val[target_name] - df_val[f"{prefix}_{basis_col}"]
            merit_va = df_val[f"{prefix}_{basis_col}"].values
        else:
            y_tr = df_train[target_name]
            y_va = df_val[target_name]
            merit_va = None

        # Weights
        weights = compute_sample_weights(df_train, decay) if decay else None

        preds_spot, iters, _ = train_eval(X_tr, y_tr, X_va, y_va, merit_va, transform, weights)
        rmse = float(np.sqrt(np.mean((y_va_spot - preds_spot) ** 2)))
        bias = float(np.mean(y_va_spot - preds_spot))

        row[f"{prefix}_rmse"] = rmse
        row[f"{prefix}_bias"] = bias
        row[f"{prefix}_iters"] = iters
        row[f"{prefix}_nfeat"] = len(features)

    row["combined"] = (row["fr_rmse"] + row["uk_rmse"]) / 2
    results.append(row)

    print(f"  {label:40s}  FR={row['fr_rmse']:7.3f} (bias={row['fr_bias']:+6.1f})  "
          f"UK={row['uk_rmse']:7.3f} (bias={row['uk_bias']:+6.1f})  "
          f"Combined={row['combined']:7.3f}")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")

# ── Summary table ────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  SUMMARY — Advanced Proxy A/B Tests")
print("=" * 90)
print(f"{'Config':42s} {'FR RMSE':>8s} {'FR Bias':>8s} {'UK RMSE':>8s} {'UK Bias':>8s} {'Combined':>9s}")
print("-" * 90)

best_combined = min(r["combined"] for r in results)
for r in results:
    marker = " ***" if r["combined"] == best_combined else ""
    print(f"{r['config']:42s} {r['fr_rmse']:8.3f} {r['fr_bias']:+8.1f} "
          f"{r['uk_rmse']:8.3f} {r['uk_bias']:+8.1f} {r['combined']:9.3f}{marker}")

print("-" * 90)
best = min(results, key=lambda r: r["combined"])
print(f"\nBEST: {best['config']}")
print(f"  FR: RMSE={best['fr_rmse']:.3f}, Bias={best['fr_bias']:+.1f}")
print(f"  UK: RMSE={best['uk_rmse']:.3f}, Bias={best['uk_bias']:+.1f}")
print(f"  Combined: {best['combined']:.3f}")

# Best per-target
best_fr = min(results, key=lambda r: r["fr_rmse"])
best_uk = min(results, key=lambda r: r["uk_rmse"])
print(f"\nBest per-target combo:")
print(f"  FR: {best_fr['config']} → {best_fr['fr_rmse']:.3f}")
print(f"  UK: {best_uk['config']} → {best_uk['uk_rmse']:.3f}")
print(f"  Cherry-pick combined: {(best_fr['fr_rmse'] + best_uk['uk_rmse'])/2:.3f}")

# ── Save results ─────────────────────────────────────────────────────────
save_data = {r["config"]: {k: v for k, v in r.items() if k != "config"} for r in results}
with open("outputs/advanced_proxy_results.json", "w") as f:
    json.dump(save_data, f, indent=2)
print("\nSaved to outputs/advanced_proxy_results.json")
