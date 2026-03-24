"""Basis Modeling A/B Test — Target engineering for stationarity.

Current targets:
  FR: y = spot - EMA(spot_la, 240h)     ← slow-moving, leaks non-stationarity
  UK: y = spot - merit_order_cost        ← already basis

Idea: predict basis = spot - fundamental_anchor instead of raw price.
The anchor captures the LEVEL of prices, so the model only learns CORRECTIONS.

Tests multiple anchors for FR:
  1. EMA(spot_la, 240)       ← current baseline
  2. merit_order_cost        ← gas CCGT/OCGT blend
  3. dynamic_marginal        ← nuclear/gas weighted by scarcity
  4. opportunity_cost        ← min(dynamic_marginal, import_price)
  5. scarcity_barrier        ← spark * convex scarcity multiplier

For each anchor, also tests arcsinh transform on the basis.
For UK, tests opportunity_cost and dynamic_marginal as alternatives.
"""

import sys, yaml, warnings, time, json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

print("=" * 90)
print("  BASIS MODELING A/B TEST")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
df = build_features(train, config)

holdout_start = config["validation"]["holdout_start"]
mask_val = (df["datetime_CET"] >= holdout_start).values
df_tr = df[~mask_val].copy()
df_va = df[mask_val].copy()

# ── Feature sets ──────────────────────────────────────────────────────
with open("outputs/feature_selection_v5_fr.json") as f:
    FR_FEAT = [f for f in json.load(f)["features"] if f in df_tr.columns]

with open("outputs/uk_feature_research.json") as f:
    UK_FEAT = [f for f in json.load(f)["confirmed_features"] if f in df_tr.columns]

print(f"  FR features: {len(FR_FEAT)}, UK features: {len(UK_FEAT)}")

hours_va = df_va["hour"].values
fr_spot_va = df_va["fr_spot"].values
uk_spot_va = df_va["uk_spot"].values


# ── Metrics ───────────────────────────────────────────────────────────
def compute_hbc(preds, actual, hours):
    errors = actual - preds
    hbc = {h: float(errors[hours == h].mean()) for h in range(24) if (hours == h).sum() > 0}
    corrected = preds + np.array([hbc.get(h, 0) for h in hours])
    rmse_raw = np.sqrt(np.mean((actual - preds) ** 2))
    rmse_hbc = np.sqrt(np.mean((actual - corrected) ** 2))
    mean_bias = np.mean(preds - actual)
    return rmse_raw, rmse_hbc, mean_bias


def compute_monthly_hbc(preds, actual, hours, months):
    corrected = preds.copy()
    for m in np.unique(months):
        for h in range(24):
            mask = (months == m) & (hours == h)
            if mask.sum() > 0:
                corrected[mask] += (actual[mask] - preds[mask]).mean()
    return np.sqrt(np.mean((actual - corrected) ** 2))


# ── Sample weights (same as pipeline) ────────────────────────────────
days_ago = (df_tr["datetime_CET"].max() - df_tr["datetime_CET"]).dt.total_seconds() / 86400
roll_std = df_tr["fr_spot_la"].rolling(168, min_periods=24).std().fillna(df_tr["fr_spot_la"].std())
fr_sample_w = np.exp(-2 * days_ago.values / 365) / np.clip(roll_std.values ** 2, 1, None)

months_va = df_va["datetime_CET"].dt.month.values


# ══════════════════════════════════════════════════════════════════════
#  FR ANCHOR SWEEP
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("  FR — ANCHOR SWEEP")
print(f"{'='*90}")

# Build anchor arrays (train + val)
fr_anchors = {}

# 1. Current: EMA(spot_la, 240)
fr_la = df["fr_spot_la"].values
ema_fr = pd.Series(fr_la).ewm(span=240).mean().values
fr_anchors["EMA_240 (current)"] = (ema_fr[~mask_val], ema_fr[mask_val])

# 2. merit_order_cost
fr_anchors["merit_order_cost"] = (df_tr["fr_merit_order_cost"].values, df_va["fr_merit_order_cost"].values)

# 3. dynamic_marginal
fr_anchors["dynamic_marginal"] = (df_tr["fr_dynamic_marginal"].values, df_va["fr_dynamic_marginal"].values)

# 4. opportunity_cost
fr_anchors["opportunity_cost"] = (df_tr["fr_opportunity_cost"].values, df_va["fr_opportunity_cost"].values)

# 5. scarcity_barrier
fr_anchors["scarcity_barrier"] = (df_tr["fr_scarcity_barrier"].values, df_va["fr_scarcity_barrier"].values)

# 6. spark_spread (simple gas cost)
fr_anchors["spark_spread"] = (df_tr["fr_spark_spread"].values, df_va["fr_spark_spread"].values)

cb_params_fr = config.get("catboost_params_fr_optuna_v2", config.get("catboost_params_fr", {}))

results_fr = []

for anchor_name, (anchor_tr, anchor_va) in fr_anchors.items():
    for use_arcsinh in [False, True]:
        label = f"{anchor_name}" + (" + arcsinh" if use_arcsinh else "")

        # Compute basis target
        y_basis_tr = df_tr["fr_spot"].values - anchor_tr
        y_basis_va = fr_spot_va - anchor_va

        valid_tr = np.isfinite(y_basis_tr) & np.isfinite(anchor_tr)
        valid_va = np.isfinite(y_basis_va) & np.isfinite(anchor_va)

        if use_arcsinh:
            y_tr_transformed = np.arcsinh(y_basis_tr)
            y_va_for_eval = y_basis_va  # not transformed, for RMSE
        else:
            y_tr_transformed = y_basis_tr

        # Adjust sample weights for valid mask
        sw = fr_sample_w.copy()
        sw[~valid_tr] = 0

        # Train CatBoost
        t1 = time.time()
        cb = CatBoostRegressor(**{**cb_params_fr, "verbose": 0})
        cb.fit(df_tr[FR_FEAT].values[valid_tr], y_tr_transformed[valid_tr],
               sample_weight=sw[valid_tr],
               eval_set=(df_va[FR_FEAT].values, y_basis_va if not use_arcsinh else np.arcsinh(y_basis_va)))

        # Predict
        preds_basis = cb.predict(df_va[FR_FEAT].values)
        if use_arcsinh:
            preds_basis = np.sinh(preds_basis)

        preds_spot = anchor_va + preds_basis
        elapsed = time.time() - t1

        # Evaluate
        rmse_raw, rmse_hbc, mean_bias = compute_hbc(preds_spot, fr_spot_va, hours_va)
        rmse_mhbc = compute_monthly_hbc(preds_spot, fr_spot_va, hours_va, months_va)

        # Basis stats
        basis_mean_tr = np.nanmean(y_basis_tr)
        basis_std_tr = np.nanstd(y_basis_tr)
        basis_mean_va = np.nanmean(y_basis_va)

        result = {
            "anchor": label, "rmse_raw": round(rmse_raw, 2), "rmse_hbc": round(rmse_hbc, 2),
            "rmse_mhbc": round(rmse_mhbc, 2), "mean_bias": round(mean_bias, 2),
            "basis_mean_tr": round(basis_mean_tr, 2), "basis_std_tr": round(basis_std_tr, 2),
            "basis_mean_va": round(basis_mean_va, 2),
            "time": round(elapsed, 1),
        }
        results_fr.append(result)

        flag = " ***" if rmse_hbc <= min(r["rmse_hbc"] for r in results_fr) else ""
        print(f"    {label:35s}  RMSE={rmse_raw:.2f}  +HBC={rmse_hbc:.2f}  +MHBC={rmse_mhbc:.2f}  "
              f"bias={mean_bias:+.1f}  basis_mu_tr={basis_mean_tr:+.1f}  basis_mu_va={basis_mean_va:+.1f}  "
              f"{elapsed:.1f}s{flag}")


# Also test with LightGBM for robustness
print(f"\n  --- FR LightGBM cross-check (top 3 anchors) ---")
top3_fr = sorted(results_fr, key=lambda x: x["rmse_hbc"])[:3]
lgb_params_fr = config.get("lightgbm_params_fr", {})
lgb_params_fr_clean = {k: v for k, v in lgb_params_fr.items() if k != "n_estimators"}

for anchor_name in [r["anchor"] for r in top3_fr]:
    base_name = anchor_name.replace(" + arcsinh", "")
    use_arcsinh = "+ arcsinh" in anchor_name
    anchor_tr, anchor_va = fr_anchors[base_name]

    y_basis_tr = df_tr["fr_spot"].values - anchor_tr
    valid_tr = np.isfinite(y_basis_tr) & np.isfinite(anchor_tr)
    sw = fr_sample_w.copy()
    sw[~valid_tr] = 0

    y_tr_use = np.arcsinh(y_basis_tr) if use_arcsinh else y_basis_tr
    y_va_use = np.arcsinh(fr_spot_va - anchor_va) if use_arcsinh else (fr_spot_va - anchor_va)

    ds_tr = lgb.Dataset(df_tr[FR_FEAT].values[valid_tr], y_tr_use[valid_tr], weight=sw[valid_tr])
    ds_va = lgb.Dataset(df_va[FR_FEAT].values, y_va_use, reference=ds_tr)
    lgb_model = lgb.train(lgb_params_fr_clean, ds_tr,
                          num_boost_round=lgb_params_fr.get("n_estimators", 5000),
                          valid_sets=[ds_va], callbacks=[lgb.early_stopping(50, verbose=False)])

    preds = lgb_model.predict(df_va[FR_FEAT].values)
    if use_arcsinh:
        preds = np.sinh(preds)
    preds_spot = anchor_va + preds

    rmse_raw, rmse_hbc, mean_bias = compute_hbc(preds_spot, fr_spot_va, hours_va)
    print(f"    LGB {anchor_name:35s}  RMSE={rmse_raw:.2f}  +HBC={rmse_hbc:.2f}  bias={mean_bias:+.1f}")


# ══════════════════════════════════════════════════════════════════════
#  UK ANCHOR SWEEP
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("  UK — ANCHOR SWEEP")
print(f"{'='*90}")

uk_anchors = {}

# 1. Current: merit_order_cost
uk_anchors["merit_order_cost (current)"] = (df_tr["uk_merit_order_cost"].values, df_va["uk_merit_order_cost"].values)

# 2. dynamic_marginal
uk_anchors["dynamic_marginal"] = (df_tr["uk_dynamic_marginal"].values, df_va["uk_dynamic_marginal"].values)

# 3. opportunity_cost
uk_anchors["opportunity_cost"] = (df_tr["uk_opportunity_cost"].values, df_va["uk_opportunity_cost"].values)

# 4. scarcity_barrier
uk_anchors["scarcity_barrier"] = (df_tr["uk_scarcity_barrier"].values, df_va["uk_scarcity_barrier"].values)

# 5. spark_spread
uk_anchors["spark_spread"] = (df_tr["uk_spark_spread"].values, df_va["uk_spark_spread"].values)

# 6. EMA for comparison
uk_la = df["uk_spot_la"].values
ema_uk = pd.Series(uk_la).ewm(span=240).mean().values
uk_anchors["EMA_240"] = (ema_uk[~mask_val], ema_uk[mask_val])

cb_params_uk = config.get("catboost_params_uk", {})

results_uk = []

for anchor_name, (anchor_tr, anchor_va) in uk_anchors.items():
    for use_arcsinh in [False, True]:
        label = f"{anchor_name}" + (" + arcsinh" if use_arcsinh else "")

        y_basis_tr = df_tr["uk_spot"].values - anchor_tr
        y_basis_va = uk_spot_va - anchor_va

        valid_tr = np.isfinite(y_basis_tr) & np.isfinite(anchor_tr)
        valid_va = np.isfinite(y_basis_va) & np.isfinite(anchor_va)

        if use_arcsinh:
            y_tr_transformed = np.arcsinh(y_basis_tr)
        else:
            y_tr_transformed = y_basis_tr

        t1 = time.time()
        cb = CatBoostRegressor(**{**cb_params_uk, "verbose": 0})
        cb.fit(df_tr[UK_FEAT].values[valid_tr], y_tr_transformed[valid_tr],
               eval_set=(df_va[UK_FEAT].values, y_basis_va if not use_arcsinh else np.arcsinh(y_basis_va)))

        preds_basis = cb.predict(df_va[UK_FEAT].values)
        if use_arcsinh:
            preds_basis = np.sinh(preds_basis)

        preds_spot = anchor_va + preds_basis
        elapsed = time.time() - t1

        rmse_raw, rmse_hbc, mean_bias = compute_hbc(preds_spot, uk_spot_va, hours_va)
        rmse_mhbc = compute_monthly_hbc(preds_spot, uk_spot_va, hours_va, months_va)

        basis_mean_tr = np.nanmean(y_basis_tr)
        basis_std_tr = np.nanstd(y_basis_tr)
        basis_mean_va = np.nanmean(y_basis_va)

        result = {
            "anchor": label, "rmse_raw": round(rmse_raw, 2), "rmse_hbc": round(rmse_hbc, 2),
            "rmse_mhbc": round(rmse_mhbc, 2), "mean_bias": round(mean_bias, 2),
            "basis_mean_tr": round(basis_mean_tr, 2), "basis_std_tr": round(basis_std_tr, 2),
            "basis_mean_va": round(basis_mean_va, 2),
            "time": round(elapsed, 1),
        }
        results_uk.append(result)

        flag = " ***" if rmse_hbc <= min(r["rmse_hbc"] for r in results_uk) else ""
        print(f"    {label:40s}  RMSE={rmse_raw:.2f}  +HBC={rmse_hbc:.2f}  +MHBC={rmse_mhbc:.2f}  "
              f"bias={mean_bias:+.1f}  basis_mu_tr={basis_mean_tr:+.1f}  basis_mu_va={basis_mean_va:+.1f}  "
              f"{elapsed:.1f}s{flag}")


# ══════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("  SUMMARY — BEST ANCHORS")
print(f"{'='*90}")

print("\n  FR (sorted by +HBC):")
for r in sorted(results_fr, key=lambda x: x["rmse_hbc"]):
    print(f"    {r['anchor']:35s}  +HBC={r['rmse_hbc']}  +MHBC={r['rmse_mhbc']}  "
          f"bias={r['mean_bias']:+.1f}  basis_shift={r['basis_mean_tr'] - r['basis_mean_va']:+.1f}")

print("\n  UK (sorted by +HBC):")
for r in sorted(results_uk, key=lambda x: x["rmse_hbc"]):
    print(f"    {r['anchor']:40s}  +HBC={r['rmse_hbc']}  +MHBC={r['rmse_mhbc']}  "
          f"bias={r['mean_bias']:+.1f}  basis_shift={r['basis_mean_tr'] - r['basis_mean_va']:+.1f}")

# Best combination
best_fr = min(results_fr, key=lambda x: x["rmse_hbc"])
best_uk = min(results_uk, key=lambda x: x["rmse_hbc"])
print(f"\n  OPTIMAL: FR={best_fr['anchor']} ({best_fr['rmse_hbc']}) + UK={best_uk['anchor']} ({best_uk['rmse_hbc']})")
print(f"  SUM = {best_fr['rmse_hbc'] + best_uk['rmse_hbc']:.2f}")

# Save
with open("outputs/basis_modeling_test.json", "w") as f:
    json.dump({"fr": results_fr, "uk": results_uk}, f, indent=2)

print(f"\n  Total time: {time.time() - t0:.0f}s")
