"""A/B test — Model diversity for ensemble (CatBoost, LightGBM, XGBoost, HistGB).

Tests standalone performance + error correlation + 2/3/4-model ensembles.
The key insight: a weaker model with uncorrelated errors can improve the ensemble.

Models tested:
  - CatBoost (current)
  - LightGBM (current)
  - XGBoost
  - HistGradientBoosting (sklearn)

Both FR and UK.
"""

import sys, yaml, warnings, time, json
import numpy as np
import pandas as pd
from itertools import combinations

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  ! XGBoost not installed — pip install xgboost")

from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

print("=" * 90)
print("  MODEL DIVERSITY A/B TEST — CatBoost / LightGBM / XGBoost / HistGB")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
df = build_features(train, config)

holdout_start = config["validation"]["holdout_start"]
mask_val = (df["datetime_CET"] >= holdout_start).values

if "fr_spot_la_roll_168h_mean" in df.columns and "uk_price_per_mw_7d" in df.columns:
    df["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
        df["fr_spot_la_roll_168h_mean"] * df["uk_price_per_mw_7d"]
    )

df_tr = df[~mask_val].copy()
df_va = df[mask_val].copy()
print(f"  Data loaded in {time.time() - t0:.0f}s  |  Train: {len(df_tr)}, Val: {len(df_va)}")


# ── Features ────────────────────────────────────────────────────────
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

with open("outputs/uk_feature_research.json") as f:
    uk_research = json.load(f)
UK_FEATURES = [f for f in uk_research["confirmed_features"] if f in df_tr.columns]


# ── Targets ──────────────────────────────────────────────────────────
# FR: EMA 240h
fr_spot_la_full = df["fr_spot_la"].values
ema_fr = pd.Series(fr_spot_la_full).ewm(span=240).mean().values
fr_anchor_tr, fr_anchor_va = ema_fr[~mask_val], ema_fr[mask_val]
fr_spot_tr, fr_spot_va = df_tr["fr_spot"].values, df_va["fr_spot"].values
fr_y_tr = fr_spot_tr - fr_anchor_tr
fr_y_va = fr_spot_va - fr_anchor_va

# UK: Basis
uk_moc_tr, uk_moc_va = df_tr["uk_merit_order_cost"].values, df_va["uk_merit_order_cost"].values
uk_spot_tr, uk_spot_va = df_tr["uk_spot"].values, df_va["uk_spot"].values
uk_y_tr = uk_spot_tr - uk_moc_tr
uk_y_va = uk_spot_va - uk_moc_va

hours_va = df_va["hour"].values

# FR weights
dates_tr = pd.to_datetime(df_tr["datetime_CET"])
days_ago = (dates_tr.max() - dates_tr).dt.total_seconds() / 86400
w_recency = np.exp(-2.0 * days_ago.values / 365)
roll_std = df_tr["fr_spot_la_roll_168h_std"].values
w_stability = 1.0 / np.clip(roll_std ** 2, 1, None)
w_stability = np.where(np.isnan(w_stability), 1.0, w_stability)
fr_weights = w_recency * w_stability

fr_valid_tr = np.isfinite(fr_y_tr) & np.isfinite(fr_anchor_tr) & np.isfinite(fr_weights) & (fr_weights > 0)
fr_valid_va = np.isfinite(fr_y_va) & np.isfinite(fr_anchor_va)
uk_valid_tr = np.isfinite(uk_y_tr)
uk_valid_va = np.isfinite(uk_y_va)

# Fill NaN for tree models that don't handle them (XGB/HistGB handle natively)
fr_feat = [f for f in FR_FEATURES if f in df_tr.columns]
uk_feat = [f for f in UK_FEATURES if f in df_tr.columns]


# ── HBC helper ──────────────────────────────────────────────────────
def apply_hbc(preds_spot, actual, hours):
    errors = actual - preds_spot
    hbc = {h: float(errors[hours == h].mean()) for h in range(24) if (hours == h).sum() > 0}
    corrected = preds_spot + np.array([hbc.get(h, 0) for h in hours])
    rmse_hbc = np.sqrt(np.mean((actual - corrected) ** 2))
    return rmse_hbc, corrected


# ══════════════════════════════════════════════════════════════════════
# TRAIN ALL MODELS
# ══════════════════════════════════════════════════════════════════════

def train_all_models(market, features, y_tr, y_va, v_tr, v_va, anchor_va, spot_va,
                     weights, hours_v, loss_uk="MAE"):
    """Train CB, LGB, XGB, HistGB and return predictions."""
    feat = [f for f in features if f in df_tr.columns]
    X_tr = df_tr.loc[df_tr.index[v_tr], feat]
    X_va = df_va.loc[df_va.index[v_va], feat]
    yt = y_tr[v_tr]
    yv = y_va[v_va]
    w = weights[v_tr] if weights is not None else None
    hrs = hours_v[v_va]

    results = {}

    # ── 1. CatBoost ──────────────────────────────────────────────
    t1 = time.time()
    if market == "fr":
        cb_params = {
            "loss_function": "RMSE", "eval_metric": "RMSE",
            "iterations": 15000, "learning_rate": 0.059, "depth": 3,
            "l2_leaf_reg": 4.42, "subsample": 0.533, "colsample_bylevel": 0.228,
            "min_child_samples": 14, "random_strength": 0.9,
            "random_seed": 42, "verbose": 0, "use_best_model": True,
            "allow_writing_files": False,
        }
    else:
        cb_params = {
            "loss_function": loss_uk, "eval_metric": "RMSE",
            "iterations": 15000, "learning_rate": 0.03, "depth": 8,
            "l2_leaf_reg": 5, "colsample_bylevel": 0.8, "subsample": 0.8,
            "random_seed": 42, "verbose": 0, "use_best_model": True,
            "allow_writing_files": False,
        }

    pool_tr = Pool(X_tr, yt, weight=w)
    pool_va = Pool(X_va, yv)
    cb = CatBoostRegressor(**cb_params)
    cb.fit(pool_tr, eval_set=pool_va, early_stopping_rounds=200, verbose=0)
    preds_cb = anchor_va[v_va] + cb.predict(X_va)
    rmse_cb = np.sqrt(np.mean((spot_va[v_va] - preds_cb) ** 2))
    rmse_cb_hbc, preds_cb_hbc = apply_hbc(preds_cb, spot_va[v_va], hrs)
    print(f"  CatBoost      RMSE={rmse_cb:7.4f}  +HBC={rmse_cb_hbc:7.4f}  "
          f"iter={cb.get_best_iteration():5d}  ({time.time()-t1:.0f}s)")
    results["CatBoost"] = {"preds": preds_cb, "preds_hbc": preds_cb_hbc,
                           "rmse": rmse_cb, "rmse_hbc": rmse_cb_hbc}

    # ── 2. LightGBM ─────────────────────────────────────────────
    t1 = time.time()
    if market == "fr":
        lgb_params = {
            "objective": "regression", "metric": "rmse",
            "n_estimators": 15000, "learning_rate": 0.03,
            "max_depth": 4, "num_leaves": 15,
            "reg_alpha": 5, "reg_lambda": 30,
            "subsample": 0.7, "colsample_bytree": 0.5,
            "min_child_samples": 50,
            "random_state": 42, "verbose": -1,
        }
    else:
        lgb_params = {
            "objective": "regression", "metric": "rmse",
            "n_estimators": 15000, "learning_rate": 0.02,
            "max_depth": 7, "num_leaves": 63,
            "reg_alpha": 1, "reg_lambda": 5,
            "subsample": 0.8, "colsample_bytree": 0.7,
            "min_child_samples": 30,
            "random_state": 42, "verbose": -1,
        }

    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_tr, yt, sample_weight=w,
                  eval_set=[(X_va, yv)],
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    preds_lgb = anchor_va[v_va] + lgb_model.predict(X_va)
    rmse_lgb = np.sqrt(np.mean((spot_va[v_va] - preds_lgb) ** 2))
    rmse_lgb_hbc, preds_lgb_hbc = apply_hbc(preds_lgb, spot_va[v_va], hrs)
    print(f"  LightGBM      RMSE={rmse_lgb:7.4f}  +HBC={rmse_lgb_hbc:7.4f}  "
          f"iter={lgb_model.best_iteration_:5d}  ({time.time()-t1:.0f}s)")
    results["LightGBM"] = {"preds": preds_lgb, "preds_hbc": preds_lgb_hbc,
                           "rmse": rmse_lgb, "rmse_hbc": rmse_lgb_hbc}

    # ── 3. XGBoost ───────────────────────────────────────────────
    if HAS_XGB:
        t1 = time.time()
        if market == "fr":
            xgb_params = {
                "objective": "reg:squarederror", "eval_metric": "rmse",
                "n_estimators": 15000, "learning_rate": 0.05,
                "max_depth": 4, "reg_alpha": 5, "reg_lambda": 10,
                "subsample": 0.6, "colsample_bytree": 0.4,
                "min_child_weight": 15,
                "random_state": 42, "verbosity": 0,
                "tree_method": "hist",
            }
        else:
            xgb_params = {
                "objective": "reg:squarederror", "eval_metric": "rmse",
                "n_estimators": 15000, "learning_rate": 0.03,
                "max_depth": 7, "reg_alpha": 2, "reg_lambda": 8,
                "subsample": 0.75, "colsample_bytree": 0.6,
                "min_child_weight": 20,
                "random_state": 42, "verbosity": 0,
                "tree_method": "hist",
            }

        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_tr, yt, sample_weight=w,
                      eval_set=[(X_va, yv)],
                      verbose=False)
        preds_xgb = anchor_va[v_va] + xgb_model.predict(X_va)
        rmse_xgb = np.sqrt(np.mean((spot_va[v_va] - preds_xgb) ** 2))
        rmse_xgb_hbc, preds_xgb_hbc = apply_hbc(preds_xgb, spot_va[v_va], hrs)
        best_iter_xgb = xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else xgb_params["n_estimators"]
        print(f"  XGBoost       RMSE={rmse_xgb:7.4f}  +HBC={rmse_xgb_hbc:7.4f}  "
              f"iter={best_iter_xgb:5d}  ({time.time()-t1:.0f}s)")
        results["XGBoost"] = {"preds": preds_xgb, "preds_hbc": preds_xgb_hbc,
                              "rmse": rmse_xgb, "rmse_hbc": rmse_xgb_hbc}

    # ── 4. HistGradientBoosting (sklearn) ────────────────────────
    t1 = time.time()
    if market == "fr":
        hgb_params = {
            "max_iter": 5000, "learning_rate": 0.05,
            "max_depth": 4, "max_leaf_nodes": 15,
            "l2_regularization": 10,
            "min_samples_leaf": 20,
            "random_state": 42, "verbose": 0,
            "early_stopping": True, "n_iter_no_change": 200,
            "validation_fraction": 0.15,
        }
    else:
        hgb_params = {
            "max_iter": 5000, "learning_rate": 0.03,
            "max_depth": 7, "max_leaf_nodes": 63,
            "l2_regularization": 5,
            "min_samples_leaf": 30,
            "random_state": 42, "verbose": 0,
            "early_stopping": True, "n_iter_no_change": 200,
            "validation_fraction": 0.15,
        }

    # HistGB uses sample_weight but doesn't support external eval set natively
    hgb_model = HistGradientBoostingRegressor(**hgb_params)
    hgb_model.fit(X_tr.values, yt, sample_weight=w)
    preds_hgb = anchor_va[v_va] + hgb_model.predict(X_va.values)
    rmse_hgb = np.sqrt(np.mean((spot_va[v_va] - preds_hgb) ** 2))
    rmse_hgb_hbc, preds_hgb_hbc = apply_hbc(preds_hgb, spot_va[v_va], hrs)
    n_iter_hgb = hgb_model.n_iter_
    print(f"  HistGB        RMSE={rmse_hgb:7.4f}  +HBC={rmse_hgb_hbc:7.4f}  "
          f"iter={n_iter_hgb:5d}  ({time.time()-t1:.0f}s)")
    results["HistGB"] = {"preds": preds_hgb, "preds_hbc": preds_hgb_hbc,
                         "rmse": rmse_hgb, "rmse_hbc": rmse_hgb_hbc}

    return results, spot_va[v_va], hrs


def analyze_diversity(results, actual, hours, market):
    """Analyze error correlation and test multi-model ensembles."""
    model_names = list(results.keys())
    n_models = len(model_names)

    # ── Error correlation matrix ─────────────────────────────────
    print(f"\n  ERROR CORRELATION MATRIX ({market.upper()}):")
    errors = {}
    for name in model_names:
        errors[name] = actual - results[name]["preds"]

    # Header
    header = "  " + " " * 14 + "".join(f"{n:>12s}" for n in model_names)
    print(header)
    for n1 in model_names:
        row = f"  {n1:12s}"
        for n2 in model_names:
            corr = np.corrcoef(errors[n1], errors[n2])[0, 1]
            row += f"  {corr:10.4f}"
        print(row)

    # ── All 2-model ensembles ────────────────────────────────────
    print(f"\n  2-MODEL ENSEMBLES ({market.upper()}):")
    print(f"  {'Combination':35s}  {'w_opt':>6s}  {'RMSE':>7s}  {'+HBC':>7s}  {'vs CB':>7s}")
    print("  " + "-" * 70)

    best_2 = {"rmse_hbc": 999, "label": ""}
    baseline_hbc = results["CatBoost"]["rmse_hbc"]

    for m1, m2 in combinations(model_names, 2):
        best_w = 0.5
        best_rmse = 999
        for w in np.arange(0.0, 1.05, 0.05):
            ens = w * results[m1]["preds"] + (1 - w) * results[m2]["preds"]
            rmse = np.sqrt(np.mean((actual - ens) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_w = w

        ens_opt = best_w * results[m1]["preds"] + (1 - best_w) * results[m2]["preds"]
        rmse_hbc, _ = apply_hbc(ens_opt, actual, hours)

        label = f"{m1} + {m2}"
        delta = rmse_hbc - baseline_hbc
        marker = " <<<" if rmse_hbc == min(best_2["rmse_hbc"], rmse_hbc) and rmse_hbc < baseline_hbc else ""
        print(f"  {label:35s}  {best_w:6.2f}  {best_rmse:7.4f}  {rmse_hbc:7.4f}  {delta:+7.4f}{marker}")

        if rmse_hbc < best_2["rmse_hbc"]:
            best_2 = {"rmse_hbc": rmse_hbc, "label": label, "w": best_w,
                      "m1": m1, "m2": m2, "rmse": best_rmse}

    # ── All 3-model ensembles ────────────────────────────────────
    if n_models >= 3:
        print(f"\n  3-MODEL ENSEMBLES ({market.upper()}):")
        print(f"  {'Combination':35s}  {'Weights':>20s}  {'RMSE':>7s}  {'+HBC':>7s}  {'vs CB':>7s}")
        print("  " + "-" * 85)

        best_3 = {"rmse_hbc": 999, "label": ""}

        for m1, m2, m3 in combinations(model_names, 3):
            best_w = (0.33, 0.33, 0.34)
            best_rmse = 999
            for w1 in np.arange(0.0, 1.05, 0.1):
                for w2 in np.arange(0.0, 1.05 - w1, 0.1):
                    w3 = 1.0 - w1 - w2
                    if w3 < -0.01:
                        continue
                    ens = w1 * results[m1]["preds"] + w2 * results[m2]["preds"] + w3 * results[m3]["preds"]
                    rmse = np.sqrt(np.mean((actual - ens) ** 2))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_w = (w1, w2, w3)

            ens_opt = best_w[0] * results[m1]["preds"] + best_w[1] * results[m2]["preds"] + best_w[2] * results[m3]["preds"]
            rmse_hbc, _ = apply_hbc(ens_opt, actual, hours)

            label = f"{m1}+{m2}+{m3}"
            w_str = f"{best_w[0]:.1f}/{best_w[1]:.1f}/{best_w[2]:.1f}"
            delta = rmse_hbc - baseline_hbc
            print(f"  {label:35s}  {w_str:>20s}  {best_rmse:7.4f}  {rmse_hbc:7.4f}  {delta:+7.4f}")

            if rmse_hbc < best_3["rmse_hbc"]:
                best_3 = {"rmse_hbc": rmse_hbc, "label": label, "w": best_w,
                          "models": [m1, m2, m3], "rmse": best_rmse}

    # ── 4-model ensemble ─────────────────────────────────────────
    if n_models >= 4:
        print(f"\n  4-MODEL ENSEMBLE ({market.upper()}):")
        best_rmse_4 = 999
        best_w_4 = (0.25, 0.25, 0.25, 0.25)
        names_4 = model_names[:4]

        for w1 in np.arange(0.0, 1.05, 0.1):
            for w2 in np.arange(0.0, 1.05 - w1, 0.1):
                for w3 in np.arange(0.0, 1.05 - w1 - w2, 0.1):
                    w4 = 1.0 - w1 - w2 - w3
                    if w4 < -0.01:
                        continue
                    ens = (w1 * results[names_4[0]]["preds"] +
                           w2 * results[names_4[1]]["preds"] +
                           w3 * results[names_4[2]]["preds"] +
                           w4 * results[names_4[3]]["preds"])
                    rmse = np.sqrt(np.mean((actual - ens) ** 2))
                    if rmse < best_rmse_4:
                        best_rmse_4 = rmse
                        best_w_4 = (w1, w2, w3, w4)

        ens_4 = sum(best_w_4[i] * results[names_4[i]]["preds"] for i in range(4))
        rmse_4_hbc, _ = apply_hbc(ens_4, actual, hours)
        delta = rmse_4_hbc - baseline_hbc
        w_str = "/".join(f"{w:.1f}" for w in best_w_4)
        print(f"  {'+'.join(names_4):35s}  {w_str:>20s}  {best_rmse_4:7.4f}  {rmse_4_hbc:7.4f}  {delta:+7.4f}")

    return best_2, best_3 if n_models >= 3 else None


# ══════════════════════════════════════════════════════════════════════
# FR
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  FR — MODEL COMPARISON")
print("=" * 90)

fr_results, fr_actual, fr_hours = train_all_models(
    "fr", FR_FEATURES, fr_y_tr, fr_y_va, fr_valid_tr, fr_valid_va,
    fr_anchor_va, fr_spot_va, fr_weights, hours_va)

fr_best_2, fr_best_3 = analyze_diversity(fr_results, fr_actual, fr_hours, "fr")


# ══════════════════════════════════════════════════════════════════════
# UK
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  UK — MODEL COMPARISON")
print("=" * 90)

uk_results, uk_actual, uk_hours = train_all_models(
    "uk", UK_FEATURES, uk_y_tr, uk_y_va, uk_valid_tr, uk_valid_va,
    uk_moc_va, uk_spot_va, None, hours_va, loss_uk="MAE")

uk_best_2, uk_best_3 = analyze_diversity(uk_results, uk_actual, uk_hours, "uk")


# ══════════════════════════════════════════════════════════════════════
# COMBINED SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  COMBINED SUMMARY")
print("=" * 90)

# Standalone
print("\n  STANDALONE MODELS (+HBC):")
print(f"  {'Model':15s}  {'FR':>7s}  {'UK':>7s}  {'SUM':>7s}")
print("  " + "-" * 40)
for name in fr_results:
    if name in uk_results:
        s = fr_results[name]["rmse_hbc"] + uk_results[name]["rmse_hbc"]
        print(f"  {name:15s}  {fr_results[name]['rmse_hbc']:7.4f}  {uk_results[name]['rmse_hbc']:7.4f}  {s:7.4f}")

# Best ensembles
print(f"\n  BEST 2-MODEL ENSEMBLE:")
print(f"    FR: {fr_best_2['label']} → {fr_best_2['rmse_hbc']:.4f}")
print(f"    UK: {uk_best_2['label']} → {uk_best_2['rmse_hbc']:.4f}")
if fr_best_3 and uk_best_3:
    print(f"\n  BEST 3-MODEL ENSEMBLE:")
    print(f"    FR: {fr_best_3['label']} → {fr_best_3['rmse_hbc']:.4f}")
    print(f"    UK: {uk_best_3['label']} → {uk_best_3['rmse_hbc']:.4f}")

# Current vs best possible
current_sum = fr_results["CatBoost"]["rmse_hbc"] + uk_results["CatBoost"]["rmse_hbc"]
best_2_sum = fr_best_2["rmse_hbc"] + uk_best_2["rmse_hbc"]
print(f"\n  CB only SUM:       {current_sum:.4f}")
print(f"  Best 2-ens SUM:    {best_2_sum:.4f} (delta={best_2_sum - current_sum:+.4f})")
if fr_best_3 and uk_best_3:
    best_3_sum = fr_best_3["rmse_hbc"] + uk_best_3["rmse_hbc"]
    print(f"  Best 3-ens SUM:    {best_3_sum:.4f} (delta={best_3_sum - current_sum:+.4f})")

# Save
output = {
    "fr_standalone": {n: {"rmse": r["rmse"], "rmse_hbc": r["rmse_hbc"]}
                      for n, r in fr_results.items()},
    "uk_standalone": {n: {"rmse": r["rmse"], "rmse_hbc": r["rmse_hbc"]}
                      for n, r in uk_results.items()},
    "fr_best_2": {k: v for k, v in fr_best_2.items() if k != "preds"},
    "uk_best_2": {k: v for k, v in uk_best_2.items() if k != "preds"},
}
if fr_best_3:
    output["fr_best_3"] = {k: v for k, v in fr_best_3.items() if k != "preds"}
if uk_best_3:
    output["uk_best_3"] = {k: v for k, v in uk_best_3.items() if k != "preds"}

with open("outputs/model_diversity_ab_test.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n  Total time: {time.time() - t0:.0f}s")
