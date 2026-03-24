"""A/B test — Ensemble weights by hour regime.

Tests: fixed global weights vs per-regime weights.
One regime split based on electricity market structure:
  - Night   (0-5):   base load, low volatility
  - Morning (6-9):   demand ramp, volatile
  - Day     (10-16): solar influence, moderate
  - Peak    (17-21): evening peak, highest vol
  - Late    (22-23): wind-down

FR and UK, 3 models (CB+LGB+XGB).
"""

import sys, yaml, warnings, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

print("=" * 90)
print("  REGIME-BASED ENSEMBLE WEIGHTS A/B TEST")
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
fr_spot_la_full = df["fr_spot_la"].values
ema_fr = pd.Series(fr_spot_la_full).ewm(span=240).mean().values
fr_anchor_va = ema_fr[mask_val]
fr_spot_va = df_va["fr_spot"].values
fr_y_tr = df_tr["fr_spot"].values - ema_fr[~mask_val]
fr_y_va = fr_spot_va - fr_anchor_va

uk_moc_va = df_va["uk_merit_order_cost"].values
uk_spot_va = df_va["uk_spot"].values
uk_y_tr = df_tr["uk_spot"].values - df_tr["uk_merit_order_cost"].values
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

fr_valid_tr = np.isfinite(fr_y_tr) & np.isfinite(ema_fr[~mask_val]) & np.isfinite(fr_weights) & (fr_weights > 0)
fr_valid_va = np.isfinite(fr_y_va) & np.isfinite(fr_anchor_va)
uk_valid_tr = np.isfinite(uk_y_tr)
uk_valid_va = np.isfinite(uk_y_va)

# ── Regime definition ───────────────────────────────────────────────
REGIMES = {
    "night":   [0, 1, 2, 3, 4, 5],
    "morning": [6, 7, 8, 9],
    "day":     [10, 11, 12, 13, 14, 15, 16],
    "peak":    [17, 18, 19, 20, 21],
    "late":    [22, 23],
}

def hour_to_regime(h):
    for name, hours in REGIMES.items():
        if h in hours:
            return name
    return "day"

regime_va = np.array([hour_to_regime(h) for h in hours_va])


def apply_hbc(preds, actual, hours):
    errors = actual - preds
    hbc = {h: float(errors[hours == h].mean()) for h in range(24) if (hours == h).sum() > 0}
    corrected = preds + np.array([hbc.get(h, 0) for h in hours])
    return np.sqrt(np.mean((actual - corrected) ** 2))


def compute_rmse(actual, preds):
    return np.sqrt(np.mean((actual - preds) ** 2))


# ══════════════════════════════════════════════════════════════════════
# TRAIN 3 MODELS (full training data)
# ══════════════════════════════════════════════════════════════════════

def train_models(market, features, y_tr, y_va, v_tr, v_va, anchor_va, spot_va, weights):
    feat = [f for f in features if f in df_tr.columns]
    X_tr = df_tr.loc[df_tr.index[v_tr], feat]
    X_va = df_va.loc[df_va.index[v_va], feat]
    w = weights[v_tr] if weights is not None else None

    preds = {}

    # CatBoost
    if market == "fr":
        cb_p = {"loss_function": "RMSE", "eval_metric": "RMSE", "iterations": 15000,
                "learning_rate": 0.059, "depth": 3, "l2_leaf_reg": 4.42,
                "subsample": 0.533, "colsample_bylevel": 0.228, "min_child_samples": 14,
                "random_strength": 0.9, "random_seed": 42, "verbose": 0,
                "use_best_model": True, "allow_writing_files": False}
    else:
        cb_p = {"loss_function": "MAE", "eval_metric": "RMSE", "iterations": 15000,
                "learning_rate": 0.03, "depth": 8, "l2_leaf_reg": 5,
                "colsample_bylevel": 0.8, "subsample": 0.8, "random_seed": 42,
                "verbose": 0, "use_best_model": True, "allow_writing_files": False}
    cb = CatBoostRegressor(**cb_p)
    cb.fit(Pool(X_tr, y_tr[v_tr], weight=w), eval_set=Pool(X_va, y_va[v_va]),
           early_stopping_rounds=200, verbose=0)
    preds["CB"] = anchor_va[v_va] + cb.predict(X_va)

    # LightGBM
    if market == "fr":
        lgb_p = {"objective": "regression", "metric": "rmse", "n_estimators": 15000,
                 "learning_rate": 0.03, "max_depth": 4, "num_leaves": 15,
                 "reg_alpha": 5, "reg_lambda": 30, "subsample": 0.7,
                 "colsample_bytree": 0.5, "min_child_samples": 50,
                 "random_state": 42, "verbose": -1}
    else:
        lgb_p = {"objective": "regression", "metric": "rmse", "n_estimators": 15000,
                 "learning_rate": 0.02, "max_depth": 7, "num_leaves": 63,
                 "reg_alpha": 1, "reg_lambda": 5, "subsample": 0.8,
                 "colsample_bytree": 0.7, "min_child_samples": 30,
                 "random_state": 42, "verbose": -1}
    lgb_m = lgb.LGBMRegressor(**lgb_p)
    lgb_m.fit(X_tr, y_tr[v_tr], sample_weight=w, eval_set=[(X_va, y_va[v_va])],
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    preds["LGB"] = anchor_va[v_va] + lgb_m.predict(X_va)

    # XGBoost
    if market == "fr":
        xgb_p = {"objective": "reg:squarederror", "eval_metric": "rmse", "n_estimators": 15000,
                 "learning_rate": 0.05, "max_depth": 4, "reg_alpha": 5, "reg_lambda": 10,
                 "subsample": 0.6, "colsample_bytree": 0.4, "min_child_weight": 15,
                 "random_state": 42, "verbosity": 0, "tree_method": "hist"}
    else:
        xgb_p = {"objective": "reg:squarederror", "eval_metric": "rmse", "n_estimators": 15000,
                 "learning_rate": 0.03, "max_depth": 7, "reg_alpha": 2, "reg_lambda": 8,
                 "subsample": 0.75, "colsample_bytree": 0.6, "min_child_weight": 20,
                 "random_state": 42, "verbosity": 0, "tree_method": "hist"}
    xgb_m = xgb.XGBRegressor(**xgb_p)
    xgb_m.fit(X_tr, y_tr[v_tr], sample_weight=w, eval_set=[(X_va, y_va[v_va])], verbose=False)
    preds["XGB"] = anchor_va[v_va] + xgb_m.predict(X_va)

    return preds


def optimize_3w(preds, actual, mask=None):
    """Find best w1,w2,w3 for CB,LGB,XGB on subset defined by mask."""
    if mask is not None:
        p = {n: preds[n][mask] for n in preds}
        a = actual[mask]
    else:
        p = preds
        a = actual

    best = {"rmse": 999, "w": (1.0, 0.0, 0.0)}
    for w1 in np.arange(0.0, 1.05, 0.1):
        for w2 in np.arange(0.0, 1.05 - w1, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < -0.01:
                continue
            ens = w1 * p["CB"] + w2 * p["LGB"] + w3 * p["XGB"]
            rmse = np.sqrt(np.mean((a - ens) ** 2))
            if rmse < best["rmse"]:
                best = {"rmse": rmse, "w": (round(w1, 1), round(w2, 1), round(w3, 1))}
    return best


# ══════════════════════════════════════════════════════════════════════
def run_market(market, features, y_tr, y_va, v_tr, v_va, anchor_va, spot_va, weights):
    print(f"\n  Training 3 models...")
    preds = train_models(market, features, y_tr, y_va, v_tr, v_va, anchor_va, spot_va, weights)
    actual = spot_va[v_va]
    hrs = hours_va[v_va]
    reg = regime_va[v_va]

    # A. Global fixed weights
    global_best = optimize_3w(preds, actual)
    global_ens = sum(global_best["w"][i] * preds[n] for i, n in enumerate(["CB", "LGB", "XGB"]))
    rmse_global = compute_rmse(actual, global_ens)
    rmse_global_hbc = apply_hbc(global_ens, actual, hrs)
    print(f"  A. Global fixed:   CB={global_best['w'][0]} LGB={global_best['w'][1]} XGB={global_best['w'][2]}  "
          f"RMSE={rmse_global:.4f}  +HBC={rmse_global_hbc:.4f}")

    # B. Per-regime weights
    regime_weights = {}
    regime_ens = np.zeros(len(actual))
    for rname in REGIMES:
        rmask = reg == rname
        if rmask.sum() == 0:
            continue
        rbest = optimize_3w(preds, actual, rmask)
        regime_weights[rname] = rbest
        regime_ens[rmask] = sum(rbest["w"][i] * preds[n][rmask] for i, n in enumerate(["CB", "LGB", "XGB"]))

    rmse_regime = compute_rmse(actual, regime_ens)
    rmse_regime_hbc = apply_hbc(regime_ens, actual, hrs)
    print(f"  B. Per-regime:     RMSE={rmse_regime:.4f}  +HBC={rmse_regime_hbc:.4f}  "
          f"({len(REGIMES)} regimes x 3 weights = {len(REGIMES)*3} params)")

    for rname, rbest in regime_weights.items():
        n_hours = len(REGIMES[rname])
        rmask = reg == rname
        r_rmse = compute_rmse(actual[rmask],
                              sum(rbest["w"][i] * preds[n][rmask] for i, n in enumerate(["CB", "LGB", "XGB"])))
        print(f"    {rname:8s} (h={REGIMES[rname]})  CB={rbest['w'][0]} LGB={rbest['w'][1]} XGB={rbest['w'][2]}  "
              f"RMSE={r_rmse:.2f}  n={rmask.sum()}")

    # Summary
    delta = rmse_regime_hbc - rmse_global_hbc
    print(f"\n  Delta (regime vs global): {delta:+.4f}")

    return {
        "global": {"rmse_hbc": rmse_global_hbc, "weights": global_best["w"]},
        "regime": {"rmse_hbc": rmse_regime_hbc, "weights": regime_weights},
    }


# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  FR — REGIME WEIGHTS")
print("=" * 90)
fr_res = run_market("fr", FR_FEATURES, fr_y_tr, fr_y_va, fr_valid_tr, fr_valid_va,
                    fr_anchor_va, fr_spot_va, fr_weights)

print("\n" + "=" * 90)
print("  UK — REGIME WEIGHTS")
print("=" * 90)
uk_res = run_market("uk", UK_FEATURES, uk_y_tr, uk_y_va, uk_valid_tr, uk_valid_va,
                    uk_moc_va, uk_spot_va, None)

# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  COMBINED")
print("=" * 90)
g_sum = fr_res["global"]["rmse_hbc"] + uk_res["global"]["rmse_hbc"]
r_sum = fr_res["regime"]["rmse_hbc"] + uk_res["regime"]["rmse_hbc"]
print(f"  Global:  FR={fr_res['global']['rmse_hbc']:.4f} + UK={uk_res['global']['rmse_hbc']:.4f} = {g_sum:.4f}")
print(f"  Regime:  FR={fr_res['regime']['rmse_hbc']:.4f} + UK={uk_res['regime']['rmse_hbc']:.4f} = {r_sum:.4f}")
print(f"  Delta:   {r_sum - g_sum:+.4f}")

with open("outputs/regime_weights_ab_test.json", "w") as f:
    json.dump({"fr": fr_res, "uk": uk_res, "global_sum": g_sum, "regime_sum": r_sum},
              f, indent=2, default=str)

print(f"\n  Total time: {time.time() - t0:.0f}s")
