"""A/B test — Elastic Net as 4th ensemble model.

Tests whether adding a linear model (Ridge / ElasticNet) to the
CB+LGB+XGB ensemble improves RMSE via error diversity.

Key hypothesis: linear models capture linear relationships exactly,
while trees approximate them with step functions. The error correlation
should be low → ensemble gain.
"""

import sys, yaml, warnings, time, json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.preprocessing import StandardScaler

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
print("  ELASTIC NET A/B TEST — Linear model as 4th ensemble member")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
df = build_features(train, config)

holdout_start = config["validation"]["holdout_start"]
mask_val = (df["datetime_CET"] >= holdout_start).values

# Interaction feature
if "fr_spot_la_roll_168h_mean" in df.columns and "uk_price_per_mw_7d" in df.columns:
    df["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
        df["fr_spot_la_roll_168h_mean"] * df["uk_price_per_mw_7d"]
    )

df_tr = df[~mask_val].copy()
df_va = df[mask_val].copy()
print(f"  Data loaded in {time.time() - t0:.0f}s  |  Train: {len(df_tr)}, Val: {len(df_va)}")

# ── Feature lists ────────────────────────────────────────────────────
with open("outputs/feature_selection_v5_fr.json") as f:
    fs_v5 = json.load(f)
FR_FEATURES = fs_v5["features"] + ["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"]
FR_FEATURES = [f for f in FR_FEATURES if f in df_tr.columns]

with open("outputs/uk_feature_research.json") as f:
    uk_research = json.load(f)
UK_FEATURES = [f for f in uk_research["confirmed_features"] if f in df_tr.columns]

# ── Targets ──────────────────────────────────────────────────────────
# FR: EMA 240h stationary
fr_spot_la_full = df["fr_spot_la"].values
ema_fr = pd.Series(fr_spot_la_full).ewm(span=240).mean().values
fr_anchor_va = ema_fr[mask_val]
fr_spot_va = df_va["fr_spot"].values
fr_y_tr = df_tr["fr_spot"].values - ema_fr[~mask_val]
fr_y_va = fr_spot_va - fr_anchor_va

# UK: basis (spot - merit_order_cost)
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


def compute_rmse(actual, preds):
    return np.sqrt(np.mean((actual - preds) ** 2))


def apply_hbc(preds, actual, hours):
    errors = actual - preds
    hbc = {h: float(errors[hours == h].mean()) for h in range(24) if (hours == h).sum() > 0}
    corrected = preds + np.array([hbc.get(h, 0) for h in hours])
    return np.sqrt(np.mean((actual - corrected) ** 2))


# ══════════════════════════════════════════════════════════════════════
# TRAIN TREE MODELS (same as pipeline)
# ══════════════════════════════════════════════════════════════════════

def train_tree_models(market, features, y_tr, y_va, v_tr, v_va, anchor_va, spot_va, weights):
    feat = [f for f in features if f in df_tr.columns]
    X_tr = df_tr.loc[df_tr.index[v_tr], feat].values
    X_va = df_va.loc[df_va.index[v_va], feat].values
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
    cb.fit(Pool(X_tr, y_tr[v_tr], weight=w),
           eval_set=Pool(X_va, y_va[v_va]),
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
    lgb_m.fit(X_tr, y_tr[v_tr], sample_weight=w,
              eval_set=[(X_va, y_va[v_va])],
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
    xgb_m.fit(X_tr, y_tr[v_tr], sample_weight=w,
              eval_set=[(X_va, y_va[v_va])], verbose=False)
    preds["XGB"] = anchor_va[v_va] + xgb_m.predict(X_va)

    return preds, feat


# ══════════════════════════════════════════════════════════════════════
# TRAIN LINEAR MODELS
# ══════════════════════════════════════════════════════════════════════

def train_linear_models(market, features, y_tr, y_va, v_tr, v_va, anchor_va, weights):
    """Train Ridge, ElasticNet, Lasso with standardized features."""
    feat = [f for f in features if f in df_tr.columns]
    X_tr_raw = df_tr.loc[df_tr.index[v_tr], feat].values
    X_va_raw = df_va.loc[df_va.index[v_va], feat].values
    w = weights[v_tr] if weights is not None else None

    # Standardize (critical for linear models)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_va = scaler.transform(X_va_raw)

    # Replace NaN with 0 after scaling
    X_tr = np.nan_to_num(X_tr, 0)
    X_va = np.nan_to_num(X_va, 0)

    preds = {}

    # Ridge (pure L2)
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_tr, y_tr[v_tr], sample_weight=w)
        preds[f"Ridge_a{alpha}"] = anchor_va[v_va] + ridge.predict(X_va)

    # Elastic Net (L1 + L2)
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        for l1 in [0.1, 0.5, 0.9]:
            en = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=10000)
            en.fit(X_tr, y_tr[v_tr])  # ElasticNet doesn't support sample_weight
            preds[f"EN_a{alpha}_l{l1}"] = anchor_va[v_va] + en.predict(X_va)

    return preds


# ══════════════════════════════════════════════════════════════════════
# RUN FOR EACH MARKET
# ══════════════════════════════════════════════════════════════════════

def run_market(market, features, y_tr, y_va, v_tr, v_va, anchor_va, spot_va, weights):
    print(f"\n{'='*90}")
    print(f"  {market.upper()} — ELASTIC NET A/B TEST")
    print(f"{'='*90}")

    hrs = hours_va[v_va]
    actual = spot_va[v_va]

    # Train tree models
    print(f"  Training tree models...")
    tree_preds, feat = train_tree_models(market, features, y_tr, y_va, v_tr, v_va, anchor_va, spot_va, weights)

    # Train linear models
    print(f"  Training linear models...")
    linear_preds = train_linear_models(market, features, y_tr, y_va, v_tr, v_va, anchor_va, weights)

    # ── Standalone results ──────────────────────────────────────────
    print(f"\n  Standalone RMSE (+HBC):")
    all_preds = {**tree_preds, **linear_preds}
    results = {}
    for name, pred in sorted(all_preds.items(), key=lambda x: compute_rmse(actual, x[1])):
        rmse = compute_rmse(actual, pred)
        rmse_hbc = apply_hbc(pred, actual, hrs)
        results[name] = {"rmse": rmse, "rmse_hbc": rmse_hbc}
        marker = " ***" if name.startswith(("Ridge", "EN_")) else ""
        print(f"    {name:20s}: RMSE={rmse:.2f}  +HBC={rmse_hbc:.2f}{marker}")

    # ── Error correlation ──────────────────────────────────────────
    # Find best linear model
    best_linear_name = min(
        [(n, r["rmse_hbc"]) for n, r in results.items() if n.startswith(("Ridge", "EN_"))],
        key=lambda x: x[1]
    )[0]
    best_linear_pred = all_preds[best_linear_name]

    print(f"\n  Best linear model: {best_linear_name} (RMSE+HBC={results[best_linear_name]['rmse_hbc']:.2f})")

    print(f"\n  Error correlation matrix (with {best_linear_name}):")
    errors = {}
    for name in ["CB", "LGB", "XGB", best_linear_name]:
        errors[name] = actual - all_preds[name]

    names = list(errors.keys())
    print(f"    {'':15s}", end="")
    for n in names:
        print(f" {n:>8s}", end="")
    print()
    for n1 in names:
        print(f"    {n1:15s}", end="")
        for n2 in names:
            corr = np.corrcoef(errors[n1], errors[n2])[0, 1]
            print(f" {corr:8.3f}", end="")
        print()

    # ── 4-model ensemble (CB + LGB + XGB + Linear) ──────────────────
    print(f"\n  Ensemble tests:")

    # A. Current 3-model (CB + LGB + XGB) — best regime weights
    REGIMES = {
        "night": [0, 1, 2, 3, 4, 5], "morning": [6, 7, 8, 9],
        "day": [10, 11, 12, 13, 14, 15, 16], "peak": [17, 18, 19, 20, 21],
        "late": [22, 23],
    }

    def optimize_regime_4model(model_preds, actual, hours, names):
        """Optimize 4-model weights per regime."""
        regime_ens = np.zeros(len(actual))
        regime_weights = {}

        for rname, rhours in REGIMES.items():
            rmask = np.isin(hours, rhours)
            if rmask.sum() == 0:
                continue

            a = actual[rmask]
            p = {n: model_preds[n][rmask] for n in names}
            best = {"rmse": 999, "w": None}

            if len(names) == 3:
                for w1 in np.arange(0.0, 1.05, 0.1):
                    for w2 in np.arange(0.0, 1.05 - w1, 0.1):
                        w3 = round(1.0 - w1 - w2, 1)
                        if w3 < -0.01:
                            continue
                        e = w1 * p[names[0]] + w2 * p[names[1]] + w3 * p[names[2]]
                        r = compute_rmse(a, e)
                        if r < best["rmse"]:
                            best = {"rmse": r, "w": {names[0]: round(w1, 1),
                                                      names[1]: round(w2, 1),
                                                      names[2]: w3}}
            elif len(names) == 4:
                # Coarser grid for 4 models (step=0.1 would be too many combos)
                for w1 in np.arange(0.0, 1.05, 0.1):
                    for w2 in np.arange(0.0, 1.05 - w1, 0.1):
                        for w3 in np.arange(0.0, 1.05 - w1 - w2, 0.1):
                            w4 = round(1.0 - w1 - w2 - w3, 1)
                            if w4 < -0.01:
                                continue
                            e = (w1 * p[names[0]] + w2 * p[names[1]] +
                                 w3 * p[names[2]] + w4 * p[names[3]])
                            r = compute_rmse(a, e)
                            if r < best["rmse"]:
                                best = {"rmse": r, "w": {
                                    names[0]: round(w1, 1), names[1]: round(w2, 1),
                                    names[2]: round(w3, 1), names[3]: w4}}

            regime_weights[rname] = best["w"]
            regime_ens[rmask] = sum(best["w"][n] * p[n] for n in names)

        return regime_weights, regime_ens

    # A. 3-model regime
    rw_3, ens_3 = optimize_regime_4model(tree_preds, actual, hrs, ["CB", "LGB", "XGB"])
    rmse_3 = compute_rmse(actual, ens_3)
    rmse_3_hbc = apply_hbc(ens_3, actual, hrs)
    print(f"    A. 3-model (CB+LGB+XGB):          RMSE={rmse_3:.2f}  +HBC={rmse_3_hbc:.2f}")
    for rname, rw in rw_3.items():
        w_str = " / ".join(f"{n}={rw.get(n, 0):.1f}" for n in ["CB", "LGB", "XGB"])
        print(f"       {rname:8s}: {w_str}")

    # B. 4-model regime (CB + LGB + XGB + best linear)
    four_preds = {**tree_preds, "LIN": best_linear_pred}
    rw_4, ens_4 = optimize_regime_4model(four_preds, actual, hrs, ["CB", "LGB", "XGB", "LIN"])
    rmse_4 = compute_rmse(actual, ens_4)
    rmse_4_hbc = apply_hbc(ens_4, actual, hrs)
    print(f"\n    B. 4-model (CB+LGB+XGB+{best_linear_name}):  RMSE={rmse_4:.2f}  +HBC={rmse_4_hbc:.2f}")
    for rname, rw in rw_4.items():
        w_str = " / ".join(f"{n}={rw.get(n, 0):.1f}" for n in ["CB", "LGB", "XGB", "LIN"])
        print(f"       {rname:8s}: {w_str}")

    # C. Try replacing XGB with Linear (CB + LGB + Linear)
    three_alt = {"CB": tree_preds["CB"], "LGB": tree_preds["LGB"], "LIN": best_linear_pred}
    rw_alt, ens_alt = optimize_regime_4model(three_alt, actual, hrs, ["CB", "LGB", "LIN"])
    rmse_alt = compute_rmse(actual, ens_alt)
    rmse_alt_hbc = apply_hbc(ens_alt, actual, hrs)
    print(f"\n    C. 3-model (CB+LGB+LIN, no XGB):  RMSE={rmse_alt:.2f}  +HBC={rmse_alt_hbc:.2f}")
    for rname, rw in rw_alt.items():
        w_str = " / ".join(f"{n}={rw.get(n, 0):.1f}" for n in ["CB", "LGB", "LIN"])
        print(f"       {rname:8s}: {w_str}")

    delta_4vs3 = rmse_4_hbc - rmse_3_hbc
    print(f"\n  Delta 4-model vs 3-model: {delta_4vs3:+.4f}")
    delta_alt = rmse_alt_hbc - rmse_3_hbc
    print(f"  Delta CB+LGB+LIN vs 3-model: {delta_alt:+.4f}")

    return {
        "3model_hbc": rmse_3_hbc, "4model_hbc": rmse_4_hbc,
        "alt_hbc": rmse_alt_hbc,
        "best_linear": best_linear_name,
        "3model_weights": rw_3, "4model_weights": rw_4,
        "standalone": {n: r for n, r in results.items()},
    }


# ══════════════════════════════════════════════════════════════════════
print("\n  Training FR models...")
fr_res = run_market("fr", FR_FEATURES, fr_y_tr, fr_y_va, fr_valid_tr, fr_valid_va,
                    fr_anchor_va, fr_spot_va, fr_weights)

print("\n  Training UK models...")
uk_res = run_market("uk", UK_FEATURES, uk_y_tr, uk_y_va, uk_valid_tr, uk_valid_va,
                    uk_moc_va, uk_spot_va, None)

# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  COMBINED RESULTS")
print("=" * 90)

sum_3 = fr_res["3model_hbc"] + uk_res["3model_hbc"]
sum_4 = fr_res["4model_hbc"] + uk_res["4model_hbc"]
sum_alt = fr_res["alt_hbc"] + uk_res["alt_hbc"]

print(f"  A. 3-model (CB+LGB+XGB):     FR={fr_res['3model_hbc']:.2f} + UK={uk_res['3model_hbc']:.2f} = {sum_3:.2f}")
print(f"  B. 4-model (+Linear):         FR={fr_res['4model_hbc']:.2f} + UK={uk_res['4model_hbc']:.2f} = {sum_4:.2f}  ({sum_4-sum_3:+.2f})")
print(f"  C. CB+LGB+LIN (no XGB):       FR={fr_res['alt_hbc']:.2f} + UK={uk_res['alt_hbc']:.2f} = {sum_alt:.2f}  ({sum_alt-sum_3:+.2f})")

with open("outputs/elastic_net_ab_test.json", "w") as f:
    json.dump({"fr": fr_res, "uk": uk_res,
               "sum_3model": sum_3, "sum_4model": sum_4, "sum_alt": sum_alt},
              f, indent=2, default=str)

print(f"\n  Total time: {time.time() - t0:.0f}s")
