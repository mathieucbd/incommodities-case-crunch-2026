"""A/B test — Stacking & conditional ensemble weighting.

Instead of fixed ensemble weights, learn WHEN to trust each model more.

Approaches:
  A. Fixed weights (current baseline — optimized on val)
  B. Ridge stacking (learn linear combination)
  C. Ridge stacking + context features (hour, volatility → conditional weights)
  D. Per-hour optimal weights (24 sets of weights)
  E. Quantile-gated: train models at q=0.25/0.5/0.75, use interval width
     as confidence → weight inversely to uncertainty

Both FR and UK, with CatBoost + LightGBM + XGBoost.
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
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

print("=" * 90)
print("  STACKING & CONDITIONAL ENSEMBLE A/B TEST")
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
fr_anchor_tr, fr_anchor_va = ema_fr[~mask_val], ema_fr[mask_val]
fr_spot_tr, fr_spot_va = df_tr["fr_spot"].values, df_va["fr_spot"].values
fr_y_tr = fr_spot_tr - fr_anchor_tr
fr_y_va = fr_spot_va - fr_anchor_va

uk_moc_tr, uk_moc_va = df_tr["uk_merit_order_cost"].values, df_va["uk_merit_order_cost"].values
uk_spot_tr, uk_spot_va = df_tr["uk_spot"].values, df_va["uk_spot"].values
uk_y_tr = uk_spot_tr - uk_moc_tr
uk_y_va = uk_spot_va - uk_moc_va

hours_tr = df_tr["hour"].values
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


def apply_hbc(preds_spot, actual, hours):
    errors = actual - preds_spot
    hbc = {h: float(errors[hours == h].mean()) for h in range(24) if (hours == h).sum() > 0}
    corrected = preds_spot + np.array([hbc.get(h, 0) for h in hours])
    rmse_hbc = np.sqrt(np.mean((actual - corrected) ** 2))
    return rmse_hbc, corrected, hbc


def compute_rmse(actual, preds):
    return np.sqrt(np.mean((actual - preds) ** 2))


# ══════════════════════════════════════════════════════════════════════
# TRAIN BASE MODELS — with temporal split for stacking
# ══════════════════════════════════════════════════════════════════════
# For proper stacking, we need OOF predictions on the validation set.
# But we also need meta-train data. So we split training into:
#   - meta_train (first 75%) → train base models
#   - meta_val (last 25%) → get OOF predictions for stacking training
#   - val (holdout) → final evaluation
#
# This avoids information leakage in stacking.

def run_market(market, features, y_tr, y_va, v_tr, v_va, anchor_tr, anchor_va,
               spot_tr, spot_va, weights, hours_t, hours_v):
    """Run all stacking experiments for one market."""
    feat = [f for f in features if f in df_tr.columns]

    X_tr_full = df_tr.loc[df_tr.index[v_tr], feat]
    X_va = df_va.loc[df_va.index[v_va], feat]
    yt_full = y_tr[v_tr]
    yv = y_va[v_va]
    w_full = weights[v_tr] if weights is not None else None
    hrs_v = hours_v[v_va]
    actual = spot_va[v_va]
    anch_va = anchor_va[v_va]

    # ── Temporal split for meta-train ───────────────────────────
    n_tr = len(X_tr_full)
    split_idx = int(n_tr * 0.75)
    X_base = X_tr_full.iloc[:split_idx]
    X_meta = X_tr_full.iloc[split_idx:]
    y_base = yt_full[:split_idx]
    y_meta = yt_full[split_idx:]
    w_base = w_full[:split_idx] if w_full is not None else None
    w_meta = w_full[split_idx:] if w_full is not None else None

    # Anchors and spots for meta set
    # The anchor values correspond to the training indices
    anchor_full = anchor_tr[v_tr] if anchor_tr is not None else None
    spot_full = spot_tr[v_tr]
    anch_meta = anchor_full[split_idx:] if anchor_full is not None else None
    spot_meta = spot_full[split_idx:]
    hrs_meta = hours_t[v_tr][split_idx:]

    print(f"\n  Base train: {len(X_base)}, Meta train: {len(X_meta)}, Val: {len(X_va)}")

    # ── Train 3 base models on base set ─────────────────────────
    models = {}

    # CatBoost
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
            "loss_function": "MAE", "eval_metric": "RMSE",
            "iterations": 15000, "learning_rate": 0.03, "depth": 8,
            "l2_leaf_reg": 5, "colsample_bylevel": 0.8, "subsample": 0.8,
            "random_seed": 42, "verbose": 0, "use_best_model": True,
            "allow_writing_files": False,
        }

    pool_b = Pool(X_base, y_base, weight=w_base)
    pool_m = Pool(X_meta, y_meta)
    cb = CatBoostRegressor(**cb_params)
    cb.fit(pool_b, eval_set=pool_m, early_stopping_rounds=200, verbose=0)
    models["CB"] = cb

    # LightGBM
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
    lgb_m = lgb.LGBMRegressor(**lgb_params)
    lgb_m.fit(X_base, y_base, sample_weight=w_base,
              eval_set=[(X_meta, y_meta)],
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    models["LGB"] = lgb_m

    # XGBoost
    if market == "fr":
        xgb_params = {
            "objective": "reg:squarederror", "eval_metric": "rmse",
            "n_estimators": 15000, "learning_rate": 0.05,
            "max_depth": 4, "reg_alpha": 5, "reg_lambda": 10,
            "subsample": 0.6, "colsample_bytree": 0.4,
            "min_child_weight": 15,
            "random_state": 42, "verbosity": 0, "tree_method": "hist",
        }
    else:
        xgb_params = {
            "objective": "reg:squarederror", "eval_metric": "rmse",
            "n_estimators": 15000, "learning_rate": 0.03,
            "max_depth": 7, "reg_alpha": 2, "reg_lambda": 8,
            "subsample": 0.75, "colsample_bytree": 0.6,
            "min_child_weight": 20,
            "random_state": 42, "verbosity": 0, "tree_method": "hist",
        }
    xgb_m = xgb.XGBRegressor(**xgb_params)
    xgb_m.fit(X_base, y_base, sample_weight=w_base,
              eval_set=[(X_meta, y_meta)], verbose=False)
    models["XGB"] = xgb_m

    # ── Get predictions on meta set and val set ──────────────────
    meta_preds = {}
    val_preds = {}
    for name, model in models.items():
        if name == "CB":
            meta_preds[name] = model.predict(X_meta)
            val_preds[name] = model.predict(X_va)
        elif name == "LGB":
            meta_preds[name] = model.predict(X_meta)
            val_preds[name] = model.predict(X_va)
        else:
            meta_preds[name] = model.predict(X_meta)
            val_preds[name] = model.predict(X_va)

    # Convert deviation predictions to spot for evaluation
    val_spot_preds = {name: anch_va + p for name, p in val_preds.items()}

    # Standalone results
    print(f"\n  STANDALONE ({market.upper()}):")
    for name in models:
        rmse = compute_rmse(actual, val_spot_preds[name])
        rmse_hbc, _, _ = apply_hbc(val_spot_preds[name], actual, hrs_v)
        print(f"    {name:6s} RMSE={rmse:7.4f}  +HBC={rmse_hbc:7.4f}")

    # ══════════════════════════════════════════════════════════════
    # A. Fixed optimal weights (grid search on val)
    # ══════════════════════════════════════════════════════════════
    best_fixed = {"rmse_hbc": 999}
    for w1 in np.arange(0.0, 1.05, 0.05):
        for w2 in np.arange(0.0, 1.05 - w1, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < -0.01:
                continue
            ens = w1 * val_spot_preds["CB"] + w2 * val_spot_preds["LGB"] + w3 * val_spot_preds["XGB"]
            rmse_hbc, _, _ = apply_hbc(ens, actual, hrs_v)
            if rmse_hbc < best_fixed["rmse_hbc"]:
                best_fixed = {"rmse_hbc": rmse_hbc, "w": (w1, w2, w3),
                              "rmse": compute_rmse(actual, ens)}

    print(f"\n  A. Fixed weights:       +HBC={best_fixed['rmse_hbc']:.4f}  "
          f"w=CB:{best_fixed['w'][0]:.2f}/LGB:{best_fixed['w'][1]:.2f}/XGB:{best_fixed['w'][2]:.2f}")

    # ══════════════════════════════════════════════════════════════
    # B. Ridge stacking (on deviation predictions)
    # ══════════════════════════════════════════════════════════════
    # Meta features: predictions from each model on meta set
    S_meta = np.column_stack([meta_preds[n] for n in ["CB", "LGB", "XGB"]])
    S_val = np.column_stack([val_preds[n] for n in ["CB", "LGB", "XGB"]])

    ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])
    ridge.fit(S_meta, y_meta)
    stack_val_dev = ridge.predict(S_val)
    stack_val_spot = anch_va + stack_val_dev
    rmse_stack = compute_rmse(actual, stack_val_spot)
    rmse_stack_hbc, _, _ = apply_hbc(stack_val_spot, actual, hrs_v)

    print(f"  B. Ridge stacking:      +HBC={rmse_stack_hbc:.4f}  "
          f"coefs={[f'{c:.3f}' for c in ridge.coef_]}  alpha={ridge.alpha_:.2f}")

    # ══════════════════════════════════════════════════════════════
    # C. Ridge stacking + context features
    # ══════════════════════════════════════════════════════════════
    # Add hour, rolling_std, scarcity as context
    context_cols_meta = []
    context_cols_val = []

    # Hour (one-hot would be too many params, use sin/cos)
    h_meta = hours_t[v_tr][split_idx:]
    h_val = hrs_v
    context_cols_meta.append(np.sin(2 * np.pi * h_meta / 24))
    context_cols_meta.append(np.cos(2 * np.pi * h_meta / 24))
    context_cols_val.append(np.sin(2 * np.pi * h_val / 24))
    context_cols_val.append(np.cos(2 * np.pi * h_val / 24))

    # Rolling std (volatility regime)
    if "fr_spot_la_roll_168h_std" in df_tr.columns:
        std_col = "fr_spot_la_roll_168h_std" if market == "fr" else "uk_spot_la_roll_168h_std"
        if std_col in df_tr.columns:
            std_meta = df_tr.loc[df_tr.index[v_tr], std_col].values[split_idx:]
            std_val = df_va.loc[df_va.index[v_va], std_col].values
            std_meta = np.nan_to_num(std_meta, nan=0)
            std_val = np.nan_to_num(std_val, nan=0)
            context_cols_meta.append(std_meta)
            context_cols_val.append(std_val)

    # Prediction spread (disagreement between models)
    spread_meta = np.std(S_meta, axis=1)
    spread_val = np.std(S_val, axis=1)
    context_cols_meta.append(spread_meta)
    context_cols_val.append(spread_val)

    SC_meta = np.column_stack([S_meta] + context_cols_meta)
    SC_val = np.column_stack([S_val] + context_cols_val)

    scaler = StandardScaler()
    SC_meta_s = scaler.fit_transform(SC_meta)
    SC_val_s = scaler.transform(SC_val)

    ridge_ctx = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])
    ridge_ctx.fit(SC_meta_s, y_meta)
    stack_ctx_dev = ridge_ctx.predict(SC_val_s)
    stack_ctx_spot = anch_va + stack_ctx_dev
    rmse_ctx = compute_rmse(actual, stack_ctx_spot)
    rmse_ctx_hbc, _, _ = apply_hbc(stack_ctx_spot, actual, hrs_v)

    print(f"  C. Ridge + context:     +HBC={rmse_ctx_hbc:.4f}  "
          f"alpha={ridge_ctx.alpha_:.2f}  n_feat={SC_meta.shape[1]}")

    # ══════════════════════════════════════════════════════════════
    # D. Per-hour optimal weights
    # ══════════════════════════════════════════════════════════════
    hourly_preds = np.zeros(len(actual))
    hourly_weights = {}

    for h in range(24):
        h_mask = hrs_v == h
        if h_mask.sum() < 10:
            hourly_preds[h_mask] = val_spot_preds["CB"][h_mask]
            hourly_weights[h] = (1.0, 0.0, 0.0)
            continue

        best_w_h = (1.0, 0.0, 0.0)
        best_rmse_h = 999
        for w1 in np.arange(0.0, 1.05, 0.1):
            for w2 in np.arange(0.0, 1.05 - w1, 0.1):
                w3 = 1.0 - w1 - w2
                if w3 < -0.01:
                    continue
                ens_h = (w1 * val_spot_preds["CB"][h_mask] +
                         w2 * val_spot_preds["LGB"][h_mask] +
                         w3 * val_spot_preds["XGB"][h_mask])
                rmse_h = compute_rmse(actual[h_mask], ens_h)
                if rmse_h < best_rmse_h:
                    best_rmse_h = rmse_h
                    best_w_h = (w1, w2, w3)

        hourly_preds[h_mask] = (best_w_h[0] * val_spot_preds["CB"][h_mask] +
                                best_w_h[1] * val_spot_preds["LGB"][h_mask] +
                                best_w_h[2] * val_spot_preds["XGB"][h_mask])
        hourly_weights[h] = best_w_h

    rmse_hourly = compute_rmse(actual, hourly_preds)
    rmse_hourly_hbc, _, _ = apply_hbc(hourly_preds, actual, hrs_v)

    print(f"  D. Per-hour weights:    +HBC={rmse_hourly_hbc:.4f}  (24 x 3 = 72 params)")

    # Show interesting hours where weights differ
    print(f"     Hour  CB   LGB  XGB")
    for h in sorted(hourly_weights.keys()):
        w = hourly_weights[h]
        if w[1] > 0.15 or w[2] > 0.15:  # only show interesting ones
            print(f"      {h:2d}  {w[0]:.1f}  {w[1]:.1f}  {w[2]:.1f}")

    # ══════════════════════════════════════════════════════════════
    # E. Quantile-gated ensemble
    # ══════════════════════════════════════════════════════════════
    print(f"\n  Training quantile models...")

    q_preds_val = {}  # {model: {quantile: preds}}

    for name in ["CB", "LGB"]:
        q_preds_val[name] = {}
        for q in [0.25, 0.5, 0.75]:
            if name == "CB":
                q_params = {**cb_params, "loss_function": f"Quantile:alpha={q}"}
                q_model = CatBoostRegressor(**q_params)
                q_model.fit(pool_b, eval_set=pool_m, early_stopping_rounds=200, verbose=0)
                q_preds_val[name][q] = q_model.predict(X_va)
            else:
                q_lgb_params = {**lgb_params, "objective": "quantile", "alpha": q}
                q_model = lgb.LGBMRegressor(**q_lgb_params)
                q_model.fit(X_base, y_base, sample_weight=w_base,
                            eval_set=[(X_meta, y_meta)],
                            callbacks=[lgb.early_stopping(200, verbose=False),
                                       lgb.log_evaluation(0)])
                q_preds_val[name][q] = q_model.predict(X_va)

    # Strategy: weight each model inversely to its prediction interval width
    # Narrow interval = more confident = higher weight
    widths = {}
    for name in q_preds_val:
        widths[name] = q_preds_val[name][0.75] - q_preds_val[name][0.25]
        widths[name] = np.clip(widths[name], 0.1, None)  # avoid div by zero

    # Inverse-width weighting (normalize to sum=1)
    inv_w_cb = 1.0 / widths["CB"]
    inv_w_lgb = 1.0 / widths["LGB"]
    total_inv = inv_w_cb + inv_w_lgb
    dyn_w_cb = inv_w_cb / total_inv
    dyn_w_lgb = inv_w_lgb / total_inv

    # Use median predictions from each model, weighted by confidence
    q_ens_dev = dyn_w_cb * q_preds_val["CB"][0.5] + dyn_w_lgb * q_preds_val["LGB"][0.5]
    q_ens_spot = anch_va + q_ens_dev
    rmse_q = compute_rmse(actual, q_ens_spot)
    rmse_q_hbc, _, _ = apply_hbc(q_ens_spot, actual, hrs_v)

    print(f"  E. Quantile-gated (CB+LGB): +HBC={rmse_q_hbc:.4f}  "
          f"mean_w_CB={dyn_w_cb.mean():.3f}  mean_w_LGB={dyn_w_lgb.mean():.3f}")

    # Also try: use quantile spread as feature in stacking
    S_q_meta = np.column_stack([S_meta])
    S_q_val_features = [S_val]

    # Add quantile widths as features for val (we need meta versions too)
    # Since we only have val quantile preds, we'll just test the gated approach

    # ── F. Simple median of all model predictions ────────────────
    median_ens_dev = np.median(S_val, axis=1)
    median_ens_spot = anch_va + median_ens_dev
    rmse_med = compute_rmse(actual, median_ens_spot)
    rmse_med_hbc, _, _ = apply_hbc(median_ens_spot, actual, hrs_v)

    print(f"  F. Simple median:       +HBC={rmse_med_hbc:.4f}")

    # ══════════════════════════════════════════════════════════════
    # RANKING
    # ══════════════════════════════════════════════════════════════
    all_results = [
        ("A. Fixed weights", best_fixed["rmse_hbc"]),
        ("B. Ridge stacking", rmse_stack_hbc),
        ("C. Ridge + context", rmse_ctx_hbc),
        ("D. Per-hour weights", rmse_hourly_hbc),
        ("E. Quantile-gated", rmse_q_hbc),
        ("F. Simple median", rmse_med_hbc),
    ]

    # Add standalone CB for reference
    cb_hbc = apply_hbc(val_spot_preds["CB"], actual, hrs_v)[0]
    all_results.append(("   CB standalone", cb_hbc))

    print(f"\n  RANKING ({market.upper()}):")
    print(f"  {'#':>3s}  {'Method':30s}  {'+HBC':>7s}  {'vs CB':>8s}")
    print("  " + "-" * 55)
    for i, (label, rmse) in enumerate(sorted(all_results, key=lambda x: x[1]), 1):
        delta = rmse - cb_hbc
        best_mark = " <<<" if i == 1 else ""
        print(f"  {i:3d}  {label:30s}  {rmse:7.4f}  {delta:+8.4f}{best_mark}")

    return {r[0].strip(): r[1] for r in all_results}, hourly_weights


# ══════════════════════════════════════════════════════════════════════
# FR
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  FR — STACKING EXPERIMENTS")
print("=" * 90)

fr_results, fr_hourly_w = run_market(
    "fr", FR_FEATURES, fr_y_tr, fr_y_va, fr_valid_tr, fr_valid_va,
    fr_anchor_tr, fr_anchor_va, fr_spot_tr, fr_spot_va, fr_weights,
    hours_tr, hours_va)


# ══════════════════════════════════════════════════════════════════════
# UK
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  UK — STACKING EXPERIMENTS")
print("=" * 90)

uk_results, uk_hourly_w = run_market(
    "uk", UK_FEATURES, uk_y_tr, uk_y_va, uk_valid_tr, uk_valid_va,
    uk_moc_tr, uk_moc_va, uk_spot_tr, uk_spot_va, None,
    hours_tr, hours_va)


# ══════════════════════════════════════════════════════════════════════
# COMBINED SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  COMBINED SUMMARY")
print("=" * 90)

methods = ["A. Fixed weights", "B. Ridge stacking", "C. Ridge + context",
           "D. Per-hour weights", "E. Quantile-gated", "F. Simple median",
           "CB standalone"]

print(f"\n  {'Method':30s}  {'FR':>7s}  {'UK':>7s}  {'SUM':>7s}")
print("  " + "-" * 55)
for m in methods:
    if m in fr_results and m in uk_results:
        s = fr_results[m] + uk_results[m]
        print(f"  {m:30s}  {fr_results[m]:7.4f}  {uk_results[m]:7.4f}  {s:7.4f}")

output = {"fr": fr_results, "uk": uk_results}
with open("outputs/stacking_ab_test.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Total time: {time.time() - t0:.0f}s")
