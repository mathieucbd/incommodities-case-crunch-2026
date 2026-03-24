"""A/B test — Loss functions for CatBoost (FR + UK).

Tests alternative loss functions while keeping eval_metric=RMSE
(competition metric). The idea: train with a more robust loss,
but measure performance on what Kaggle scores.

Variants:
  - RMSE (baseline)
  - Huber:delta=D  (quadratic near 0, linear beyond delta)
  - MAE (L1)
  - Lq:q=Q  (generalized: q=2 is MSE, q=1 is MAE)
  - Quantile:alpha=0.5  (another L1 variant)

Both FR and UK.
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

print("=" * 90)
print("  LOSS FUNCTION A/B TEST — FR + UK")
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


# ── Features & base params ────────────────────────────────────────────
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

FR_BASE_PARAMS = {
    "eval_metric": "RMSE",
    "iterations": 15000, "learning_rate": 0.059, "depth": 3,
    "l2_leaf_reg": 4.42, "subsample": 0.533, "colsample_bylevel": 0.228,
    "min_child_samples": 14, "random_strength": 0.9,
    "random_seed": 42, "verbose": 0, "use_best_model": True,
    "allow_writing_files": False,
}

with open("outputs/uk_feature_research.json") as f:
    uk_research = json.load(f)
UK_FEATURES = [f for f in uk_research["confirmed_features"] if f in df_tr.columns]

UK_BASE_PARAMS = {
    "eval_metric": "RMSE",
    "iterations": 15000, "learning_rate": 0.03, "depth": 8,
    "l2_leaf_reg": 5, "colsample_bylevel": 0.8, "subsample": 0.8,
    "random_seed": 42, "verbose": 0, "use_best_model": True,
    "allow_writing_files": False,
}


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

# Target stats for choosing Huber delta
fr_target_valid = fr_y_tr[fr_valid_tr]
uk_target_valid = uk_y_tr[uk_valid_tr]
fr_mad = np.median(np.abs(fr_target_valid - np.median(fr_target_valid)))
uk_mad = np.median(np.abs(uk_target_valid - np.median(uk_target_valid)))
print(f"  FR target: median={np.median(fr_target_valid):.1f}, MAD={fr_mad:.1f}, "
      f"std={np.std(fr_target_valid):.1f}, IQR=[{np.percentile(fr_target_valid,25):.1f}, {np.percentile(fr_target_valid,75):.1f}]")
print(f"  UK target: median={np.median(uk_target_valid):.1f}, MAD={uk_mad:.1f}, "
      f"std={np.std(uk_target_valid):.1f}, IQR=[{np.percentile(uk_target_valid,25):.1f}, {np.percentile(uk_target_valid,75):.1f}]")


# ── Loss variants ────────────────────────────────────────────────────
# FR: MAD~29, std~42 → Huber delta should span from ~15 (aggressive) to ~60 (mild)
# UK: MAD~14, std~30 → Huber delta from ~10 to ~40

FR_LOSSES = [
    ("RMSE", "RMSE (baseline)"),
    ("MAE", "MAE (L1)"),
    ("Huber:delta=10", "Huber delta=10 (aggressive)"),
    ("Huber:delta=20", "Huber delta=20"),
    ("Huber:delta=30", "Huber delta=30 (~MAD)"),
    ("Huber:delta=50", "Huber delta=50 (~std)"),
    ("Huber:delta=80", "Huber delta=80 (mild)"),
    ("Lq:q=1.5", "Lq q=1.5 (between MAE/MSE)"),
    ("Lq:q=1.8", "Lq q=1.8 (close to MSE)"),
    ("Quantile:alpha=0.5", "Quantile 0.5 (=MAE)"),
]

UK_LOSSES = [
    ("RMSE", "RMSE (baseline)"),
    ("MAE", "MAE (L1)"),
    ("Huber:delta=5", "Huber delta=5 (aggressive)"),
    ("Huber:delta=10", "Huber delta=10"),
    ("Huber:delta=15", "Huber delta=15 (~MAD)"),
    ("Huber:delta=25", "Huber delta=25"),
    ("Huber:delta=40", "Huber delta=40 (~std)"),
    ("Lq:q=1.5", "Lq q=1.5 (between MAE/MSE)"),
    ("Lq:q=1.8", "Lq q=1.8 (close to MSE)"),
]


# ── Evaluate helper ──────────────────────────────────────────────────
def evaluate(features, base_params, loss_fn, y_train, y_valid, v_tr, v_va,
             anchor_va, spot_va, weights, label):
    """Train with given loss, evaluate on RMSE."""
    feat = [f for f in features if f in df_tr.columns]
    w = weights[v_tr] if weights is not None else None

    params = {**base_params, "loss_function": loss_fn}
    pool_tr = Pool(df_tr.loc[df_tr.index[v_tr], feat], y_train[v_tr], weight=w)
    pool_va = Pool(df_va.loc[df_va.index[v_va], feat], y_valid[v_va])

    model = CatBoostRegressor(**params)
    model.fit(pool_tr, eval_set=pool_va, early_stopping_rounds=200, verbose=0)

    preds_dev = model.predict(df_va.loc[df_va.index[v_va], feat])
    preds_spot = anchor_va[v_va] + preds_dev
    actual = spot_va[v_va]
    hrs = hours_va[v_va]

    rmse = np.sqrt(np.mean((actual - preds_spot) ** 2))
    bias = float(np.mean(actual - preds_spot))

    # HBC
    errors = actual - preds_spot
    hbc = {h: float(errors[hrs == h].mean()) for h in range(24) if (hrs == h).sum() > 0}
    corrected = preds_spot + np.array([hbc.get(h, 0) for h in hrs])
    rmse_hbc = np.sqrt(np.mean((actual - corrected) ** 2))

    iters = model.get_best_iteration()

    print(f"  {label:40s}  RMSE={rmse:7.4f}  +HBC={rmse_hbc:7.4f}  "
          f"bias={bias:+6.2f}  iters={iters:5d}")
    sys.stdout.flush()

    return {"label": label, "loss": loss_fn, "rmse": round(rmse, 4),
            "rmse_hbc": round(rmse_hbc, 4), "bias": round(bias, 2), "iters": iters}


# ══════════════════════════════════════════════════════════════════════
# FR TESTS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  FR — LOSS FUNCTION COMPARISON")
print("=" * 90)
print(f"\n  {'Method':40s}  {'RMSE':>7s}  {'+HBC':>7s}  {'Bias':>6s}  {'Iter':>6s}")
print("  " + "-" * 75)

fr_results = []
for loss_fn, label in FR_LOSSES:
    fr_results.append(evaluate(
        FR_FEATURES, FR_BASE_PARAMS, loss_fn,
        fr_y_tr, fr_y_va, fr_valid_tr, fr_valid_va,
        fr_anchor_va, fr_spot_va, fr_weights, label))

# FR ranking
print(f"\n  FR RANKING (by RMSE+HBC):")
baseline_fr = fr_results[0]["rmse_hbc"]
for i, r in enumerate(sorted(fr_results, key=lambda x: x["rmse_hbc"]), 1):
    delta = f"{r['rmse_hbc'] - baseline_fr:+.4f}"
    best = " <<<" if i == 1 else ""
    print(f"  {i:3d}  {r['label']:40s}  {r['rmse_hbc']:7.4f}  {delta:>8s}{best}")


# ══════════════════════════════════════════════════════════════════════
# UK TESTS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  UK — LOSS FUNCTION COMPARISON")
print("=" * 90)
print(f"\n  {'Method':40s}  {'RMSE':>7s}  {'+HBC':>7s}  {'Bias':>6s}  {'Iter':>6s}")
print("  " + "-" * 75)

uk_results = []
for loss_fn, label in UK_LOSSES:
    uk_results.append(evaluate(
        UK_FEATURES, UK_BASE_PARAMS, loss_fn,
        uk_y_tr, uk_y_va, uk_valid_tr, uk_valid_va,
        uk_moc_va, uk_spot_va, None, label))

# UK ranking
print(f"\n  UK RANKING (by RMSE+HBC):")
baseline_uk = uk_results[0]["rmse_hbc"]
for i, r in enumerate(sorted(uk_results, key=lambda x: x["rmse_hbc"]), 1):
    delta = f"{r['rmse_hbc'] - baseline_uk:+.4f}"
    best = " <<<" if i == 1 else ""
    print(f"  {i:3d}  {r['label']:40s}  {r['rmse_hbc']:7.4f}  {delta:>8s}{best}")


# ══════════════════════════════════════════════════════════════════════
# COMBINED
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  COMBINED — Best per market")
print("=" * 90)
best_fr = min(fr_results, key=lambda x: x["rmse_hbc"])
best_uk = min(uk_results, key=lambda x: x["rmse_hbc"])
current = fr_results[0]["rmse_hbc"] + uk_results[0]["rmse_hbc"]
best_combo = best_fr["rmse_hbc"] + best_uk["rmse_hbc"]
print(f"  Current:    FR={fr_results[0]['rmse_hbc']:.4f} + UK={uk_results[0]['rmse_hbc']:.4f} = {current:.4f}")
print(f"  Best FR:    {best_fr['label']} ({best_fr['loss']}) → {best_fr['rmse_hbc']:.4f}")
print(f"  Best UK:    {best_uk['label']} ({best_uk['loss']}) → {best_uk['rmse_hbc']:.4f}")
print(f"  Best combo: {best_combo:.4f} (delta={best_combo - current:+.4f})")

with open("outputs/loss_function_ab_test.json", "w") as f:
    json.dump({"fr": fr_results, "uk": uk_results}, f, indent=2)

print(f"\n  Total time: {time.time() - t0:.0f}s")
