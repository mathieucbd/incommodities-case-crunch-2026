"""A/B test — MAD-based target winsorization vs current approach.

Tests winsorizing the TRAINING TARGET (deviation) at ±z*MAD to reduce
the model's focus on extreme spikes.

MAD = median(|x - median(x)|)
Bounds = median ± z * 1.4826 * MAD  (1.4826 = consistency constant for normal)

Variants:
  - Baseline: no winsorization (current)
  - MAD z=2.5, 3, 4, 5 on training target
  - Percentile 1%/99%, 2%/98% on training target
  - MAD z=3 on predictions (post-processing clipping)

Tests on both FR and UK.
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
print("  MAD CLIPPING A/B TEST — Target Winsorization")
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


# ── Features & params ────────────────────────────────────────────────────
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

FR_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 15000, "learning_rate": 0.059, "depth": 3,
    "l2_leaf_reg": 4.42, "subsample": 0.533, "colsample_bylevel": 0.228,
    "min_child_samples": 14, "random_strength": 0.9,
    "random_seed": 42, "verbose": 0, "use_best_model": True,
    "allow_writing_files": False,
}

with open("outputs/uk_feature_research.json") as f:
    uk_research = json.load(f)
UK_FEATURES = [f for f in uk_research["confirmed_features"] if f in df_tr.columns]

UK_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 15000, "learning_rate": 0.03, "depth": 8,
    "l2_leaf_reg": 5, "colsample_bylevel": 0.8, "subsample": 0.8,
    "random_seed": 42, "verbose": 0, "use_best_model": True,
    "allow_writing_files": False,
}


# ── Targets ──────────────────────────────────────────────────────────────
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


# ── MAD helper ───────────────────────────────────────────────────────────
def mad_bounds(y, z):
    """Compute MAD-based bounds: median ± z * 1.4826 * MAD."""
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med))
    scale = 1.4826 * mad  # consistent estimator of std for normal
    return med - z * scale, med + z * scale


def winsorize(y, low, high):
    """Clip y to [low, high]."""
    return np.clip(y, low, high)


# ── Evaluate helper ──────────────────────────────────────────────────────
def evaluate(market, features, params, y_train, y_valid, v_tr, v_va,
             anchor_va, spot_va, weights, label, clip_preds=None):
    """Train and evaluate."""
    feat = [f for f in features if f in df_tr.columns]
    w = weights[v_tr] if weights is not None else None

    pool_tr = Pool(df_tr.loc[df_tr.index[v_tr], feat], y_train[v_tr], weight=w)
    pool_va = Pool(df_va.loc[df_va.index[v_va], feat], y_valid[v_va])

    model = CatBoostRegressor(**params)
    model.fit(pool_tr, eval_set=pool_va, early_stopping_rounds=200, verbose=0)

    preds_dev = model.predict(df_va.loc[df_va.index[v_va], feat])

    if clip_preds is not None:
        preds_dev = np.clip(preds_dev, clip_preds[0], clip_preds[1])

    preds_spot = anchor_va[v_va] + preds_dev
    actual = spot_va[v_va]
    hrs = hours_va[v_va]

    rmse = np.sqrt(np.mean((actual - preds_spot) ** 2))
    bias = float(np.mean(actual - preds_spot))

    errors = actual - preds_spot
    hbc = {h: float(errors[hrs == h].mean()) for h in range(24) if (hrs == h).sum() > 0}
    corrected = preds_spot + np.array([hbc.get(h, 0) for h in hrs])
    rmse_hbc = np.sqrt(np.mean((actual - corrected) ** 2))

    iters = model.get_best_iteration()

    print(f"  {label:45s}  RMSE={rmse:7.4f}  +HBC={rmse_hbc:7.4f}  "
          f"bias={bias:+6.2f}  iters={iters:4d}")
    sys.stdout.flush()

    return {"label": label, "rmse": round(rmse, 4), "rmse_hbc": round(rmse_hbc, 4),
            "bias": round(bias, 2), "iters": iters}


# ══════════════════════════════════════════════════════════════════════════
# FR TESTS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  FR — MAD TARGET WINSORIZATION")
print("=" * 90)

# Stats on the deviation target
y_valid = fr_y_tr[fr_valid_tr]
med = np.median(y_valid)
mad_val = np.median(np.abs(y_valid - med))
print(f"  Target stats: median={med:.2f}, MAD={mad_val:.2f}, scaled_MAD={1.4826*mad_val:.2f}")
print(f"  Range: [{y_valid.min():.1f}, {y_valid.max():.1f}]")
for z in [2.5, 3, 4, 5]:
    lo, hi = mad_bounds(y_valid, z)
    n_clip = ((y_valid < lo) | (y_valid > hi)).sum()
    print(f"  z={z}: bounds=[{lo:.1f}, {hi:.1f}], clipped={n_clip} ({100*n_clip/len(y_valid):.2f}%)")

print(f"\n  {'Method':45s}  {'RMSE':>7s}  {'+HBC':>7s}  {'Bias':>6s}  {'Iter':>5s}")
print("  " + "-" * 75)

fr_results = []

# Baseline
fr_results.append(evaluate(
    "fr", FR_FEATURES, FR_PARAMS, fr_y_tr, fr_y_va,
    fr_valid_tr, fr_valid_va, fr_anchor_va, fr_spot_va, fr_weights,
    "Baseline (no winsorization)"))

# MAD winsorization on training target
for z in [2.5, 3, 4, 5]:
    lo, hi = mad_bounds(fr_y_tr[fr_valid_tr], z)
    y_clipped = winsorize(fr_y_tr.copy(), lo, hi)
    fr_results.append(evaluate(
        "fr", FR_FEATURES, FR_PARAMS, y_clipped, fr_y_va,
        fr_valid_tr, fr_valid_va, fr_anchor_va, fr_spot_va, fr_weights,
        f"MAD z={z} on train target"))

# Percentile winsorization
for lo_p, hi_p in [(1, 99), (2, 98), (5, 95)]:
    lo = np.percentile(fr_y_tr[fr_valid_tr], lo_p)
    hi = np.percentile(fr_y_tr[fr_valid_tr], hi_p)
    y_clipped = winsorize(fr_y_tr.copy(), lo, hi)
    n_clip = ((fr_y_tr[fr_valid_tr] < lo) | (fr_y_tr[fr_valid_tr] > hi)).sum()
    fr_results.append(evaluate(
        "fr", FR_FEATURES, FR_PARAMS, y_clipped, fr_y_va,
        fr_valid_tr, fr_valid_va, fr_anchor_va, fr_spot_va, fr_weights,
        f"Percentile {lo_p}%/{hi_p}% on train target"))

# MAD z=3 on predictions (post-processing)
lo, hi = mad_bounds(fr_y_tr[fr_valid_tr], 3)
fr_results.append(evaluate(
    "fr", FR_FEATURES, FR_PARAMS, fr_y_tr, fr_y_va,
    fr_valid_tr, fr_valid_va, fr_anchor_va, fr_spot_va, fr_weights,
    "MAD z=3 on predictions (post-proc)", clip_preds=(lo, hi)))

# FR ranking
print(f"\n  FR RANKING:")
baseline_fr = fr_results[0]["rmse_hbc"]
for i, r in enumerate(sorted(fr_results, key=lambda x: x["rmse_hbc"]), 1):
    delta = f"{r['rmse_hbc'] - baseline_fr:+.4f}"
    best = " <<<" if i == 1 else ""
    print(f"  {i:3d}  {r['label']:45s}  {r['rmse_hbc']:7.4f}  {delta:>8s}{best}")


# ══════════════════════════════════════════════════════════════════════════
# UK TESTS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  UK — MAD TARGET WINSORIZATION")
print("=" * 90)

y_valid_uk = uk_y_tr[uk_valid_tr]
med_uk = np.median(y_valid_uk)
mad_uk = np.median(np.abs(y_valid_uk - med_uk))
print(f"  Target stats: median={med_uk:.2f}, MAD={mad_uk:.2f}, scaled_MAD={1.4826*mad_uk:.2f}")
print(f"  Range: [{y_valid_uk.min():.1f}, {y_valid_uk.max():.1f}]")
for z in [2.5, 3, 4, 5]:
    lo, hi = mad_bounds(y_valid_uk, z)
    n_clip = ((y_valid_uk < lo) | (y_valid_uk > hi)).sum()
    print(f"  z={z}: bounds=[{lo:.1f}, {hi:.1f}], clipped={n_clip} ({100*n_clip/len(y_valid_uk):.2f}%)")

print(f"\n  {'Method':45s}  {'RMSE':>7s}  {'+HBC':>7s}  {'Bias':>6s}  {'Iter':>5s}")
print("  " + "-" * 75)

uk_results = []

# Baseline
uk_results.append(evaluate(
    "uk", UK_FEATURES, UK_PARAMS, uk_y_tr, uk_y_va,
    uk_valid_tr, uk_valid_va, uk_moc_va, uk_spot_va, None,
    "Baseline (no winsorization)"))

# MAD on UK training target
for z in [2.5, 3, 4, 5]:
    lo, hi = mad_bounds(uk_y_tr[uk_valid_tr], z)
    y_clipped = winsorize(uk_y_tr.copy(), lo, hi)
    uk_results.append(evaluate(
        "uk", UK_FEATURES, UK_PARAMS, y_clipped, uk_y_va,
        uk_valid_tr, uk_valid_va, uk_moc_va, uk_spot_va, None,
        f"MAD z={z} on train target"))

# Percentile on UK
for lo_p, hi_p in [(1, 99), (2, 98)]:
    lo = np.percentile(uk_y_tr[uk_valid_tr], lo_p)
    hi = np.percentile(uk_y_tr[uk_valid_tr], hi_p)
    y_clipped = winsorize(uk_y_tr.copy(), lo, hi)
    uk_results.append(evaluate(
        "uk", UK_FEATURES, UK_PARAMS, y_clipped, uk_y_va,
        uk_valid_tr, uk_valid_va, uk_moc_va, uk_spot_va, None,
        f"Percentile {lo_p}%/{hi_p}% on train target"))

# UK ranking
print(f"\n  UK RANKING:")
baseline_uk = uk_results[0]["rmse_hbc"]
for i, r in enumerate(sorted(uk_results, key=lambda x: x["rmse_hbc"]), 1):
    delta = f"{r['rmse_hbc'] - baseline_uk:+.4f}"
    best = " <<<" if i == 1 else ""
    print(f"  {i:3d}  {r['label']:45s}  {r['rmse_hbc']:7.4f}  {delta:>8s}{best}")


# ══════════════════════════════════════════════════════════════════════════
# COMBINED
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  COMBINED — Best per market")
print("=" * 90)
best_fr = min(fr_results, key=lambda x: x["rmse_hbc"])
best_uk = min(uk_results, key=lambda x: x["rmse_hbc"])
current = fr_results[0]["rmse_hbc"] + uk_results[0]["rmse_hbc"]
best_combo = best_fr["rmse_hbc"] + best_uk["rmse_hbc"]
print(f"  Current:    FR={fr_results[0]['rmse_hbc']:.4f} + UK={uk_results[0]['rmse_hbc']:.4f} = {current:.4f}")
print(f"  Best FR:    {best_fr['label']} → {best_fr['rmse_hbc']:.4f}")
print(f"  Best UK:    {best_uk['label']} → {best_uk['rmse_hbc']:.4f}")
print(f"  Best combo: {best_combo:.4f} (delta={best_combo - current:+.4f})")

with open("outputs/mad_clipping_ab_test.json", "w") as f:
    json.dump({"fr": fr_results, "uk": uk_results}, f, indent=2)

print(f"\n  Total time: {time.time() - t0:.0f}s")
