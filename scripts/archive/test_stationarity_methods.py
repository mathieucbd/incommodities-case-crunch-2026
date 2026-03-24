"""Rigorous A/B test: stationarity methods for FR spot price.

Tests every relevant method to make the target stationary before CatBoost:

  1. Raw spot (baseline)
  2. arcsinh(spot) (current best)
  3. spot - spot_la (24h seasonal diff)
  4. spot - roll_168h_mean (deviation from 7d mean)
  5. Rolling z-score (mean+std, 168h window)
  6. Robust rolling z-score (median+MAD, 168h window)
  7. Rolling z-score 24h window
  8. Robust rolling z-score 24h window
  9. STL decomposition — predict residual
  10. Fractional differencing (min d for stationarity)
  11. Spike clipping (1%/99%) + arcsinh
  12. Mirror-log: sign(x)*log(1+|x|)
  13. spot - spot_lag_168h (weekly seasonal diff)
  14. Parametric arcsinh with rolling calibration
  15. Combined: rolling z-score + arcsinh

All with Cat32 + weights(2.0).

Usage: python scripts/test_stationarity_methods.py
"""

import sys, json, warnings
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

# ── Common setup ──────────────────────────────────────────────────────────
CB_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 5000, "learning_rate": 0.03, "depth": 8,
    "l2_leaf_reg": 5, "subsample": 0.8, "random_seed": 42,
    "verbose": 0, "allow_writing_files": False, "use_best_model": True,
}

CAT32_FR = [
    "fr_opportunity_cost", "fr_dynamic_marginal", "fr_import_price",
    "fr_scarcity_barrier", "fr_load_price_signal_7d",
    "fr_load_price_signal_load", "fr_hydro_opp_cost",
    "fr_basis_v2", "fr_basis_v2_lag_48h", "fr_basis_v2_roll_24h_mean",
    "fr_price_per_mw_7d",
]

base_feat = [f for f in clean_ranking["fr_spot"][:20] if f in df_train.columns]
extras = [f for f in CAT32_FR if f in df_train.columns and f not in base_feat]
features = base_feat + extras

spot_tr = df_train["fr_spot"].values.copy()
spot_va = df_val["fr_spot"].values.copy()


def compute_weights(n, decay=2.0):
    dt = pd.to_datetime(df_train["datetime_CET"])
    days_ago = (dt.max() - dt).dt.total_seconds() / 86400
    return np.exp(-decay * days_ago / 365).values


weights = compute_weights(len(df_train), 2.0)


def run_experiment(label, y_tr, y_va_transformed, reconstruct_fn, feat_override=None):
    """Train CatBoost on transformed target, reconstruct spot, compute RMSE."""
    feat = feat_override if feat_override is not None else features

    # Remove NaN rows
    valid_tr = np.isfinite(y_tr)
    valid_va = np.isfinite(y_va_transformed)

    if valid_tr.sum() < len(y_tr) * 0.9:
        print(f"  {label:55s}  SKIPPED (too many NaN: {(~valid_tr).sum()})")
        return None

    X_tr = df_train.loc[valid_tr, feat] if valid_tr.sum() < len(y_tr) else df_train[feat]
    X_va = df_val.loc[valid_va, feat] if valid_va.sum() < len(y_va_transformed) else df_val[feat]
    w = weights[valid_tr] if valid_tr.sum() < len(y_tr) else weights

    model = CatBoostRegressor(**CB_PARAMS)
    model.fit(Pool(X_tr, y_tr[valid_tr], weight=w),
              eval_set=Pool(X_va, y_va_transformed[valid_va]),
              early_stopping_rounds=100, verbose=0)

    preds_t = model.predict(df_val[feat])
    preds_spot = reconstruct_fn(preds_t)

    rmse = np.sqrt(np.mean((spot_va - preds_spot) ** 2))
    bias = np.mean(spot_va - preds_spot)
    mae = np.mean(np.abs(spot_va - preds_spot))

    print(f"  {label:55s}  RMSE={rmse:7.3f}  Bias={bias:+6.1f}  MAE={mae:5.2f}")
    return rmse


# ══════════════════════════════════════════════════════════════════════════
# Compute all transforms on train + val
# ══════════════════════════════════════════════════════════════════════════

# Precompute rolling stats on the FULL dataset (before split) to avoid NaN at boundaries
full_spot = train_fe["fr_spot"].values
full_spot_la = train_fe["fr_spot_la"].values

# Rolling stats (use spot_la to avoid leakage — it's D-1 price)
full_la = train_fe["fr_spot_la"]

roll_168_mean_full = full_la.rolling(168, min_periods=24).mean().values
roll_168_std_full = full_la.rolling(168, min_periods=24).std().values
roll_168_median_full = full_la.rolling(168, min_periods=24).median().values
roll_168_mad_full = full_la.rolling(168, min_periods=24).apply(
    lambda x: np.median(np.abs(x - np.median(x))), raw=True).values

roll_24_mean_full = full_la.rolling(24, min_periods=6).mean().values
roll_24_std_full = full_la.rolling(24, min_periods=6).std().values
roll_24_median_full = full_la.rolling(24, min_periods=6).median().values
roll_24_mad_full = full_la.rolling(24, min_periods=6).apply(
    lambda x: np.median(np.abs(x - np.median(x))), raw=True).values

# Split back
n_tr = len(df_train)
n_va = len(df_val)

roll_168_mean_tr = roll_168_mean_full[:n_tr]
roll_168_std_tr = roll_168_std_full[:n_tr]
roll_168_median_tr = roll_168_median_full[:n_tr]
roll_168_mad_tr = roll_168_mad_full[:n_tr]
roll_24_mean_tr = roll_24_mean_full[:n_tr]
roll_24_std_tr = roll_24_std_full[:n_tr]
roll_24_median_tr = roll_24_median_full[:n_tr]
roll_24_mad_tr = roll_24_mad_full[:n_tr]

roll_168_mean_va = roll_168_mean_full[n_tr:n_tr+n_va]
roll_168_std_va = roll_168_std_full[n_tr:n_tr+n_va]
roll_168_median_va = roll_168_median_full[n_tr:n_tr+n_va]
roll_168_mad_va = roll_168_mad_full[n_tr:n_tr+n_va]
roll_24_mean_va = roll_24_mean_full[n_tr:n_tr+n_va]
roll_24_std_va = roll_24_std_full[n_tr:n_tr+n_va]
roll_24_median_va = roll_24_median_full[n_tr:n_tr+n_va]
roll_24_mad_va = roll_24_mad_full[n_tr:n_tr+n_va]

# spot_la values
spot_la_tr = df_train["fr_spot_la"].values
spot_la_va = df_val["fr_spot_la"].values

# Lag 168h (same hour, 7 days ago)
spot_lag168_tr = df_train["fr_spot_lag_168h"].values if "fr_spot_lag_168h" in df_train.columns else full_la.shift(168).values[:n_tr]
spot_lag168_va = df_val["fr_spot_lag_168h"].values if "fr_spot_lag_168h" in df_val.columns else full_la.shift(168).values[n_tr:n_tr+n_va]

# ── Stationarity check ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STATIONARITY CHECK — train quarterly means")
print("=" * 70)

dt_tr = pd.to_datetime(df_train["datetime_CET"])
quarters = dt_tr.dt.to_period("Q")

checks = {
    "raw spot": spot_tr,
    "spot - spot_la": spot_tr - spot_la_tr,
    "spot - roll_168h_mean": spot_tr - roll_168_mean_tr,
    "z-score 168h": (spot_tr - roll_168_mean_tr) / np.clip(roll_168_std_tr, 1, None),
    "robust z 168h": (spot_tr - roll_168_median_tr) / np.clip(roll_168_mad_tr * 1.4826, 1, None),
    "spot - lag_168h": spot_tr - spot_lag168_tr,
}

for name, series in checks.items():
    valid = np.isfinite(series)
    s = pd.Series(series[valid], index=quarters[valid])
    means = s.groupby(level=0).mean()
    stds = s.groupby(level=0).std()
    cv = (means.std() / means.abs().mean() * 100) if means.abs().mean() > 0 else 999
    print(f"\n  {name} (CV of quarterly means: {cv:.0f}%):")
    for q in means.index:
        print(f"    {q}: mean={means[q]:+8.2f}  std={stds[q]:7.2f}")


# ══════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  EXPERIMENTS")
print("=" * 70)

results = []

# ── 1. Raw spot ───────────────────────────────────────────────────────────
r = run_experiment("1) Raw spot",
                   spot_tr, spot_va,
                   lambda p: p)
if r: results.append(("1) Raw spot", r))

# ── 2. arcsinh(spot) — current best ──────────────────────────────────────
r = run_experiment("2) arcsinh(spot)",
                   np.arcsinh(spot_tr), np.arcsinh(spot_va),
                   lambda p: np.sinh(p))
if r: results.append(("2) arcsinh(spot)", r))

# ── 3. spot - spot_la (24h seasonal diff) ────────────────────────────────
y_diff24_tr = spot_tr - spot_la_tr
y_diff24_va = spot_va - spot_la_va
r = run_experiment("3) spot - spot_la (24h diff)",
                   y_diff24_tr, y_diff24_va,
                   lambda p: spot_la_va + p)
if r: results.append(("3) spot - spot_la", r))

# ── 4. spot - roll_168h_mean ─────────────────────────────────────────────
y_dev168_tr = spot_tr - roll_168_mean_tr
y_dev168_va = spot_va - roll_168_mean_va
r = run_experiment("4) spot - roll_168h_mean",
                   y_dev168_tr, y_dev168_va,
                   lambda p: roll_168_mean_va + p)
if r: results.append(("4) spot - roll_168h_mean", r))

# ── 5. Rolling z-score (mean+std, 168h) ──────────────────────────────────
std168_tr_clip = np.clip(roll_168_std_tr, 1.0, None)
std168_va_clip = np.clip(roll_168_std_va, 1.0, None)
y_z168_tr = (spot_tr - roll_168_mean_tr) / std168_tr_clip
y_z168_va = (spot_va - roll_168_mean_va) / std168_va_clip
r = run_experiment("5) Z-score 168h (mean+std)",
                   y_z168_tr, y_z168_va,
                   lambda p: roll_168_mean_va + p * std168_va_clip)
if r: results.append(("5) Z-score 168h", r))

# ── 6. Robust z-score (median+MAD, 168h) ─────────────────────────────────
# MAD × 1.4826 ≈ std for normal distribution
mad168_tr_clip = np.clip(roll_168_mad_tr * 1.4826, 1.0, None)
mad168_va_clip = np.clip(roll_168_mad_va * 1.4826, 1.0, None)
y_rz168_tr = (spot_tr - roll_168_median_tr) / mad168_tr_clip
y_rz168_va = (spot_va - roll_168_median_va) / mad168_va_clip
r = run_experiment("6) Robust z-score 168h (median+MAD)",
                   y_rz168_tr, y_rz168_va,
                   lambda p: roll_168_median_va + p * mad168_va_clip)
if r: results.append(("6) Robust z 168h", r))

# ── 7. Rolling z-score 24h ───────────────────────────────────────────────
std24_tr_clip = np.clip(roll_24_std_tr, 1.0, None)
std24_va_clip = np.clip(roll_24_std_va, 1.0, None)
y_z24_tr = (spot_tr - roll_24_mean_tr) / std24_tr_clip
y_z24_va = (spot_va - roll_24_mean_va) / std24_va_clip
r = run_experiment("7) Z-score 24h (mean+std)",
                   y_z24_tr, y_z24_va,
                   lambda p: roll_24_mean_va + p * std24_va_clip)
if r: results.append(("7) Z-score 24h", r))

# ── 8. Robust z-score 24h ────────────────────────────────────────────────
mad24_tr_clip = np.clip(roll_24_mad_tr * 1.4826, 1.0, None)
mad24_va_clip = np.clip(roll_24_mad_va * 1.4826, 1.0, None)
y_rz24_tr = (spot_tr - roll_24_median_tr) / mad24_tr_clip
y_rz24_va = (spot_va - roll_24_median_va) / mad24_va_clip
r = run_experiment("8) Robust z-score 24h (median+MAD)",
                   y_rz24_tr, y_rz24_va,
                   lambda p: roll_24_median_va + p * mad24_va_clip)
if r: results.append(("8) Robust z 24h", r))

# ── 9. STL decomposition ─────────────────────────────────────────────────
print("\n  --- STL decomposition ---")
try:
    from statsmodels.tsa.seasonal import STL, MSTL

    # Use full series for decomposition
    full_series = pd.Series(train_fe["fr_spot"].values,
                            index=pd.to_datetime(train_fe["datetime_CET"]))
    full_series = full_series.asfreq("h")

    # Fill any gaps
    full_series = full_series.interpolate(method="linear")

    # MSTL with daily (24) and weekly (168) seasonality
    mstl = MSTL(full_series, periods=[24, 168], stl_kwargs={"robust": True})
    res_stl = mstl.fit()

    trend_full = res_stl.trend.values
    seasonal_full = res_stl.seasonal.sum(axis=1).values  # sum of all seasonal components
    residual_full = res_stl.resid.values

    trend_tr = trend_full[:n_tr]
    seasonal_tr = seasonal_full[:n_tr]
    residual_tr = residual_full[:n_tr]

    trend_va = trend_full[n_tr:n_tr+n_va]
    seasonal_va = seasonal_full[n_tr:n_tr+n_va]
    residual_va = residual_full[n_tr:n_tr+n_va]

    # Method A: predict residual, add trend+seasonal from STL
    r = run_experiment("9a) MSTL residual (trend+seasonal fixed)",
                       residual_tr, residual_va,
                       lambda p: trend_va + seasonal_va + p)
    if r: results.append(("9a) MSTL residual", r))

    # Method B: predict (residual + trend), add seasonal
    deseason_tr = spot_tr - seasonal_tr
    deseason_va = spot_va - seasonal_va
    r = run_experiment("9b) MSTL deseasoned (seasonal fixed)",
                       deseason_tr, deseason_va,
                       lambda p: seasonal_va + p)
    if r: results.append(("9b) MSTL deseasoned", r))

except Exception as e:
    print(f"  STL failed: {e}")

# ── 10. Fractional differencing ───────────────────────────────────────────
print("\n  --- Fractional differencing ---")
try:
    from fracdiff import fdiff

    # Find min d for stationarity (ADF test)
    from statsmodels.tsa.stattools import adfuller

    spot_series = train_fe["fr_spot"].values
    best_d = None
    for d_test in np.arange(0.1, 1.05, 0.05):
        fd = fdiff(spot_series.reshape(-1, 1), d_test, mode="valid")
        if len(fd) > 100:
            adf_stat = adfuller(fd.flatten(), maxlag=24, autolag=None)[1]
            if adf_stat < 0.01:  # p-value < 1%
                best_d = d_test
                print(f"  Min d for ADF p<0.01: d={d_test:.2f}")
                break

    if best_d is not None:
        fd_full = fdiff(spot_series.reshape(-1, 1), best_d, mode="same").flatten()
        fd_tr = fd_full[:n_tr]
        fd_va = fd_full[n_tr:n_tr+n_va]

        # Reconstruction: for tree models, we approximate by using the
        # naive anchor + model correction approach
        # The fracdiff inverse is complex, so we use a practical approximation:
        # pred_spot ≈ spot_la + sinh(arcsinh(fd_pred)) correction
        # Actually, we just evaluate if the fracdiff target helps the model
        r = run_experiment("10) Fractional diff (d={:.2f})".format(best_d),
                           fd_tr, fd_va,
                           # Approximation: use spot_la as anchor + fracdiff prediction
                           # This isn't perfect but tests if the target is learnable
                           lambda p: spot_la_va + p)
        if r: results.append(("10) Fractional diff d={:.2f}".format(best_d), r))
    else:
        print("  Could not find suitable d")

except ImportError:
    print("  fracdiff not installed, skipping")
except Exception as e:
    print(f"  Fractional diff failed: {e}")

# ── 11. Spike clipping + arcsinh ──────────────────────────────────────────
q01 = np.percentile(spot_tr, 1)
q99 = np.percentile(spot_tr, 99)
spot_tr_clipped = np.clip(spot_tr, q01, q99)
spot_va_clipped = np.clip(spot_va, q01, q99)  # clip val too for target
y_clip_tr = np.arcsinh(spot_tr_clipped)
y_clip_va = np.arcsinh(spot_va)  # don't clip val target
r = run_experiment("11) Clip(1%/99%) + arcsinh",
                   y_clip_tr, np.arcsinh(spot_va),
                   lambda p: np.sinh(p))
if r: results.append(("11) Clip + arcsinh", r))

# ── 12. Mirror-log ───────────────────────────────────────────────────────
def mirror_log(x):
    return np.sign(x) * np.log1p(np.abs(x))

def mirror_log_inv(y):
    return np.sign(y) * (np.exp(np.abs(y)) - 1)

r = run_experiment("12) Mirror-log: sign(x)*log(1+|x|)",
                   mirror_log(spot_tr), mirror_log(spot_va),
                   lambda p: mirror_log_inv(p))
if r: results.append(("12) Mirror-log", r))

# ── 13. spot - spot_lag_168h (weekly seasonal diff) ──────────────────────
y_wdiff_tr = spot_tr - spot_lag168_tr
y_wdiff_va = spot_va - spot_lag168_va
r = run_experiment("13) spot - spot_lag_168h (weekly diff)",
                   y_wdiff_tr, y_wdiff_va,
                   lambda p: spot_lag168_va + p)
if r: results.append(("13) spot - lag_168h", r))

# ── 14. Parametric arcsinh with rolling calibration ──────────────────────
# arcsinh((x - b) / a) with rolling b=median, a=MAD
a_tr = np.clip(roll_168_mad_tr * 1.4826, 1.0, None)
b_tr = roll_168_median_tr
a_va = np.clip(roll_168_mad_va * 1.4826, 1.0, None)
b_va = roll_168_median_va

y_pasinh_tr = np.arcsinh((spot_tr - b_tr) / a_tr)
y_pasinh_va = np.arcsinh((spot_va - b_va) / a_va)

r = run_experiment("14) Parametric arcsinh (rolling median+MAD)",
                   y_pasinh_tr, y_pasinh_va,
                   lambda p: b_va + a_va * np.sinh(p))
if r: results.append(("14) Param arcsinh", r))

# ── 15. arcsinh(spot - spot_la) ──────────────────────────────────────────
r = run_experiment("15) arcsinh(spot - spot_la)",
                   np.arcsinh(y_diff24_tr), np.arcsinh(y_diff24_va),
                   lambda p: spot_la_va + np.sinh(p))
if r: results.append(("15) arcsinh(diff24)", r))

# ── 16. spot - roll_24h_mean ─────────────────────────────────────────────
y_dev24_tr = spot_tr - roll_24_mean_tr
y_dev24_va = spot_va - roll_24_mean_va
r = run_experiment("16) spot - roll_24h_mean",
                   y_dev24_tr, y_dev24_va,
                   lambda p: roll_24_mean_va + p)
if r: results.append(("16) spot - roll_24h_mean", r))

# ── 17. Double: z-score 168h + arcsinh ───────────────────────────────────
r = run_experiment("17) arcsinh(z-score 168h)",
                   np.arcsinh(y_z168_tr), np.arcsinh(y_z168_va),
                   lambda p: roll_168_mean_va + np.sinh(p) * std168_va_clip)
if r: results.append(("17) arcsinh(z168)", r))

# ── 18. Deviation from roll_168h + arcsinh ───────────────────────────────
r = run_experiment("18) arcsinh(spot - roll_168h_mean)",
                   np.arcsinh(y_dev168_tr), np.arcsinh(y_dev168_va),
                   lambda p: roll_168_mean_va + np.sinh(p))
if r: results.append(("18) arcsinh(dev168)", r))

# ── 19. Robust z 168h + arcsinh ──────────────────────────────────────────
r = run_experiment("19) arcsinh(robust z 168h)",
                   np.arcsinh(y_rz168_tr), np.arcsinh(y_rz168_va),
                   lambda p: roll_168_median_va + np.sinh(p) * mad168_va_clip)
if r: results.append(("19) arcsinh(robust z168)", r))


# ══════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  SUMMARY — STATIONARITY METHODS A/B TEST")
print("=" * 70)

results.sort(key=lambda x: x[1])
baseline_rmse = next(r[1] for r in results if "arcsinh(spot)" in r[0])

for i, (label, rmse) in enumerate(results):
    delta = rmse - baseline_rmse
    marker = " <<<" if i == 0 else ""
    print(f"  {i+1:2d}. {label:45s}  RMSE={rmse:7.3f}  Δ={delta:+6.2f}{marker}")

print(f"\n  Baseline (arcsinh spot): {baseline_rmse:.3f}")
print(f"  Best: {results[0][0]} → {results[0][1]:.3f} (Δ={results[0][1]-baseline_rmse:+.2f})")
