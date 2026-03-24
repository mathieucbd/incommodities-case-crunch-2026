"""Multi-window ensemble averaging — EPF competition winning technique.

Instead of training on the full history, train the same models on different
calibration windows and average predictions. This smooths regime-specific biases.

Windows (all ending at holdout_start = 2024-02-01):
  - Full:  Jul 2022 — Jan 2024  (~18 months, includes crisis)
  - 18m:   Aug 2022 — Jan 2024  (trim first month)
  - 12m:   Feb 2023 — Jan 2024  (1 year, mostly normal regime)
  - 9m:    May 2023 — Jan 2024  (recent only)
  - 6m:    Aug 2023 — Jan 2024  (very recent, fully normal regime)

Tests:
  1. Each window individually (CB+LGB+EN per window)
  2. Average of all windows vs best single window
  3. Weighted average (optimized on val)
"""

import sys, yaml, warnings, time, json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features

warnings.filterwarnings("ignore")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

print("=" * 90)
print("  MULTI-WINDOW ENSEMBLE TEST")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
df = build_features(train, config)

holdout_start = config["validation"]["holdout_start"]
mask_val = (df["datetime_CET"] >= holdout_start).values
df_full_tr = df[~mask_val].copy()
df_va = df[mask_val].copy()

# ── Features ──────────────────────────────────────────────────────────
with open("outputs/feature_selection_v5_fr.json") as f:
    FR_FEAT = [f for f in json.load(f)["features"] if f in df_full_tr.columns]
with open("outputs/uk_feature_research.json") as f:
    UK_FEAT = [f for f in json.load(f)["confirmed_features"] if f in df_full_tr.columns]

print(f"  FR features: {len(FR_FEAT)}, UK features: {len(UK_FEAT)}")

# ── Targets & anchors ────────────────────────────────────────────────
fr_la = df["fr_spot_la"].values
ema_fr_full = pd.Series(fr_la).ewm(span=240).mean().values
fr_anchor_va = ema_fr_full[mask_val]
fr_spot_va = df_va["fr_spot"].values
uk_moc_va = df_va["uk_merit_order_cost"].values
uk_spot_va = df_va["uk_spot"].values
hours_va = df_va["hour"].values


def compute_hbc(preds, actual, hours):
    errors = actual - preds
    hbc = {h: float(errors[hours == h].mean()) for h in range(24) if (hours == h).sum() > 0}
    corrected = preds + np.array([hbc.get(h, 0) for h in hours])
    rmse_raw = np.sqrt(np.mean((actual - preds) ** 2))
    rmse_hbc = np.sqrt(np.mean((actual - corrected) ** 2))
    return rmse_raw, rmse_hbc


# ── Windows ───────────────────────────────────────────────────────────
WINDOWS = [
    {"name": "full",  "start": None},                # All available data
    {"name": "18m",   "start": "2022-08-01"},
    {"name": "12m",   "start": "2023-02-01"},
    {"name": "9m",    "start": "2023-05-01"},
    {"name": "6m",    "start": "2023-08-01"},
]

cb_params_fr = config.get("catboost_params_fr_optuna_v2", config.get("catboost_params_fr", {}))
cb_params_uk = config.get("catboost_params_uk", {})
lgb_params_fr = config.get("lightgbm_params_fr", {})
lgb_params_fr_clean = {k: v for k, v in lgb_params_fr.items() if k != "n_estimators"}
lgb_params_uk = config.get("lightgbm_params_uk", {})
lgb_params_uk_clean = {k: v for k, v in lgb_params_uk.items() if k != "n_estimators"}


# ══════════════════════════════════════════════════════════════════════
#  TRAIN MODELS PER WINDOW
# ══════════════════════════════════════════════════════════════════════
all_preds_fr = {}  # window_name -> spot preds array
all_preds_uk = {}

for win in WINDOWS:
    t1 = time.time()
    win_name = win["name"]

    # Filter training data by window
    if win["start"] is not None:
        mask_win = df_full_tr["datetime_CET"] >= win["start"]
        df_tr = df_full_tr[mask_win].copy()
    else:
        df_tr = df_full_tr.copy()

    print(f"\n  --- Window: {win_name} ({len(df_tr)} rows, "
          f"{df_tr['datetime_CET'].min().strftime('%Y-%m')} to "
          f"{df_tr['datetime_CET'].max().strftime('%Y-%m')}) ---")

    # Recompute EMA for this window (important: EMA depends on history)
    # For shorter windows, we still use the full EMA as anchor (it's precomputed on full data)
    # The window only affects TRAINING data, not the anchor
    ema_tr = ema_fr_full[~mask_val]
    if win["start"] is not None:
        # Get indices within the full training set
        win_idx = mask_win.values
        ema_tr_win = ema_tr[win_idx]
    else:
        win_idx = np.ones(len(df_full_tr), dtype=bool)
        ema_tr_win = ema_tr

    # FR targets
    fr_y_tr = df_tr["fr_spot"].values - ema_tr_win
    fr_valid_tr = np.isfinite(fr_y_tr) & np.isfinite(ema_tr_win)
    fr_y_va = fr_spot_va - fr_anchor_va
    fr_valid_va = np.isfinite(fr_y_va) & np.isfinite(fr_anchor_va)

    # UK targets
    uk_y_tr = df_tr["uk_spot"].values - df_tr["uk_merit_order_cost"].values
    uk_y_va = uk_spot_va - uk_moc_va
    uk_valid_tr = np.isfinite(uk_y_tr)
    uk_valid_va = np.isfinite(uk_y_va)

    # Sample weights (recency relative to THIS window's end)
    days_ago = (df_tr["datetime_CET"].max() - df_tr["datetime_CET"]).dt.total_seconds() / 86400
    roll_std = df_tr["fr_spot_la"].rolling(168, min_periods=24).std().fillna(df_tr["fr_spot_la"].std())
    fr_sw = np.exp(-2 * days_ago.values / 365) / np.clip(roll_std.values ** 2, 1, None)
    fr_sw[~fr_valid_tr] = 0

    # ── FR: CB + LGB + EN ──────────────────────────────────────────
    # CatBoost
    cb_fr = CatBoostRegressor(**{**cb_params_fr, "verbose": 0})
    cb_fr.fit(df_tr[FR_FEAT].values[fr_valid_tr], fr_y_tr[fr_valid_tr],
              sample_weight=fr_sw[fr_valid_tr],
              eval_set=(df_va[FR_FEAT].values, fr_y_va))
    p_fr_cb = fr_anchor_va + cb_fr.predict(df_va[FR_FEAT].values)

    # LightGBM
    ds_tr = lgb.Dataset(df_tr[FR_FEAT].values[fr_valid_tr], fr_y_tr[fr_valid_tr],
                        weight=fr_sw[fr_valid_tr])
    ds_va = lgb.Dataset(df_va[FR_FEAT].values, fr_y_va, reference=ds_tr)
    lgb_fr = lgb.train(lgb_params_fr_clean, ds_tr,
                       num_boost_round=lgb_params_fr.get("n_estimators", 5000),
                       valid_sets=[ds_va], callbacks=[lgb.early_stopping(50, verbose=False)])
    p_fr_lgb = fr_anchor_va + lgb_fr.predict(df_va[FR_FEAT].values)

    # Elastic Net
    fr_scaler = StandardScaler()
    X_en_tr = fr_scaler.fit_transform(np.nan_to_num(df_tr[FR_FEAT].values[fr_valid_tr], 0))
    X_en_va = fr_scaler.transform(np.nan_to_num(df_va[FR_FEAT].values, 0))
    en_fr = ElasticNet(alpha=10.0, l1_ratio=0.9, max_iter=10000)
    en_fr.fit(X_en_tr, fr_y_tr[fr_valid_tr])
    p_fr_en = fr_anchor_va + en_fr.predict(X_en_va)

    # Simple average of 3 models
    p_fr_avg = (p_fr_cb + p_fr_lgb + p_fr_en) / 3
    all_preds_fr[win_name] = p_fr_avg

    rmse_raw, rmse_hbc = compute_hbc(p_fr_avg, fr_spot_va, hours_va)
    print(f"    FR 3-model avg:  RMSE={rmse_raw:.2f}  +HBC={rmse_hbc:.2f}")

    # ── UK: CB + LGB + EN ──────────────────────────────────────────
    cb_uk = CatBoostRegressor(**{**cb_params_uk, "verbose": 0})
    cb_uk.fit(df_tr[UK_FEAT].values[uk_valid_tr], uk_y_tr[uk_valid_tr],
              eval_set=(df_va[UK_FEAT].values, uk_y_va))
    p_uk_cb = uk_moc_va + cb_uk.predict(df_va[UK_FEAT].values)

    ds_tr_uk = lgb.Dataset(df_tr[UK_FEAT].values[uk_valid_tr], uk_y_tr[uk_valid_tr])
    ds_va_uk = lgb.Dataset(df_va[UK_FEAT].values, uk_y_va, reference=ds_tr_uk)
    lgb_uk = lgb.train(lgb_params_uk_clean, ds_tr_uk,
                       num_boost_round=lgb_params_uk.get("n_estimators", 5000),
                       valid_sets=[ds_va_uk], callbacks=[lgb.early_stopping(50, verbose=False)])
    p_uk_lgb = uk_moc_va + lgb_uk.predict(df_va[UK_FEAT].values)

    uk_scaler = StandardScaler()
    X_uk_en_tr = uk_scaler.fit_transform(np.nan_to_num(df_tr[UK_FEAT].values[uk_valid_tr], 0))
    X_uk_en_va = uk_scaler.transform(np.nan_to_num(df_va[UK_FEAT].values, 0))
    en_uk = ElasticNet(alpha=1.0, l1_ratio=0.9, max_iter=10000)
    en_uk.fit(X_uk_en_tr, uk_y_tr[uk_valid_tr])
    p_uk_en = uk_moc_va + en_uk.predict(X_uk_en_va)

    p_uk_avg = (p_uk_cb + p_uk_lgb + p_uk_en) / 3
    all_preds_uk[win_name] = p_uk_avg

    rmse_raw, rmse_hbc = compute_hbc(p_uk_avg, uk_spot_va, hours_va)
    elapsed = time.time() - t1
    print(f"    UK 3-model avg:  RMSE={rmse_raw:.2f}  +HBC={rmse_hbc:.2f}  ({elapsed:.0f}s)")


# ══════════════════════════════════════════════════════════════════════
#  MULTI-WINDOW AVERAGING
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("  MULTI-WINDOW AVERAGING")
print(f"{'='*90}")

window_names = list(all_preds_fr.keys())

# Individual windows
print("\n  Individual windows:")
for wn in window_names:
    _, fr_hbc = compute_hbc(all_preds_fr[wn], fr_spot_va, hours_va)
    _, uk_hbc = compute_hbc(all_preds_uk[wn], uk_spot_va, hours_va)
    print(f"    {wn:8s}  FR+HBC={fr_hbc:.2f}  UK+HBC={uk_hbc:.2f}  SUM={fr_hbc + uk_hbc:.2f}")

# Average of all windows
print("\n  Window averages:")
combos = [
    ("all_5", window_names),
    ("no_full", [w for w in window_names if w != "full"]),
    ("recent_3", ["12m", "9m", "6m"]),
    ("recent_2", ["12m", "6m"]),
    ("full+12m", ["full", "12m"]),
    ("full+9m", ["full", "9m"]),
    ("full+6m", ["full", "6m"]),
]

results = []
for combo_name, combo_windows in combos:
    fr_avg = np.mean([all_preds_fr[w] for w in combo_windows], axis=0)
    uk_avg = np.mean([all_preds_uk[w] for w in combo_windows], axis=0)
    _, fr_hbc = compute_hbc(fr_avg, fr_spot_va, hours_va)
    _, uk_hbc = compute_hbc(uk_avg, uk_spot_va, hours_va)
    total = fr_hbc + uk_hbc
    results.append({"combo": combo_name, "windows": combo_windows,
                     "fr_hbc": fr_hbc, "uk_hbc": uk_hbc, "sum": total})
    print(f"    {combo_name:15s}  FR+HBC={fr_hbc:.2f}  UK+HBC={uk_hbc:.2f}  SUM={total:.2f}")

# Weighted average (optimize on val)
print("\n  Optimized weighted averages:")
from scipy.optimize import minimize

for market, all_preds, actual in [
    ("FR", all_preds_fr, fr_spot_va),
    ("UK", all_preds_uk, uk_spot_va),
]:
    preds_list = [all_preds[w] for w in window_names]
    n = len(preds_list)

    def objective(w):
        w_full = np.append(w, 1.0 - np.sum(w))
        if w_full[-1] < -0.05:
            return 1e6
        pred = sum(w_full[i] * preds_list[i] for i in range(n))
        # Apply HBC internally
        errors = actual - pred
        hbc = {h: float(errors[hours_va == h].mean()) for h in range(24)}
        corrected = pred + np.array([hbc.get(h, 0) for h in hours_va])
        return np.sqrt(np.mean((actual - corrected) ** 2))

    best_result = None
    for _ in range(50):
        w0 = np.random.dirichlet(np.ones(n))[:-1]
        res = minimize(objective, w0, method='L-BFGS-B',
                      bounds=[(0, 1)] * (n - 1))
        if best_result is None or res.fun < best_result.fun:
            best_result = res

    w_opt = np.append(best_result.x, 1.0 - np.sum(best_result.x))
    w_opt = np.maximum(w_opt, 0)
    w_opt /= w_opt.sum()

    w_str = " / ".join(f"{window_names[i]}={w_opt[i]:.2f}" for i in range(n))
    print(f"    {market} optimal: {w_str}  → RMSE+HBC={best_result.fun:.2f}")


# ══════════════════════════════════════════════════════════════════════
#  COMPARE WITH CURRENT PIPELINE
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("  COMPARISON WITH CURRENT PIPELINE")
print(f"{'='*90}")

# Current pipeline: single full-window, 3-model average
_, fr_current = compute_hbc(all_preds_fr["full"], fr_spot_va, hours_va)
_, uk_current = compute_hbc(all_preds_uk["full"], uk_spot_va, hours_va)
current_sum = fr_current + uk_current

best_combo = min(results, key=lambda x: x["sum"])
print(f"\n  Current (full window, CB+LGB+EN):    FR={fr_current:.2f}  UK={uk_current:.2f}  SUM={current_sum:.2f}")
print(f"  Best multi-window ({best_combo['combo']:15s}): FR={best_combo['fr_hbc']:.2f}  UK={best_combo['uk_hbc']:.2f}  SUM={best_combo['sum']:.2f}")
print(f"  Delta: {best_combo['sum'] - current_sum:+.2f}")

print(f"\n  Note: Current pipeline v7 SUM=25.12 (5 models + regime weights)")
print(f"  This test uses only 3 models (CB+LGB+EN) with simple average + HBC")
print(f"  The multi-window gain would apply ON TOP of the full pipeline")

print(f"\n  Total time: {time.time() - t0:.0f}s")

# Save results
with open("outputs/multi_window_test.json", "w") as f:
    json.dump({"windows": {w: {"fr_hbc": float(compute_hbc(all_preds_fr[w], fr_spot_va, hours_va)[1]),
                                "uk_hbc": float(compute_hbc(all_preds_uk[w], uk_spot_va, hours_va)[1])}
               for w in window_names},
               "combos": [{k: v for k, v in r.items() if k != "windows"} for r in results]},
              f, indent=2)
