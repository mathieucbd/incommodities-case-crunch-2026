"""Temporal cross-validation v1 — Expanding window validation.

Validates that our v7 results (RMSE=17.84) aren't overfit to the
Feb-Jun 2024 holdout by testing on 4 temporal folds.

Folds (3-month validation windows):
  Fold 1: Train Jul22-Jun23, Val Jul23-Sep23
  Fold 2: Train Jul22-Sep23, Val Oct23-Dec23
  Fold 3: Train Jul22-Dec23, Val Jan24-Mar24
  Fold 4: Train Jul22-Mar24, Val Apr24-Jun24  ← current holdout

If fold 4 RMSE is much better than folds 1-3, we've likely overfit
our hyperparameters to this specific period.

Usage: cd "INCOMO 3" && python scripts/temporal_cv_v1.py
"""

import sys, json, time, warnings
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

# ── Load ──────────────────────────────────────────────────────────────────
print("=" * 90)
print("  TEMPORAL CV v1 — Expanding window (4 folds)")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
train_fe = build_features(train, config)
print(f"  Data loaded in {time.time() - t0:.0f}s — shape: {train_fe.shape}")

# Features
with open("outputs/shap_ranking_v4_stationary.json") as f:
    v4_ranking = json.load(f)
feat_all = [f for f in v4_ranking["fr_spot"] if f in train_fe.columns]

with open("outputs/feature_selection_v5_fr.json") as f:
    fs_v5 = json.load(f)
feat_27 = fs_v5["features"]

# Configs to test
V7_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 15000, "learning_rate": 0.03, "depth": 3,
    "l2_leaf_reg": 30, "subsample": 0.7, "colsample_bylevel": 0.5,
    "random_seed": 42, "verbose": 0, "allow_writing_files": False,
    "use_best_model": True,
}

OLD_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 5000, "learning_rate": 0.03, "depth": 8,
    "l2_leaf_reg": 5, "subsample": 0.8, "random_seed": 42,
    "verbose": 0, "allow_writing_files": False, "use_best_model": True,
}

# Fold definitions
FOLDS = [
    {"name": "Fold 1", "val_start": "2023-07-01", "val_end": "2023-10-01"},
    {"name": "Fold 2", "val_start": "2023-10-01", "val_end": "2024-01-01"},
    {"name": "Fold 3", "val_start": "2024-01-01", "val_end": "2024-04-01"},
    {"name": "Fold 4", "val_start": "2024-04-01", "val_end": "2024-07-01"},
]

# Rolling stats on full dataset
fr_la = train_fe["fr_spot_la"]
roll_mean_full = fr_la.rolling(168, min_periods=24).mean().values
roll_std_full = fr_la.rolling(168, min_periods=24).std().values
dt_full = pd.to_datetime(train_fe["datetime_CET"])


def run_fold(fold_def, feat_list, params, label=""):
    """Run one fold of the CV."""
    val_start = fold_def["val_start"]
    val_end = fold_def["val_end"]

    dt_col = train_fe["datetime_CET"]
    mask_train = dt_col < val_start
    mask_val = (dt_col >= val_start) & (dt_col < val_end)

    df_tr = train_fe[mask_train].copy()
    df_va = train_fe[mask_val].copy()

    if len(df_tr) < 500 or len(df_va) < 100:
        return None

    # Indices in full dataset
    tr_idx = np.where(mask_train.values)[0]
    va_idx = np.where(mask_val.values)[0]

    # Rolling stats for this fold
    rm_tr = roll_mean_full[tr_idx]
    rm_va = roll_mean_full[va_idx]
    rs_tr = roll_std_full[tr_idx]

    spot_tr = df_tr["fr_spot"].values
    spot_va = df_va["fr_spot"].values

    # Target
    y_dev_tr = spot_tr - rm_tr
    y_dev_va = spot_va - rm_va
    valid_tr = np.isfinite(y_dev_tr)
    valid_va = np.isfinite(y_dev_va)

    # Weights
    dt_tr = pd.to_datetime(df_tr["datetime_CET"])
    days_ago = (dt_tr.max() - dt_tr).dt.total_seconds() / 86400
    time_decay = np.exp(-2.0 * days_ago.values / 365)
    var_168h = np.clip(rs_tr ** 2, 1.0, None)
    var_168h = np.where(np.isnan(var_168h), 1.0, var_168h)
    w = time_decay / var_168h

    # Check features exist
    feat = [f for f in feat_list if f in df_tr.columns]

    model = CatBoostRegressor(**params)
    model.fit(
        Pool(df_tr.loc[df_tr.index[valid_tr], feat], y_dev_tr[valid_tr], weight=w[valid_tr]),
        eval_set=Pool(df_va.loc[df_va.index[valid_va], feat], y_dev_va[valid_va]),
        early_stopping_rounds=200, verbose=0,
    )

    preds_dev = model.predict(df_va[feat])
    preds_spot = rm_va + preds_dev
    rmse = np.sqrt(np.mean((spot_va - preds_spot) ** 2))
    bias = np.mean(spot_va - preds_spot)
    best_iter = model.get_best_iteration()

    # HBC
    hours_va = df_va["hour"].values
    errors = spot_va - preds_spot
    hb = {}
    for h in range(24):
        mask = hours_va == h
        if mask.sum() > 0:
            hb[h] = errors[mask].mean()
    corrected = preds_spot + np.array([hb.get(h, 0) for h in hours_va])
    rmse_hbc = np.sqrt(np.mean((spot_va - corrected) ** 2))

    # Mean price for context
    mean_price = spot_va.mean()
    std_price = spot_va.std()

    return {
        "rmse": rmse, "rmse_hbc": rmse_hbc, "bias": bias,
        "best_iter": best_iter, "n_train": len(df_tr), "n_val": len(df_va),
        "mean_price": mean_price, "std_price": std_price,
    }


# ══════════════════════════════════════════════════════════════════════════
# RUN CV
# ══════════════════════════════════════════════════════════════════════════

configs = [
    ("v7 (d=3, 27 feat)", feat_27, V7_PARAMS),
    ("v7 (d=3, all feat)", feat_all, V7_PARAMS),
    ("old (d=8, 27 feat)", feat_27, OLD_PARAMS),
    ("old (d=8, all feat)", feat_all, OLD_PARAMS),
]

all_results = {}

for config_name, feat_list, params in configs:
    print(f"\n{'=' * 90}")
    print(f"  CONFIG: {config_name}")
    print(f"{'=' * 90}")

    header = (f"  {'Fold':10s}  {'Train':>6s}  {'Val':>5s}  {'MeanP':>6s}  {'StdP':>5s}  "
              f"{'RMSE':>6s}  {'+HBC':>6s}  {'Bias':>6s}  {'Iter':>5s}")
    print(header)
    print("  " + "-" * 80)

    fold_results = []
    for fold_def in FOLDS:
        result = run_fold(fold_def, feat_list, params)
        if result is None:
            print(f"  {fold_def['name']:10s}  SKIPPED (insufficient data)")
            continue
        fold_results.append(result)
        print(f"  {fold_def['name']:10s}  {result['n_train']:6d}  {result['n_val']:5d}  "
              f"{result['mean_price']:6.1f}  {result['std_price']:5.1f}  "
              f"{result['rmse']:6.2f}  {result['rmse_hbc']:6.2f}  "
              f"{result['bias']:+6.1f}  {result['best_iter']:5d}")

    if fold_results:
        avg_rmse = np.mean([r["rmse"] for r in fold_results])
        std_rmse = np.std([r["rmse"] for r in fold_results])
        avg_hbc = np.mean([r["rmse_hbc"] for r in fold_results])
        print("  " + "-" * 80)
        print(f"  {'AVG':10s}  {'':6s}  {'':5s}  {'':6s}  {'':5s}  "
              f"{avg_rmse:6.2f}  {avg_hbc:6.2f}  {'':6s}  {'':5s}  "
              f"(±{std_rmse:.2f})")

        all_results[config_name] = {
            "folds": fold_results,
            "avg_rmse": avg_rmse,
            "std_rmse": std_rmse,
            "avg_hbc": avg_hbc,
        }


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  CV SUMMARY — Is our improvement real?")
print("=" * 90)

print(f"\n  {'Config':30s}  {'Avg RMSE':>8s}  {'±Std':>6s}  {'Avg+HBC':>8s}  {'F1':>6s}  "
      f"{'F2':>6s}  {'F3':>6s}  {'F4':>6s}")
print("  " + "-" * 90)

for config_name, data in all_results.items():
    folds_str = "  ".join(f"{r['rmse']:6.2f}" for r in data["folds"])
    print(f"  {config_name:30s}  {data['avg_rmse']:8.2f}  ±{data['std_rmse']:5.2f}  "
          f"{data['avg_hbc']:8.2f}  {folds_str}")

# Check: does v7 consistently beat old?
if "v7 (d=3, 27 feat)" in all_results and "old (d=8, 27 feat)" in all_results:
    v7_folds = all_results["v7 (d=3, 27 feat)"]["folds"]
    old_folds = all_results["old (d=8, 27 feat)"]["folds"]

    print("\n  v7 vs old improvement per fold:")
    for i, (v7_r, old_r) in enumerate(zip(v7_folds, old_folds)):
        delta = v7_r["rmse"] - old_r["rmse"]
        print(f"    Fold {i+1}: v7={v7_r['rmse']:.2f}, old={old_r['rmse']:.2f}, "
              f"Δ={delta:+.2f} {'✓' if delta < 0 else '✗'}")

    n_wins = sum(1 for v, o in zip(v7_folds, old_folds) if v["rmse"] < o["rmse"])
    print(f"    v7 wins: {n_wins}/{len(v7_folds)} folds")

# Check: is fold 4 an outlier?
if "v7 (d=3, 27 feat)" in all_results:
    v7_data = all_results["v7 (d=3, 27 feat)"]
    fold4_rmse = v7_data["folds"][-1]["rmse"]
    other_rmse = np.mean([r["rmse"] for r in v7_data["folds"][:-1]])
    print(f"\n  Fold 4 vs others: F4={fold4_rmse:.2f}, avg(F1-F3)={other_rmse:.2f}, "
          f"Δ={fold4_rmse - other_rmse:+.2f}")
    if fold4_rmse < other_rmse - 2:
        print("  ⚠ WARNING: Fold 4 significantly better → possible overfitting to holdout!")
    else:
        print("  ✓ Fold 4 consistent with other folds → results likely robust")

print(f"\n  Total time: {time.time() - t0:.0f}s")
