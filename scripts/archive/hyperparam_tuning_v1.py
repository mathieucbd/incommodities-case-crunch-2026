"""Hyperparameter tuning v1 — FR stationary target.

Finding: feature selection v5 shows best_iter stuck at 50-66 regardless of
feature count. This means lr=0.03 + depth=8 converges too fast.
Need: lower lr, shallower trees → longer training, finer-grained learning.

Tests:
  A) Learning rate sweep: 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05
  B) Depth sweep: 4, 5, 6, 7, 8, 9, 10
  C) Grid: lr × depth (top combos)
  D) Regularization: l2_leaf_reg, min_child_samples, colsample_bylevel
  E) Full grid on best combos

Usage: cd "INCOMO 3" && python scripts/hyperparam_tuning_v1.py
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

# ── Load data ─────────────────────────────────────────────────────────────
print("=" * 90)
print("  HYPERPARAMETER TUNING v1 — FR stationary target")
print("=" * 90)

print("\nLoading data...")
t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
train_fe = build_features(train, config)
print(f"  Done in {time.time() - t0:.0f}s")

holdout_start = config["validation"]["holdout_start"]
mask_val = train_fe["datetime_CET"] >= holdout_start
df_train = train_fe[~mask_val].copy()
df_val = train_fe[mask_val].copy()

# ── Rolling stats ─────────────────────────────────────────────────────────
fr_la = train_fe["fr_spot_la"]
roll_mean = fr_la.rolling(168, min_periods=24).mean()
roll_std = fr_la.rolling(168, min_periods=24).std()

n_tr = len(df_train)
roll_mean_tr = roll_mean.iloc[:n_tr].values
roll_std_tr = roll_std.iloc[:n_tr].values
roll_mean_va = roll_mean.iloc[n_tr:n_tr + len(df_val)].values

spot_tr = df_train["fr_spot"].values
spot_va = df_val["fr_spot"].values
hours_va = df_val["hour"].values

# ── Target & weights ──────────────────────────────────────────────────────
y_dev_tr = spot_tr - roll_mean_tr
y_dev_va = spot_va - roll_mean_va
valid_tr = np.isfinite(y_dev_tr)
valid_va = np.isfinite(y_dev_va)

dt = pd.to_datetime(df_train["datetime_CET"])
days_ago = (dt.max() - dt).dt.total_seconds() / 86400
time_decay = np.exp(-2.0 * days_ago.values / 365)
var_168h = np.clip(roll_std_tr ** 2, 1.0, None)
var_168h = np.where(np.isnan(var_168h), 1.0, var_168h)
weights = time_decay / var_168h

# ── Features ──────────────────────────────────────────────────────────────
with open("outputs/shap_ranking_v4_stationary.json") as f:
    v4_ranking = json.load(f)

# Test with different feature sets
feat_all = [f for f in v4_ranking["fr_spot"] if f in df_train.columns]
feat_27 = None
try:
    with open("outputs/feature_selection_v5_fr.json") as f:
        fs_v5 = json.load(f)
    feat_27 = fs_v5["features"]
except FileNotFoundError:
    feat_27 = feat_all[:27]

feat_sets = {
    "top-20": feat_all[:20],
    "v5-27": feat_27,
    "top-50": feat_all[:50],
    "all": feat_all,
}

print(f"  Train: {df_train.shape[0]}, Val: {df_val.shape[0]}")
print(f"  Feature sets: {', '.join(f'{k}({len(v)})' for k, v in feat_sets.items())}")


def train_eval(feat_list, params, label=""):
    """Train CatBoost, return dict with metrics."""
    model = CatBoostRegressor(**params)
    X_tr = df_train.loc[df_train.index[valid_tr], feat_list]
    X_va = df_val.loc[df_val.index[valid_va], feat_list]
    model.fit(
        Pool(X_tr, y_dev_tr[valid_tr], weight=weights[valid_tr]),
        eval_set=Pool(X_va, y_dev_va[valid_va]),
        early_stopping_rounds=200, verbose=0,
    )
    preds_dev = model.predict(df_val[feat_list])
    preds_spot = roll_mean_va + preds_dev
    rmse = np.sqrt(np.mean((spot_va - preds_spot) ** 2))
    bias = np.mean(spot_va - preds_spot)
    best_iter = model.get_best_iteration()

    # Hourly bias correction
    errors = spot_va - preds_spot
    hb = {}
    for h in range(24):
        mask = hours_va == h
        if mask.sum() > 0:
            hb[h] = errors[mask].mean()
    corrected = preds_spot + np.array([hb.get(h, 0) for h in hours_va])
    rmse_hbc = np.sqrt(np.mean((spot_va - corrected) ** 2))

    return {
        "rmse": rmse, "rmse_hbc": rmse_hbc, "bias": bias,
        "best_iter": best_iter, "model": model,
    }


# ══════════════════════════════════════════════════════════════════════════
# A) LEARNING RATE SWEEP
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  A) LEARNING RATE SWEEP (depth=8, feat=top-20)")
print("=" * 90)

lr_results = []
for lr in [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08]:
    params = {
        "loss_function": "RMSE", "eval_metric": "RMSE",
        "iterations": 15000, "learning_rate": lr, "depth": 8,
        "l2_leaf_reg": 5, "subsample": 0.8, "random_seed": 42,
        "verbose": 0, "allow_writing_files": False, "use_best_model": True,
    }
    r = train_eval(feat_sets["top-20"], params)
    lr_results.append({"lr": lr, **r})
    print(
        f"  lr={lr:.3f}  RMSE={r['rmse']:6.2f}  +HBC={r['rmse_hbc']:6.2f}  "
        f"iter={r['best_iter']:5d}  bias={r['bias']:+5.1f}"
    )

best_lr = min(lr_results, key=lambda x: x["rmse"])
print(f"\n  Best lr: {best_lr['lr']}, RMSE={best_lr['rmse']:.3f}")


# ══════════════════════════════════════════════════════════════════════════
# B) DEPTH SWEEP
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print(f"  B) DEPTH SWEEP (lr={best_lr['lr']}, feat=top-20)")
print("=" * 90)

depth_results = []
for depth in [3, 4, 5, 6, 7, 8, 9, 10]:
    params = {
        "loss_function": "RMSE", "eval_metric": "RMSE",
        "iterations": 15000, "learning_rate": best_lr["lr"], "depth": depth,
        "l2_leaf_reg": 5, "subsample": 0.8, "random_seed": 42,
        "verbose": 0, "allow_writing_files": False, "use_best_model": True,
    }
    r = train_eval(feat_sets["top-20"], params)
    depth_results.append({"depth": depth, **r})
    print(
        f"  depth={depth}  RMSE={r['rmse']:6.2f}  +HBC={r['rmse_hbc']:6.2f}  "
        f"iter={r['best_iter']:5d}  bias={r['bias']:+5.1f}"
    )

best_depth = min(depth_results, key=lambda x: x["rmse"])
print(f"\n  Best depth: {best_depth['depth']}, RMSE={best_depth['rmse']:.3f}")


# ══════════════════════════════════════════════════════════════════════════
# C) LR × DEPTH × FEATURES GRID
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  C) LR × DEPTH × FEATURES GRID")
print("=" * 90)

# Test top combos from A and B across feature sets
top_lrs = sorted(lr_results, key=lambda x: x["rmse"])[:3]
top_depths = sorted(depth_results, key=lambda x: x["rmse"])[:3]

grid_results = []
for lr_r in top_lrs:
    lr = lr_r["lr"]
    for depth_r in top_depths:
        depth = depth_r["depth"]
        for feat_name, feat_list in feat_sets.items():
            params = {
                "loss_function": "RMSE", "eval_metric": "RMSE",
                "iterations": 15000, "learning_rate": lr, "depth": depth,
                "l2_leaf_reg": 5, "subsample": 0.8, "random_seed": 42,
                "verbose": 0, "allow_writing_files": False,
                "use_best_model": True,
            }
            r = train_eval(feat_list, params)
            grid_results.append({
                "lr": lr, "depth": depth, "feat": feat_name,
                "n_feat": len(feat_list), **r,
            })
            print(
                f"  lr={lr:.3f} d={depth} {feat_name:6s}({len(feat_list):3d})  "
                f"RMSE={r['rmse']:6.2f}  +HBC={r['rmse_hbc']:6.2f}  "
                f"iter={r['best_iter']:5d}  bias={r['bias']:+5.1f}"
            )

best_grid = min(grid_results, key=lambda x: x["rmse"])
print(
    f"\n  Best grid: lr={best_grid['lr']}, depth={best_grid['depth']}, "
    f"feat={best_grid['feat']}({best_grid['n_feat']}), "
    f"RMSE={best_grid['rmse']:.3f}"
)


# ══════════════════════════════════════════════════════════════════════════
# D) REGULARIZATION on best combo
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  D) REGULARIZATION SWEEP")
print("=" * 90)

best_feat_name = best_grid["feat"]
best_feat_list = feat_sets[best_feat_name]

reg_results = []
for l2 in [1, 3, 5, 8, 10, 15, 20, 30]:
    for min_child in [1, 5, 10, 20, 50]:
        params = {
            "loss_function": "RMSE", "eval_metric": "RMSE",
            "iterations": 15000, "learning_rate": best_grid["lr"],
            "depth": best_grid["depth"],
            "l2_leaf_reg": l2, "subsample": 0.8,
            "min_child_samples": min_child,
            "random_seed": 42, "verbose": 0,
            "allow_writing_files": False, "use_best_model": True,
        }
        r = train_eval(best_feat_list, params)
        reg_results.append({
            "l2": l2, "min_child": min_child, **r,
        })
        print(
            f"  l2={l2:2d}  min_child={min_child:2d}  "
            f"RMSE={r['rmse']:6.2f}  +HBC={r['rmse_hbc']:6.2f}  "
            f"iter={r['best_iter']:5d}  bias={r['bias']:+5.1f}"
        )

best_reg = min(reg_results, key=lambda x: x["rmse"])
print(
    f"\n  Best reg: l2={best_reg['l2']}, min_child={best_reg['min_child']}, "
    f"RMSE={best_reg['rmse']:.3f}"
)


# ══════════════════════════════════════════════════════════════════════════
# E) COLUMN SAMPLING on best combo
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  E) COLUMN SAMPLING + SUBSAMPLE")
print("=" * 90)

colsamp_results = []
for colsample in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for subsample in [0.6, 0.7, 0.8, 0.9, 1.0]:
        params = {
            "loss_function": "RMSE", "eval_metric": "RMSE",
            "iterations": 15000, "learning_rate": best_grid["lr"],
            "depth": best_grid["depth"],
            "l2_leaf_reg": best_reg["l2"],
            "min_child_samples": best_reg["min_child"],
            "subsample": subsample,
            "colsample_bylevel": colsample,
            "random_seed": 42, "verbose": 0,
            "allow_writing_files": False, "use_best_model": True,
        }
        r = train_eval(best_feat_list, params)
        colsamp_results.append({
            "colsample": colsample, "subsample": subsample, **r,
        })
        print(
            f"  colsample={colsample:.1f}  subsample={subsample:.1f}  "
            f"RMSE={r['rmse']:6.2f}  +HBC={r['rmse_hbc']:6.2f}  "
            f"iter={r['best_iter']:5d}  bias={r['bias']:+5.1f}"
        )

best_col = min(colsamp_results, key=lambda x: x["rmse"])
print(
    f"\n  Best: colsample={best_col['colsample']}, "
    f"subsample={best_col['subsample']}, RMSE={best_col['rmse']:.3f}"
)


# ══════════════════════════════════════════════════════════════════════════
# F) FINAL CONFIG — all feature sets
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  F) FINAL CONFIG — ALL FEATURE SETS")
print("=" * 90)

final_params = {
    "loss_function": "RMSE", "eval_metric": "RMSE",
    "iterations": 15000,
    "learning_rate": best_grid["lr"],
    "depth": best_grid["depth"],
    "l2_leaf_reg": best_reg["l2"],
    "min_child_samples": best_reg["min_child"],
    "subsample": best_col["subsample"],
    "colsample_bylevel": best_col["colsample"],
    "random_seed": 42, "verbose": 0,
    "allow_writing_files": False, "use_best_model": True,
}

# Remove model from dict for printing
print(f"\n  Final params: {json.dumps({k: v for k, v in final_params.items() if k != 'model'}, indent=2)}")

final_results = []
for feat_name, feat_list in feat_sets.items():
    r = train_eval(feat_list, final_params)
    final_results.append({
        "feat": feat_name, "n_feat": len(feat_list), **r,
    })
    print(
        f"  {feat_name:6s}({len(feat_list):3d})  RMSE={r['rmse']:6.2f}  "
        f"+HBC={r['rmse_hbc']:6.2f}  iter={r['best_iter']:5d}  "
        f"bias={r['bias']:+5.1f}"
    )

# Also test with more feature counts
for n in [30, 40, 60, 80, 100, 150, 200]:
    feat = feat_all[:n]
    r = train_eval(feat, final_params)
    final_results.append({
        "feat": f"top-{n}", "n_feat": n, **r,
    })
    print(
        f"  top-{n:3d}   ({n:3d})  RMSE={r['rmse']:6.2f}  "
        f"+HBC={r['rmse_hbc']:6.2f}  iter={r['best_iter']:5d}  "
        f"bias={r['bias']:+5.1f}"
    )

best_final = min(final_results, key=lambda x: x["rmse"])
best_final_hbc = min(final_results, key=lambda x: x["rmse_hbc"])


# ══════════════════════════════════════════════════════════════════════════
# GRAND SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  GRAND SUMMARY")
print("=" * 90)

print(f"\n  Previous best:     RMSE=19.99 (dev168 + decay/std², 31 feat, lr=0.03, d=8)")
print(f"  Selection v5 best: RMSE=18.55 (all feat, lr=0.03, d=8, iter=56)")
print(f"\n  TUNED best:        RMSE={best_final['rmse']:.3f} "
      f"({best_final['feat']}, {best_final['n_feat']} feat, "
      f"iter={best_final['best_iter']})")
print(f"  TUNED best +HBC:   RMSE={best_final_hbc['rmse_hbc']:.3f} "
      f"({best_final_hbc['feat']}, {best_final_hbc['n_feat']} feat)")
print(f"\n  Optimal params:")
for k, v in final_params.items():
    if k not in ("verbose", "allow_writing_files", "use_best_model", "model"):
        print(f"    {k}: {v}")

# Save best config
output = {
    "target": "FR stationary (spot - roll_168h_mean)",
    "weights": "exp_decay(2.0) / clip(roll_std², 1)",
    "best_rmse": best_final["rmse"],
    "best_rmse_hbc": best_final_hbc["rmse_hbc"],
    "best_features": best_final["feat"],
    "best_n_features": best_final["n_feat"],
    "best_iter": best_final["best_iter"],
    "params": {k: v for k, v in final_params.items()
               if k not in ("verbose", "allow_writing_files", "use_best_model")},
}
with open("outputs/hyperparam_tuning_v1_fr.json", "w") as fout:
    json.dump(output, fout, indent=2)
print(f"\n  Saved to outputs/hyperparam_tuning_v1_fr.json")
print(f"\n  Total time: {time.time() - t0:.0f}s")
