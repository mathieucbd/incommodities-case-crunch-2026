"""Hyperparameter tuning v2 FR — Optuna full search.

v1 was sequential grid (each param independently). v2 explores ALL
interactions between params simultaneously via Optuna TPE.

Key improvements over v1:
  - Joint optimization (depth × lr × l2 × csbl × subsample × min_child × random_strength)
  - Low LR regime (0.005-0.02) that v1 barely explored
  - Feature count as HP (jointly optimized)
  - 300 trials with TPE sampler

Current best: depth=3, lr=0.03, l2=30, csbl=0.5, ss=0.7 → RMSE=17.84 (+HBC=17.44)

Usage: cd "INCOMO 3" && python scripts/hyperparam_tuning_v2_fr.py
"""

import sys, json, time, warnings
import numpy as np
import pandas as pd
import optuna

sys.path.insert(0, ".")
from src.data_loading import load_data, merge_train
from src.feature_engineering import build_features
from catboost import CatBoostRegressor, Pool
import yaml

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# ── Load data ─────────────────────────────────────────────────────────────
print("=" * 90)
print("  HYPERPARAMETER TUNING v2 FR — Optuna (300 trials)")
print("=" * 90)

t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train = merge_train(x_train, y_train)
train_fe = build_features(train, config)
print(f"  Data loaded in {time.time() - t0:.0f}s")

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

# ── Feature lists ────────────────────────────────────────────────────────
with open("outputs/shap_ranking_v4_stationary.json") as f:
    v4_ranking = json.load(f)
feat_all = [f for f in v4_ranking["fr_spot"] if f in train_fe.columns]

with open("outputs/feature_selection_v5_fr.json") as f:
    fs_v5 = json.load(f)
feat_27 = fs_v5["features"]

# Add interaction feature
df_train["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
    df_train["fr_spot_la_roll_168h_mean"] * df_train["uk_price_per_mw_7d"]
)
df_val["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
    df_val["fr_spot_la_roll_168h_mean"] * df_val["uk_price_per_mw_7d"]
)
feat_28 = feat_27 + ["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"]

# Feature sets to choose from
FEAT_SETS = {
    "v5-27": feat_27,
    "v5-28": feat_28,
    "top-40": feat_all[:40],
    "top-50": feat_all[:50],
    "top-75": feat_all[:75],
    "top-100": feat_all[:100],
    "all": feat_all,
}

print(f"  Train: {len(df_train)}, Val: {len(df_val)}")
print(f"  Feature sets: {list(FEAT_SETS.keys())}")


# ══════════════════════════════════════════════════════════════════════════
# OPTUNA OBJECTIVE
# ══════════════════════════════════════════════════════════════════════════

def objective(trial):
    depth = trial.suggest_int("depth", 2, 8)
    lr = trial.suggest_float("learning_rate", 0.005, 0.08, log=True)
    l2 = trial.suggest_float("l2_leaf_reg", 0.5, 100, log=True)
    csbl = trial.suggest_float("colsample_bylevel", 0.2, 1.0)
    ss = trial.suggest_float("subsample", 0.5, 1.0)
    min_child = trial.suggest_int("min_child_samples", 1, 100)
    rand_str = trial.suggest_float("random_strength", 0.1, 10, log=True)
    feat_set = trial.suggest_categorical("feature_set",
                                          ["v5-27", "v5-28", "top-40", "top-50",
                                           "top-75", "top-100", "all"])

    feat = FEAT_SETS[feat_set]

    params = {
        "loss_function": "RMSE", "eval_metric": "RMSE",
        "iterations": 15000, "learning_rate": lr, "depth": depth,
        "l2_leaf_reg": l2, "colsample_bylevel": csbl, "subsample": ss,
        "min_child_samples": min_child, "random_strength": rand_str,
        "random_seed": 42, "verbose": 0, "allow_writing_files": False,
        "use_best_model": True,
    }

    model = CatBoostRegressor(**params)
    model.fit(
        Pool(df_train.loc[df_train.index[valid_tr], feat], y_dev_tr[valid_tr],
             weight=weights[valid_tr]),
        eval_set=Pool(df_val.loc[df_val.index[valid_va], feat], y_dev_va[valid_va]),
        early_stopping_rounds=200, verbose=0,
    )

    preds_dev = model.predict(df_val[feat])
    preds_spot = roll_mean_va + preds_dev
    rmse = np.sqrt(np.mean((spot_va - preds_spot) ** 2))

    # Store extra info
    trial.set_user_attr("best_iter", model.get_best_iteration())
    trial.set_user_attr("bias", float(np.mean(spot_va - preds_spot)))

    # Also compute HBC RMSE
    errors = spot_va - preds_spot
    hb = {}
    for h in range(24):
        mask = hours_va == h
        if mask.sum() > 0:
            hb[h] = errors[mask].mean()
    corrected = preds_spot + np.array([hb.get(h, 0) for h in hours_va])
    rmse_hbc = np.sqrt(np.mean((spot_va - corrected) ** 2))
    trial.set_user_attr("rmse_hbc", float(rmse_hbc))

    return rmse


# ══════════════════════════════════════════════════════════════════════════
# RUN OPTUNA
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  RUNNING OPTUNA — 300 trials")
print("=" * 90)

study = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=42))

# Seed with our known best config
study.enqueue_trial({
    "depth": 3, "learning_rate": 0.03, "l2_leaf_reg": 30.0,
    "colsample_bylevel": 0.5, "subsample": 0.7,
    "min_child_samples": 1, "random_strength": 1.0,
    "feature_set": "v5-28",
})

# Also seed with v1 runner-ups
study.enqueue_trial({
    "depth": 3, "learning_rate": 0.03, "l2_leaf_reg": 30.0,
    "colsample_bylevel": 0.3, "subsample": 0.7,
    "min_child_samples": 1, "random_strength": 1.0,
    "feature_set": "v5-27",
})

study.enqueue_trial({
    "depth": 4, "learning_rate": 0.02, "l2_leaf_reg": 10.0,
    "colsample_bylevel": 0.5, "subsample": 0.7,
    "min_child_samples": 1, "random_strength": 1.0,
    "feature_set": "v5-28",
})

# Progress callback
def progress_callback(study, trial):
    if (trial.number + 1) % 25 == 0 or trial.number < 5:
        best = study.best_trial
        print(f"  Trial {trial.number+1:3d}/300: RMSE={trial.value:.4f} "
              f"| Best: {best.value:.4f} (trial {best.number}, "
              f"d={best.params['depth']}, lr={best.params['learning_rate']:.4f}, "
              f"feat={best.params['feature_set']})")

study.optimize(objective, n_trials=300, callbacks=[progress_callback])


# ══════════════════════════════════════════════════════════════════════════
# RESULTS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  OPTUNA RESULTS")
print("=" * 90)

best = study.best_trial
print(f"\n  Best trial: #{best.number}")
print(f"  RMSE:      {best.value:.4f}")
print(f"  RMSE+HBC:  {best.user_attrs['rmse_hbc']:.4f}")
print(f"  Bias:      {best.user_attrs['bias']:+.2f}")
print(f"  Best iter: {best.user_attrs['best_iter']}")
print(f"\n  Parameters:")
for k, v in best.params.items():
    if isinstance(v, float):
        print(f"    {k:25s}: {v:.6f}")
    else:
        print(f"    {k:25s}: {v}")

# Improvement vs v1
v1_rmse = 17.84
v1_hbc = 17.44
print(f"\n  vs v1 (RMSE={v1_rmse}):")
print(f"    RMSE:     {best.value:.4f} (Δ={best.value - v1_rmse:+.4f})")
print(f"    RMSE+HBC: {best.user_attrs['rmse_hbc']:.4f} (Δ={best.user_attrs['rmse_hbc'] - v1_hbc:+.4f})")

# Top-10 trials
print(f"\n  TOP-10 trials:")
print(f"  {'#':>4s}  {'RMSE':>7s}  {'+HBC':>7s}  {'d':>2s}  {'lr':>7s}  {'l2':>7s}  {'csbl':>5s}  {'ss':>5s}  {'feat':>8s}  {'iter':>5s}")
print("  " + "-" * 75)

sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else 999)
for t in sorted_trials[:10]:
    if t.value is None:
        continue
    p = t.params
    print(f"  {t.number:4d}  {t.value:7.4f}  {t.user_attrs.get('rmse_hbc', 0):7.4f}  "
          f"{p['depth']:2d}  {p['learning_rate']:7.4f}  {p['l2_leaf_reg']:7.2f}  "
          f"{p['colsample_bylevel']:5.2f}  {p['subsample']:5.2f}  "
          f"{p['feature_set']:>8s}  {t.user_attrs.get('best_iter', 0):5d}")

# Parameter importance
print(f"\n  Parameter importance (Optuna fANOVA):")
try:
    importances = optuna.importance.get_param_importances(study)
    for k, v in importances.items():
        bar = "█" * int(v * 40)
        print(f"    {k:25s}: {v:.3f}  {bar}")
except Exception:
    print("    (Could not compute — need more trials)")

# Best by feature set
print(f"\n  Best RMSE by feature set:")
feat_best = {}
for t in study.trials:
    if t.value is None:
        continue
    fs = t.params["feature_set"]
    if fs not in feat_best or t.value < feat_best[fs]["rmse"]:
        feat_best[fs] = {"rmse": t.value, "hbc": t.user_attrs.get("rmse_hbc", 999),
                          "trial": t.number}

for fs in ["v5-27", "v5-28", "top-40", "top-50", "top-75", "top-100", "all"]:
    if fs in feat_best:
        fb = feat_best[fs]
        marker = " ◄ BEST" if fb["rmse"] == best.value else ""
        print(f"    {fs:>8s}: RMSE={fb['rmse']:.4f}  +HBC={fb['hbc']:.4f}  (trial {fb['trial']}){marker}")

# Best by depth
print(f"\n  Best RMSE by depth:")
depth_best = {}
for t in study.trials:
    if t.value is None:
        continue
    d = t.params["depth"]
    if d not in depth_best or t.value < depth_best[d]:
        depth_best[d] = t.value
for d in sorted(depth_best.keys()):
    marker = " ◄" if depth_best[d] == best.value else ""
    print(f"    depth={d}: {depth_best[d]:.4f}{marker}")


# ══════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════
output = {
    "method": "optuna_v2",
    "n_trials": len(study.trials),
    "best_rmse": best.value,
    "best_rmse_hbc": best.user_attrs["rmse_hbc"],
    "best_iter": best.user_attrs["best_iter"],
    "best_params": best.params,
    "top10": [
        {"trial": t.number, "rmse": t.value, "rmse_hbc": t.user_attrs.get("rmse_hbc"),
         "params": t.params, "best_iter": t.user_attrs.get("best_iter")}
        for t in sorted_trials[:10] if t.value is not None
    ],
    "v1_comparison": {
        "v1_rmse": v1_rmse, "v2_rmse": best.value,
        "delta": best.value - v1_rmse,
    },
}

with open("outputs/hyperparam_tuning_v2_fr.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n  Saved to outputs/hyperparam_tuning_v2_fr.json")
print(f"  Total time: {time.time() - t0:.0f}s")
