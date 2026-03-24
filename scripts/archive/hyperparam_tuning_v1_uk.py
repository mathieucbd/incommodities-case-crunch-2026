"""Hyperparameter tuning v1 UK — Optuna focused search.

UK-specific: basis target (spot - merit_order_cost), NO sample weights.
Focused search: d=6-10, confirmed features only, 150 trials.
Unbuffered output for progress tracking.

Current best: depth=8, lr=0.03, l2=5, csbl=0.8, ss=0.8 → RMSE=10.19 (+HBC=9.97)

Usage: cd "INCOMO 3" && python -u scripts/hyperparam_tuning_v1_uk.py
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
print("  HYPERPARAMETER TUNING v1 UK — Optuna (300 trials)")
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

# ── Basis target ─────────────────────────────────────────────────────────
uk_spot_tr = df_train["uk_spot"].values
uk_spot_va = df_val["uk_spot"].values
uk_moc_tr = df_train["uk_merit_order_cost"].values
uk_moc_va = df_val["uk_merit_order_cost"].values

y_basis_tr = uk_spot_tr - uk_moc_tr
y_basis_va = uk_spot_va - uk_moc_va
valid_tr = np.isfinite(y_basis_tr)
valid_va = np.isfinite(y_basis_va)

hours_va = df_val["hour"].values

# ── Feature lists ────────────────────────────────────────────────────────
with open("outputs/uk_feature_research.json") as f:
    uk_research = json.load(f)

# Confirmed features (150) ranked by SHAP importance
feat_confirmed = [f for f in uk_research["confirmed_features"] if f in df_train.columns]

# Focused feature sets (no all-numeric — too slow and marginal)
FEAT_SETS = {
    "confirmed-75": feat_confirmed[:75],
    "confirmed-100": feat_confirmed[:100],
    "confirmed-150": feat_confirmed,
}

print(f"  Train: {len(df_train)}, Val: {len(df_val)}")
print(f"  Feature sets: {', '.join(f'{k}({len(v)})' for k, v in FEAT_SETS.items())}")
print(f"  Basis target: mean={y_basis_tr[valid_tr].mean():.1f}, std={y_basis_tr[valid_tr].std():.1f}")


# ══════════════════════════════════════════════════════════════════════════
# OPTUNA OBJECTIVE
# ══════════════════════════════════════════════════════════════════════════

def objective(trial):
    depth = trial.suggest_int("depth", 6, 10)
    lr = trial.suggest_float("learning_rate", 0.01, 0.08, log=True)
    l2 = trial.suggest_float("l2_leaf_reg", 0.5, 50, log=True)
    csbl = trial.suggest_float("colsample_bylevel", 0.3, 1.0)
    ss = trial.suggest_float("subsample", 0.5, 1.0)
    min_child = trial.suggest_int("min_child_samples", 1, 50)
    rand_str = trial.suggest_float("random_strength", 0.1, 5, log=True)
    feat_set = trial.suggest_categorical("feature_set",
                                          list(FEAT_SETS.keys()))

    feat = FEAT_SETS[feat_set]

    params = {
        "loss_function": "RMSE", "eval_metric": "RMSE",
        "iterations": 15000, "learning_rate": lr, "depth": depth,
        "l2_leaf_reg": l2, "colsample_bylevel": csbl, "subsample": ss,
        "min_child_samples": min_child, "random_strength": rand_str,
        "random_seed": 42, "verbose": 0, "allow_writing_files": False,
        "use_best_model": True,
    }

    # UK: NO sample weights
    model = CatBoostRegressor(**params)
    model.fit(
        Pool(df_train.loc[df_train.index[valid_tr], feat], y_basis_tr[valid_tr]),
        eval_set=Pool(df_val.loc[df_val.index[valid_va], feat], y_basis_va[valid_va]),
        early_stopping_rounds=200, verbose=0,
    )

    preds_basis = model.predict(df_val[feat])
    preds_spot = uk_moc_va + preds_basis
    rmse = np.sqrt(np.mean((uk_spot_va - preds_spot) ** 2))

    # Store extra info
    trial.set_user_attr("best_iter", model.get_best_iteration())
    trial.set_user_attr("bias", float(np.mean(uk_spot_va - preds_spot)))

    # HBC RMSE
    errors = uk_spot_va - preds_spot
    hb = {}
    for h in range(24):
        mask = hours_va == h
        if mask.sum() > 0:
            hb[h] = errors[mask].mean()
    corrected = preds_spot + np.array([hb.get(h, 0) for h in hours_va])
    rmse_hbc = np.sqrt(np.mean((uk_spot_va - corrected) ** 2))
    trial.set_user_attr("rmse_hbc", float(rmse_hbc))

    return rmse


# ══════════════════════════════════════════════════════════════════════════
# RUN OPTUNA
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  RUNNING OPTUNA — 150 trials (focused)")
print("=" * 90)
sys.stdout.flush()

study = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=42))

# Seed with our known best config
study.enqueue_trial({
    "depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 5.0,
    "colsample_bylevel": 0.8, "subsample": 0.8,
    "min_child_samples": 1, "random_strength": 1.0,
    "feature_set": "confirmed-150",
})

# Seed with variants
study.enqueue_trial({
    "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 5.0,
    "colsample_bylevel": 0.7, "subsample": 0.8,
    "min_child_samples": 1, "random_strength": 1.0,
    "feature_set": "confirmed-100",
})

study.enqueue_trial({
    "depth": 10, "learning_rate": 0.02, "l2_leaf_reg": 10.0,
    "colsample_bylevel": 0.6, "subsample": 0.7,
    "min_child_samples": 10, "random_strength": 1.0,
    "feature_set": "confirmed-150",
})

# Progress callback with flush
def progress_callback(study, trial):
    if (trial.number + 1) % 10 == 0 or trial.number < 5:
        best = study.best_trial
        print(f"  Trial {trial.number+1:3d}/150: RMSE={trial.value:.4f} "
              f"| Best: {best.value:.4f} (trial {best.number}, "
              f"d={best.params['depth']}, lr={best.params['learning_rate']:.4f}, "
              f"feat={best.params['feature_set']})")
        sys.stdout.flush()

study.optimize(objective, n_trials=150, callbacks=[progress_callback])


# ══════════════════════════════════════════════════════════════════════════
# RESULTS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  OPTUNA RESULTS — UK")
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

# Improvement vs current
v0_rmse = 10.19
v0_hbc = 9.97
print(f"\n  vs current (RMSE={v0_rmse}):")
print(f"    RMSE:     {best.value:.4f} (Δ={best.value - v0_rmse:+.4f})")
print(f"    RMSE+HBC: {best.user_attrs['rmse_hbc']:.4f} (Δ={best.user_attrs['rmse_hbc'] - v0_hbc:+.4f})")

# Top-10 trials
print(f"\n  TOP-10 trials:")
print(f"  {'#':>4s}  {'RMSE':>7s}  {'+HBC':>7s}  {'d':>2s}  {'lr':>7s}  {'l2':>7s}  {'csbl':>5s}  {'ss':>5s}  {'mc':>3s}  {'feat':>14s}  {'iter':>5s}")
print("  " + "-" * 85)

sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else 999)
for t in sorted_trials[:10]:
    if t.value is None:
        continue
    p = t.params
    print(f"  {t.number:4d}  {t.value:7.4f}  {t.user_attrs.get('rmse_hbc', 0):7.4f}  "
          f"{p['depth']:2d}  {p['learning_rate']:7.4f}  {p['l2_leaf_reg']:7.2f}  "
          f"{p['colsample_bylevel']:5.2f}  {p['subsample']:5.2f}  "
          f"{p['min_child_samples']:3d}  "
          f"{p['feature_set']:>14s}  {t.user_attrs.get('best_iter', 0):5d}")

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

for fs in sorted(FEAT_SETS.keys()):
    if fs in feat_best:
        fb = feat_best[fs]
        marker = " ◄ BEST" if fb["rmse"] == best.value else ""
        print(f"    {fs:>14s}: RMSE={fb['rmse']:.4f}  +HBC={fb['hbc']:.4f}  (trial {fb['trial']}){marker}")

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
    print(f"    depth={d:2d}: {depth_best[d]:.4f}{marker}")


# ══════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════
output = {
    "method": "optuna_v1_uk",
    "target": "basis (spot - merit_order_cost)",
    "weights": "none",
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
    "v0_comparison": {
        "v0_rmse": v0_rmse, "optuna_rmse": best.value,
        "delta": best.value - v0_rmse,
    },
}

with open("outputs/hyperparam_tuning_v1_uk.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n  Saved to outputs/hyperparam_tuning_v1_uk.json")
print(f"  Total time: {time.time() - t0:.0f}s")
