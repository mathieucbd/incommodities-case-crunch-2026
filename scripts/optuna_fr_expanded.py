"""Optuna tune for FR CatBoost on an expanded ~35-feature set.

Motivation
----------
The current 28-feature FR set misses fr_gas_margin (rank-1 SHAP, score=7.67)
because the Optuna params (colsample_bylevel=0.104) were tuned for exactly 28
features. Adding new features without re-tuning colsample causes them to be
sampled so rarely they have no effect.

This script:
  1. Builds a ~35-feature set: current 28 + top SHAP newcomers not yet included
  2. Runs Optuna on this expanded set, letting colsample re-tune to reflect
     the richer feature pool
  3. Saves to data/outputs/optuna_fr_expanded.json (separate from the main
     retune file so it can be evaluated before overwriting)
  4. Reports val HBC vs current 28-feat baseline

After running, evaluate manually and if better, copy into
data/outputs/optuna_catboost_retune.json (fr key) and update
feature_selection_v6_fr.json (or create v7_fr).

Usage:
  python scripts/optuna_fr_expanded.py [--trials N]   # default 80
"""

import sys, os, json, time, warnings, argparse
import numpy as np
import yaml
import optuna

sys.path.insert(0, ".")
from src.data_loading import load_data
from src.feature_engineering import build_features
from src.models import compute_rmse, compute_hbc
from src.models.targets import prepare_stationary
from src.models.tree_models import train_tree, predict_tree

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=80)
args = parser.parse_args()

# ── Data ──────────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    config = yaml.safe_load(f)

print("Loading data + features...")
t0 = time.time()
x_train, y_train, x_test = load_data("data/raw")
train_fe = build_features(x_train.copy(), config)
train_fe = train_fe.join(y_train[["fr_spot", "uk_spot"]])

holdout_start = config["validation"]["holdout_start"]
mask_val = train_fe["datetime_CET"] >= holdout_start
df_train = train_fe[~mask_val].copy()
df_val   = train_fe[mask_val].copy()

for df in [df_train, df_val]:
    if "fr_spot_la_roll_168h_mean" in df.columns and "uk_price_per_mw_7d" in df.columns:
        df["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
            df["fr_spot_la_roll_168h_mean"] * df["uk_price_per_mw_7d"]
        )
print(f"  Done in {time.time()-t0:.0f}s  train={len(df_train)} val={len(df_val)}")

# ── Feature set ───────────────────────────────────────────────────────────────
# Current 28 (v6 / v5 + interaction)
with open("data/outputs/feature_selection_v6_fr.json") as f:
    _v6 = json.load(f)
feat_current = _v6["features"]  # 28 features

# Newcomers from SHAP top-30 not already in the current set (ranked by SHAP importance)
# fr_gas_margin is rank-1 (SHAP=7.67) — this is the primary target
SHAP_NEWCOMERS = [
    "fr_gas_margin",                    # rank 1  — gas spark spread vs carbon cost
    "de_river_temp_danube_donauworth_f",# rank 3  — continental hydro/nuclear signal
    "fr_spot_la_h19",                   # rank 4  — hour-19 lag (evening peak)
    "ch_spot_la",                       # rank 5  — Swiss price (hydro-rich neighbour)
    "euro_adequacy_deficit",            # rank 8  — scarcity measure
    "uk_residual_ramp_3h",              # rank 11 — UK supply/demand ramp
    "uk_thermal_need",                  # rank 12 — UK gas generation need
    "fr_gas_spot_rolling_corr",         # rank 16 — FR gas-price correlation regime
    "uk_load_surprise",                 # rank 17 — UK demand surprise
    "ch_hydro_res_f",                   # rank 18 — Swiss hydro reservoir (weekly)
]

# Build expanded set: current 28 + available newcomers (deduped, order preserved)
newcomers_avail = [f for f in SHAP_NEWCOMERS if f in df_train.columns]
feat_expanded = list(dict.fromkeys(feat_current + newcomers_avail))

print(f"\n  Current features : {len(feat_current)}")
print(f"  Newcomers avail  : {len(newcomers_avail)} / {len(SHAP_NEWCOMERS)}")
print(f"  Expanded set     : {len(feat_expanded)} features")
print(f"  New vs current   : {newcomers_avail}")

# ── Stationary target ─────────────────────────────────────────────────────────
fr_stat = prepare_stationary("fr_spot_la", "fr_spot", train_fe, df_train, df_val)
hours_va  = df_val["hour"].values
valid_tr  = fr_stat["valid_tr"]
valid_va  = fr_stat["valid_va"]

feat_exp_avail = [f for f in feat_expanded if f in df_train.columns]
X_tr = df_train.loc[df_train.index[valid_tr], feat_exp_avail]
y_tr = fr_stat["y_dev_tr"][valid_tr]
w_tr = fr_stat["weights"][valid_tr]
X_va = df_val.loc[df_val.index[valid_va], feat_exp_avail]
y_va = fr_stat["y_dev_va"][valid_va]
spot_va = fr_stat["spot_va"][valid_va]
rm_va   = fr_stat["rm_va"][valid_va]

# ── Baseline: current 28 features + current Optuna params ─────────────────────
_retune_path = "data/outputs/optuna_catboost_retune.json"
BASELINE_PARAMS = {
    "loss_function": "Huber:delta=15", "eval_metric": "RMSE",
    "iterations": 15000, "learning_rate": 0.01478,
    "depth": 7, "l2_leaf_reg": 0.931, "subsample": 0.751,
    "colsample_bylevel": 0.104, "min_child_samples": 42,
    "random_strength": 2.836,
    "random_seed": 42, "verbose": 0,
    "allow_writing_files": False, "use_best_model": True,
}
if os.path.exists(_retune_path):
    with open(_retune_path) as f:
        _rt = json.load(f)
    if "fr" in _rt:
        BASELINE_PARAMS.update({**_rt["fr"]["params"],
                                 "eval_metric": "RMSE", "iterations": 15000,
                                 "random_seed": 42, "verbose": 0,
                                 "allow_writing_files": False, "use_best_model": True})

feat_base_avail = [f for f in feat_current if f in df_train.columns]
X_tr_base = df_train.loc[df_train.index[valid_tr], feat_base_avail]
X_va_base = df_val.loc[df_val.index[valid_va], feat_base_avail]

cb_base = train_tree("catboost", BASELINE_PARAMS, X_tr_base, y_tr, X_va_base, y_va, sample_weight=w_tr)
pred_base = rm_va + predict_tree(cb_base.model, df_val.loc[df_val.index[valid_va], feat_base_avail].values)
_, base_hbc = compute_hbc(pred_base, spot_va, hours_va[valid_va])
base_rmse = compute_rmse(spot_va, pred_base)
print(f"\n  Baseline (28 feats, current params): RMSE={base_rmse:.2f}  HBC={base_hbc:.2f}  iter={cb_base.best_iteration}")

# Quick sanity: expanded set with baseline params (should show if gas_margin helps at all)
print(f"  Sanity: expanded {len(feat_exp_avail)} feats with baseline colsample=0.104 ...")
cb_sanity = train_tree("catboost", BASELINE_PARAMS, X_tr, y_tr, X_va, y_va, sample_weight=w_tr)
pred_sanity = rm_va + predict_tree(cb_sanity.model, X_va.values)
_, sanity_hbc = compute_hbc(pred_sanity, spot_va, hours_va[valid_va])
sanity_rmse = compute_rmse(spot_va, pred_sanity)
print(f"  Sanity ({len(feat_exp_avail)} feats, old colsample): RMSE={sanity_rmse:.2f}  HBC={sanity_hbc:.2f}  "
      f"(delta vs base: {sanity_hbc - base_hbc:+.2f})")

# ── Optuna on expanded set ─────────────────────────────────────────────────────
print(f"\n  Starting Optuna ({args.trials} trials) on {len(feat_exp_avail)}-feature set...")
print("=" * 70)

def objective(trial):
    loss = trial.suggest_categorical("loss_function",
                                     ["RMSE", "MAE", "Huber:delta=10", "Huber:delta=15", "Huber:delta=30"])
    params = {
        "loss_function": loss,
        "eval_metric": "RMSE",
        "iterations": 15000,
        "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.08, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.3, 30.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        # colsample range widened to allow proper sampling with ~35 features
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 0.6),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "random_strength": trial.suggest_float("random_strength", 0.1, 5.0),
        "random_seed": 42,
        "verbose": 0,
        "allow_writing_files": False,
        "use_best_model": True,
    }
    try:
        res = train_tree("catboost", params, X_tr, y_tr, X_va, y_va, sample_weight=w_tr)
        pred = rm_va + predict_tree(res.model, X_va.values)
        _, hbc = compute_hbc(pred, spot_va, hours_va[valid_va])
        return hbc
    except Exception:
        return 999.0

storage = "sqlite:///data/outputs/optuna_fr_expanded.db"
study = optuna.create_study(
    direction="minimize",
    study_name="fr_catboost_expanded",
    storage=storage,
    load_if_exists=True,
)
study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

best = study.best_trial
print(f"\n  BEST: HBC={best.value:.2f}  (baseline={base_hbc:.2f}  delta={best.value - base_hbc:+.2f})")
print(f"  Params: {best.params}")

# ── Rebuild best for full metrics ─────────────────────────────────────────────
best_params = {
    "loss_function": best.params["loss_function"],
    "eval_metric": "RMSE",
    "iterations": 15000,
    "learning_rate": best.params["learning_rate"],
    "depth": best.params["depth"],
    "l2_leaf_reg": best.params["l2_leaf_reg"],
    "subsample": best.params["subsample"],
    "colsample_bylevel": best.params["colsample_bylevel"],
    "min_child_samples": best.params["min_child_samples"],
    "random_strength": best.params["random_strength"],
    "random_seed": 42,
    "verbose": 0,
    "allow_writing_files": False,
    "use_best_model": True,
}
cb_best = train_tree("catboost", best_params, X_tr, y_tr, X_va, y_va, sample_weight=w_tr)
pred_best = rm_va + predict_tree(cb_best.model, X_va.values)
rmse_best = compute_rmse(spot_va, pred_best)
_, hbc_best = compute_hbc(pred_best, spot_va, hours_va[valid_va])
print(f"  Verified: RMSE={rmse_best:.2f}  HBC={hbc_best:.2f}  iter={cb_best.best_iteration}")
print(f"  Delta vs baseline: RMSE={rmse_best - base_rmse:+.2f}  HBC={hbc_best - base_hbc:+.2f}")

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    "fr_expanded": {
        "params": {k: v for k, v in best_params.items()
                   if k not in ("verbose", "allow_writing_files", "use_best_model")},
        "features": feat_exp_avail,
        "n_features": len(feat_exp_avail),
        "hbc": round(hbc_best, 4),
        "rmse": round(rmse_best, 4),
        "best_iteration": cb_best.best_iteration,
        "baseline_hbc": round(base_hbc, 4),
        "baseline_n_features": len(feat_base_avail),
        "delta_hbc": round(hbc_best - base_hbc, 4),
        "delta_rmse": round(rmse_best - base_rmse, 4),
        "new_features": newcomers_avail,
    }
}
out_path = "data/outputs/optuna_fr_expanded.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved to {out_path}")

print("\n" + "=" * 70)
print("  NEXT STEPS")
print("=" * 70)
if hbc_best < base_hbc:
    print(f"  ✓ Improvement confirmed: HBC {base_hbc:.2f} -> {hbc_best:.2f} ({hbc_best - base_hbc:+.2f})")
    print()
    print("  To promote to main pipeline:")
    print("  1. Update data/outputs/optuna_catboost_retune.json  (fr key)")
    print("     with the params from data/outputs/optuna_fr_expanded.json")
    print("  2. Update data/outputs/feature_selection_v6_fr.json (or create v7)")
    print("     with the expanded feature list")
    print("  3. Re-run scripts/final_pipeline_v11.py")
else:
    print(f"  ✗ No improvement: HBC {base_hbc:.2f} -> {hbc_best:.2f} ({hbc_best - base_hbc:+.2f})")
    print("  The new features don't add value on the current STL target.")
    print("  Consider: adding more newcomers, or longer Optuna runs.")
