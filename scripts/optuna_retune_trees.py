"""Optuna re-tuning of FR & UK CatBoost for STL/basis targets.

The current FR_PARAMS (depth=3, lr=0.059) were tuned for EMA(240h) detrending.
STL(168) changes the residual distribution, so we re-optimise.
Also tunes UK CatBoost (basis target) and explores loss functions.

Usage: python scripts/optuna_retune_trees.py [--fr-only | --uk-only] [--trials N]
"""

import sys, json, time, warnings, argparse
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

# -- Args -----------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--fr-only", action="store_true")
parser.add_argument("--uk-only", action="store_true")
parser.add_argument("--trials", type=int, default=60)
args = parser.parse_args()

# -- Data -----------------------------------------------------------------
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

# Interaction feature
for df in [df_train, df_val]:
    if "fr_spot_la_roll_168h_mean" in df.columns and "uk_price_per_mw_7d" in df.columns:
        df["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"] = (
            df["fr_spot_la_roll_168h_mean"] * df["uk_price_per_mw_7d"]
        )

print(f"  Data ready in {time.time() - t0:.0f}s  train={len(df_train)} val={len(df_val)}")

# -- Feature lists --------------------------------------------------------
with open("data/outputs/feature_selection_v5_fr.json") as f:
    fs_v5 = json.load(f)
feat_fr_27 = fs_v5["features"]
feat_fr_28 = feat_fr_27 + ["X_fr_spot_la_roll_168h_mean_x_uk_price_per_mw_7d"]
feat_fr = [f for f in feat_fr_28 if f in df_train.columns]

with open("data/outputs/uk_feature_research.json") as f:
    uk_research = json.load(f)
feat_uk = [f for f in uk_research["confirmed_features"] if f in df_train.columns]


# =========================================================================
# FR CatBoost -- STL stationary target
# =========================================================================
def tune_fr(n_trials: int):
    fr_stat = prepare_stationary("fr_spot_la", "fr_spot", train_fe, df_train, df_val)
    hours_va = df_val["hour"].values

    valid_tr = fr_stat["valid_tr"]
    valid_va = fr_stat["valid_va"]
    X_tr = df_train.loc[df_train.index[valid_tr], feat_fr]
    y_tr = fr_stat["y_dev_tr"][valid_tr]
    w_tr = fr_stat["weights"][valid_tr]
    X_va = df_val.loc[df_val.index[valid_va], feat_fr]
    y_va = fr_stat["y_dev_va"][valid_va]
    spot_va = fr_stat["spot_va"]
    rm_va   = fr_stat["rm_va"]

    # Current baseline
    CURRENT_FR = {
        "loss_function": "RMSE", "eval_metric": "RMSE",
        "iterations": 15000, "learning_rate": 0.059, "depth": 3,
        "l2_leaf_reg": 4.42, "subsample": 0.533, "colsample_bylevel": 0.228,
        "min_child_samples": 14, "random_strength": 0.9,
        "random_seed": 42, "verbose": 0,
        "allow_writing_files": False, "use_best_model": True,
    }
    cb_base = train_tree("catboost", CURRENT_FR, X_tr, y_tr, X_va, y_va, sample_weight=w_tr)
    pred_base = rm_va + predict_tree(cb_base.model, df_val[feat_fr])
    _, base_hbc = compute_hbc(pred_base, spot_va, hours_va)
    base_rmse = compute_rmse(spot_va, pred_base)
    print(f"\n  FR baseline: RMSE={base_rmse:.2f}  HBC={base_hbc:.2f}  iter={cb_base.best_iteration}")

    def objective(trial):
        loss = trial.suggest_categorical("loss_function", ["RMSE", "MAE", "Huber:delta=15", "Huber:delta=30"])
        params = {
            "loss_function": loss,
            "eval_metric": "RMSE",
            "iterations": 15000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "depth": trial.suggest_int("depth", 3, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 0.8),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "random_strength": trial.suggest_float("random_strength", 0.1, 5.0),
            "random_seed": 42,
            "verbose": 0,
            "allow_writing_files": False,
            "use_best_model": True,
        }
        try:
            res = train_tree("catboost", params, X_tr, y_tr, X_va, y_va, sample_weight=w_tr)
            pred = rm_va + predict_tree(res.model, df_val[feat_fr])
            _, hbc = compute_hbc(pred, spot_va, hours_va)
            return hbc
        except Exception:
            return 999.0

    storage = "sqlite:///data/outputs/optuna_retune.db"
    study = optuna.create_study(direction="minimize", study_name="fr_catboost_stl",
                                storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\n  FR BEST: HBC={best.value:.2f}  (baseline={base_hbc:.2f}  delta={best.value - base_hbc:+.2f})")
    print(f"  Params: {best.params}")

    # Rebuild best model for detailed metrics
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
    pred_best = rm_va + predict_tree(cb_best.model, df_val[feat_fr])
    rmse_best = compute_rmse(spot_va, pred_best)
    _, hbc_best = compute_hbc(pred_best, spot_va, hours_va)
    print(f"  Verified: RMSE={rmse_best:.2f}  HBC={hbc_best:.2f}  iter={cb_best.best_iteration}")

    return {"params": best_params, "hbc": hbc_best, "rmse": rmse_best,
            "best_iteration": cb_best.best_iteration, "baseline_hbc": base_hbc}


# =========================================================================
# UK CatBoost -- basis target
# =========================================================================
def tune_uk(n_trials: int):
    uk_spot_tr = df_train["uk_spot"].values
    uk_spot_va = df_val["uk_spot"].values
    uk_moc_tr  = df_train["uk_merit_order_cost"].values
    uk_moc_va  = df_val["uk_merit_order_cost"].values
    y_basis_tr = uk_spot_tr - uk_moc_tr
    y_basis_va = uk_spot_va - uk_moc_va
    valid_tr   = np.isfinite(y_basis_tr)
    valid_va   = np.isfinite(y_basis_va)
    hours_va   = df_val["hour"].values

    X_tr = df_train.loc[df_train.index[valid_tr], feat_uk]
    y_tr = y_basis_tr[valid_tr]
    X_va = df_val.loc[df_val.index[valid_va], feat_uk]
    y_va = y_basis_va[valid_va]

    # Current baseline
    CURRENT_UK = {
        "loss_function": "MAE", "eval_metric": "RMSE",
        "iterations": 15000, "learning_rate": 0.03, "depth": 8,
        "l2_leaf_reg": 5, "colsample_bylevel": 0.8, "subsample": 0.8,
        "random_seed": 42, "verbose": 0,
        "allow_writing_files": False, "use_best_model": True,
    }
    cb_base = train_tree("catboost", CURRENT_UK, X_tr, y_tr, X_va, y_va)
    pred_base = uk_moc_va + predict_tree(cb_base.model, df_val[feat_uk])
    _, base_hbc = compute_hbc(pred_base, uk_spot_va, hours_va)
    base_rmse = compute_rmse(uk_spot_va, pred_base)
    print(f"\n  UK baseline: RMSE={base_rmse:.2f}  HBC={base_hbc:.2f}  iter={cb_base.best_iteration}")

    def objective(trial):
        loss = trial.suggest_categorical("loss_function", ["RMSE", "MAE", "Huber:delta=10", "Huber:delta=20"])
        params = {
            "loss_function": loss,
            "eval_metric": "RMSE",
            "iterations": 15000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 0.8),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "random_strength": trial.suggest_float("random_strength", 0.1, 5.0),
            "random_seed": 42,
            "verbose": 0,
            "allow_writing_files": False,
            "use_best_model": True,
        }
        try:
            res = train_tree("catboost", params, X_tr, y_tr, X_va, y_va)
            pred = uk_moc_va + predict_tree(res.model, df_val[feat_uk])
            _, hbc = compute_hbc(pred, uk_spot_va, hours_va)
            return hbc
        except Exception:
            return 999.0

    storage = "sqlite:///data/outputs/optuna_retune.db"
    study = optuna.create_study(direction="minimize", study_name="uk_catboost_basis",
                                storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\n  UK BEST: HBC={best.value:.2f}  (baseline={base_hbc:.2f}  delta={best.value - base_hbc:+.2f})")
    print(f"  Params: {best.params}")

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
    cb_best = train_tree("catboost", best_params, X_tr, y_tr, X_va, y_va)
    pred_best = uk_moc_va + predict_tree(cb_best.model, df_val[feat_uk])
    rmse_best = compute_rmse(uk_spot_va, pred_best)
    _, hbc_best = compute_hbc(pred_best, uk_spot_va, hours_va)
    print(f"  Verified: RMSE={rmse_best:.2f}  HBC={hbc_best:.2f}  iter={cb_best.best_iteration}")

    return {"params": best_params, "hbc": hbc_best, "rmse": rmse_best,
            "best_iteration": cb_best.best_iteration, "baseline_hbc": base_hbc}


# =========================================================================
# Run
# =========================================================================
print("=" * 70)
print(f"  OPTUNA RE-TUNE -- CatBoost for STL/Basis targets ({args.trials} trials)")
print("=" * 70)

results = {}
if not args.uk_only:
    print("\n>>> FR CatBoost (STL target, feat_fr_28)")
    results["fr"] = tune_fr(args.trials)

if not args.fr_only:
    print("\n>>> UK CatBoost (basis target, feat_uk_confirmed)")
    results["uk"] = tune_uk(args.trials)

# -- Summary ---------------------------------------------------------------
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
for mkt, r in results.items():
    delta = r["hbc"] - r["baseline_hbc"]
    print(f"  {mkt.upper()}: baseline HBC={r['baseline_hbc']:.2f} -> best HBC={r['hbc']:.2f}  (delta={delta:+.2f})")
    print(f"        loss={r['params']['loss_function']}  depth={r['params']['depth']}  lr={r['params']['learning_rate']:.4f}")
    print(f"        l2={r['params']['l2_leaf_reg']:.2f}  sub={r['params']['subsample']:.3f}  col={r['params']['colsample_bylevel']:.3f}")

# Save results
out = {mkt: {
    "params": {k: v for k, v in r["params"].items() if k not in ("verbose", "allow_writing_files", "use_best_model")},
    "hbc": round(r["hbc"], 4),
    "rmse": round(r["rmse"], 4),
    "best_iteration": r["best_iteration"],
    "baseline_hbc": round(r["baseline_hbc"], 4),
} for mkt, r in results.items()}

with open("data/outputs/optuna_catboost_retune.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Results saved to data/outputs/optuna_catboost_retune.json")
