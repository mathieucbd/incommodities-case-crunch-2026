"""
Exhaustive Bayesian Hyperparameter Optimization with Checkpointing.
Uses the Tree-structured Parzen Estimator (TPE) algorithm via hyperopt.
Supports fault-tolerant resuming via pickle-based Trials checkpointing.
"""

import logging
import pickle
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
from sklearn.exceptions import ConvergenceWarning
import torch
from sklearn.linear_model import LassoLarsIC

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Suppress convergence warnings from coordinate descent with large feature matrices
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from lightgbm import LGBMRegressor

from src.data_ingestion import load_and_merge_zone
from src.features import build_features
from src.preprocessing import chronological_train_val_test_split, scale_data
from src.constants import TARGET_COL
from src.evaluation.metrics import MAE
from src.evaluation.probabilistic import pinball_loss

from src.models.tree_models import (
    train_lightgbm,
    train_xgboost,
    train_catboost,
    train_random_forest,
)
from src.models.deep_learning import reshape_to_daily, train_pytorch_dnn
from src.models.baselines import predict_lear as train_lear

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global data container — populated once in __main__ before fmin loops
# ---------------------------------------------------------------------------
_D = {}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _load_trials(path: str) -> Trials:
    """Load existing Trials from disk, or return a fresh one."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            trials = pickle.load(f)
        logger.info(
            f"[Checkpoint] Resuming from {path} ({len(trials.trials)} completed trials)."
        )
    else:
        trials = Trials()
        logger.info(f"[Checkpoint] No checkpoint at {path}. Starting fresh.")
    return trials


def _save_trials(trials: Trials, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(trials, f)


def _load_prediction_matrix_zone(folder: Path, zone: str) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction CSV files found in {folder}.")

    series_list = []
    for file_path in files:
        pred_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        if pred_df.empty:
            continue
        if zone in pred_df.columns:
            pred_s = pred_df[zone]
            pred_s.name = file_path.stem.lower()
            series_list.append(pred_s)

    if not series_list:
        raise ValueError(f"No usable prediction series found for {zone} in {folder}.")

    return pd.concat(series_list, axis=1, sort=False).sort_index()


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------


def objective_lgb(params):
    seed = _D["seed"]
    p = {
        "n_estimators": int(params["n_estimators"]),
        "learning_rate": params["learning_rate"],
        "num_leaves": int(params["num_leaves"]),
        "colsample_bytree": params["colsample_bytree"],
        "subsample": params["subsample"],
        "reg_alpha": params["reg_alpha"],
        "reg_lambda": params["reg_lambda"],
        "early_stopping_rounds": 50,
        "random_state": seed,
        "n_jobs": -1,
    }
    model = train_lightgbm(_D["X_tr"], _D["y_tr"], _D["X_va"], _D["y_va"], params=p)
    val_mae = MAE(
        _D["y_va"], pd.Series(model.predict(_D["X_va"]), index=_D["X_va"].index)
    )
    _save_trials(_D["trials_lgb"], _D["ckpt_lgb"])
    return {"loss": val_mae, "status": STATUS_OK}


def objective_xgb(params):
    seed = _D["seed"]
    p = {
        "n_estimators": int(params["n_estimators"]),
        "learning_rate": params["learning_rate"],
        "max_depth": int(params["max_depth"]),
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "min_child_weight": int(params["min_child_weight"]),
        "reg_alpha": params["reg_alpha"],
        "reg_lambda": params["reg_lambda"],
        "early_stopping_rounds": 50,
        "random_state": seed,
        "n_jobs": -1,
    }
    model = train_xgboost(_D["X_tr"], _D["y_tr"], _D["X_va"], _D["y_va"], params=p)
    val_mae = MAE(
        _D["y_va"], pd.Series(model.predict(_D["X_va"]), index=_D["X_va"].index)
    )
    _save_trials(_D["trials_xgb"], _D["ckpt_xgb"])
    return {"loss": val_mae, "status": STATUS_OK}


def objective_cat(params):
    seed = _D["seed"]
    p = {
        "n_estimators": int(params["n_estimators"]),
        "learning_rate": params["learning_rate"],
        "depth": int(params["depth"]),
        "l2_leaf_reg": params["l2_leaf_reg"],
        "random_strength": params["random_strength"],
        "early_stopping_rounds": 50,
        "random_state": seed,
        "train_dir": "data/outputs/catboost/optimize",
    }
    model = train_catboost(_D["X_tr"], _D["y_tr"], _D["X_va"], _D["y_va"], params=p)
    val_mae = MAE(
        _D["y_va"], pd.Series(model.predict(_D["X_va"]), index=_D["X_va"].index)
    )
    _save_trials(_D["trials_cat"], _D["ckpt_cat"])
    return {"loss": val_mae, "status": STATUS_OK}


def objective_rf(params):
    seed = _D["seed"]
    p = {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "min_samples_split": int(params["min_samples_split"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "random_state": seed,
        "n_jobs": -1,
    }
    model = train_random_forest(_D["X_tr"], _D["y_tr"], params=p)
    val_mae = MAE(
        _D["y_va"], pd.Series(model.predict(_D["X_va"]), index=_D["X_va"].index)
    )
    _save_trials(_D["trials_rf"], _D["ckpt_rf"])
    return {"loss": val_mae, "status": STATUS_OK}


def objective_dnn(params):
    seed = _D["seed"]
    p = {
        "lr": params["lr"],
        "dropout_rate": params["dropout_rate"],
        "weight_decay": params["weight_decay"],
        "batch_size": int(params["batch_size"]),
        "epochs": 150,
        "patience": 15,
        "seed": seed,
    }

    model, device = train_pytorch_dnn(
        _D["X_tr_d"], _D["y_tr_d"], _D["X_va_d"], _D["y_va_d"], params=p
    )

    # Evaluate on (scaled) validation daily blocks → compute MAE in original scale
    model.eval()
    with torch.no_grad():
        preds_scaled = (
            model(torch.tensor(_D["X_va_d"]).to(device)).cpu().numpy().flatten()
        )

    preds_unscaled = (
        _D["y_scaler"].inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    )

    # Re-align to valid (full-day) val indices
    y_va_raw = _D["y_va_raw"]
    valid_idx = []
    for date, grp in pd.DataFrame(
        {"T": y_va_raw.values, "D": y_va_raw.index.date}, index=y_va_raw.index
    ).groupby("D"):
        if len(grp) == 24:
            valid_idx.extend(grp.index)

    val_mae = MAE(y_va_raw.loc[valid_idx], pd.Series(preds_unscaled, index=valid_idx))
    _save_trials(_D["trials_dnn"], _D["ckpt_dnn"])

    if np.isnan(val_mae) or np.isinf(val_mae):
        return {'loss': 99999.0, 'status': STATUS_OK}

    return {"loss": val_mae, "status": STATUS_OK}


def _predict_lear_with_alpha(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    test_indices: pd.DatetimeIndex,
    calibration_window_days: int,
) -> pd.Series:
    """Equivalent LEAR-like hourly rolling calibration using Lasso(alpha)."""
    unique_test_days = np.unique(test_indices.date)
    predictions = pd.Series(index=test_indices, dtype=float)

    for current_date in unique_test_days:
        current_date_ts = pd.Timestamp(
            current_date, tz=test_indices.tz if hasattr(test_indices, "tz") else "UTC"
        )
        window_start = current_date_ts - pd.Timedelta(days=calibration_window_days)

        mask_train = (X_full.index >= window_start) & (X_full.index < current_date_ts)
        X_calib = X_full.loc[mask_train]
        y_calib = y_full.loc[mask_train]

        mask_test = test_indices.date == current_date
        day_test_indices = test_indices[mask_test]
        X_test_day = X_full.loc[day_test_indices]

        for hour in range(24):
            hour_mask_calib = X_calib.index.hour == hour
            if not hour_mask_calib.any():
                continue

            X_calib_h = X_calib.loc[hour_mask_calib]
            y_calib_h = y_calib.loc[hour_mask_calib]

            hour_mask_test = X_test_day.index.hour == hour
            if not hour_mask_test.any():
                continue

            X_test_h = X_test_day.loc[hour_mask_test]

            noise_var = float(np.var(y_calib_h)) if len(y_calib_h) <= X_calib_h.shape[1] else None
            model = LassoLarsIC(criterion='bic', max_iter=1000, noise_variance=noise_var)
            try:
                model.fit(X_calib_h, y_calib_h)
                preds = model.predict(X_test_h)
                predictions.loc[X_test_h.index] = preds
            except Exception:
                predictions.loc[X_test_h.index] = y_calib_h.mean()

    return predictions


def objective_lear(params):
    calibration_window = int(config.get("model_settings", {}).get("lear", {}).get("calibration_window", 56))

    X_full = pd.concat([_D["X_tr"], _D["X_va"]]).sort_index()
    y_full = pd.concat([_D["y_tr"], _D["y_va"]]).sort_index()
    val_idx = _D["X_va"].index

    try:
        # Keep compatibility with baseline LEAR implementation if signature is extended.
        val_preds = train_lear(
            X_full,
            y_full,
            val_idx,
            calibration_window_days=calibration_window,
        )
    except TypeError:
        # Fallback equivalent LEAR implementation with tunable alpha.
        val_preds = _predict_lear_with_alpha(
            X_full,
            y_full,
            val_idx,
            calibration_window_days=calibration_window,
        )

    valid_mask = ~val_preds.isna()
    val_mae = MAE(_D["y_va"].loc[valid_mask], val_preds.loc[valid_mask])
    _save_trials(_D["trials_lear"], _D["ckpt_lear"])
    return {"loss": val_mae, "status": STATUS_OK}


def objective_qra(params):
    p = {
        "objective": "quantile",
        "alpha": 0.5,
        "n_estimators": int(params["n_estimators"]),
        "learning_rate": params["learning_rate"],
        "num_leaves": int(params["num_leaves"]),
        "colsample_bytree": params["colsample_bytree"],
        "subsample": params["subsample"],
        "reg_alpha": params["reg_alpha"],
        "reg_lambda": params["reg_lambda"],
        "random_state": _D["seed"],
        "verbose": -1,
    }

    model = LGBMRegressor(**p)
    model.fit(_D["X_qra_va"], _D["y_va_aligned"])
    qra_pred = model.predict(_D["X_qra_va"])
    val_pinball = pinball_loss(_D["y_va_aligned"], qra_pred, 0.5)
    _save_trials(_D["trials_qra"], _D["ckpt_qra"])
    return {"loss": val_pinball, "status": STATUS_OK}


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

SPACE_LGB = {
    "n_estimators": hp.choice("lgb_n_est", [500, 1000, 1500]),
    "learning_rate": hp.loguniform("lgb_lr", np.log(0.001), np.log(0.1)),
    "num_leaves": hp.quniform("lgb_leaves", 20, 256, 1),
    "colsample_bytree": hp.uniform("lgb_col", 0.5, 1.0),
    "subsample": hp.uniform("lgb_sub", 0.5, 1.0),
    "reg_alpha": hp.loguniform("lgb_ra", np.log(1e-4), np.log(10.0)),
    "reg_lambda": hp.loguniform("lgb_rl", np.log(1e-4), np.log(10.0)),
}

SPACE_XGB = {
    "n_estimators": hp.choice("xgb_n_est", [500, 1000, 1500]),
    "learning_rate": hp.loguniform("xgb_lr", np.log(0.001), np.log(0.1)),
    "max_depth": hp.quniform("xgb_depth", 3, 12, 1),
    "subsample": hp.uniform("xgb_sub", 0.5, 1.0),
    "colsample_bytree": hp.uniform("xgb_col", 0.5, 1.0),
    "min_child_weight": hp.quniform("xgb_mcw", 1, 10, 1),
    "reg_alpha": hp.loguniform("xgb_ra", np.log(1e-4), np.log(10.0)),
    "reg_lambda": hp.loguniform("xgb_rl", np.log(1e-4), np.log(10.0)),
}

SPACE_CAT = {
    "n_estimators": hp.choice("cat_n_est", [500, 1000, 1500]),
    "learning_rate": hp.loguniform("cat_lr", np.log(0.001), np.log(0.1)),
    "depth": hp.quniform("cat_depth", 4, 10, 1),
    "l2_leaf_reg": hp.loguniform("cat_l2", np.log(1), np.log(100)),
    "random_strength": hp.loguniform("cat_rs", np.log(1e-3), np.log(10.0)),
}

SPACE_RF = {
    "n_estimators": hp.choice("rf_n_est", [100, 200, 300]),
    "max_depth": hp.quniform("rf_depth", 10, 30, 1),
    "min_samples_split": hp.quniform("rf_mss", 2, 20, 1),
    "min_samples_leaf": hp.quniform("rf_msl", 1, 10, 1),
}

SPACE_DNN = {
    "lr": hp.loguniform("dnn_lr", np.log(0.0001), np.log(0.01)),
    "dropout_rate": hp.uniform("dnn_drop", 0.1, 0.5),
    "weight_decay": hp.loguniform("dnn_wd", np.log(1e-6), np.log(1e-2)),
    "batch_size": hp.choice("dnn_bs", [32, 64, 128]),
}

SPACE_LEAR = {}

SPACE_QRA = {
    "n_estimators": hp.choice("qra_n_est", [500, 1000, 1500]),
    "learning_rate": hp.loguniform("qra_lr", np.log(0.001), np.log(0.1)),
    "num_leaves": hp.quniform("qra_leaves", 20, 256, 1),
    "colsample_bytree": hp.uniform("qra_col", 0.5, 1.0),
    "subsample": hp.uniform("qra_sub", 0.5, 1.0),
    "reg_alpha": hp.loguniform("qra_ra", np.log(1e-4), np.log(10.0)),
    "reg_lambda": hp.loguniform("qra_rl", np.log(1e-4), np.log(10.0)),
}


# ---------------------------------------------------------------------------
# Execution block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    # Keep our own logger at INFO
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_dir = config["data"]["raw_dir"]
    target_zones = config["data"]["target_zones"]
    # Load only target zones (those with complete spot price, load, generation data)
    # Flow-only zones (DK2, NO2, SE4) lack spot prices and won't contribute to neighbor features
    raw_data_dict = {z: load_and_merge_zone(z, raw_dir) for z in target_zones}

    missing_targets = [z for z in target_zones if z not in raw_data_dict]
    if missing_targets:
        raise ValueError(
            f"Missing required target zones in preloaded raw_data_dict: {missing_targets}"
        )

    seed = config.get("pipeline", {}).get("global_seed", 42)
    trials_dir = (
        config.get("model_settings", {})
        .get("hyperopt", {})
        .get("trials_dir", "data/outputs/trials/")
    )
    max_evals = (
        config.get("model_settings", {}).get("hyperopt", {}).get("max_evals", 50)
    )
    aug = (
        config.get("model_settings", {})
        .get("dnn", {})
        .get("use_data_augmentation", True)
    )

    all_best_params = {}

    for zone in target_zones:
        # -------------------------------------------------------------------
        # 1. Load & feature-engineer for current zone
        # -------------------------------------------------------------------
        logger.info("========================================")
        logger.info(
            f"Loading data (zone={zone}, seed={seed}, max_evals={max_evals})..."
        )

        df, active_features = build_features(raw_data_dict, zone, lag_actual_flows=True)

        train_df, val_df, test_df = chronological_train_val_test_split(
            df, val_start=config["data"]["val_start"], test_start=config["data"]["test_start"]
        )

        # Raw (for trees)
        X_tr = train_df[active_features]
        y_tr = train_df[TARGET_COL]
        X_va = val_df[active_features]
        y_va = val_df[TARGET_COL]

        # --- HYBRID SEASONAL IMPUTATION (CAUSAL) ---
        # 1. Try same hour yesterday (t-24)
        # 2. Fallback to last known hour (t-1)
        # 3. Fallback to next known hour ONLY for row 0 edge-cases
        X_tr_clean = X_tr.fillna(X_tr.shift(24)).ffill().bfill()
        X_va_clean = X_va.fillna(X_va.shift(24)).ffill().bfill()
        test_df_clean = test_df[active_features].fillna(test_df[active_features].shift(24)).ffill().bfill()
        # -------------------------------------------

        # --- DNN FEATURE SELECTION (Reduce Noise) ---
        from sklearn.feature_selection import SelectFromModel
        from lightgbm import LGBMRegressor

        # Train a quick tree to find the most important features dynamically
        top_k = config.get("model_settings", {}).get("dnn", {}).get("top_k_features", 40)
        selector = SelectFromModel(
            LGBMRegressor(n_estimators=50, random_state=seed, n_jobs=-1), 
            max_features=top_k, 
            threshold=-np.inf, # Force it to strictly take the top_k
            prefit=False
        )
        
        # Fit on train, transform train, val, and test
        X_tr_dnn_clean = pd.DataFrame(selector.fit_transform(X_tr_clean, y_tr), index=X_tr_clean.index)
        X_va_dnn_clean = pd.DataFrame(selector.transform(X_va_clean), index=X_va_clean.index)
        test_df_dnn_clean = pd.DataFrame(selector.transform(test_df_clean), index=test_df_clean.index)
        # --------------------------------------------

        # Scaled (for DNN) - Pass the cleanly filtered matrices!
        X_tr_s, X_va_s, _, _ = scale_data(X_tr_dnn_clean, X_va_dnn_clean, test_df_dnn_clean)
        y_tr_s_df, y_va_s_df, _, y_scaler = scale_data(
            y_tr.to_frame(), y_va.to_frame(), test_df[[TARGET_COL]]
        )
        y_tr_s = y_tr_s_df[TARGET_COL]
        y_va_s = y_va_s_df[TARGET_COL]

        X_tr_d, y_tr_d_raw = reshape_to_daily(X_tr_s, y_tr_s, augment=aug)
        X_va_d, y_va_d_raw = reshape_to_daily(X_va_s, y_va_s, augment=False)

        # Sanitize targets once per zone rather than per-trial inside objective_dnn
        y_tr_d = np.nan_to_num(y_tr_d_raw, nan=np.nanmean(y_tr_d_raw)).astype(np.float32)
        y_va_d = np.nan_to_num(y_va_d_raw, nan=np.nanmean(y_va_d_raw)).astype(np.float32)

        logger.info(f"Train DNN tensors ({zone}): {X_tr_d.shape}  Val: {X_va_d.shape}")

        # -------------------------------------------------------------------
        # 2. Populate global data container and zone-specific checkpoints
        # -------------------------------------------------------------------
        zone_trials_dir = os.path.join(trials_dir, zone)
        Path(zone_trials_dir).mkdir(parents=True, exist_ok=True)

        ckpt_lgb = os.path.join(zone_trials_dir, "lgb_trials.pkl")
        ckpt_xgb = os.path.join(zone_trials_dir, "xgb_trials.pkl")
        ckpt_cat = os.path.join(zone_trials_dir, "cat_trials.pkl")
        ckpt_rf = os.path.join(zone_trials_dir, "rf_trials.pkl")
        ckpt_dnn = os.path.join(zone_trials_dir, "dnn_trials.pkl")
        ckpt_lear = os.path.join(zone_trials_dir, "lear_trials.pkl")
        ckpt_qra = os.path.join(zone_trials_dir, "qra_trials.pkl")

        trials_lgb = _load_trials(ckpt_lgb)
        trials_xgb = _load_trials(ckpt_xgb)
        trials_cat = _load_trials(ckpt_cat)
        trials_rf = _load_trials(ckpt_rf)
        trials_dnn = _load_trials(ckpt_dnn)
        trials_lear = _load_trials(ckpt_lear)
        trials_qra = _load_trials(ckpt_qra)

        _D.update(
            {
                "seed": seed,
                "X_tr": X_tr,
                "y_tr": y_tr,
                "X_va": X_va,
                "y_va": y_va,
                "X_tr_d": X_tr_d,
                "y_tr_d": y_tr_d,
                "X_va_d": X_va_d,
                "y_va_d": y_va_d,
                "y_va_raw": y_va,
                "y_scaler": y_scaler,
                "trials_lgb": trials_lgb,
                "ckpt_lgb": ckpt_lgb,
                "trials_xgb": trials_xgb,
                "ckpt_xgb": ckpt_xgb,
                "trials_cat": trials_cat,
                "ckpt_cat": ckpt_cat,
                "trials_rf": trials_rf,
                "ckpt_rf": ckpt_rf,
                "trials_dnn": trials_dnn,
                "ckpt_dnn": ckpt_dnn,
                "trials_lear": trials_lear,
                "ckpt_lear": ckpt_lear,
                "trials_qra": trials_qra,
                "ckpt_qra": ckpt_qra,
            }
        )

        # -------------------------------------------------------------------
        # 2.5 Load Prediction Lake features for QRA optimization
        # -------------------------------------------------------------------
        qra_ready = False
        try:
            val_pred_dir = Path("data/outputs/predictions/val")
            pred_val_all = _load_prediction_matrix_zone(val_pred_dir, zone)

            common_val_idx = pred_val_all.index.intersection(y_va.index)
            X_qra_va = pred_val_all.loc[common_val_idx].sort_index()
            y_va_aligned = y_va.loc[X_qra_va.index]
            val_mask = ~X_qra_va.isna().any(axis=1)
            X_qra_va = X_qra_va.loc[val_mask]
            y_va_aligned = y_va_aligned.loc[val_mask]

            if X_qra_va.empty:
                raise ValueError(
                    "No aligned validation prediction rows available for QRA."
                )

            _D["X_qra_va"] = X_qra_va
            _D["y_va_aligned"] = y_va_aligned
            qra_ready = True
        except (FileNotFoundError, ValueError):
            qra_ready = False
            logger.warning(
                f"Prediction lake missing for {zone}. Skipping QRA optimization."
            )

        # -------------------------------------------------------------------
        # 3. Run optimizations — each model picks up from zone checkpoint
        # -------------------------------------------------------------------
        models_config = [
            ("LEAR", objective_lear, SPACE_LEAR, trials_lear, ckpt_lear),
            ("LightGBM", objective_lgb, SPACE_LGB, trials_lgb, ckpt_lgb),
            ("XGBoost", objective_xgb, SPACE_XGB, trials_xgb, ckpt_xgb),
            ("CatBoost", objective_cat, SPACE_CAT, trials_cat, ckpt_cat),
            ("RandomForest", objective_rf, SPACE_RF, trials_rf, ckpt_rf),
            ("PyTorch DNN", objective_dnn, SPACE_DNN, trials_dnn, ckpt_dnn),
        ]

        if qra_ready:
            models_config.append(
                ("QRA", objective_qra, SPACE_QRA, trials_qra, ckpt_qra)
            )

        best_params = {}
        for name, obj_fn, space, trials, ckpt in models_config:
            # Asymmetric Tuning Budget
            if name == "LEAR":
                current_max_evals = 1
            elif name in ["CatBoost", "RandomForest", "PyTorch DNN"]:
                current_max_evals = min(10, max_evals)
            else:
                current_max_evals = min(15, max_evals)

            already_done = len(trials.trials)
            remaining = max(0, current_max_evals - already_done)
            logger.info(f"========================================")
            logger.info(
                f"[{zone}] Tuning {name}: {already_done}/{current_max_evals} done, running {remaining} more..."
            )

            if remaining > 0:
                fmin(
                    fn=obj_fn,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=current_max_evals,
                    trials=trials,
                )
                _save_trials(trials, ckpt)
            else:
                logger.info(
                    f"[{zone}] {name} already fully optimized (checkpoint intact)."
                )

            raw_best = space_eval(space, trials.argmin)
            best_params[name] = {
                k: round(v, 4) if isinstance(v, float) else v 
                for k, v in raw_best.items()
            }
            logger.info(f"[{zone}] [BEST] {name}: {best_params[name]}")

        all_best_params[zone] = best_params

    # -----------------------------------------------------------------------
    # 4. Final summary across all zones
    # -----------------------------------------------------------------------
    logger.info("========================================")
    logger.info("=== PAN-EUROPEAN OPTIMAL HYPERPARAMETERS ===")
    for zone, params_by_model in all_best_params.items():
        logger.info(f"\nZONE: {zone}")
        logger.info("----------------------------------------")
        for name, params in params_by_model.items():
            logger.info(f"{name:15s}: {params}")

    artifact_by_model = {}
    for zone, params_by_model in all_best_params.items():
        for model_name, params in params_by_model.items():
            artifact_by_model.setdefault(model_name, {})[zone] = params

    artifact_path = Path("best_hyperparameters.yaml")
    with open(artifact_path, "w") as f:
        yaml.dump(artifact_by_model, f, default_flow_style=False)

    logger.info(f"Saved parameter artifact: {artifact_path}")
    logger.info("========================================")
    logger.info("Copy these values into config.yaml -> model_settings!")
