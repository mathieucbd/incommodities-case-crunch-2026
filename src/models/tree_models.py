import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ensure src is in standard path for execution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_ingestion import load_and_merge_zone
from src.features import build_features
from src.preprocessing import chronological_train_val_test_split
from src.evaluation.metrics import MAE, sMAPE, rMAE, save_metrics_to_csv
from src.constants import TARGET_COL

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

INT_PARAMS = [
    "num_leaves",
    "max_depth",
    "n_estimators",
    "min_child_weight",
    "min_samples_leaf",
    "min_samples_split",
    "batch_size",
    "depth",
]


def sanitize_int_params(params: dict) -> dict:
    for param in INT_PARAMS:
        if param in params and params[param] is not None:
            params[param] = int(params[param])
    return params


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict = None,
):
    """Trains a LightGBM Regressor utilizing early stopping against the validation set."""
    if params is None:
        params = {}

    fit_params = {}
    if "early_stopping_rounds" in params:
        es_rounds = params.pop("early_stopping_rounds")
        try:
            # Modern LightGBM requires passing early stopping via callbacks list
            fit_params["callbacks"] = [
                lgb.early_stopping(stopping_rounds=es_rounds, verbose=False)
            ]
        except AttributeError:
            # Legacy LightGBM uses explicit fit parameter
            fit_params["early_stopping_rounds"] = es_rounds

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **fit_params)
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict = None,
):
    """Trains an XGBoost Regressor utilizing early stopping against the validation set."""
    if params is None:
        params = {}

    # Utilizing safe extractor preventing conflict against older XBG matrices
    fit_params = {"verbose": False}
    if "early_stopping_rounds" in params:
        # Latest XBG allows `early_stopping_rounds` internally mapping straight into constructing node
        # But we ensure we protect against dual declarations
        pass

    params.setdefault("tree_method", "hist")

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **fit_params)
    return model


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict = None,
    zone: str = "default",
):
    """Trains a CatBoost Regressor mapping explicit categorical features perfectly without one-hot requirements."""
    if params is None:
        params = {}

    params["train_dir"] = f"data/outputs/catboost/{zone}"
    params.setdefault("thread_count", -1)

    model = CatBoostRegressor(verbose=False, **params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, params: dict = None):
    """Trains a RandomForest Regressor strictly on Training arrays (Internal Bagging eliminates Val mapping need)."""
    if params is None:
        params = {}

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    zone: str, model_name: str, model, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.Series:
    """Orchestrates test generation and routes metrics mapping identically to terminal validation outputs."""
    logger.info(f"--- Evaluating {model_name} ---")

    preds_raw = model.predict(X_test)
    preds = pd.Series(preds_raw, index=X_test.index)

    valid_mask = ~preds.isna()
    y_t = y_test.loc[valid_mask]
    p_t = preds.loc[valid_mask]

    mae_score = MAE(y_t, p_t)
    smape_score = sMAPE(y_t, p_t) * 100
    rmae_score = rMAE(y_t, p_t, m="W")

    logger.info(f"[{model_name}] MAE:   {mae_score:.3f} EUR/MWh")
    logger.info(f"[{model_name}] sMAPE: {smape_score:.3f} %")
    logger.info(f"[{model_name}] rMAE:  {rmae_score:.3f}")

    save_metrics_to_csv(
        zone=zone,
        model_name=model_name,
        metrics_dict={"MAE": mae_score, "sMAPE": smape_score, "rMAE": rmae_score},
    )

    return preds


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_directory = config.get("data", {}).get("raw_dir", "data/raw/auhack_legacy/")

    val_pred_dir = Path("data/outputs/predictions/val")
    test_pred_dir = Path("data/outputs/predictions/test")
    val_pred_dir.mkdir(parents=True, exist_ok=True)
    test_pred_dir.mkdir(parents=True, exist_ok=True)

    base_lgb_params = (
        config.get("model_settings", {}).get("trees", {}).get("lgb", {}).copy()
    )
    base_xgb_params = (
        config.get("model_settings", {}).get("trees", {}).get("xgb", {}).copy()
    )
    base_cat_params = (
        config.get("model_settings", {}).get("trees", {}).get("cat", {}).copy()
    )
    base_rf_params = (
        config.get("model_settings", {}).get("trees", {}).get("rf", {}).copy()
    )

    try:
        with open("best_hyperparameters.yaml", "r") as f:
            best_hyperparams = yaml.safe_load(f) or {}
    except FileNotFoundError:
        best_hyperparams = {}
        logger.warning(
            "best_hyperparameters.yaml not found; falling back to default config.yaml parameters."
        )

    target_zones = config.get("data", {}).get("target_zones", ["DE"])
    flow_only_zones = config["data"].get("flow_only_zones", [])
    all_zones = target_zones + flow_only_zones
    raw_data_dict = {z: load_and_merge_zone(z, raw_directory) for z in all_zones}

    missing_targets = [z for z in target_zones if z not in raw_data_dict]
    if missing_targets:
        raise ValueError(
            f"Missing required target zones in preloaded raw_data_dict: {missing_targets}"
        )

    val_preds_dict_lgbm = {}
    val_preds_dict_xgb = {}
    val_preds_dict_cat = {}
    val_preds_dict_rf = {}
    test_preds_dict_lgbm = {}
    test_preds_dict_xgb = {}
    test_preds_dict_cat = {}
    test_preds_dict_rf = {}

    zone_mae_lgbm = {}
    zone_mae_xgb = {}
    zone_mae_cat = {}
    zone_mae_rf = {}

    for target_zone in target_zones:
        logger.info(f"========================================")
        logger.info(f"Loading Tree Evaluation Pipeline natively for {target_zone}...")

        df, active_features = build_features(
            raw_data_dict, target_zone, lag_actual_flows=True
        )

        # 2. Chronological Subsets Extracting Scaling Limits
        # Trees operate utilizing relative branching nodes meaning variance Standardization is useless computational overhead.
        train_df, val_df, test_df = chronological_train_val_test_split(
            df, val_ratio=0.15, test_ratio=0.15
        )

        X_train = train_df[active_features]
        y_train = train_df[TARGET_COL]
        X_val = val_df[active_features]
        y_val = val_df[TARGET_COL]
        X_test = test_df[active_features]
        y_test = test_df[TARGET_COL]

        logger.info(
            "Executing Train / Val / Test (Tree matrices bypassing Standard Scaler)..."
        )
        logger.info("========================================")

        # 3. Model Orchestration
        # LightGBM
        lgb_params_zone = (
            best_hyperparams.get("LightGBM", {}).get(target_zone, base_lgb_params)
            or base_lgb_params
        ).copy()
        lgb_params_zone = sanitize_int_params(lgb_params_zone)
        lgb_params_zone.setdefault("early_stopping_rounds", 50)
        logger.info("Training LightGBM (Optimized Config)...")
        lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, lgb_params_zone)
        val_preds_lgbm = pd.Series(lgb_model.predict(X_val), index=X_val.index)
        test_preds_lgbm = evaluate_model(
            target_zone, "LightGBM", lgb_model, X_test, y_test
        )
        val_preds_dict_lgbm[target_zone] = val_preds_lgbm
        test_preds_dict_lgbm[target_zone] = test_preds_lgbm
        zone_mae_lgbm[target_zone] = MAE(y_test, test_preds_lgbm)

        # XGBoost
        xgb_params_zone = (
            best_hyperparams.get("XGBoost", {}).get(target_zone, base_xgb_params)
            or base_xgb_params
        ).copy()
        xgb_params_zone = sanitize_int_params(xgb_params_zone)
        xgb_params_zone.setdefault("early_stopping_rounds", 50)
        logger.info("Training XGBoost (Optimized Config)...")
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val, xgb_params_zone)
        val_preds_xgb = pd.Series(xgb_model.predict(X_val), index=X_val.index)
        test_preds_xgb = evaluate_model(
            target_zone, "XGBoost", xgb_model, X_test, y_test
        )
        val_preds_dict_xgb[target_zone] = val_preds_xgb
        test_preds_dict_xgb[target_zone] = test_preds_xgb
        zone_mae_xgb[target_zone] = MAE(y_test, test_preds_xgb)

        # CatBoost
        cat_params_zone = (
            best_hyperparams.get("CatBoost", {}).get(target_zone, base_cat_params)
            or base_cat_params
        ).copy()
        cat_params_zone = sanitize_int_params(cat_params_zone)
        cat_params_zone.setdefault("early_stopping_rounds", 50)
        logger.info("Training CatBoost (Optimized Config)...")
        cat_model = train_catboost(
            X_train, y_train, X_val, y_val, cat_params_zone, zone=target_zone
        )
        val_preds_cat = pd.Series(cat_model.predict(X_val), index=X_val.index)
        test_preds_cat = evaluate_model(
            target_zone, "CatBoost", cat_model, X_test, y_test
        )
        val_preds_dict_cat[target_zone] = val_preds_cat
        test_preds_dict_cat[target_zone] = test_preds_cat
        zone_mae_cat[target_zone] = MAE(y_test, test_preds_cat)

        # RandomForest
        rf_params_zone = (
            best_hyperparams.get("RandomForest", {}).get(target_zone, base_rf_params)
            or base_rf_params
        ).copy()
        rf_params_zone = sanitize_int_params(rf_params_zone)
        logger.info("Training RandomForest (Optimized Config)...")
        rf_model = train_random_forest(X_train, y_train, rf_params_zone)
        val_preds_rf = pd.Series(rf_model.predict(X_val), index=X_val.index)
        test_preds_rf = evaluate_model(
            target_zone, "RandomForest", rf_model, X_test, y_test
        )
        val_preds_dict_rf[target_zone] = val_preds_rf
        test_preds_dict_rf[target_zone] = test_preds_rf
        zone_mae_rf[target_zone] = MAE(y_test, test_preds_rf)

        logger.info("========================================")

    # Save multi-zone predictions as DataFrames
    pd.DataFrame(val_preds_dict_lgbm).to_csv(val_pred_dir / "lgbm.csv")
    pd.DataFrame(test_preds_dict_lgbm).to_csv(test_pred_dir / "lgbm.csv")
    pd.DataFrame(val_preds_dict_xgb).to_csv(val_pred_dir / "xgb.csv")
    pd.DataFrame(test_preds_dict_xgb).to_csv(test_pred_dir / "xgb.csv")
    pd.DataFrame(val_preds_dict_cat).to_csv(val_pred_dir / "cat.csv")
    pd.DataFrame(test_preds_dict_cat).to_csv(test_pred_dir / "cat.csv")
    pd.DataFrame(val_preds_dict_rf).to_csv(val_pred_dir / "rf.csv")
    pd.DataFrame(test_preds_dict_rf).to_csv(test_pred_dir / "rf.csv")

    # Macro-averages
    logger.info("\n========== MACRO AVERAGES ACROSS ZONES ==========")
    if zone_mae_lgbm:
        avg_mae_lgbm = np.mean(list(zone_mae_lgbm.values()))
        logger.info(f"[LightGBM] Avg MAE: {avg_mae_lgbm:.3f} EUR/MWh")
    if zone_mae_xgb:
        avg_mae_xgb = np.mean(list(zone_mae_xgb.values()))
        logger.info(f"[XGBoost] Avg MAE: {avg_mae_xgb:.3f} EUR/MWh")
    if zone_mae_cat:
        avg_mae_cat = np.mean(list(zone_mae_cat.values()))
        logger.info(f"[CatBoost] Avg MAE: {avg_mae_cat:.3f} EUR/MWh")
    if zone_mae_rf:
        avg_mae_rf = np.mean(list(zone_mae_rf.values()))
        logger.info(f"[RandomForest] Avg MAE: {avg_mae_rf:.3f} EUR/MWh")
    logger.info("================================================")
