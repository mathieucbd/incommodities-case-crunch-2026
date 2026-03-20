import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ensure src is in standard path for execution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_ingestion import load_and_merge_zone
from src.features import create_lags, add_deterministic_features, apply_mad_filter
from src.preprocessing import chronological_train_val_test_split
from src.evaluation.metrics import MAE, sMAPE, rMAE
from src.constants import TARGET_COL

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


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

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **fit_params)
    return model


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict = None,
):
    """Trains a CatBoost Regressor mapping explicit categorical features perfectly without one-hot requirements."""
    if params is None:
        params = {}

    potential_cats = ["Hour", "DayOfWeek", "Month"]
    cat_features = [col for col in potential_cats if col in X_train.columns]

    params["train_dir"] = "data/outputs/catboost_info"

    model = CatBoostRegressor(verbose=False, **params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features)
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, params: dict = None):
    """Trains a RandomForest Regressor strictly on Training arrays (Internal Bagging eliminates Val mapping need)."""
    if params is None:
        params = {}

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model_name: str, model, X_test: pd.DataFrame, y_test: pd.Series
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

    return preds


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_directory = config.get("data", {}).get("raw_dir", "data/raw/auhack_legacy/")

    lgb_params = config.get("model_settings", {}).get("trees", {}).get("lgb", {}).copy()
    xgb_params = config.get("model_settings", {}).get("trees", {}).get("xgb", {}).copy()
    cat_params = config.get("model_settings", {}).get("trees", {}).get("cat", {}).copy()
    rf_params = config.get("model_settings", {}).get("trees", {}).get("rf", {}).copy()

    target_zone = "DE"
    logger.info(f"========================================")
    logger.info(f"Loading Tree Evaluation Pipeline natively for {target_zone}...")

    df = load_and_merge_zone(target_zone, raw_directory)

    # 1. Pipeline Feature Alignment
    df["Spot_Price_Filtered"] = apply_mad_filter(df[TARGET_COL], window="24h", z=3.0)
    df = add_deterministic_features(df)

    lag_targets = ["Spot_Price_Filtered", "Residual_Load"]
    lags_list = [24, 48, 168]
    df = create_lags(df, lag_targets, lags_list)

    active_features = ["Hour", "DayOfWeek", "Month"]
    for col in lag_targets:
        for lag in lags_list:
            active_features.append(f"{col}_lag_{lag}")

    df = df.dropna(subset=active_features + [TARGET_COL])

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
    lgb_params["early_stopping_rounds"] = 50
    logger.info("Training LightGBM (Optimized Config)...")
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, lgb_params)
    _ = evaluate_model("LightGBM", lgb_model, X_test, y_test)

    # XGBoost
    xgb_params["early_stopping_rounds"] = 50
    logger.info("Training XGBoost (Optimized Config)...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, xgb_params)
    _ = evaluate_model("XGBoost", xgb_model, X_test, y_test)

    # CatBoost
    cat_params["early_stopping_rounds"] = 50
    logger.info("Training CatBoost (Optimized Config)...")
    cat_model = train_catboost(X_train, y_train, X_val, y_val, cat_params)
    _ = evaluate_model("CatBoost", cat_model, X_test, y_test)

    # RandomForest
    logger.info("Training RandomForest (Optimized Config)...")
    rf_model = train_random_forest(X_train, y_train, rf_params)
    _ = evaluate_model("RandomForest", rf_model, X_test, y_test)

    logger.info("========================================")
