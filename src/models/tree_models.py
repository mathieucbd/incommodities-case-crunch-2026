import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

from src.constants import TARGET_COLS
from src.data_ingestion import load_competition_data
from src.evaluation.metrics import MAE, RMSE, sMAPE, save_metrics_to_csv
from src.features import apply_full_feature_engineering
from src.preprocessing import chronological_train_val_test_split

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


def cast_tree_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["hour", "day_of_week", "dayofweek", "month"]:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def prepare_sklearn_tree_matrix(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in out.columns:
        if str(out[col].dtype) == "category":
            out[col] = out[col].astype(float)
    return out


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict | None = None,
):
    if params is None:
        params = {}

    fit_params = {}
    es_rounds = params.pop("early_stopping_rounds", None)
    if es_rounds is not None:
        fit_params["callbacks"] = [
            lgb.early_stopping(stopping_rounds=int(es_rounds), verbose=False)
        ]

    params.setdefault("objective", "regression")
    params.setdefault("use_missing", True)
    params.setdefault("random_state", 42)

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature="auto",
        **fit_params,
    )
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict | None = None,
):
    if params is None:
        params = {}

    es_rounds = params.pop("early_stopping_rounds", None)
    if es_rounds is not None:
        params["early_stopping_rounds"] = int(es_rounds)

    params.setdefault("objective", "reg:squarederror")
    params.setdefault("tree_method", "hist")
    params.setdefault("enable_categorical", True)
    params.setdefault("random_state", 42)

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict | None = None,
    target_col: str = "default",
):
    if params is None:
        params = {}

    train_dir = Path("data/outputs/catboost") / target_col
    if train_dir.exists() and not train_dir.is_dir():
        raise ValueError(
            f"CatBoost train_dir exists and is not a directory: {train_dir}"
        )
    train_dir.mkdir(parents=True, exist_ok=True)

    params["train_dir"] = str(train_dir)
    params.setdefault("loss_function", "RMSE")
    params.setdefault("thread_count", -1)
    params.setdefault("random_seed", 42)

    cat_features = [
        col for col in X_train.columns if str(X_train[col].dtype) == "category"
    ]
    model = CatBoostRegressor(verbose=False, **params)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
    )
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
):
    if params is None:
        params = {}

    params.setdefault("random_state", 42)
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    target_col: str,
    model_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> None:
    valid_mask = ~y_pred.isna()
    y_t = y_true.loc[valid_mask]
    p_t = y_pred.loc[valid_mask]

    if y_t.empty:
        logger.warning(
            "[%s - %s] No rows for metric evaluation", model_name, target_col
        )
        return

    mae_score = MAE(y_t, p_t)
    rmse_score = RMSE(y_t, p_t)
    smape_score = sMAPE(y_t, p_t) * 100

    logger.info("[%s - %s] MAE: %.3f EUR/MWh", model_name, target_col, mae_score)
    logger.info("[%s - %s] RMSE: %.3f EUR/MWh", model_name, target_col, rmse_score)
    logger.info("[%s - %s] sMAPE: %.3f %%", model_name, target_col, smape_score)

    save_metrics_to_csv(
        zone=target_col,
        model_name=f"{model_name} (val)",
        metrics_dict={"MAE": mae_score, "RMSE": rmse_score, "sMAPE": smape_score},
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    split_cfg = config.get("data", {}).get("splits", {})
    val_start = split_cfg.get("val_start")
    test_start = split_cfg.get("test_start")
    if val_start is None or test_start is None:
        raise ValueError("Missing data.splits.val_start or data.splits.test_start")

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
        with open("best_hyperparameters.yaml", "r", encoding="utf-8") as f:
            best_hyperparams = yaml.safe_load(f) or {}
    except FileNotFoundError:
        best_hyperparams = {}
        logger.warning("best_hyperparameters.yaml not found; using config defaults")

    logger.info("Loading Kaggle training data...")
    df_train = load_competition_data(mode="train")
    df_train = apply_full_feature_engineering(df_train)
    df_train = cast_tree_categoricals(df_train)

    target_cols = [c for c in TARGET_COLS if c in df_train.columns]
    if not target_cols:
        raise ValueError("No configured target columns found in training dataframe")

    feature_cols = [c for c in df_train.columns if c not in target_cols]

    logger.info("Applying chronological train/val/test split...")
    train_df, val_df, test_df = chronological_train_val_test_split(
        df_train,
        val_start=val_start,
        test_start=test_start,
    )
    logger.info(
        "Split sizes (from labeled train data): train=%d, val=%d, test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]

    models_by_target: dict[str, dict[str, Any]] = {col: {} for col in target_cols}
    val_preds_by_model: dict[str, dict[str, pd.Series]] = {
        "lgbm": {},
        "xgb": {},
        "cat": {},
        "rf": {},
    }

    for target_col in TARGET_COLS:
        if target_col not in target_cols:
            logger.warning("Skipping missing target column: %s", target_col)
            continue

        logger.info("========================================")
        logger.info("Training tree models for target: %s", target_col)

        y_train = train_df[target_col]
        y_val = val_df[target_col]

        lgb_params = (
            best_hyperparams.get("LightGBM", {}).get(target_col, base_lgb_params)
            or base_lgb_params
        ).copy()
        lgb_params = sanitize_int_params(lgb_params)
        lgb_params.setdefault("early_stopping_rounds", 50)
        lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, lgb_params)
        models_by_target[target_col]["lgbm"] = lgb_model
        val_preds_lgbm = pd.Series(
            np.asarray(lgb_model.predict(X_val)).reshape(-1), index=X_val.index
        )
        val_preds_by_model["lgbm"][target_col] = val_preds_lgbm
        evaluate_model(target_col, "LightGBM", y_val, val_preds_lgbm)

        xgb_params = (
            best_hyperparams.get("XGBoost", {}).get(target_col, base_xgb_params)
            or base_xgb_params
        ).copy()
        xgb_params = sanitize_int_params(xgb_params)
        xgb_params.setdefault("early_stopping_rounds", 50)
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val, xgb_params)
        models_by_target[target_col]["xgb"] = xgb_model
        val_preds_xgb = pd.Series(
            np.asarray(xgb_model.predict(X_val)).reshape(-1), index=X_val.index
        )
        val_preds_by_model["xgb"][target_col] = val_preds_xgb
        evaluate_model(target_col, "XGBoost", y_val, val_preds_xgb)

        cat_params = (
            best_hyperparams.get("CatBoost", {}).get(target_col, base_cat_params)
            or base_cat_params
        ).copy()
        cat_params = sanitize_int_params(cat_params)
        cat_params.setdefault("early_stopping_rounds", 50)
        cat_model = train_catboost(
            X_train, y_train, X_val, y_val, cat_params, target_col=target_col
        )
        models_by_target[target_col]["cat"] = cat_model
        val_preds_cat = pd.Series(
            np.asarray(cat_model.predict(X_val)).reshape(-1), index=X_val.index
        )
        val_preds_by_model["cat"][target_col] = val_preds_cat
        evaluate_model(target_col, "CatBoost", y_val, val_preds_cat)

        rf_params = (
            best_hyperparams.get("RandomForest", {}).get(target_col, base_rf_params)
            or base_rf_params
        ).copy()
        rf_params = sanitize_int_params(rf_params)
        X_train_rf = prepare_sklearn_tree_matrix(X_train)
        X_val_rf = prepare_sklearn_tree_matrix(X_val)
        rf_model = train_random_forest(X_train_rf, y_train, rf_params)
        models_by_target[target_col]["rf"] = rf_model
        val_preds_rf = pd.Series(
            np.asarray(rf_model.predict(X_val_rf)).reshape(-1), index=X_val.index
        )
        val_preds_by_model["rf"][target_col] = val_preds_rf
        evaluate_model(target_col, "RandomForest", y_val, val_preds_rf)

    pd.DataFrame(val_preds_by_model["lgbm"]).to_csv(val_pred_dir / "lgbm.csv")
    pd.DataFrame(val_preds_by_model["xgb"]).to_csv(val_pred_dir / "xgb.csv")
    pd.DataFrame(val_preds_by_model["cat"]).to_csv(val_pred_dir / "cat.csv")
    pd.DataFrame(val_preds_by_model["rf"]).to_csv(val_pred_dir / "rf.csv")

    # Kaggle submission block: predict on x_test and save model-wise files.
    logger.info("Loading Kaggle test features...")
    df_test = load_competition_data(mode="test")
    df_test = apply_full_feature_engineering(df_test)
    df_test = cast_tree_categoricals(df_test)

    missing_test_features = [c for c in feature_cols if c not in df_test.columns]
    if missing_test_features:
        logger.warning(
            "Adding %d missing test features as NaN for schema alignment",
            len(missing_test_features),
        )
        for col in missing_test_features:
            df_test[col] = np.nan

    X_kaggle = df_test[feature_cols]
    X_kaggle_rf = prepare_sklearn_tree_matrix(X_kaggle)

    config_abs_path = Path("config.yaml").resolve()
    raw_dir = config_abs_path.parent / config["data"]["raw_dir"]
    test_ids = pd.read_csv(
        raw_dir / config["data"]["test_features_file"], usecols=["id"]
    )["id"]
    if len(test_ids) != len(X_kaggle):
        raise ValueError(
            "Kaggle id length mismatch between raw and engineered test set"
        )

    test_preds_by_model: dict[str, dict[str, np.ndarray]] = {
        "lgbm": {},
        "xgb": {},
        "cat": {},
        "rf": {},
    }

    for target_col in TARGET_COLS:
        if target_col not in models_by_target:
            continue
        model_pack = models_by_target[target_col]
        test_preds_by_model["lgbm"][target_col] = model_pack["lgbm"].predict(X_kaggle)
        test_preds_by_model["xgb"][target_col] = model_pack["xgb"].predict(X_kaggle)
        test_preds_by_model["cat"][target_col] = model_pack["cat"].predict(X_kaggle)
        test_preds_by_model["rf"][target_col] = model_pack["rf"].predict(X_kaggle_rf)

    for model_key in ["lgbm", "xgb", "cat", "rf"]:
        out_df = pd.DataFrame(
            {col: test_preds_by_model[model_key][col] for col in target_cols},
            index=test_ids.to_numpy(),
        )
        out_df.index.name = "id"
        out_df.to_csv(test_pred_dir / f"{model_key}.csv")

    logger.info("Saved Kaggle test predictions to %s", test_pred_dir)
