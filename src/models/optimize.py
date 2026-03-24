import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import yaml

from src.constants import TARGET_COLS
from src.data_ingestion import load_competition_data
from src.evaluation.metrics import RMSE
from src.features import apply_full_feature_engineering
from src.preprocessing import chronological_train_val_test_split

logger = logging.getLogger(__name__)
_DATA: dict[str, pd.DataFrame | pd.Series] = {}


def cast_tree_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["hour", "day_of_week", "dayofweek", "month"]:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def objective(trial: optuna.Trial, target_col: str, model_type: str) -> float:
    X_train = _DATA["X_train"]
    X_val = _DATA["X_val"]
    y_train = _DATA[f"y_train_{target_col}"]
    y_val = _DATA[f"y_val_{target_col}"]

    if model_type == "lightgbm":
        params = {
            "objective": "regression",
            "use_missing": True,
            "random_state": 42,
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
            categorical_feature="auto",
        )

    elif model_type == "xgboost":
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "enable_categorical": True,
            "random_state": 42,
            "n_estimators": 1500,
            "early_stopping_rounds": 100,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    preds = pd.Series(np.asarray(model.predict(X_val)).reshape(-1), index=X_val.index)
    val_rmse = RMSE(y_val, preds)
    logger.info(
        "[%s | %s] trial=%d RMSE=%.5f",
        target_col,
        model_type,
        trial.number,
        val_rmse,
    )
    return float(val_rmse)


def _best_params_with_fixed(model_type: str, best_params: dict) -> dict:
    if model_type == "lightgbm":
        return {
            **best_params,
            "objective": "regression",
            "use_missing": True,
            "early_stopping_rounds": 100,
        }
    if model_type == "xgboost":
        return {
            **best_params,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "enable_categorical": True,
            "n_estimators": 1500,
            "early_stopping_rounds": 100,
        }
    return dict(best_params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    split_cfg = config.get("data", {}).get("splits", {})
    val_start = split_cfg.get("val_start")
    test_start = split_cfg.get("test_start")
    if val_start is None or test_start is None:
        raise ValueError("Missing data.splits.val_start or data.splits.test_start")

    logger.info("Loading Kaggle training data...")
    df_train = load_competition_data(mode="train")
    df_train = apply_full_feature_engineering(df_train)
    df_train = cast_tree_categoricals(df_train)

    train_df, val_df, _ = chronological_train_val_test_split(
        df_train,
        val_start=val_start,
        test_start=test_start,
    )

    target_cols = [c for c in TARGET_COLS if c in df_train.columns]
    if not target_cols:
        raise ValueError("No configured target columns found in training dataframe")

    feature_cols = [c for c in df_train.columns if c not in target_cols]
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    _DATA["X_train"] = X_train
    _DATA["X_val"] = X_val

    best_cfg = {}
    model_types = ["lightgbm", "xgboost"]

    for target_col in target_cols:
        y_train = train_df[target_col]
        y_val = val_df[target_col]
        _DATA[f"y_train_{target_col}"] = y_train
        _DATA[f"y_val_{target_col}"] = y_val

        for model_type in model_types:
            logger.info("Running Optuna: target=%s model=%s", target_col, model_type)
            study = optuna.create_study(direction="minimize")
            study.optimize(
                lambda trial: objective(trial, target_col, model_type),
                n_trials=50,
                show_progress_bar=False,
            )

            key = f"{target_col}_{model_type}"
            best_cfg[key] = _best_params_with_fixed(model_type, study.best_params)
            logger.info(
                "Best for %s: RMSE=%.5f params=%s",
                key,
                study.best_value,
                best_cfg[key],
            )

    out_path = Path("best_hyperparameters.yaml")
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=True)

    logger.info("Saved best hyperparameters to %s", out_path)
