import argparse
import logging
import os

from catboost import CatBoostRegressor
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
_YAML_PATH = "best_hyperparameters.yaml"


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

    elif model_type == "catboost":
        os.makedirs("data/outputs/catboost", exist_ok=True)
        params = {
            "loss_function": "RMSE",
            "random_seed": 42,
            "n_estimators": 1500,
            "early_stopping_rounds": 100,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "train_dir": "data/outputs/catboost",
        }
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


def save_best_params_callback(study, trial):
    if study.best_trial.number == trial.number:
        if os.path.exists(_YAML_PATH):
            with open(_YAML_PATH, "r", encoding="utf-8") as f:
                all_best_params = yaml.safe_load(f) or {}
        else:
            all_best_params = {}

        all_best_params[study.study_name] = study.best_params
        yaml_dir = os.path.dirname(_YAML_PATH)
        if yaml_dir:
            os.makedirs(yaml_dir, exist_ok=True)
        with open(_YAML_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(all_best_params, f, sort_keys=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    os.makedirs("data/outputs", exist_ok=True)
    db_url = "sqlite:///data/outputs/optuna_history.db"

    parser = argparse.ArgumentParser(
        description="Tune tree model hyperparameters with Optuna"
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["fr_spot", "uk_spot"],
        default=["fr_spot", "uk_spot"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lightgbm", "xgboost", "catboost"],
        default=["catboost"],
    )
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

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

    available_target_cols = [c for c in TARGET_COLS if c in df_train.columns]
    if not available_target_cols:
        raise ValueError("No configured target columns found in training dataframe")

    target_cols = [c for c in args.targets if c in available_target_cols]
    if not target_cols:
        raise ValueError("None of the requested --targets exist in training dataframe")

    feature_cols = [c for c in df_train.columns if c not in target_cols]
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    _DATA["X_train"] = X_train
    _DATA["X_val"] = X_val

    # Requested path was config/best_hyperparameters.yaml; adjusted to repo structure.
    yaml_path = "best_hyperparameters.yaml"
    _YAML_PATH = yaml_path
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            all_best_params = yaml.safe_load(f) or {}
    else:
        all_best_params = {}

    for target_col in target_cols:
        y_train = train_df[target_col]
        y_val = val_df[target_col]
        _DATA[f"y_train_{target_col}"] = y_train
        _DATA[f"y_val_{target_col}"] = y_val

        for model_type in args.models:
            logger.info("Running Optuna: target=%s model=%s", target_col, model_type)
            study_name = f"{target_col}_{model_type}"
            study = optuna.create_study(
                study_name=study_name,
                storage=db_url,
                load_if_exists=True,
                direction="minimize",
            )
            study.optimize(
                lambda trial: objective(trial, target_col, model_type),
                n_trials=args.trials,
                callbacks=[save_best_params_callback],
                show_progress_bar=False,
            )

            key = f"{target_col}_{model_type}"
            all_best_params[key] = study.best_params
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(all_best_params, f, sort_keys=True)
            logger.info(
                "Best for %s: RMSE=%.5f params=%s",
                key,
                study.best_value,
                all_best_params[key],
            )

    logger.info("Saved/updated best hyperparameters in %s", yaml_path)
