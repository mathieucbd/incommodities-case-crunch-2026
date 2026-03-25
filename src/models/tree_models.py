"""Tree model wrappers (CatBoost, LightGBM, XGBoost)."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

ModelType = Literal["catboost", "lightgbm", "xgboost"]


@dataclass
class TreeResult:
    """Result of training a tree model."""
    model: object
    best_iteration: int


def train_tree(
    model_type: ModelType,
    params: dict,
    X_train, y_train,
    X_val, y_val,
    sample_weight=None,
    early_stopping_rounds=200,
) -> TreeResult:
    """Train a tree model with early stopping.

    CatBoost: uses Pool for weights, eval_set for early stopping.
    LightGBM: uses callbacks for early stopping.
    XGBoost: uses eval_set for early stopping.
    """
    if model_type == "catboost":
        from catboost import CatBoostRegressor, Pool
        model = CatBoostRegressor(**params)
        train_pool = Pool(X_train, y_train, weight=sample_weight)
        eval_pool = Pool(X_val, y_val)
        model.fit(train_pool, eval_set=eval_pool,
                  early_stopping_rounds=early_stopping_rounds, verbose=0)
        best_iter = model.get_best_iteration()

    elif model_type == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                       lgb.log_evaluation(0)],
        )
        best_iter = model.best_iteration_

    elif model_type == "xgboost":
        import xgboost as xgb
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=early_stopping_rounds,
        )
        best_iter = model.best_iteration
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return TreeResult(model=model, best_iteration=best_iter)


def retrain_tree(
    model_type: ModelType,
    params: dict,
    X_full, y_full,
    best_iteration: int,
    sample_weight=None,
    iter_pad: int = 50,
    min_iter: int = 500,
) -> object:
    """Retrain a tree model on full data with fixed iterations (no early stopping)."""
    n_iter = max(best_iteration + iter_pad, min_iter)

    if model_type == "catboost":
        from catboost import CatBoostRegressor, Pool
        final_params = {**params, "iterations": n_iter, "use_best_model": False}
        model = CatBoostRegressor(**final_params)
        model.fit(Pool(X_full, y_full, weight=sample_weight), verbose=0)

    elif model_type == "lightgbm":
        import lightgbm as lgb
        final_params = {**params, "n_estimators": n_iter}
        model = lgb.LGBMRegressor(**final_params)
        model.fit(X_full, y_full, sample_weight=sample_weight)

    elif model_type == "xgboost":
        import xgboost as xgb
        final_params = {**params, "n_estimators": n_iter}
        model = xgb.XGBRegressor(**final_params)
        model.fit(X_full, y_full, sample_weight=sample_weight, verbose=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model


def predict_tree(model, X) -> np.ndarray:
    """Predict with any tree model."""
    return model.predict(X)
