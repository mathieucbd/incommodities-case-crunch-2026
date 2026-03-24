"""CatBoost model for electricity price forecasting.

OOP interface with Optuna hyperparameter optimization, early stopping,
and feature importance extraction. Separate models for FR and UK.

Usage:
    from src.models.catboost_model import CatBoostForecaster

    model = CatBoostForecaster(config, target="fr_spot")
    model.optimize(X_train, y_train, X_val, y_val)  # Optuna
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_test)
    model.save("outputs/models/catboost_fr.cbm")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None


class CatBoostForecaster:
    """CatBoost regressor with Optuna tuning for EPF."""

    def __init__(self, config: dict, target: str = "fr_spot"):
        self.config = config
        self.target = target
        self.cb_config = config["models"]["catboost"]
        self.model: CatBoostRegressor | None = None
        self.best_params: dict | None = None
        self.feature_importances_: pd.Series | None = None

    def _base_params(self) -> dict:
        """Base CatBoost parameters from config."""
        return {
            "loss_function": self.cb_config.get("loss_function", "RMSE"),
            "eval_metric": self.cb_config.get("eval_metric", "RMSE"),
            "iterations": self.cb_config.get("iterations", 5000),
            "learning_rate": self.cb_config.get("learning_rate", 0.03),
            "depth": self.cb_config.get("depth", 8),
            "l2_leaf_reg": self.cb_config.get("l2_leaf_reg", 5),
            "subsample": self.cb_config.get("subsample", 0.8),
            "random_seed": self.cb_config.get("random_seed", 42),
            "verbose": 0,
            "allow_writing_files": False,
            "use_best_model": True,
        }

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int | None = None,
    ) -> dict:
        """Run Optuna hyperparameter optimization.

        Returns:
            Best parameters found.
        """
        if optuna is None:
            raise ImportError("optuna is required for optimization. Install with: uv add optuna")

        n_trials = n_trials or self.cb_config.get("optuna_trials", 80)
        early_stop = self.cb_config.get("early_stopping_rounds", 100)

        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)

        def objective(trial: optuna.Trial) -> float:
            params = self._base_params()
            params.update({
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
                "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            })

            model = CatBoostRegressor(**params)
            model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=early_stop,
                verbose=0,
            )

            preds = model.predict(X_val)
            rmse = float(np.sqrt(np.mean((y_val.values - preds) ** 2)))
            return rmse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        print(f"[Optuna {self.target}] Best RMSE: {study.best_value:.3f}")
        print(f"  Params: {self.best_params}")

        return self.best_params

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        params_override: dict | None = None,
    ) -> "CatBoostForecaster":
        """Train the model with given or optimized parameters.

        If ``params_override`` is given, uses those. Otherwise uses
        ``best_params`` from Optuna (if available) merged with base config.
        """
        params = self._base_params()

        if params_override:
            params.update(params_override)
        elif self.best_params:
            params.update(self.best_params)

        early_stop = self.cb_config.get("early_stopping_rounds", 100)

        self.model = CatBoostRegressor(**params)

        fit_kwargs = {"verbose": 100}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = Pool(X_val, y_val)
            fit_kwargs["early_stopping_rounds"] = early_stop

        self.model.fit(Pool(X_train, y_train), **fit_kwargs)

        # Feature importances
        importances = self.model.get_feature_importance()
        self.feature_importances_ = pd.Series(
            importances, index=X_train.columns
        ).sort_values(ascending=False)

        best_iter = self.model.get_best_iteration() if X_val is not None else params["iterations"]
        print(f"[CatBoost {self.target}] Trained — {best_iter} iterations, "
              f"{len(X_train.columns)} features")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        return self.model.predict(X)

    def fit_final(
        self,
        X_full: pd.DataFrame,
        y_full: pd.Series,
        n_iterations: int | None = None,
    ) -> "CatBoostForecaster":
        """Retrain on full data with fixed iterations (no early stopping).

        Used for final submission: train on ALL available data.
        """
        params = self._base_params()
        if self.best_params:
            params.update(self.best_params)

        if n_iterations:
            params["iterations"] = n_iterations
        params["use_best_model"] = False

        self.model = CatBoostRegressor(**params)
        self.model.fit(Pool(X_full, y_full), verbose=100)

        self.feature_importances_ = pd.Series(
            self.model.get_feature_importance(), index=X_full.columns
        ).sort_values(ascending=False)

        print(f"[CatBoost {self.target}] Final train — {params['iterations']} iterations, "
              f"{len(X_full)} rows")

        return self

    def save(self, path: str) -> None:
        """Save model and metadata."""
        if self.model is None:
            raise RuntimeError("No model to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save_model(str(path))

        meta = {
            "target": self.target,
            "best_params": self.best_params,
            "n_features": len(self.feature_importances_) if self.feature_importances_ is not None else 0,
        }
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2))

        if self.feature_importances_ is not None:
            imp_path = path.with_name(path.stem + "_importances.csv")
            self.feature_importances_.to_csv(imp_path, header=["importance"])

        print(f"  Saved to {path}")

    def load(self, path: str) -> "CatBoostForecaster":
        """Load model from file."""
        self.model = CatBoostRegressor()
        self.model.load_model(path)

        meta_path = Path(path).with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.best_params = meta.get("best_params")
            self.target = meta.get("target", self.target)

        print(f"  Loaded from {path}")
        return self

    def get_top_features(self, n: int = 30) -> pd.Series:
        """Return top-N features by importance."""
        if self.feature_importances_ is None:
            raise RuntimeError("No feature importances. Call .fit() first.")
        return self.feature_importances_.head(n)
