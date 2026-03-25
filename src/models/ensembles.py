"""Block-specific ensemble with Inverse-RMSE weighted averaging."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import TimeSeriesSplit

from .metrics import compute_rmse

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


class BlockSpecificEnsemble(BaseEstimator, RegressorMixin):
    """
    Ensemble using three mutually exclusive feature blocks with Inverse-RMSE weights.

    Each of three blocks (A: calendar/AR, B: fundamentals, C: interconnections)
    trains a separate base estimator. Weights are computed via out-of-fold predictions
    using TimeSeriesSplit (no lookahead bias), then applied at test time.

    Parameters
    ----------
    base_estimator : Any, optional
        Regressor (sklearn-compatible or CatBoost). Defaults to LGBMRegressor(max_depth=4, n_estimators=100).
    n_splits : int, default=3
        Number of folds in TimeSeriesSplit for OOF weight computation.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        base_estimator: Any = None,
        n_splits: int = 3,
        random_state: int = 42,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_splits = n_splits
        self.random_state = random_state

        # Learned state
        self.model_a_: Any | None = None
        self.model_b_: Any | None = None
        self.model_c_: Any | None = None
        self.blocks_: tuple[list[str], list[str], list[str]] | None = None
        self.weights_: dict[str, float] | None = None
        self.oof_rmses_: dict[str, float] | None = None

    def _get_default_estimator(self) -> Any:
        """Return default estimator if not set."""
        if not HAS_LGBM:
            raise ImportError("LightGBM not available. Provide base_estimator explicitly.")
        return LGBMRegressor(
            max_depth=4,
            n_estimators=100,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=-1
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BlockSpecificEnsemble":
        """
        Fit block-specific ensemble with OOF weight computation.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with column names.
        y : np.ndarray
            Target values (1-D array).

        Returns
        -------
        self
        """
        from ..features import get_feature_blocks

        # Import here to avoid circular dependency
        X = X.copy()
        y = np.asarray(y, dtype=np.float64)

        # Get feature blocks
        blocks = get_feature_blocks(list(X.columns))
        block_a, block_b, block_c = blocks
        self.blocks_ = (block_a, block_b, block_c)

        # Validate all blocks have features
        for block_name, block in [("A", block_a), ("B", block_b), ("C", block_c)]:
            if not block:
                raise ValueError(f"Block {block_name} has no features. Cannot proceed.")

        # Filter X to available columns
        X_a = X[[f for f in block_a if f in X.columns]]
        X_b = X[[f for f in block_b if f in X.columns]]
        X_c = X[[f for f in block_c if f in X.columns]]

        # Compute OOF predictions for each block using TimeSeriesSplit
        base_est = self.base_estimator or self._get_default_estimator()
        tss = TimeSeriesSplit(n_splits=self.n_splits)

        oof_a = np.zeros(len(y))
        oof_b = np.zeros(len(y))
        oof_c = np.zeros(len(y))

        for train_idx, test_idx in tss.split(X_a):
            # Block A
            m_a = clone(base_est).fit(X_a.iloc[train_idx], y[train_idx])
            oof_a[test_idx] = m_a.predict(X_a.iloc[test_idx])

            # Block B
            m_b = clone(base_est).fit(X_b.iloc[train_idx], y[train_idx])
            oof_b[test_idx] = m_b.predict(X_b.iloc[test_idx])

            # Block C
            m_c = clone(base_est).fit(X_c.iloc[train_idx], y[train_idx])
            oof_c[test_idx] = m_c.predict(X_c.iloc[test_idx])

        # Compute RMSE for each block's OOF predictions
        rmse_a = compute_rmse(y, oof_a)
        rmse_b = compute_rmse(y, oof_b)
        rmse_c = compute_rmse(y, oof_c)
        self.oof_rmses_ = {"A": rmse_a, "B": rmse_b, "C": rmse_c}

        # Compute Inverse-RMSE weights
        inv_rmses = np.array([1.0 / rmse_a, 1.0 / rmse_b, 1.0 / rmse_c])
        weights_norm = inv_rmses / inv_rmses.sum()
        self.weights_ = {
            "A": float(weights_norm[0]),
            "B": float(weights_norm[1]),
            "C": float(weights_norm[2]),
        }

        # Retrain final models on all data
        self.model_a_ = clone(base_est).fit(X_a, y)
        self.model_b_ = clone(base_est).fit(X_b, y)
        self.model_c_ = clone(base_est).fit(X_c, y)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using weighted average of three block models.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with column names.

        Returns
        -------
        np.ndarray
            Weighted ensemble predictions (1-D array).
        """
        if self.model_a_ is None or self.blocks_ is None or self.weights_ is None:
            raise RuntimeError("Call .fit() before .predict()")

        block_a, block_b, block_c = self.blocks_
        X = X.copy()

        # Filter to available columns
        X_a = X[[f for f in block_a if f in X.columns]]
        X_b = X[[f for f in block_b if f in X.columns]]
        X_c = X[[f for f in block_c if f in X.columns]]

        # Generate block predictions
        pred_a = self.model_a_.predict(X_a)
        pred_b = self.model_b_.predict(X_b)
        pred_c = self.model_c_.predict(X_c)

        # Weighted average
        w_a = self.weights_["A"]
        w_b = self.weights_["B"]
        w_c = self.weights_["C"]
        ensemble_pred = w_a * pred_a + w_b * pred_b + w_c * pred_c

        return ensemble_pred
