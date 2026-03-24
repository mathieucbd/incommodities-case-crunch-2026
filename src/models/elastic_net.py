"""Elastic Net model wrapper with StandardScaler."""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


@dataclass
class ElasticNetResult:
    """Result of training an Elastic Net model."""
    model: ElasticNet
    scaler: StandardScaler
    preds_val: np.ndarray
    n_nonzero: int


def train_elastic_net(
    X_train, y_train, X_val,
    alpha=10.0, l1_ratio=0.9, max_iter=10000,
) -> ElasticNetResult:
    """Train ElasticNet with StandardScaler preprocessing."""
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(np.nan_to_num(X_train, 0))
    X_va_scaled = scaler.transform(np.nan_to_num(X_val, 0))

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)
    model.fit(X_tr_scaled, y_train)
    preds = model.predict(X_va_scaled)

    return ElasticNetResult(
        model=model, scaler=scaler,
        preds_val=preds,
        n_nonzero=int(np.sum(model.coef_ != 0)),
    )


def retrain_elastic_net(
    X_full, y_full,
    alpha=10.0, l1_ratio=0.9, max_iter=10000,
) -> tuple[ElasticNet, StandardScaler]:
    """Retrain ElasticNet on full data. Returns (model, scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.nan_to_num(X_full, 0))
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)
    model.fit(X_scaled, y_full)
    return model, scaler


def predict_elastic_net(model, scaler, X) -> np.ndarray:
    """Predict with ElasticNet (applies scaler)."""
    return model.predict(scaler.transform(np.nan_to_num(X, 0)))
