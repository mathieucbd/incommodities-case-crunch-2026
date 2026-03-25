"""Elastic Net model wrapper with StandardScaler."""

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


@dataclass
class ElasticNetResult:
    """Result of training an Elastic Net model."""
    model: ElasticNet | ElasticNetCV
    scaler: StandardScaler
    preds_val: np.ndarray
    n_nonzero: int
    # Populated only when trained via train_elastic_net_cv
    alpha_chosen: float | None = field(default=None)
    l1_ratio_chosen: float | None = field(default=None)


def train_elastic_net(
    X_train, y_train, X_val,
    alpha=10.0, l1_ratio=0.9, max_iter=10000,
) -> ElasticNetResult:
    """Train ElasticNet with StandardScaler preprocessing."""
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(np.nan_to_num(np.asarray(X_train, dtype=np.float64), nan=0.0, copy=True))
    X_va_scaled = scaler.transform(np.nan_to_num(np.asarray(X_val, dtype=np.float64), nan=0.0, copy=True))

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)
    model.fit(X_tr_scaled, y_train)
    preds = model.predict(X_va_scaled)

    return ElasticNetResult(
        model=model, scaler=scaler,
        preds_val=preds,
        n_nonzero=int(np.sum(model.coef_ != 0)),
    )


def train_elastic_net_cv(
    X_train, y_train, X_val,
    alphas: np.ndarray | None = None,
    l1_ratios: list[float] | None = None,
    n_splits: int = 5,
    max_iter: int = 50_000,
) -> ElasticNetResult:
    """Train ElasticNet with cross-validated alpha and l1_ratio selection.

    Uses TimeSeriesSplit to respect temporal ordering — no future leakage
    into the CV folds. Both alpha and l1_ratio are selected jointly.

    Parameters
    ----------
    alphas:
        Log-spaced grid of regularisation strengths. Defaults to 50 values
        from 1e-3 to 1e3.
    l1_ratios:
        L1/L2 mixing candidates. Defaults to [0.1, 0.5, 0.7, 0.9, 0.95, 1.0].
    n_splits:
        Number of TimeSeriesSplit folds (fit on training data only).
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 50)
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(np.nan_to_num(np.asarray(X_train, dtype=np.float64), nan=0.0, copy=True))
    X_va_scaled = scaler.transform(np.nan_to_num(np.asarray(X_val, dtype=np.float64), nan=0.0, copy=True))

    cv = TimeSeriesSplit(n_splits=n_splits)
    model = ElasticNetCV(
        alphas=alphas,
        l1_ratio=l1_ratios,
        cv=cv,
        max_iter=max_iter,
        tol=1e-3,
        selection="random",
        n_jobs=-1,
    )
    model.fit(X_tr_scaled, y_train)
    preds = model.predict(X_va_scaled)

    return ElasticNetResult(
        model=model,
        scaler=scaler,
        preds_val=preds,
        n_nonzero=int(np.sum(model.coef_ != 0)),
        alpha_chosen=float(model.alpha_),
        l1_ratio_chosen=float(model.l1_ratio_),
    )


def retrain_elastic_net(
    X_full, y_full,
    alpha=10.0, l1_ratio=0.9, max_iter=10000,
) -> tuple[ElasticNet, StandardScaler]:
    """Retrain ElasticNet on full data. Returns (model, scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.nan_to_num(np.asarray(X_full, dtype=np.float64), nan=0.0, copy=True))
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)
    model.fit(X_scaled, y_full)
    return model, scaler


def predict_elastic_net(model, scaler, X) -> np.ndarray:
    """Predict with ElasticNet (applies scaler)."""
    return model.predict(scaler.transform(np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, copy=True)))
