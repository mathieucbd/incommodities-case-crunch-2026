"""Metrics and hourly bias correction."""

import numpy as np


def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_hbc(preds_spot, spot_va, hours_va):
    """Compute hourly bias correction (24 params)."""
    errors = spot_va - preds_spot
    hb = {}
    for h in range(24):
        mask = hours_va == h
        if mask.sum() > 0:
            hb[h] = errors[mask].mean()
    corrected = preds_spot + np.array([hb.get(h, 0) for h in hours_va])
    rmse_hbc = np.sqrt(np.mean((spot_va - corrected) ** 2))
    return hb, rmse_hbc


def compute_hbc_monthly(preds_spot, spot_va, hours_va, months_va, alpha=1.0):
    """Compute monthly x hourly bias correction (120 params)."""
    errors = spot_va - preds_spot
    hb = {}
    for m in sorted(set(months_va)):
        for h in range(24):
            mask = (months_va == m) & (hours_va == h)
            if mask.sum() >= 5:
                hb[(m, h)] = alpha * errors[mask].mean()
    corrected = preds_spot + np.array([hb.get((m, h), 0) for m, h in zip(months_va, hours_va)])
    rmse_hbc = np.sqrt(np.mean((spot_va - corrected) ** 2))
    return hb, rmse_hbc
