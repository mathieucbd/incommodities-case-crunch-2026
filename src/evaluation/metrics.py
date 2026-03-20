"""
Metric functions used across deterministic model evaluation.
"""

import numpy as np
import pandas as pd


def _process_inputs_for_metrics(p_real, p_pred):
    """Safely aligns and flattens arbitrary-dimension pandas/numpy arrays."""
    p_real = np.asarray(p_real).flatten()
    p_pred = np.asarray(p_pred).flatten()
    return p_real, p_pred


def MAE(p_real, p_pred):
    r"""
    Author: Jesus Lago (Adapted for standalone pipeline)
    Function that computes the mean absolute error (MAE) between two forecasts.

    .. math::
        \mathrm{MAE} = \frac{1}{N}\sum_{i=1}^N \bigl|p_\mathrm{real}[i]-p_\mathrm{pred}[i]\bigr|

    Parameters
    ----------
    p_real : array-like
        The actual prices.
    p_pred : array-like
        The predicted prices.

    Returns
    -------
    float
        The calculated metric.
    """
    p_real, p_pred = _process_inputs_for_metrics(p_real, p_pred)
    return np.mean(np.abs(p_real - p_pred))


def sMAPE(p_real, p_pred):
    r"""
    Author: Jesus Lago (Adapted for standalone pipeline)
    Function that computes the symmetric mean absolute percentage error (sMAPE) between two forecasts.

    .. math::
        \mathrm{sMAPE} = \frac{1}{N}\sum_{i=1}^N \frac{2\bigl|p_\mathrm{real}[i]−p_\mathrm{pred}[i]\bigr|}{
        \bigl|P_\mathrm{real}[i]\bigr|+\bigl|P_\mathrm{pred}[i]\bigr|}

    Parameters
    ----------
    p_real : array-like
        The actual prices.
    p_pred : array-like
        The predicted prices.

    Returns
    -------
    float
        The calculated metric.
    """
    p_real, p_pred = _process_inputs_for_metrics(p_real, p_pred)
    epsilon = np.finfo(np.float64).eps
    return np.mean(
        np.abs(p_real - p_pred) / (((np.abs(p_real) + np.abs(p_pred)) / 2) + epsilon)
    )


def rMAE(p_real, p_pred, m=None, freq="1h"):
    r"""
    Author: Jesus Lago (Adapted for standalone pipeline)
    Function that computes the relative mean absolute error (rMAE) between two forecasts.

    .. math::
        \mathrm{rMAE}_\mathrm{m} = \frac{1}{N}\sum_{i=1}^N
                         \frac{\bigl|p_\mathrm{real}[i]−p_\mathrm{pred}[i]\bigr|}
                         {\mathrm{MAE}(p_\mathrm{real}, p_\mathrm{naive})}.

    Parameters
    ----------
    p_real : array-like
        The actual prices.
    p_pred : array-like
        The predicted prices.

    Returns
    -------
    float
        The calculated metric.
    """
    p_real_s = pd.Series(np.asarray(p_real).flatten())
    shift_val = 168 if m == "W" else 24
    p_pred_naive = p_real_s.shift(shift_val)

    valid_mask = ~p_pred_naive.isna()
    mae_naive_train = MAE(p_real_s[valid_mask], p_pred_naive[valid_mask])

    p_real, p_pred = _process_inputs_for_metrics(p_real, p_pred)
    return np.mean(np.abs(p_real - p_pred) / mae_naive_train)
