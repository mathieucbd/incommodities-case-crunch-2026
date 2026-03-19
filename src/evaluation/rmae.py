"""
Function that implements the relative mean absolute error (rMAE) metric.
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

import numpy as np
import pandas as pd
from src.evaluation.mae import MAE

def _process_inputs_for_metrics(p_real, p_pred):
    """Safely aligns and flattens arbitrary dimension pandas/numpy arrays."""
    p_real = np.asarray(p_real).flatten()
    p_pred = np.asarray(p_pred).flatten()
    return p_real, p_pred


def rMAE(p_real, p_pred, m=None, freq='1h'):
    """
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

    # Computing the MAE of the naive forecast
    # We use a 1-week shift as a naive forecast for day-ahead electricity markets
    p_real_s = pd.Series(np.asarray(p_real).flatten())
    # Identify shift (freq implies hourly array, 1 week = 168 hours, 1 day = 24 hours)
    shift_val = 168 if m == 'W' else 24
    p_pred_naive = p_real_s.shift(shift_val)
    
    # Drop NaNs to compute naive MAE properly
    valid_mask = ~p_pred_naive.isna()
    
    MAE_naive_train = MAE(p_real_s[valid_mask], p_pred_naive[valid_mask])
    
    # Checking if standard inputs are compatible
    p_real, p_pred = _process_inputs_for_metrics(p_real, p_pred)

    return np.mean(np.abs(p_real - p_pred) / MAE_naive_train)
