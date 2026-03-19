"""
Function that implements the mean absolute error (MAE) metric.
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

import numpy as np

def _process_inputs_for_metrics(p_real, p_pred):
    """Safely aligns and flattens arbitrary dimension pandas/numpy arrays."""
    p_real = np.asarray(p_real).flatten()
    p_pred = np.asarray(p_pred).flatten()
    return p_real, p_pred


def MAE(p_real, p_pred):
    """
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

    # Checking if inputs are compatible
    p_real, p_pred = _process_inputs_for_metrics(p_real, p_pred)

    return np.mean(np.abs(p_real - p_pred))
