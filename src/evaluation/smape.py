"""
Function that implements the symmetric mean absolute percentage error (sMAPE) metric.
"""

# Author: Jesus Lago

# License: AGPL-3.0 License


import numpy as np

def _process_inputs_for_metrics(p_real, p_pred):
    """Safely aligns and flattens arbitrary dimension pandas/numpy arrays."""
    p_real = np.asarray(p_real).flatten()
    p_pred = np.asarray(p_pred).flatten()
    return p_real, p_pred


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

    # Checking if inputs are compatible
    p_real, p_pred = _process_inputs_for_metrics(p_real, p_pred)

    epsilon = np.finfo(np.float64).eps
    return np.mean(np.abs(p_real - p_pred) / (((np.abs(p_real) + np.abs(p_pred)) / 2) + epsilon))