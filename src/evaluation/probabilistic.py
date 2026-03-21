import numpy as np


def pinball_loss(y_true, y_pred, tau):
    r"""
    Author: Jesus Lago (Adapted for standalone pipeline)
    Calculates the pinball loss for a specific quantile.

    .. math::
        L(y, \hat{y}, \tau) = \max(\tau(y - \hat{y}), (1 - \tau)(\hat{y} - y))

    Parameters
    ----------
    y_true : array-like
        The actual prices.
    y_pred : array-like
        The predicted prices for the quantile.
    tau : float
        The quantile (e.g., 0.05, 0.5, 0.95).

    Returns
    -------
    float
        The calculated pinball loss.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return np.mean(np.maximum(tau * (y_true - y_pred), (tau - 1) * (y_true - y_pred)))


def winkler_score(y_true, lower, upper, alpha=0.1):
    r"""
    Author: Jesus Lago (Adapted for standalone pipeline)
    Calculates the Winkler score for a prediction interval.

    .. math::
        W = (U - L) + \frac{2}{\alpha}(L - y)I(y < L) + \frac{2}{\alpha}(y - U)I(y > U)

    Parameters
    ----------
    y_true : array-like
        The actual prices.
    lower : array-like
        The lower bound of the interval.
    upper : array-like
        The upper bound of the interval.
    alpha : float
        The significance level (1 - coverage).

    Returns
    -------
    float
        The calculated Winkler score.
    """
    y_true = np.array(y_true).flatten()
    lower = np.array(lower).flatten()
    upper = np.array(upper).flatten()

    diff = upper - lower
    lower_penalty = (2 / alpha) * (lower - y_true) * (y_true < lower)
    upper_penalty = (2 / alpha) * (y_true - upper) * (y_true > upper)

    return np.mean(diff + lower_penalty + upper_penalty)
