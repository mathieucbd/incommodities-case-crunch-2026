"""
Functions to compute and plot the univariate and multivariate versions of the Diebold-Mariano (DM) test.
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


def DM(p_real, p_pred_1, p_pred_2, norm=1, version='univariate'):
    """
    Author: Jesus Lago (Adapted for standalone pipeline)
    Function that performs the one-sided DM test to compare predictive accuracy between two forecasts.

    Parameters
    ----------
    p_real : array-like
        The actual prices.
    p_pred_1 : array-like
        The first predicted prices.
    p_pred_2 : array-like
        The second predicted prices.
    
    Returns
    -------
    float
        The calculated metric.
    """

    # Checking that all time series have the same shape
    if p_real.shape != p_pred_1.shape or p_real.shape != p_pred_2.shape:
        raise ValueError('The three time series must have the same shape')

    # Ensuring that time series have shape (n_days, n_prices_day)
    if len(p_real.shape) == 1 or (len(p_real.shape) == 2 and p_real.shape[1] == 1):
        raise ValueError('The time series must have shape (n_days, n_prices_day')

    # Computing the errors of each forecast
    errors_pred_1 = p_real - p_pred_1
    errors_pred_2 = p_real - p_pred_2

    # Computing the test statistic
    if version == 'univariate':

        # Computing the loss differential series for the univariate test
        if norm == 1:
            d = np.abs(errors_pred_1) - np.abs(errors_pred_2)
        if norm == 2:
            d = errors_pred_1**2 - errors_pred_2**2

        # Computing the loss differential size
        N = d.shape[0]

        # Computing the test statistic
        mean_d = np.mean(d, axis=0)
        var_d = np.var(d, ddof=0, axis=0)
        DM_stat = mean_d / np.sqrt((1 / N) * var_d)

    elif version == 'multivariate':

        # Computing the loss differential series for the multivariate test
        if norm == 1:
            d = np.mean(np.abs(errors_pred_1), axis=1) - np.mean(np.abs(errors_pred_2), axis=1)
        if norm == 2:
            d = np.mean(errors_pred_1**2, axis=1) - np.mean(errors_pred_2**2, axis=1)

        # Computing the loss differential size
        N = d.size

        # Computing the test statistic
        mean_d = np.mean(d)
        var_d = np.var(d, ddof=0)
        DM_stat = mean_d / np.sqrt((1 / N) * var_d)
        
    p_value = 1 - stats.norm.cdf(DM_stat)

    return p_value

def plot_multivariate_DM_test(real_price, forecasts, norm=1, title='DM test', savefig=False, path=''):
    """
    Author: Jesus Lago (Adapted for standalone pipeline)
    Plotting the results of comparing forecasts using the multivariate DM test.

    Parameters
    ----------
    real_price : array-like
        Dataframe that contains the real prices
    forecasts : array-like
        Dataframe that contains the forecasts of different models.
    
    Returns
    -------
    None
    """

    # Computing the multivariate DM test for each forecast pair
    p_values = pd.DataFrame(index=forecasts.columns, columns=forecasts.columns) 

    for model1 in forecasts.columns:
        for model2 in forecasts.columns:
            # For the diagonal elemnts representing comparing the same model we directly set a 
            # p-value of 1
            if model1 == model2:
                p_values.loc[model1, model2] = 1
            else:
                p_values.loc[model1, model2] = DM(p_real=real_price.values.reshape(-1, 24), 
                                                  p_pred_1=forecasts.loc[:, model1].values.reshape(-1, 24), 
                                                  p_pred_2=forecasts.loc[:, model2].values.reshape(-1, 24), 
                                                  norm=norm, version='multivariate')

    # Defining color map
    red = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0.5, 50)[1:], [0]])
    green = np.concatenate([np.linspace(0.5, 1, 50), np.zeros(50)])
    blue = np.zeros(100)
    rgb_color_map = np.concatenate([red.reshape(-1, 1), green.reshape(-1, 1), 
                                    blue.reshape(-1, 1)], axis=1)
    rgb_color_map = mpl.colors.ListedColormap(rgb_color_map)

    # Generating figure
    plt.imshow(p_values.astype(float).values, cmap=rgb_color_map, vmin=0, vmax=0.1)
    plt.xticks(range(len(forecasts.columns)), forecasts.columns, rotation=90.)
    plt.yticks(range(len(forecasts.columns)), forecasts.columns)
    plt.plot(range(p_values.shape[0]), range(p_values.shape[0]), 'wx')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()

    if savefig:
        plt.savefig(title + '.png', dpi=300)
        plt.savefig(title + '.eps')

    plt.show()
