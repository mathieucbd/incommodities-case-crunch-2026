"""Target preparation for stationary modeling."""

import numpy as np
import pandas as pd


def prepare_stationary(spot_col_la, spot_col, train_fe_full, df_tr, df_va):
    """Prepare stationary deviation target and weights.

    Returns dict with y_dev_tr, y_dev_va, valid_tr, valid_va,
    weights, rm_tr, rm_va, spot_tr, spot_va.
    """
    la_col = train_fe_full[spot_col_la]
    roll_mean = la_col.ewm(span=240).mean()
    roll_std = la_col.rolling(168, min_periods=24).std()

    n_tr = len(df_tr)
    rm_tr = roll_mean.iloc[:n_tr].values
    rs_tr = roll_std.iloc[:n_tr].values
    rm_va = roll_mean.iloc[n_tr:n_tr + len(df_va)].values

    spot_tr = df_tr[spot_col].values
    spot_va = df_va[spot_col].values

    y_dev_tr = spot_tr - rm_tr
    y_dev_va = spot_va - rm_va
    valid_tr = np.isfinite(y_dev_tr)
    valid_va = np.isfinite(y_dev_va)

    dt = pd.to_datetime(df_tr["datetime_CET"])
    days_ago = (dt.max() - dt).dt.total_seconds() / 86400
    time_decay = np.exp(-2.0 * days_ago.values / 365)
    var_168h = np.clip(rs_tr ** 2, 1.0, None)
    var_168h = np.where(np.isnan(var_168h), 1.0, var_168h)
    w = time_decay / var_168h

    return {
        "y_dev_tr": y_dev_tr, "y_dev_va": y_dev_va,
        "valid_tr": valid_tr, "valid_va": valid_va,
        "weights": w, "rm_tr": rm_tr, "rm_va": rm_va,
        "spot_tr": spot_tr, "spot_va": spot_va,
    }
