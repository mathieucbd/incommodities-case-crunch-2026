"""
Metric functions used across deterministic model evaluation.
"""

import csv
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Mapping, SupportsFloat


def _process_inputs_for_metrics(p_real, p_pred):
    """Safely aligns and flattens arbitrary-dimension pandas/numpy arrays."""
    p_real = np.asarray(p_real).flatten()
    p_pred = np.asarray(p_pred).flatten()
    return p_real, p_pred


def MAE(p_real, p_pred):
    r"""
    Function that computes the mean absolute error (MAE) between two forecasts.
    """
    p_real, p_pred = _process_inputs_for_metrics(p_real, p_pred)
    return np.mean(np.abs(p_real - p_pred))


def RMSE(p_real, p_pred):
    r"""
    Function that computes the root mean squared error (RMSE) between two forecasts.
    """
    p_real, p_pred = _process_inputs_for_metrics(p_real, p_pred)
    return np.sqrt(np.mean((p_real - p_pred) ** 2))


def sMAPE(p_real, p_pred):
    r"""
    Function that computes the symmetric mean absolute percentage error (sMAPE) between two forecasts.

    """
    p_real, p_pred = _process_inputs_for_metrics(p_real, p_pred)
    epsilon = np.finfo(np.float64).eps
    return np.mean(
        np.abs(p_real - p_pred) / (((np.abs(p_real) + np.abs(p_pred)) / 2) + epsilon)
    )


def rMAE(p_real, p_pred, m=None, freq="1h"):
    r"""
    Function that computes the relative mean absolute error (rMAE) between two forecasts.
    """
    p_real_s = pd.Series(np.asarray(p_real).flatten())
    shift_val = 168 if m == "W" else 24
    p_pred_naive = p_real_s.shift(shift_val)

    valid_mask = ~p_pred_naive.isna()
    mae_naive_train = MAE(p_real_s[valid_mask], p_pred_naive[valid_mask])

    p_real, p_pred = _process_inputs_for_metrics(p_real, p_pred)
    return np.mean(np.abs(p_real - p_pred) / mae_naive_train)


def save_metrics_to_csv(
    zone: str,
    model_name: str,
    metrics_dict: Mapping[str, SupportsFloat],
    output_dir: str = "data/outputs/evaluation",
) -> None:
    """Append metrics to a centralized CSV log file.

    Each metric is written as one row:
    Timestamp, Zone, Model, Metric, Value
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.csv"
    file_exists = metrics_path.exists()

    with metrics_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Zone", "Model", "Metric", "Value"])

        timestamp = datetime.now().isoformat(timespec="seconds")
        for metric_name, metric_value in metrics_dict.items():
            writer.writerow(
                [timestamp, zone, model_name, metric_name, float(metric_value)]
            )
