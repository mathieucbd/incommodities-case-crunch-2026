"""Validation module.

Provides chronological holdout and expanding-window splits for time-series
electricity price forecasting. Never shuffles data.

Usage:
    from src.validation import create_holdout_split, create_expanding_splits, evaluate

    train_df, val_df = create_holdout_split(df, config)
    folds = create_expanding_splits(df, config)
    metrics = evaluate(y_true, y_pred, tag="catboost_fr")
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Holdout split (primary)
# ---------------------------------------------------------------------------

def create_holdout_split(
    df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological holdout split.

    Args:
        df: Full DataFrame with ``datetime_CET`` column.
        config: Parsed config.yaml.

    Returns:
        (train_df, val_df) — disjoint, chronologically ordered.
    """
    holdout_start = pd.Timestamp(config["validation"]["holdout_start"])
    dt = pd.to_datetime(df["datetime_CET"])
    train_mask = dt < holdout_start
    val_mask = dt >= holdout_start

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()

    print(f"Holdout split @ {holdout_start.date()}")
    print(f"  Train: {len(train_df):,} rows  ({dt[train_mask].min().date()} → {dt[train_mask].max().date()})")
    print(f"  Val:   {len(val_df):,} rows  ({dt[val_mask].min().date()} → {dt[val_mask].max().date()})")

    return train_df, val_df


# ---------------------------------------------------------------------------
# 2. Expanding-window splits (robustness)
# ---------------------------------------------------------------------------

# Default fold boundaries (can be overridden via config)
_DEFAULT_FOLDS = [
    {"train_end": "2023-07-01", "val_end": "2023-10-01"},  # Fold 1
    {"train_end": "2023-10-01", "val_end": "2024-01-01"},  # Fold 2
    {"train_end": "2024-01-01", "val_end": "2024-07-01"},  # Fold 3
]


def create_expanding_splits(
    df: pd.DataFrame,
    config: dict,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window cross-validation splits.

    Training always starts from the beginning; validation window advances.

    Returns:
        List of (train_df, val_df) tuples.
    """
    folds_cfg = config["validation"].get("expanding_folds", _DEFAULT_FOLDS)
    dt = pd.to_datetime(df["datetime_CET"])

    splits = []
    for i, fold in enumerate(folds_cfg):
        train_end = pd.Timestamp(fold["train_end"])
        val_end = pd.Timestamp(fold["val_end"])

        train_mask = dt < train_end
        val_mask = (dt >= train_end) & (dt < val_end)

        train_df = df.loc[train_mask].copy()
        val_df = df.loc[val_mask].copy()

        print(f"Fold {i+1}: train {len(train_df):,} rows → val {len(val_df):,} rows  "
              f"({train_end.date()} | {val_end.date()})")
        splits.append((train_df, val_df))

    return splits


# ---------------------------------------------------------------------------
# 3. Feature / target separation
# ---------------------------------------------------------------------------

def split_X_y(
    df: pd.DataFrame,
    target: str,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from target.

    Drops target columns, datetime, and any explicitly excluded columns.

    Args:
        df: DataFrame with features + targets.
        target: Target column name (``fr_spot`` or ``uk_spot``).
        drop_cols: Additional columns to drop.

    Returns:
        (X, y) where X is feature matrix and y is target series.
    """
    always_drop = ["datetime_CET", "datetime_UTC", "fr_spot", "uk_spot"]
    if drop_cols:
        always_drop.extend(drop_cols)

    existing_drops = [c for c in always_drop if c in df.columns]
    X = df.drop(columns=existing_drops)
    y = df[target]

    return X, y


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (0-200 scale)."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def evaluate(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    tag: str = "",
    hours: np.ndarray | pd.Series | None = None,
) -> dict:
    """Compute all metrics and optionally per-hour RMSE.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        tag: Label for printing (e.g. "catboost_fr").
        hours: Hour-of-day array for per-hour breakdown.

    Returns:
        Dictionary with metrics.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    results = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "n": len(y_true),
    }

    if tag:
        print(f"[{tag}]  RMSE={results['rmse']:.3f}  MAE={results['mae']:.3f}  "
              f"sMAPE={results['smape']:.1f}%  (n={results['n']:,})")

    # Per-hour RMSE breakdown
    if hours is not None:
        hours = np.asarray(hours)
        hourly_rmse = {}
        for h in range(24):
            mask = hours == h
            if mask.sum() > 0:
                hourly_rmse[h] = rmse(y_true[mask], y_pred[mask])
        results["hourly_rmse"] = hourly_rmse

    return results


def evaluate_both_targets(
    y_true_fr: np.ndarray,
    y_pred_fr: np.ndarray,
    y_true_uk: np.ndarray,
    y_pred_uk: np.ndarray,
    tag: str = "",
) -> dict:
    """Evaluate both FR and UK targets and compute combined RMSE."""
    rmse_fr = rmse(y_true_fr, y_pred_fr)
    rmse_uk = rmse(y_true_uk, y_pred_uk)
    combined = (rmse_fr + rmse_uk) / 2.0

    if tag:
        print(f"[{tag}]  FR_RMSE={rmse_fr:.3f}  UK_RMSE={rmse_uk:.3f}  "
              f"Combined={combined:.3f}")

    return {
        "fr_rmse": rmse_fr,
        "uk_rmse": rmse_uk,
        "combined_rmse": combined,
    }
