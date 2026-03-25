#!/usr/bin/env python
"""
Temporal Segmentation Evaluation: Global vs. Cluster vs. Hourly Models

Tests three modeling strategies:
1. Global: Single model for all hours
2. Cluster: 4 models (night/morning/day/peak)
3. Hourly: 24 individual hourly models

Evaluates RMSE, residual correlation, and decorrelation metrics.
"""

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import load_data
from src.feature_engineering import build_features


def load_feature_selection():
    """Load FR and UK feature selections from JSON files."""
    feat_fr = []
    feat_uk = []

    # FR features
    try:
        with open(PROJECT_ROOT / "outputs" / "feature_selection_v5_fr.json") as f:
            feat_fr_dict = json.load(f)
            if "features" in feat_fr_dict:
                feat_fr = feat_fr_dict["features"]
            elif "selected" in feat_fr_dict:
                feat_fr = feat_fr_dict["selected"]
    except Exception as e:
        print(f"Warning: Could not load FR features from JSON: {e}")

    # UK features
    try:
        with open(PROJECT_ROOT / "outputs" / "uk_feature_research.json") as f:
            uk_dict = json.load(f)
            if "basis_importance_ranking" in uk_dict:
                feat_uk = uk_dict["basis_importance_ranking"]
            elif "confirmed_features" in uk_dict:
                feat_uk = uk_dict["confirmed_features"]
    except Exception as e:
        print(f"Warning: Could not load UK features from JSON: {e}")

    return feat_fr, feat_uk


def main():
    print("\n" + "="*80)
    print("  Temporal Segmentation Analysis: Global vs. Cluster vs. Hourly Models")
    print("="*80)

    # Load data
    print("\nLoading data...")
    with open(PROJECT_ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    x_train, y_train, x_test = load_data(PROJECT_ROOT / "data" / "raw")
    train_fe = build_features(pd.concat([x_train], axis=0), config)
    train_fe = train_fe.join(y_train[["fr_spot", "uk_spot"]])

    # Split train/val
    holdout_start = config["validation"]["holdout_start"]
    mask_val = train_fe["datetime_CET"] >= holdout_start
    df_train = train_fe[~mask_val].copy()
    df_val = train_fe[mask_val].copy()

    print(f"Train: {len(df_train)} rows  |  Val: {len(df_val)} rows")

    # Load features
    feat_fr, feat_uk = load_feature_selection()

    # Filter to available columns
    feat_fr = [f for f in feat_fr if f in df_train.columns]
    feat_uk = [f for f in feat_uk if f in df_train.columns]

    # Fallback
    if len(feat_fr) == 0:
        exclude = {"fr_spot", "uk_spot", "datetime_CET", "datetime_UTC", "date", "id"}
        feat_fr = [c for c in df_train.columns
                   if c not in exclude and df_train[c].dtype in (float, np.float64, int, np.int64)]

    if len(feat_uk) == 0:
        exclude = {"fr_spot", "uk_spot", "datetime_CET", "datetime_UTC", "date", "id"}
        feat_uk = [c for c in df_train.columns
                   if c not in exclude and df_train[c].dtype in (float, np.float64, int, np.int64)]

    print(f"Features: FR={len(feat_fr)}, UK={len(feat_uk)}")

    # Model hyperparameters
    MODEL_PARAMS = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.05,
        "random_state": 42,
        "subsample": 0.8,
        "verbose": 0
    }

    # ─────────────────────────────────────────────────────────────────────────
    # STRATEGY 1: GLOBAL
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-"*80)
    print("Strategy 1: Global Models (1 model per market)")
    print("-"*80)

    X_train_fr = df_train[feat_fr].fillna(0)
    y_train_fr = df_train["fr_spot"].values
    model_global_fr = GradientBoostingRegressor(**MODEL_PARAMS)
    model_global_fr.fit(X_train_fr, y_train_fr)

    X_val_fr = df_val[feat_fr].fillna(0)
    y_val_fr = df_val["fr_spot"].values
    pred_global_fr = model_global_fr.predict(X_val_fr)
    rmse_global_fr = np.sqrt(mean_squared_error(y_val_fr, pred_global_fr))

    X_train_uk = df_train[feat_uk].fillna(0)
    y_train_uk = df_train["uk_spot"].values
    model_global_uk = GradientBoostingRegressor(**MODEL_PARAMS)
    model_global_uk.fit(X_train_uk, y_train_uk)

    X_val_uk = df_val[feat_uk].fillna(0)
    y_val_uk = df_val["uk_spot"].values
    pred_global_uk = model_global_uk.predict(X_val_uk)
    rmse_global_uk = np.sqrt(mean_squared_error(y_val_uk, pred_global_uk))

    print(f"FR RMSE: {rmse_global_fr:.2f}")
    print(f"UK RMSE: {rmse_global_uk:.2f}")
    print(f"Combined: {rmse_global_fr + rmse_global_uk:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # STRATEGY 2: CLUSTER (4 models per market)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-"*80)
    print("Strategy 2: Cluster Models (4 clusters per market)")
    print("-"*80)

    CLUSTERS = {
        "night": list(range(0, 6)),      # 0-5
        "morning": list(range(6, 12)),   # 6-11
        "day": list(range(12, 18)),      # 12-17
        "peak": list(range(18, 24))      # 18-23
    }

    # FR cluster models
    cluster_models_fr = {}
    for cluster_name, hours in CLUSTERS.items():
        mask = df_train["hour"].isin(hours)
        X = df_train.loc[mask, feat_fr].fillna(0)
        y = df_train.loc[mask, "fr_spot"].values
        model = GradientBoostingRegressor(**MODEL_PARAMS)
        model.fit(X, y)
        cluster_models_fr[cluster_name] = model

    pred_cluster_fr = np.zeros(len(df_val))
    for cluster_name, hours in CLUSTERS.items():
        mask = df_val["hour"].isin(hours)
        X = df_val.loc[mask, feat_fr].fillna(0)
        pred_cluster_fr[mask] = cluster_models_fr[cluster_name].predict(X)
    rmse_cluster_fr = np.sqrt(mean_squared_error(y_val_fr, pred_cluster_fr))

    # UK cluster models
    cluster_models_uk = {}
    for cluster_name, hours in CLUSTERS.items():
        mask = df_train["hour"].isin(hours)
        X = df_train.loc[mask, feat_uk].fillna(0)
        y = df_train.loc[mask, "uk_spot"].values
        model = GradientBoostingRegressor(**MODEL_PARAMS)
        model.fit(X, y)
        cluster_models_uk[cluster_name] = model

    pred_cluster_uk = np.zeros(len(df_val))
    for cluster_name, hours in CLUSTERS.items():
        mask = df_val["hour"].isin(hours)
        X = df_val.loc[mask, feat_uk].fillna(0)
        pred_cluster_uk[mask] = cluster_models_uk[cluster_name].predict(X)
    rmse_cluster_uk = np.sqrt(mean_squared_error(y_val_uk, pred_cluster_uk))

    print(f"FR RMSE: {rmse_cluster_fr:.2f}")
    print(f"UK RMSE: {rmse_cluster_uk:.2f}")
    print(f"Combined: {rmse_cluster_fr + rmse_cluster_uk:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # STRATEGY 3: HOURLY (24 models per market)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-"*80)
    print("Strategy 3: Hourly Models (24 models per market)")
    print("-"*80)

    # FR hourly models
    hourly_models_fr = {}
    for hour in range(24):
        mask = df_train["hour"] == hour
        if mask.sum() == 0:
            continue
        X = df_train.loc[mask, feat_fr].fillna(0)
        y = df_train.loc[mask, "fr_spot"].values
        model = GradientBoostingRegressor(**MODEL_PARAMS)
        model.fit(X, y)
        hourly_models_fr[hour] = model

    pred_hourly_fr = np.zeros(len(df_val))
    for hour, model in hourly_models_fr.items():
        mask = df_val["hour"] == hour
        X = df_val.loc[mask, feat_fr].fillna(0)
        pred_hourly_fr[mask] = model.predict(X)
    rmse_hourly_fr = np.sqrt(mean_squared_error(y_val_fr, pred_hourly_fr))

    # UK hourly models
    hourly_models_uk = {}
    for hour in range(24):
        mask = df_train["hour"] == hour
        if mask.sum() == 0:
            continue
        X = df_train.loc[mask, feat_uk].fillna(0)
        y = df_train.loc[mask, "uk_spot"].values
        model = GradientBoostingRegressor(**MODEL_PARAMS)
        model.fit(X, y)
        hourly_models_uk[hour] = model

    pred_hourly_uk = np.zeros(len(df_val))
    for hour, model in hourly_models_uk.items():
        mask = df_val["hour"] == hour
        X = df_val.loc[mask, feat_uk].fillna(0)
        pred_hourly_uk[mask] = model.predict(X)
    rmse_hourly_uk = np.sqrt(mean_squared_error(y_val_uk, pred_hourly_uk))

    print(f"FR RMSE: {rmse_hourly_fr:.2f}")
    print(f"UK RMSE: {rmse_hourly_uk:.2f}")
    print(f"Combined: {rmse_hourly_fr + rmse_hourly_uk:.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY & ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("SUMMARY: RMSE COMPARISON")
    print("="*80)

    results = pd.DataFrame({
        "Strategy": ["Global", "Cluster (4)", "Hourly (24)"],
        "FR_RMSE": [rmse_global_fr, rmse_cluster_fr, rmse_hourly_fr],
        "UK_RMSE": [rmse_global_uk, rmse_cluster_uk, rmse_hourly_uk],
    })
    results["Combined"] = results["FR_RMSE"] + results["UK_RMSE"]

    print(results.to_string(index=False))

    print("\n" + "-"*80)
    print("IMPROVEMENT vs GLOBAL BASELINE:")
    print("-"*80)

    fr_cluster_pct = ((rmse_global_fr - rmse_cluster_fr) / rmse_global_fr) * 100
    fr_hourly_pct = ((rmse_global_fr - rmse_hourly_fr) / rmse_global_fr) * 100
    uk_cluster_pct = ((rmse_global_uk - rmse_cluster_uk) / rmse_global_uk) * 100
    uk_hourly_pct = ((rmse_global_uk - rmse_hourly_uk) / rmse_global_uk) * 100
    combined_cluster_pct = ((rmse_global_fr + rmse_global_uk) - (rmse_cluster_fr + rmse_cluster_uk)) / (rmse_global_fr + rmse_global_uk) * 100
    combined_hourly_pct = ((rmse_global_fr + rmse_global_uk) - (rmse_hourly_fr + rmse_hourly_uk)) / (rmse_global_fr + rmse_global_uk) * 100

    print(f"FR Cluster:  {fr_cluster_pct:+6.2f}%")
    print(f"FR Hourly:   {fr_hourly_pct:+6.2f}%")
    print(f"UK Cluster:  {uk_cluster_pct:+6.2f}%")
    print(f"UK Hourly:   {uk_hourly_pct:+6.2f}%")
    print(f"\nCombined Cluster: {combined_cluster_pct:+6.2f}%")
    print(f"Combined Hourly:  {combined_hourly_pct:+6.2f}%")

    # Residual correlation analysis
    print("\n" + "="*80)
    print("RESIDUAL CORRELATION (decorrelation analysis)")
    print("="*80)

    residual_global_fr = y_val_fr - pred_global_fr
    residual_cluster_fr = y_val_fr - pred_cluster_fr
    residual_hourly_fr = y_val_fr - pred_hourly_fr

    residual_global_uk = y_val_uk - pred_global_uk
    residual_cluster_uk = y_val_uk - pred_cluster_uk
    residual_hourly_uk = y_val_uk - pred_hourly_uk

    # FR correlations
    corr_fr = pd.DataFrame({
        "Global": residual_global_fr,
        "Cluster": residual_cluster_fr,
        "Hourly": residual_hourly_fr
    }).corr()

    print("\nFR Residual Correlations (lower = better for ensemble):")
    print(corr_fr.round(3).to_string())

    # UK correlations
    corr_uk = pd.DataFrame({
        "Global": residual_global_uk,
        "Cluster": residual_cluster_uk,
        "Hourly": residual_hourly_uk
    }).corr()

    print("\nUK Residual Correlations (lower = better for ensemble):")
    print(corr_uk.round(3).to_string())

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    best_strategy = results.loc[results["Combined"].idxmin(), "Strategy"]
    best_combined = results["Combined"].min()

    print(f"\nBest overall strategy: {best_strategy}")
    print(f"Combined RMSE: {best_combined:.2f}")

    if abs(combined_hourly_pct) > 2:
        print(f"\nHourly models provide {abs(combined_hourly_pct):.1f}% improvement.")
        print("Hourly segmentation worth exploring for ensemble integration.")
    else:
        print(f"\nHourly models provide only {abs(combined_hourly_pct):.1f}% improvement.")
        print("Limited benefit from full 24-hour segmentation.")

    if abs(combined_cluster_pct) > 2:
        print(f"\nCluster models provide {abs(combined_cluster_pct):.1f}% improvement.")
        print("4-cluster approach may be more practical than 24 hourly models.")

    print("\n" + "="*80 + "\n")

    # Save results
    results.to_csv(PROJECT_ROOT / "outputs" / "temporal_segmentation_results.csv", index=False)
    print(f"Results saved to outputs/temporal_segmentation_results.csv")


if __name__ == "__main__":
    main()
