import pandas as pd
import numpy as np
import logging
import yaml
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

from src.data_ingestion import load_and_merge_zone
from src.constants import TARGET_COL

# Configure standard terminal logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def add_deterministic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts Hour, DayOfWeek, and Month from the datetime index."""
    df_feat = df.copy()
    df_feat["Hour"] = df_feat.index.hour
    df_feat["DayOfWeek"] = df_feat.index.dayofweek
    df_feat["Month"] = df_feat.index.month
    return df_feat


def apply_mad_filter(
    series: pd.Series, window: int | str = "24h", z: float = 3.0
) -> pd.Series:
    """Applies the rolling Median Absolute Deviation filter to cap spikes."""
    win_str = f"{window}h" if isinstance(window, int) else window
    rolling_median = series.rolling(win_str).median()

    # Calculate rolling MAD manually
    mad = series.rolling(win_str).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )

    upper_bound = rolling_median + z * mad
    lower_bound = rolling_median - z * mad

    return series.clip(lower=lower_bound, upper=upper_bound)


def create_lags(
    df: pd.DataFrame, columns: List[str], lags: List[int] = [24, 48, 168]
) -> pd.DataFrame:
    """Creates shifted lag features and drops the resulting NaNs strictly for the generated columns."""
    df_feat = df.copy()
    new_cols = []
    for col in columns:
        for lag in lags:
            col_name = f"{col}_lag_{lag}"
            df_feat[col_name] = df_feat[col].shift(lag)
            new_cols.append(col_name)
    return df_feat.dropna(subset=new_cols)


def walk_forward_cv(df: pd.DataFrame, features: List[str], target: str) -> float:
    """Evaluates a LightGBM model using TimeSeriesSplit (n_splits=5) and returns average MAE."""
    tscv = TimeSeriesSplit(n_splits=5)
    maes = []

    df_sorted = df.sort_index()
    X = df_sorted[features]
    y = df_sorted[target]

    model = LGBMRegressor(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1)

    for train_index, test_index in tscv.split(X):
        # We must index via iloc for numpy/pandas alignment
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        maes.append(mean_absolute_error(y_test, preds))

    return float(np.mean(maes))


def greedy_feature_selector(
    target_zone: str, candidate_zones: List[str], raw_dir: str
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Orchestrates the pipeline Step 2. Loads the target zone, calculates a baseline MAE,
    iterates through neighboring candidates, tests their bundled lags via CV,
    and returns the optimal concatenated DataFrame along with the accepted zone list.
    """
    logger.info(f"Initializing Greedy Feature Selector. Base Target: {target_zone}")

    # 1. Build Base Target DataFrame
    df_target = load_and_merge_zone(target_zone, raw_dir)

    df_target["Spot_Price_Filtered"] = apply_mad_filter(
        df_target[TARGET_COL], window="24h", z=3.0
    )
    df_target = add_deterministic_features(df_target)

    lag_targets = ["Spot_Price_Filtered", "Residual_Load"]
    lags_list = [24, 48, 168]
    df_target = create_lags(df_target, lag_targets, lags=lags_list)

    # Identify active base features dynamically
    base_features = ["Hour", "DayOfWeek", "Month"]
    for col in lag_targets:
        for lag in lags_list:
            base_features.append(f"{col}_lag_{lag}")

    # Protect against global dropped rows by strictly dropping NAs in evaluated features
    df_target = df_target.dropna(subset=base_features + [TARGET_COL])

    best_mae = walk_forward_cv(df_target, base_features, TARGET_COL)
    logger.info(f"Base Domestic MAE ({target_zone} only): {best_mae:.3f} EUR/MWh")

    accepted_zones = [target_zone]
    current_features = base_features.copy()
    optimal_df = df_target.copy()

    # 2. Iterate through neighbors
    for neighbor in candidate_zones:
        logger.info(f"Evaluating Candidate Neighbor: {neighbor}...")
        try:
            df_neighbor = load_and_merge_zone(neighbor, raw_dir)
            df_neighbor["Spot_Price_Filtered"] = apply_mad_filter(
                df_neighbor[TARGET_COL], window="24h", z=3.0
            )
            df_neighbor_lagged = create_lags(df_neighbor, lag_targets, lags=lags_list)

            # Isolate the lag columns and prefix them appropriately
            neighbor_cols = [c for c in df_neighbor_lagged.columns if "lag" in c]
            df_slice = df_neighbor_lagged[neighbor_cols].rename(
                columns=lambda x: f"{neighbor}_{x}"
            )

            # Inner join aligns time exactly
            test_df = optimal_df.join(df_slice, how="inner")
            test_features = current_features + list(df_slice.columns)

            # Secure dropna specific to active subset
            test_df = test_df.dropna(subset=test_features + [TARGET_COL])

            new_mae = walk_forward_cv(test_df, test_features, TARGET_COL)

            if new_mae < best_mae:
                logger.info(
                    f" -> ACCEPTED {neighbor}! MAE dropped: {best_mae:.3f} -> {new_mae:.3f}"
                )
                best_mae = new_mae
                optimal_df = test_df
                current_features = test_features
                accepted_zones.append(neighbor)
            else:
                logger.info(
                    f" -> REJECTED {neighbor}. MAE worsened/flat: {best_mae:.3f} vs {new_mae:.3f}"
                )

        except Exception as e:
            logger.error(f"Failed processing candidate {neighbor}: {e}")

    logger.info("--------------------------------------------------")
    logger.info(f"Final Selection: {accepted_zones}")
    logger.info(f"Final Bundle MAE: {best_mae:.3f} EUR/MWh")

    return optimal_df, accepted_zones


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_directory = config["data"]["raw_dir"]

    greedy_cfg = config.get("greedy_algorithm", {})
    configured_zones = (
        greedy_cfg.get("zones")
        or greedy_cfg.get("flow_zones")
        or config.get("data", {}).get("target_zones", [])
    )

    if not configured_zones:
        raise ValueError(
            "No zones configured. Set greedy_algorithm.zones (or flow_zones) in config.yaml."
        )

    test_target = greedy_cfg.get("target_zone", configured_zones[0])
    test_candidates = [z for z in configured_zones if z != test_target]

    logger.info(
        f"Greedy config loaded -> target_zone={test_target}, candidates={test_candidates}"
    )

    final_df, accepted = greedy_feature_selector(
        test_target, test_candidates, raw_directory
    )
    print(f"\nResulting DataFrame Shape: {final_df.shape}")
