import pandas as pd
import numpy as np
import logging
import yaml
from typing import List, Tuple

from src.data_ingestion import load_and_merge_zone
from src.constants import TARGET_COL

# Configure standard terminal logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _safe_sum_existing(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    existing = [c for c in columns if c in df.columns]
    if not existing:
        return pd.Series(0.0, index=df.index)
    return df[existing].sum(axis=1)


def _normalize_zone_df(zone_df: pd.DataFrame) -> pd.DataFrame:
    df = zone_df.copy().sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="first")]
    rename_map = {
        "value (EUR/MWh)": TARGET_COL,
        "value (MW)": "Total_Load",
        "total_load": "Total_Load",
        "load": "Total_Load",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def _ensure_residual_load(df: pd.DataFrame) -> pd.DataFrame:
    if "Residual_Load" in df.columns:
        return df
    out = df.copy()
    if "Total_Load" in out.columns:
        renewables = _safe_sum_existing(
            out, ["Renewables", "SOLAR", "WIND-OFFSHORE", "WIND-ONSHORE"]
        )
        out["Residual_Load"] = out["Total_Load"] - renewables
    else:
        out["Residual_Load"] = np.nan
    return out


def _lag_block(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            out[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return out


def add_lags(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """Adds lagged copies of selected columns without dropping rows."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        for lag in lags:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)
    return out


def build_features(
    raw_data_dict: dict, target_zone: str, lag_actual_flows: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Builds a production-grade EPF feature matrix for target_zone and returns:
        - engineered dataframe (target included)
        - dynamic feature list (target excluded)

    Pan-European God-Matrix design:
        - Build full target-zone matrix (deterministic + autoregressive + rolling)
        - Add lag blocks from all other available zones in raw_data_dict
        - Return target + all candidate features for model-native feature selection
    """
    if target_zone not in raw_data_dict:
        raise ValueError(
            f"Target zone '{target_zone}' not found in raw_data_dict keys: {list(raw_data_dict.keys())}"
        )

    df_final = _normalize_zone_df(raw_data_dict[target_zone].copy())

    if not isinstance(df_final.index, pd.DatetimeIndex):
        raise TypeError("Target dataframe index must be a pandas DatetimeIndex.")

    df_final = _ensure_residual_load(df_final)

    if TARGET_COL not in df_final.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found for zone '{target_zone}'."
        )

    # 1) Start from full target dataframe and preserve all raw exogenous columns.
    if "Spot_Price_Filtered" not in df_final.columns:
        df_final["Spot_Price_Filtered"] = apply_mad_filter(
            df_final[TARGET_COL], window="24h", z=3.0
        )

    # 2) Cyclical time encoding
    dt_index = pd.DatetimeIndex(df_final.index)
    hour = dt_index.hour
    dow = dt_index.dayofweek
    month = dt_index.month

    df_final["Hour_Sin"] = np.sin(2 * np.pi * hour / 24.0)
    df_final["Hour_Cos"] = np.cos(2 * np.pi * hour / 24.0)
    df_final["DayOfWeek_Sin"] = np.sin(2 * np.pi * dow / 7.0)
    df_final["DayOfWeek_Cos"] = np.cos(2 * np.pi * dow / 7.0)
    df_final["Month_Sin"] = np.sin(2 * np.pi * month / 12.0)
    df_final["Month_Cos"] = np.cos(2 * np.pi * month / 12.0)
    df_final = df_final.drop(columns=["Hour", "DayOfWeek", "Month"], errors="ignore")

    # 3) Target-zone lags
    lag_base_cols = ["Spot_Price_Filtered"]
    if "Residual_Load" in df_final.columns:
        lag_base_cols.append("Residual_Load")
    df_final = add_lags(df_final, lag_base_cols, lags=[24, 48, 168])
    
    # --- DANGEROUS TARGET LEAK PREVENTED ---
    # Drop the original t=0 columns so the model doesn't peek at today's answers!
    df_final = df_final.drop(columns=lag_base_cols, errors="ignore")
    # ---------------------------------------

    # 4) Neighbor-zone processing and join
    neighbor_lags = [24, 48, 168]

    for zone_name, zone_df in raw_data_dict.items():
        if zone_name == target_zone:
            continue
        if not isinstance(zone_df, pd.DataFrame):
            continue

        neighbor_df = _normalize_zone_df(zone_df)
        neighbor_df = _ensure_residual_load(neighbor_df)

        lag_sources: List[str] = []

        if TARGET_COL in neighbor_df.columns:
            try:
                neighbor_df["Spot_Price_Filtered"] = apply_mad_filter(
                    neighbor_df[TARGET_COL], window="24h", z=3.0
                )
            except Exception as exc:
                logger.warning(f"Skipping Spot_Price_Filtered for {zone_name}: {exc}")
            else:
                lag_sources.append("Spot_Price_Filtered")

        if "Residual_Load" in neighbor_df.columns:
            lag_sources.append("Residual_Load")

        if not lag_sources:
            continue

        neighbor_lag_df = _lag_block(neighbor_df, lag_sources, neighbor_lags)
        neighbor_lag_df = neighbor_lag_df.rename(
            columns=lambda col: f"{zone_name}_{col}"
        )
        df_final = df_final.join(neighbor_lag_df, how="left")

    # Keep only rows where target exists.
    df_final = df_final.dropna(subset=[TARGET_COL])

    # --- MODULAR LOOKAHEAD BIAS PREVENTION ---
    if lag_actual_flows:
        # Identify all columns related to cross-border flows
        flow_cols = [c for c in df_final.columns if "Flow" in c or "MW" in c]
        if flow_cols:
            df_final = add_lags(df_final, flow_cols, lags=[24, 168])
            df_final.drop(columns=flow_cols, inplace=True)
    # -----------------------------------------

    # 6) Dynamic output
    active_features = [col for col in df_final.columns if col != TARGET_COL]

    # Log available feature categories
    agg_features = [c for c in active_features if "_lag_" in c]
    deterministic_features = [
        c
        for c in active_features
        if any(x in c for x in ["_sin", "_cos", "_Sin", "_Cos", "_roll_"])
    ]
    raw_exog_features = [
        c
        for c in active_features
        if c not in agg_features and c not in deterministic_features
    ]

    logger.info(f"Active features breakdown for {target_zone}:")
    logger.info(f"  - Raw exogenous (weather/load): {len(raw_exog_features)} cols")
    logger.info(f"  - Deterministic (cyclical): {len(deterministic_features)} cols")
    logger.info(f"  - Autoregressive (lags/rolling): {len(agg_features)} cols")
    logger.info(f"Total active features: {len(active_features)}")

    return df_final, active_features


def add_deterministic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts Hour, DayOfWeek, and Month from the datetime index."""
    df_feat = df.copy()
    dt_index = pd.DatetimeIndex(df_feat.index)
    df_feat["Hour"] = dt_index.hour
    df_feat["DayOfWeek"] = dt_index.dayofweek
    df_feat["Month"] = dt_index.month
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


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_directory = config["data"]["raw_dir"]
    target_zones = config.get("data", {}).get("target_zones", [])
    flow_only_zones = config.get("data", {}).get("flow_only_zones", [])
    all_zones = list(dict.fromkeys(target_zones + flow_only_zones))

    if not target_zones:
        raise ValueError("No zones configured in config.yaml.")

    loaded = {}
    for zone in all_zones:
        try:
            loaded[zone] = load_and_merge_zone(zone, raw_directory)
        except FileNotFoundError as exc:
            logger.warning(f"Skipping zone {zone} during local smoke test: {exc}")

    if not loaded:
        raise ValueError("No zone data could be loaded for build_features smoke test.")

    target = target_zones[0]
    if target not in loaded:
        raise ValueError(f"Configured target zone '{target}' could not be loaded.")

    df_out, features_out = build_features(loaded, target)
    print(f"Target zone: {target}")
    print(f"Resulting DataFrame Shape: {df_out.shape}")
    print(f"Active feature count: {len(features_out)}")
