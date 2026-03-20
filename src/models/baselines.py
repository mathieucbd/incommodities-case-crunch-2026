import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LassoLarsIC
import sys

# Ensure src is in standard path for execution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_ingestion import load_and_merge_zone
from src.features import create_lags, add_deterministic_features, apply_mad_filter
from src.preprocessing import chronological_train_val_test_split, scale_data
from src.evaluation.mae import MAE
from src.evaluation.smape import sMAPE
from src.evaluation.rmae import rMAE
from src.constants import TARGET_COL

logger = logging.getLogger(__name__)

def predict_naive(y_full: pd.Series, test_indices: pd.DatetimeIndex, lag_hours: int = 168) -> pd.Series:
    """
    Predicts using a direct 168-hour persistence model sequentially mapped exactly mapping
    historical indices natively preventing alignment skew.
    """
    predictions = y_full.shift(lag_hours).reindex(test_indices)
    return predictions

def predict_lear(
    X_full: pd.DataFrame, 
    y_full: pd.Series, 
    test_indices: pd.DatetimeIndex, 
    calibration_window_days: int
) -> pd.Series:
    """
    Trains 24 independent hourly LassoLarsIC models evaluating a daily moving calibration window.
    Extracts the core logical architecture from Jesus Lago's epftoolbox tracking structural
    mathematics without the excessive computational overhead.
    """
    unique_test_days = np.unique(test_indices.date)
    predictions = pd.Series(index=test_indices, dtype=float, name="LEAR_Predictions")
    
    # Loop over every unique day in the testing partition
    for current_date in unique_test_days:
        current_date_ts = pd.Timestamp(current_date, tz=test_indices.tz if hasattr(test_indices, 'tz') else 'UTC')
        window_start = current_date_ts - pd.Timedelta(days=calibration_window_days)
        
        # 1. Geometrically slice strictly prior data bounds safely ensuring 0 leakage
        mask_train = (X_full.index >= window_start) & (X_full.index < current_date_ts)
        X_calib = X_full.loc[mask_train]
        y_calib = y_full.loc[mask_train]
        
        # Identify the target slices available on the specific test date
        mask_test = (test_indices.date == current_date)
        day_test_indices = test_indices[mask_test]
        
        # Subset features
        X_test_day = X_full.loc[day_test_indices]
        
        # 2. Iterate structurally through the 24 internal electricity models
        for h in range(24):
            # Select samples specifically matching this hour from the calibration window
            hour_mask_calib = X_calib.index.hour == h
            if not hour_mask_calib.any():
                continue
                
            X_calib_h = X_calib.loc[hour_mask_calib]
            y_calib_h = y_calib.loc[hour_mask_calib]
            
            # Verify the current testing day actually contains this target hour
            hour_mask_test = X_test_day.index.hour == h
            if not hour_mask_test.any():
                continue
                
            X_test_h = X_test_day.loc[hour_mask_test]
            
            # 3. Instantiate, Fit, and Predict isolated LassoLarsIC
            model = LassoLarsIC(criterion='aic')
            try:
                model.fit(X_calib_h, y_calib_h)
                preds = model.predict(X_test_h)
                predictions.loc[X_test_h.index] = preds
            except Exception as e:
                logger.warning(f"Failed LEAR fit Day: {current_date}, Hour: {h}. Reason: {e}")
                # Naive fallback mean assignment
                predictions.loc[X_test_h.index] = y_calib_h.mean()
                
    return predictions

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    raw_directory = config.get('data', {}).get('raw_dir', 'data/raw/auhack_legacy/')
    calibration_window = config.get('model_settings', {}).get('lear', {}).get('calibration_window_days', 182)
    
    target_zone = "DE"
    logger.info(f"Establishing Baselines natively for {target_zone}...")
    
    df = load_and_merge_zone(target_zone, raw_directory)
    
    # Rapid feature application mimicking our execution loop
    df['Spot_Price_Filtered'] = apply_mad_filter(df[TARGET_COL], window='24h', z=3.0)
    df = add_deterministic_features(df)
    
    lag_targets = ['Spot_Price_Filtered', 'Residual_Load']
    lags_list = [24, 48, 168]
    df = create_lags(df, lag_targets, lags_list)
    
    # Filter bounds specifically targeting the requested columns
    active_features = ['Hour', 'DayOfWeek', 'Month']
    for col in lag_targets:
        for lag in lags_list:
            active_features.append(f'{col}_lag_{lag}')
            
    df = df.dropna(subset=active_features + [TARGET_COL])
    
    logger.info("Executing Train / Val / Test exact chronological splits...")
    train_df, val_df, test_df = chronological_train_val_test_split(df, val_ratio=0.15, test_ratio=0.15)
    
    X_train = train_df[active_features]
    y_train = train_df[TARGET_COL]
    
    X_val = val_df[active_features]
    y_val = val_df[TARGET_COL]
    
    X_test = test_df[active_features]
    y_test = test_df[TARGET_COL]
    
    logger.info("Scaling features strictly referencing Training distributions...")
    X_train_s, X_val_s, X_test_s, scaler = scale_data(X_train, X_val, X_test)
    
    # We reconstruct the sequentially shifted mathematical arrays uniformly allowing specific slices
    X_full = pd.concat([X_train_s, X_val_s, X_test_s]).sort_index()
    y_full = pd.concat([y_train, y_val, y_test]).sort_index()
    test_idx = X_test.index
    
    logger.info("========================================")
    logger.info(f"Targeting Naive (168h Persistence) across {len(test_idx)} points...")
    preds_naive = predict_naive(y_full, test_idx, lag_hours=168)
    
    logger.info(f"Establishing LEAR pseudo-online calibrations mapping {calibration_window} days...")
    preds_lear = predict_lear(X_full, y_full, test_idx, calibration_window_days=calibration_window)
    
    logger.info("========================================")
    logger.info("--------- Evaluation Metrics -----------")
    
    # Validate mathematical alignment maps cleanly
    valid_naive = ~preds_naive.isna()
    y_t_n = y_test.loc[valid_naive]
    p_n = preds_naive.loc[valid_naive]
    
    logger.info("[NAIVE 168h Baseline]")
    logger.info(f"  MAE:   {MAE(y_t_n, p_n):.3f} EUR/MWh")
    logger.info(f"  sMAPE: {sMAPE(y_t_n, p_n)*100:.3f} %")
    logger.info(f"  rMAE:  {rMAE(y_t_n, p_n, m='W'):.3f}")
    
    logger.info("----------------------------------------")
    
    # Validate LEAR natively dropping execution gaps dynamically
    valid_lear = ~preds_lear.isna()
    y_t_l = y_test.loc[valid_lear]
    p_l = preds_lear.loc[valid_lear]
    
    logger.info("[LEAR LassoLarsIC (24-Hour Windows)]")
    logger.info(f"  MAE:   {MAE(y_t_l, p_l):.3f} EUR/MWh")
    logger.info(f"  sMAPE: {sMAPE(y_t_l, p_l)*100:.3f} %")
    logger.info(f"  rMAE:  {rMAE(y_t_l, p_l, m='W'):.3f}")
    logger.info("========================================")
