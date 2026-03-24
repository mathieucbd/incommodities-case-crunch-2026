import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from src.features import apply_full_feature_engineering

def apply_feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations defined in src/features.py.
    This should be called immediately after ingestion.
    """
    return apply_full_feature_engineering(df)

def chronological_train_val_test_split(
    df: pd.DataFrame, 
    val_start: str, 
    test_start: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataframe strictly chronologically to prevent lookahead bias.
    """
    train_df = df[df.index < val_start].copy()
    val_df = df[(df.index >= val_start) & (df.index < test_start)].copy()
    test_df = df[df.index >= test_start].copy()

    return train_df, val_df, test_df


def scale_data(
    train_df: pd.DataFrame, 
    val_df: Optional[pd.DataFrame] = None, 
    test_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], StandardScaler, StandardScaler]:
    """
    Scales features and targets using separate StandardScalers fit STRICTLY on train_df.
    """
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    target_cols = [c for c in ['fr_spot', 'uk_spot'] if c in train_df.columns]
    feature_cols = [c for c in train_df.columns if c not in target_cols]
    
    train_df_scaled = train_df.copy()
    if feature_cols:
        train_df_scaled[feature_cols] = feature_scaler.fit_transform(train_df[feature_cols])
    if target_cols:
        train_df_scaled[target_cols] = target_scaler.fit_transform(train_df[target_cols])
        
    val_df_scaled = None
    if val_df is not None:
        val_df_scaled = val_df.copy()
        if feature_cols:
            val_df_scaled[feature_cols] = feature_scaler.transform(val_df[feature_cols])
        if target_cols:
            val_df_scaled[target_cols] = target_scaler.transform(val_df[target_cols])
            
    test_df_scaled = None
    if test_df is not None:
        test_df_scaled = test_df.copy()
        test_feature_cols = [c for c in feature_cols if c in test_df.columns]
        if test_feature_cols:
            test_df_scaled[test_feature_cols] = feature_scaler.transform(test_df[test_feature_cols])
        test_target_cols = [c for c in target_cols if c in test_df.columns]
        if test_target_cols:
            test_df_scaled[test_target_cols] = target_scaler.transform(test_df[test_target_cols])
            
    return train_df_scaled, val_df_scaled, test_df_scaled, feature_scaler, target_scaler
