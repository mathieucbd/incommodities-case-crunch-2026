import pandas as pd
import numpy as np
import holidays
import logging

logger = logging.getLogger(__name__)

def engineer_residual_load(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Residual Load = load_f - (solar_f + wind_f) for relevant countries."""
    out = df.copy()
    zones = ['fr', 'uk']
    for z in zones:
        load = f"{z}_load_f"
        solar = f"{z}_solar_f"
        wind = f"{z}_wind_f"
        if all(col in out.columns for col in [load, solar, wind]):
            out[f"{z}_res_load"] = out[load] - (out[solar] + out[wind])
    return out

def engineer_thermal_floor(df: pd.DataFrame) -> pd.DataFrame:
    """Create a proxy marginal cost using forward-filled gas and emissions."""
    out = df.copy()
    # Cost_Thermal = (gas_price / 0.5) + (emission_price * (0.202 / 0.5))
    # FR Proxy
    if 'fr_gas' in out.columns and 'eu_emission' in out.columns:
        out['fr_thermal_floor'] = (out['fr_gas'] / 0.5) + (out['eu_emission'] * (0.202 / 0.5))
    # UK Proxy
    if 'uk_gas' in out.columns and 'uk_emission' in out.columns:
        out['uk_thermal_floor'] = (out['uk_gas'] / 0.5) + (out['uk_emission'] * (0.202 / 0.5))
    return out

def engineer_interconnectors(df: pd.DataFrame) -> pd.DataFrame:
    """Add online indicator masks for sparse links."""
    out = df.copy()
    atc_cols = [c for c in out.columns if 'atc_' in c]
    for atc_col in atc_cols:
        # Add binary indicator mask for interconnectors with large missing blocks (>1000 rows)
        if out[atc_col].isna().sum() > 1000:
            mask_name = f"is_{atc_col}_online"
            out[mask_name] = out[atc_col].notna().astype(int)
            
    return out

def engineer_calendar_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features and holiday boolean flags."""
    out = df.copy()
    dt_index = pd.DatetimeIndex(out.index)
    out['hour'] = dt_index.hour
    out['dayofweek'] = dt_index.dayofweek
    out['month'] = dt_index.month
    
    # Holidays
    fr_holidays = holidays.France()
    uk_holidays = holidays.UnitedKingdom()
    
    # Vectorized check via map
    dates = dt_index.date
    out['is_fr_holiday'] = pd.Series(dates, index=out.index).map(lambda x: x in fr_holidays).astype(int)
    out['is_uk_holiday'] = pd.Series(dates, index=out.index).map(lambda x: x in uk_holidays).astype(int)
    
    return out

def engineer_nuclear_shortfall(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Nuclear_Shortfall using physical grid capacities instead of dynamic max()."""
    out = df.copy()
    if 'fr_nuclear_avcap_f' in out.columns:
        out['fr_nuclear_shortfall'] = 61400 - out['fr_nuclear_avcap_f']
    if 'uk_nuclear_avcap_f' in out.columns:
        out['uk_nuclear_shortfall'] = 5800 - out['uk_nuclear_avcap_f']
    return out

def engineer_weekly_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Generate 168-hour lags for target spot prices."""
    out = df.copy()
    for target in ['fr_spot_la', 'uk_spot_la']:
        if target in out.columns:
            out[f"{target}_lag_168"] = out[target].shift(144) # 144 (6 days) as features are already lagged by 24h
    return out

def engineer_rolling_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 24h and 72h rolling standard deviation of lagged spot prices."""
    out = df.copy()
    for target in ['fr_spot_la', 'uk_spot_la']:
        if target in out.columns:
            out[f"{target}_vol_24h"] = out[target].rolling(window=24).std()
            out[f"{target}_vol_72h"] = out[target].rolling(window=72).std()
    return out

def engineer_asinh_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Apply np.arcsinh() to target lags."""
    out = df.copy()
    for col in ['fr_spot_la', 'uk_spot_la', 'ch_spot_la']:
        if col in out.columns:
            out[f"{col}_asinh"] = np.arcsinh(out[col])
    return out

def engineer_residual_ramps(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate .diff(3) on residual loads."""
    out = df.copy()
    for col in ['fr_res_load', 'uk_res_load']:
        if col in out.columns:
            out[f"{col}_ramp_3h"] = out[col].diff(3)
    return out

def engineer_scarcity(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate scarcity margins and ratios."""
    out = df.copy()
    if all(c in out.columns for c in ['fr_nuclear_avcap_f', 'fr_gas_avcap_f', 'fr_hydro_res_f', 'fr_hydro_ror_f', 'fr_load_f']):
        out['fr_total_cap_f'] = out['fr_nuclear_avcap_f'] + out['fr_gas_avcap_f'] + out['fr_hydro_res_f'] + out['fr_hydro_ror_f']
        out['fr_security_margin'] = out['fr_total_cap_f'] - out['fr_load_f']
        out['fr_scarcity_ratio'] = out['fr_load_f'] / out['fr_total_cap_f'].replace(0, np.nan)
        out['fr_scarcity_ratio'] = out['fr_scarcity_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
    return out

def engineer_regime_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate a 14-day (336-hour) rolling Z-score for residual loads."""
    out = df.copy()
    for col in ['fr_res_load', 'uk_res_load']:
        if col in out.columns:
            rolling_mean = out[col].rolling(window=336).mean()
            rolling_std = out[col].rolling(window=336).std()
            out[f"{col}_zscore_14d"] = (out[col] - rolling_mean) / rolling_std.replace(0, np.nan)
            out[f"{col}_zscore_14d"] = out[f"{col}_zscore_14d"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return out

def apply_full_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Sequentially call all feature engineering functions."""
    logger.info("Applying full feature engineering pipeline...")
    df = engineer_residual_load(df)
    df = engineer_thermal_floor(df)
    df = engineer_interconnectors(df)
    df = engineer_calendar_effects(df)
    df = engineer_nuclear_shortfall(df)
    df = engineer_weekly_lags(df)
    df = engineer_rolling_volatility(df)
    df = engineer_asinh_lags(df)
    df = engineer_residual_ramps(df)
    df = engineer_scarcity(df)
    df = engineer_regime_zscores(df)
    
    logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    return df
