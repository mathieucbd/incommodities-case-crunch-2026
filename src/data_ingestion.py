import pandas as pd
from pathlib import Path
import yaml
from src.constants import TARGET_COL, RENEWABLE_COLS, BASELOAD_COLS, DISPATCHABLE_COLS, FILE_SUFFIXES

def load_and_merge_zone(target_zone: str, raw_dir: str) -> pd.DataFrame:
    """
    Loads spot price, total load, and generation data for a specified zone.
    Standardizes timezone to UTC, resamples to hourly frequency, merges,
    imputes missing data strictly without lookahead bias, and engineers Residual Load.
    """
    base_path = Path(raw_dir)
    
    # 1. Spot Price
    price_file = base_path / 'spot-price' / f"{target_zone}{FILE_SUFFIXES['price']}"
    if not price_file.exists():
        raise FileNotFoundError(f"Missing spot price file: {price_file}")
    
    df_price = pd.read_csv(price_file)
    df_price['time'] = pd.to_datetime(df_price['time'], utc=True)
    df_price = df_price.rename(columns={'value (EUR/MWh)': TARGET_COL})
    df_price = df_price.set_index('time')
    df_price = df_price.resample('1h').mean()
    
    # 2. Total Load
    # The load file name might be e.g. "DE-total-load.csv"
    load_file = base_path / 'total-load' / f"{target_zone}{FILE_SUFFIXES['load']}"
    if not load_file.exists():
        raise FileNotFoundError(f"Missing total load file: {load_file}")
    
    df_load = pd.read_csv(load_file)
    df_load['time'] = pd.to_datetime(df_load['time'], utc=True)
    df_load = df_load.rename(columns={'value (MW)': 'Total_Load'})
    df_load = df_load.set_index('time')
    df_load = df_load.resample('1h').mean()
    
    # 3. Generation
    gen_file = base_path / 'generation' / f"{target_zone}{FILE_SUFFIXES['generation']}"
    if not gen_file.exists():
        raise FileNotFoundError(f"Missing generation file: {gen_file}")
    
    df_gen_raw = pd.read_csv(gen_file)
    df_gen_raw['time'] = pd.to_datetime(df_gen_raw['time'], utc=True)
    
    # Pivot generation 
    df_gen = df_gen_raw.pivot_table(index='time', columns='type', values='value (MW)', aggfunc='mean')
    
    # Replace underscores with dashes to cleanly map to constants.py
    df_gen.columns = [str(c).replace('_', '-') for c in df_gen.columns]
    
    df_gen = df_gen.resample('1h').mean()
    
    # 4. Merge all together using outer join via concat to preserve the full hourly index
    merged = pd.concat([df_price, df_load, df_gen], axis=1)
    
    # 5. Impute missing values (Strict NO LOOKAHEAD rule)
    # Forward fill up to 3 hours
    merged = merged.ffill(limit=3)
    
    # Fill remaining with T-24 (shift 24)
    merged = merged.fillna(merged.shift(24))
    
    # Fill final remaining with T-168 (shift 168)
    merged = merged.fillna(merged.shift(168))
    
    # 6. Feature Engineering
    # Helper to calculate sum safely ignoring missing generation types
    def safe_sum(df, cols):
        existing = [c for c in cols if c in df.columns]
        return df[existing].sum(axis=1) if existing else pd.Series(0, index=df.index)
        
    merged['Renewables'] = safe_sum(merged, RENEWABLE_COLS)
    merged['Baseload'] = safe_sum(merged, BASELOAD_COLS)
    merged['Dispatchable'] = safe_sum(merged, DISPATCHABLE_COLS)
    
    merged['Residual_Load'] = merged['Total_Load'] - merged['Renewables']
    
    return merged

if __name__ == "__main__":
    # Load config file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    raw_directory = config['data']['raw_dir']
    test_zone = "DE"
    
    print(f"Testing load_and_merge_zone for '{test_zone}' from '{raw_directory}'...")
    
    df = load_and_merge_zone(test_zone, raw_directory)
    
    print(f"\nShape: {df.shape}")
    print("\nHead (First 5 rows):")
    print(df.head()[[TARGET_COL, 'Total_Load', 'Renewables', 'Residual_Load']])
    
    print("\nMissing Values Summary:")
    missing = df.isna().sum()
    if missing.sum() == 0:
        print("Perfect! 0 missing values remain.")
    else:
        print(missing[missing > 0])
