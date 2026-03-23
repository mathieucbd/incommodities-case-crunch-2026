import pandas as pd
from pathlib import Path
import yaml

def load_competition_data(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Resolve raw_dir relative to the config file's location to handle sub-directory execution (e.g. notebooks/)
    config_abs_path = Path(config_path).resolve()
    raw_dir = config_abs_path.parent / config['data']['raw_dir']
    
    # Load features and targets
    x_train = pd.read_csv(raw_dir / config['data']['features_file'])
    y_train = pd.read_csv(raw_dir / config['data']['targets_file'])
    
    # Merge on 'id'
    # Features and targets both contain datetime_CET/UTC. Merge on all shared ID/temporal columns to avoid suffix collisions.
    df = pd.merge(x_train, y_train, on=['id', 'datetime_CET', 'datetime_UTC'], how='inner')
    
    # Set datetime index
    index_col = config['data'].get('index_col', 'datetime_CET')
    df[index_col] = pd.to_datetime(df[index_col])
    df.set_index(index_col, inplace=True)
    
    # Drop redundant columns
    df.drop(columns=['id', 'datetime_UTC'], inplace=True, errors='ignore')
    
    # Forward-fill sparse daily features mechanically
    sparse_features = config.get('data', {}).get('sparse_daily_features', [])
    for col in sparse_features:
        if col in df.columns:
            df[col] = df[col].ffill()
            
    return df