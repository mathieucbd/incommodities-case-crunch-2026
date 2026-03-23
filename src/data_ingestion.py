import pandas as pd
from pathlib import Path
import yaml

def load_competition_data(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    raw_dir = Path(config['data']['raw_dir'])
    
    # Load features and targets
    x_train = pd.read_csv(raw_dir / config['data']['features_file'])
    y_train = pd.read_csv(raw_dir / config['data']['targets_file'])
    
    # Merge on 'id'
    df = pd.merge(x_train, y_train, on='id', how='inner')
    
    # Set datetime index
    df['datetime_CET'] = pd.to_datetime(df['datetime_CET'])
    df.set_index('datetime_CET', inplace=True)
    
    # Drop redundant columns
    df.drop(columns=['id', 'datetime_UTC'], inplace=True, errors='ignore')
    
    return df