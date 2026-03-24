import sys
from pathlib import Path
sys.path.append(str(Path("/Users/paul/Desktop/Incommodities - Cruch/INCOMO 3")))
from src.data_loading import load_data, merge_train
df_raw = load_data("/Users/paul/Desktop/Incommodities - Cruch/INCOMO 3/data/raw")
df = merge_train(df_raw['x_train'], df_raw['y_train'])
from src.feature_engineering import build_features
from omegaconf import OmegaConf
config = OmegaConf.create({"data": {"raw_dir": "data/raw"}, "features": {"rolling_windows": [24, 168], "lags": [24, 48, 168], "quantiles": [0.1, 0.5, 0.9]}})
X = build_features(df, config)
print(X.columns[X.columns.str.contains('flow|spot|gas|load')].tolist())
