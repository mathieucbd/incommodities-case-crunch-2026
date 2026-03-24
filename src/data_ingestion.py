import pandas as pd
from pathlib import Path
import yaml


def load_competition_data(config_path="config.yaml", mode="train"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve raw_dir relative to the config file's location to handle sub-directory execution (e.g. notebooks/)
    config_abs_path = Path(config_path).resolve()
    raw_dir = config_abs_path.parent / config["data"]["raw_dir"]

    if mode == "train":
        # Load features and targets then merge on shared id/timestamps.
        x_df = pd.read_csv(raw_dir / config["data"]["features_file"])
        y_df = pd.read_csv(raw_dir / config["data"]["targets_file"])
        df = pd.merge(
            x_df,
            y_df,
            on=["id", "datetime_CET", "datetime_UTC"],
            how="inner",
        )
    elif mode == "test":
        # Kaggle test set has no targets; ingest feature matrix only.
        df = pd.read_csv(raw_dir / config["data"]["test_features_file"])
    else:
        raise ValueError("mode must be either 'train' or 'test'")

    # Set datetime index
    index_col = config["data"].get("index_col", "datetime_CET")
    if index_col not in df.columns:
        raise KeyError(f"Configured index_col '{index_col}' not found in loaded data")
    df[index_col] = pd.to_datetime(df[index_col])
    df.set_index(index_col, inplace=True)

    # Drop redundant columns
    df.drop(columns=["id", "datetime_UTC"], inplace=True, errors="ignore")

    # Forward-fill sparse daily features mechanically
    sparse_features = config.get("data", {}).get("sparse_daily_features", [])
    for col in sparse_features:
        if col in df.columns:
            df[col] = df[col].ffill()

    return df
