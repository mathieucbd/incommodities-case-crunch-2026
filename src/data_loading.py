"""Data loading module for INCOMO 3."""

from pathlib import Path

import pandas as pd


DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_data(
    data_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load x_train, y_train, x_test from data/raw/."""
    data_dir = Path(data_dir) if data_dir else DATA_RAW

    x_train = pd.read_csv(data_dir / "x_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv")
    x_test = pd.read_csv(data_dir / "x_test.csv")

    for df in [x_train, y_train, x_test]:
        df["datetime_CET"] = pd.to_datetime(df["datetime_CET"])
        df["datetime_UTC"] = pd.to_datetime(df["datetime_UTC"])
        df.set_index("id", inplace=True)

    return x_train, y_train, x_test


def merge_train(
    x_train: pd.DataFrame, y_train: pd.DataFrame
) -> pd.DataFrame:
    """Merge features and targets for training."""
    return x_train.join(y_train[["fr_spot", "uk_spot"]])
