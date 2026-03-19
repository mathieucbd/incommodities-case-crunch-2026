import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler

def chronological_train_val_test_split(
    df: pd.DataFrame, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataframe strictly chronologically to prevent lookahead bias.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The full dataset sorted chronologically.
    val_ratio : float
        Proportion of the dataset to allocate to the validation set.
    test_ratio : float
        Proportion of the dataset to allocate to the test set.
        
    Returns:
    --------
    train_df, val_df, test_df : tuple of pd.DataFrames
    """
    n = len(df)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_size = n - val_size - test_size

    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size:].copy()

    return train_df, val_df, test_df


def scale_data(
    X_train: pd.DataFrame, 
    X_val: Optional[pd.DataFrame] = None, 
    X_test: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], StandardScaler]:
    """
    Scales features using StandardScaler fit STRICTLY on X_train to prevent data leakage.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features.
    X_val : pd.DataFrame, optional
        Validation features.
    X_test : pd.DataFrame, optional
        Testing features.
        
    Returns:
    --------
    X_train_scaled, X_val_scaled, X_test_scaled, scaler : Tuple
        Scaled datasets as DataFrames (keeping indices/columns) and the fitted scaler.
    """
    scaler = StandardScaler()
    
    # Fit and transform on training data only
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    
    # Transform validation and test sets purely using train weights
    X_val_scaled = None
    if X_val is not None:
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val), 
            columns=X_val.columns, 
            index=X_val.index
        )
        
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), 
            columns=X_test.columns, 
            index=X_test.index
        )
        
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
