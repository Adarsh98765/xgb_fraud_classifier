import numpy as np
import pandas as pd

def load_and_clean_data(filepath: str, target_col: str = "Class") -> tuple:
    """
    Loads and preprocesses credit card data.
    Removes duplicates, handles nulls, drops highly correlated features.
    Returns features (X) and target (y).
    """
    df = pd.read_csv(filepath)

    # Drop duplicates
    df = df.drop_duplicates()

    # Handle nulls
    if df.isnull().sum().sum() > 0:
        df = df.dropna() 

    # Drop constant columns (if any)
    nunique = df.nunique()
    const_cols = nunique[nunique == 1].index.tolist()
    df.drop(columns=const_cols, inplace=True)
    
    # Drop non-informative columns
    if "id" in df.columns:
        df = df.drop(columns=["id"])


    # Drop highly correlated features (if any)
    corr_matrix = df.drop(columns=[target_col]).corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

    )
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    df.drop(columns=to_drop, inplace=True)

    # Return split data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
