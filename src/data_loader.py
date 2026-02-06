
import pandas as pd
from typing import Tuple
from src.config import DATA_PATH, TARGET_COLUMN

def load_data(path: str = str(DATA_PATH)) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {path}")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataframe based on EDA findings:
    1. Drop customerID (not needed for ML)
    2. Convert TotalCharges to numeric (coerce errors to NaN)
    3. Drop NaN values
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = df.copy()
    
    # 1. Drop customerID if it exists
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)
        
    # 2. Convert TotalCharges to numeric
    # Always try to convert, as some versions of pandas might infer mixed types differently
    # or if we want to be strictly safe against " " strings.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = (
            df["TotalCharges"]
            .astype(str)  # Ensure string methods work
            .str.strip()
            .pipe(pd.to_numeric, errors="coerce")
        )
        
    # 3. Drop missing values (creates 11 drops in TotalCharges)
    df = df.dropna()
    
    return df

def get_data_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: X (features), y (target)
    """
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    return X, y
