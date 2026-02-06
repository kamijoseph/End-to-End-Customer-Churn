
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import (
    YES_NO_MAP, GENDER_MAP, BINARY_MAP_COLS, MULTI_CAT_COLS
)

class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to handle:
    1. Binary mapping (Yes/No -> 1/0, Gender -> 1/0)
    2. One-Hot Encoding for multi-category columns
    3. Alignment of columns to ensure inference consistency
    """
    
    def __init__(self):
        self.columns_ = None  # To store the final column order seen during fit
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer. Applies mappings and OHE to learn the resulting columns.
        """
        # We need to simulate the transformation to know the final columns
        X_transformed = self._transform(X)
        self.columns_ = X_transformed.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data.
        """
        X_transformed = self._transform(X)
        
        # Ensure columns match the fit time columns (for inference consistency)
        if self.columns_:
            # Add missing columns with 0s
            for col in self.columns_:
                if col not in X_transformed.columns:
                    X_transformed[col] = 0
            
            # Reorder columns to match fit and drop extra columns (if any unseen categories appeared)
            X_transformed = X_transformed[self.columns_]
            
        return X_transformed

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # 1. Binary Mappings
        # Apply Yes/No map
        for col in BINARY_MAP_COLS:
            if col in X.columns:
                X[col] = X[col].map(YES_NO_MAP)
        
        # Apply Gender map
        if "gender" in X.columns:
            X["gender"] = X["gender"].map(GENDER_MAP)
            
        # 2. One-Hot Encoding
        # pd.get_dummies is used in the notebook. 
        # We use it here but we must control for column consistency via fit/transform pattern.
        X = pd.get_dummies(
            X,
            columns=[c for c in MULTI_CAT_COLS if c in X.columns],
            drop_first=True
        )
        
        return X
