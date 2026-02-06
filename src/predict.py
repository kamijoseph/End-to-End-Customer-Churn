
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
from src.config import ARTIFACTS_DIR, THRESHOLD

class Predictor:
    """
    Class to load model artifacts and make predictions.
    """
    
    def __init__(self, model_dir: Path = ARTIFACTS_DIR):
        self.model_dir = model_dir
        self.model = None
        self.preprocessor = None
        self._load_artifacts()
        
    def _load_artifacts(self):
        """
        Load the trained model and preprocessor from disk.
        """
        model_path = self.model_dir / "xgb_model.pkl"
        preprocessor_path = self.model_dir / "preprocessor.pkl"
        
        if not model_path.exists() or not preprocessor_path.exists():
            raise FileNotFoundError(f"Artifacts not found in {self.model_dir}. Please train the model first.")
            
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
            
        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)
            
    def predict(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Make a prediction for a single instance or batch.
        
        Args:
            data: Dictionary of features or DataFrame.
            
        Returns:
            Dictionary containing 'churn_prediction' (int), 'churn_probability' (float).
        """
        # Convert dict to DataFrame if necessary
        if isinstance(data, dict):
            # If scalar values, wrap in list
            data = {k: [v] if not isinstance(v, list) else v for k, v in data.items()}
            df = pd.DataFrame(data)
        else:
            df = data.copy()
            
        # Preprocess
        # Note: clean_data handles "TotalCharges" numeric conversion.
        # We should replicate basic cleaning for inference if raw input comes in.
        # Assuming input matches schema (no customerID, but TotalCharges as string maybe?)
        
        if "TotalCharges" in df.columns and df["TotalCharges"].dtype == "object":
             df["TotalCharges"] = (
                df["TotalCharges"]
                .str.strip()
                .pipe(pd.to_numeric, errors="coerce")
            )
            # In inference, we can't just drop na if it's a single request. 
            # We should probably fill with 0 or mean, or let it fail. 
            # For now, let's fill with 0 to prevent crash, or assume valid input.
            # config.py doesn't specify inference fill strategy.
            # Notebook just dropped na.
             df["TotalCharges"] = df["TotalCharges"].fillna(0)

        # Transform
        X_processed = self.preprocessor.transform(df)
        
        # Predict
        prob = self.model.predict_proba(X_processed)[:, 1]
        pred = (prob >= THRESHOLD).astype(int)
        
        return {
            "churn_prediction": pred.tolist(),
            "churn_probability": prob.tolist()
        }
