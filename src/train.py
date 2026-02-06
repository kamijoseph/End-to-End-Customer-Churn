
import mlflow
import mlflow.xgboost
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

from src.config import (
    RANDOM_STATE, TEST_SIZE, XGB_PARAMS, ARTIFACTS_DIR, THRESHOLD
)
from src.data_loader import load_data, clean_data, get_data_split
from src.preprocessing import FeatureTransformer

def train_model():
    """
    Main training function.
    1. Loads and cleans data
    2. Splits data
    3. Preprocesses data (fit_transform on train, transform on test)
    4. Calculates scale_pos_weight
    5. Trains XGBoost
    6. Logs to MLflow
    7. Saves artifacts
    """
    # 1. Load Data
    print("Loading data...")
    df = load_data()
    df = clean_data(df)
    
    X, y = get_data_split(df)
    
    # 2. Split Data
    # Note: Preprocessing (OHE) handles binary columns including 'Churn' if passed, 
    # but 'Churn' is separated in y. y needs mapping too since get_data_split returns original 'Yes'/'No'
    # Wait, clean_data drops NaN but config.py has target mapping
    # Let's map target manually here or ensure cleaner handles it.
    # The notebook mapped target *before* split.
    # My FeatureTransformer handles 'Churn' if it's in X. But here y is 'Churn'.
    # I should map y explicitly.
    from src.config import YES_NO_MAP
    y = y.map(YES_NO_MAP)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    # 3. Preprocessing
    print("Preprocessing...")
    preprocessor = FeatureTransformer()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 4. Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")
    
    # Update params
    params = XGB_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight
    
    # 5. Train Model
    print("Training model...")
    model = XGBClassifier(**params)
    model.fit(X_train_processed, y_train)
    
    # 6. Evaluation
    print("Evaluating...")
    # Predict probability
    y_prob = model.predict_proba(X_test_processed)[:, 1]
    # Apply threshold
    y_pred = (y_prob >= THRESHOLD).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    
    print(f"Metrics: {metrics}")
    
    # 7. MLflow Logging
    mlflow.set_experiment("Customer_Churn_Prediction")
    
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("threshold", THRESHOLD)
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
        
        # Save artifacts locally
        model_path = ARTIFACTS_DIR / "xgb_model.pkl"
        preprocessor_path = ARTIFACTS_DIR / "preprocessor.pkl"
        
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            
        with open(preprocessor_path, "wb") as f:
            pickle.dump(preprocessor, f)
            
        mlflow.log_artifact(str(model_path), "artifacts")
        mlflow.log_artifact(str(preprocessor_path), "artifacts")
        
    print(f"Training complete. Artifacts saved to {ARTIFACTS_DIR}")

if __name__ == "__main__":
    train_model()
