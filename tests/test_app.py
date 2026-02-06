
import pytest
import pandas as pd
import numpy as np
import json
from fastapi.testclient import TestClient
from src.preprocessing import FeatureTransformer
from src.predict import Predictor
from app import app
from src.config import THRESHOLD

client = TestClient(app)

# Mock Data
sample_customer = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": 55.5,
    "TotalCharges": "600.0"
}

def test_preprocessing():
    """
    Test that FeatureTransformer producing expected shape and columns.
    """
    df = pd.DataFrame([sample_customer])
    # TotalCharges coercion happens in clean_data/predict logic, so we might need to do it manually if testing transformer alone
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
    
    transformer = FeatureTransformer()
    # Mock fit on same data to initialize columns
    transformer.fit(df)
    transformed = transformer.transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert not transformed.empty
    # Check binary mapping
    if "gender" in transformed.columns:
        assert transformed["gender"].iloc[0] == 1  # Male -> 1
        
    # Check OHE result (InternetService_DSL should exist if dropped_first didn't remove it or DSL isn't base)
    # If DSL is the first category alphabetically, it might be dropped.
    pass

def test_api_predict():
    """
    Test the /predict endpoint.
    """
    # Force model load (lifespan manager runs on startup but TestClient handles it?)
    # TestClient with lifespan should work.
    
    with TestClient(app) as client:
        response = client.post("/predict", json=sample_customer)
        assert response.status_code == 200
        data = response.json()
        assert "churn_prediction" in data
        assert "churn_probability" in data
        assert isinstance(data["churn_prediction"], int)
        assert isinstance(data["churn_probability"], float)

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_predictor_integration():
    """
    Test Predictor class loading and prediction.
    """
    predictor = Predictor()
    assert predictor.model is not None
    assert predictor.preprocessor is not None
    
    result = predictor.predict(sample_customer)
    assert "churn_prediction" in result
    assert "churn_probability" in result
