
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any
import uvicorn
import pandas as pd
from contextlib import asynccontextmanager

from src.predict import Predictor

# Global predictor instance
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model artifacts on startup.
    """
    global predictor
    try:
        predictor = Predictor()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # We don't raise here to allow health check to run, 
        # but predict will fail if predictor is None.
    yield
    # Cleanup if needed

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using XGBoost",
    version="1.0.0",
    lifespan=lifespan
)

# Input Schema
class CustomerData(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["No phone service", "No", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No internet service", "No", "Yes"]
    OnlineBackup: Literal["No internet service", "No", "Yes"]
    DeviceProtection: Literal["No internet service", "No", "Yes"]
    TechSupport: Literal["No internet service", "No", "Yes"]
    StreamingTV: Literal["No internet service", "No", "Yes"]
    StreamingMovies: Literal["No internet service", "No", "Yes"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: str  # Kept as string to match training handling, though float is cleaner. The model pipeline handles string coercion.

class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """
    if predictor and predictor.model:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    """
    Predict churn for a single customer.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert Pydantic model to dict
        data = customer.model_dump()
        
        # Predict
        result = predictor.predict(data)
        
        return PredictionResponse(
            churn_prediction=result["churn_prediction"][0],
            churn_probability=result["churn_probability"][0]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
