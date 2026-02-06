
import os
import pandas as pd
import mlflow
import glob

MODEL_DIR = "/app/model/"

try:
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(
        f"model loaded succesfully from {MODEL_DIR}"
    )
except Exception as e:
    print(
        f"error loading model from {MODEL_DIR}\n: {e}"
    )

    try:
        local_model_paths = glob.glob("./mlruns/*/*/artifacts/model")
        if local_model_paths:
            latest_model = max(local_model_paths, key=os.path.getmtime)
            model = mlflow.pyfunc.load_model(latest_model)
            MODEL_DIR = latest_model
            print(f"✅ Fallback: Loaded model from {latest_model}")
        else:
            raise Exception("No model found in local mlruns")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")
    
try:
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

# deterministic binary feature mappings (consistent with training)
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1}, 
    "Partner": {"No": 0, "Yes": 1},  
    "Dependents": {"No": 0, "Yes": 1},     
    "PhoneService": {"No": 0, "Yes": 1},   
    "PaperlessBilling": {"No": 0, "Yes": 1}, 
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(data: pd.DataFrame) -> pd.DataFrame:

    """
    apply identical feature transformation as used during training
    
    :param data: Description
    :type data: pd.DataFrame
    :return: Description
    :rtype: DataFrame
    """

    data = data.copy()
    data.columns = data.columns.str.strip()

    # ensure numeric columns are properly typed (handle string inputs)
    for c in NUMERIC_COLS:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")
            data[c] = data[c].fillna(0)
    
    # apply deterministic mappings for binary features
    for c, mapping in BINARY_MAP.items():
        if c in data.columns:
            data[c] = (
                data[c]
                .astype(str)                   
                .str.strip()                 
                .map(mapping)               
                .astype("Int64")               
                .fillna(0)                     
                .astype(int)
            )
    
    obj_cols = [c for c in data.select_dtypes(include=["object"]).columns]
    if obj_cols:
        data = pd.get_dummies(data, columns=obj_cols, drop_first=True)
    
    bool_cols = data.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        data[bool_cols] = data[bool_cols].astype(int)

    data = data.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return data

def predict(input_dict: dict) -> str:
    data = pd.DataFrame([input_dict])
    data_enc = _serve_transform(data)

    try:
        preds = model.predict(data_enc)
        
        # normalize prediction output to consistent format
        if hasattr(preds, "tolist"):
            preds = preds.tolist()  # Convert numpy array to list
            
        # extract single prediction value (for single-row input)
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    # convert binary prediction (0/1) to actionable business language
    if result == 1:
        return "Likely to churn" 
    else:
        return "Not likely to churn" 