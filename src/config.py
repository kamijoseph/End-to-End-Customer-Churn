
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "churn.csv"
ARTIFACTS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Data Processing
RANDOM_STATE = 21
TEST_SIZE = 0.2
TARGET_COLUMN = "Churn"
THRESHOLD = 0.3

# Features
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

NUMERICAL_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# Mappings (from EDA.ipynb)
YES_NO_MAP = {"Yes": 1, "No": 0}
GENDER_MAP = {"Male": 1, "Female": 0}

# Columns to apply YES_NO_MAP to (before OHE, binary columns)
BINARY_MAP_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"
]

# Columns for OneHotEncoding (multiclass)
# Note: "gender" is handled separately via GENDER_MAP
MULTI_CAT_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod"
]

# XGBoost Hyperparameters (Optuna best params + fixed params)
XGB_PARAMS = {
    'n_estimators': 611,
    'learning_rate': 0.010166070843080646,
    'max_depth': 3,
    'subsample': 0.5751505439042802,
    'colsample_bytree': 0.5133132195701255,
    'min_child_weight': 1,
    'gamma': 2.6591161997534876,
    'reg_alpha': 4.922806509383804,
    'reg_lambda': 4.477119923542774,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'eval_metric': 'logloss'
    # scale_pos_weight will be calculated dynamically during training
}
