
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost

def train_model(data: pd.DataFrame, target_column:str):
    """
    trains an xgboost model and logs with mlflow
    
    :param data: Description
    :type data: pd.DataFrame
    :param target_column: Description
    :type target_column: str
    """

    # X features, y targets
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = 0.2,
        random_state = 21
    )