
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

    # model initialization
    model = XGBClassifier(
        n_estimators = 300,
        learning_rate = 0.1,
        max_depth = 6,
        random_state = 21,
        n_jobs = -1,
        eval_metric = "logloss"
    )

    with mlflow.start_run():

        # train model
        model.fit(
            X_train,
            y_train
        )

        # prediction
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rec_score = recall_score(y_test, y_pred)

        # log params, metrics, and model
        mlflow.log_param("n_estimators", 300)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metrics("recall", rec_score)
        mlflow.xgboost.log_model(model, "model")

        train_ds = mlflow.data.from_pandas(
            data,
            source = "training_data"
        )
        mlflow.log_input(
            train_ds,
            context = "training"
        )

        print(
            f"model trained. accuracy: {acc:.4f}, recall: {rec_score:.4f}"
        )