
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):

    """
    evaluate an xgboost model on test data
    
    :param model: Description
    :param X_test: Description
    :param y_test: Description
    """

    y_pred = model.predict(X_test)
    clf_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("classification report:\n", clf_report)
    print(f"confusion matrix:\n", conf_matrix)