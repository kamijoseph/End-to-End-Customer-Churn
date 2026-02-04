
import pandas as pd

def preprocess_data(data: pd.DataFrame, target_column: str = "Churn") -> pd.DataFrame:

    # clean headers
    data.columns = data.columns.str.strip()

    # dropping ids
    id_columns = [
        "CustomerID",
        "customerID",
        "Customer_id",
        "customer_id",
    ]

    # drop column ids
    for id_column in id_columns:
        if id_column in data.columns:
            data = data.drop(columns=[id_column])
    
    # mapping labels to 0/1
    if target_column in data.columns and data[target_column].dtype == "object":
        data[target_column] = data[target_column].map(
            {
                "No": 0,
                "Yes": 1
            }
        )
    
    # converting TotalCharges from object dtype to float dtype --> coerce blanks
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"],
            errors = "coerce"
        )
    
    # SeniorCitizen -->  0/1 if present
    if "seniorCitizen" in data.columns:
        data["SeniorCitizen"] = data["SeniorCitizen"].fillna(0).astype(int)
    
    # fill missing numeric data with zer0's
    numerical_columns = data.select_dtypes(include=["number"]).columns
    data[numerical_columns] = data[numerical_columns].fillna(0)

    return data