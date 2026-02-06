
import great_expectations as ge
from typing import Tuple, List
import pandas as pd

def validate_telco_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:

    """
    comprehensive data validation for telco customer churn dataset using great expectations
    
    :param data: Description
    :type data: pd.DataFrame
    :return: Description
    :rtype: Tuple[bool, List[str]]
    """

    print(" staarting validation with great expectations..........")

    # convert pd Dataframe to great expectation dataset
    ge_df = ge.dataset.PandasDataset(data)

    # customer identifier must exist (required for business operations)  
    ge_df.expect_column_to_exist("customerID")
    ge_df.expect_column_values_to_not_be_null("customerID")

    # core demographic features
    ge_df.expect_column_to_exist("gender") 
    ge_df.expect_column_to_exist("Partner")
    ge_df.expect_column_to_exist("Dependents")

    # service features (critical for churn analysis)
    ge_df.expect_column_to_exist("PhoneService")
    ge_df.expect_column_to_exist("InternetService")
    ge_df.expect_column_to_exist("Contract")
    
    # financial features (key churn predictors)
    ge_df.expect_column_to_exist("tenure")
    ge_df.expect_column_to_exist("MonthlyCharges")
    ge_df.expect_column_to_exist("TotalCharges")

    # gender must be one of expected values (data integrity)
    ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
    
    # Yes/No fields must have valid values
    ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])
    
    # contract types must be valid (business constraint)
    ge_df.expect_column_values_to_be_in_set(
        "Contract", 
        ["Month-to-month", "One year", "Two year"]
    )
    
    # internet service types (business constraint)
    ge_df.expect_column_values_to_be_in_set(
        "InternetService",
        ["DSL", "Fiber optic", "No"]
    )