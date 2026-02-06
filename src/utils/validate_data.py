
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

    # tenure must be non-negative (business logic - can't have negative tenure)
    ge_df.expect_column_values_to_be_between("tenure", min_value=0)
    
    # monthly charges must be positive (business logic - no free service)
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0)
    
    # total charges should be non-negative (business logic)
    ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)

    # tenure should be reasonable (max ~10 years = 120 months for telecom)
    ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
    
    # monthly charges should be within reasonable business range
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)
    
    # no missing values in critical numeric features  
    ge_df.expect_column_values_to_not_be_null("tenure")
    ge_df.expect_column_values_to_not_be_null("MonthlyCharges")
    
    # total charges should generally be >= Monthly charges (except for very new customers)
    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95
    )
    
    # running validation suite
    print("running complete validation suite..........")
    results = ge_df.validate()

    # Extract failed expectations for detailed error reporting
    failed_expectations = []
    for r in results["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_type"]
            failed_expectations.append(expectation_type)
    
    # print validation summary
    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks
    
    if results["success"]:
        print(f"data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"failed expectations: {failed_expectations}")
    
    return results["success"], failed_expectations