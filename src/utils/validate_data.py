
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