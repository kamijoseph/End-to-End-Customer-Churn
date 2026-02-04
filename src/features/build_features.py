
import pandas as pd

def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    :param s: Description
    :type s: pd.Series
    :return: Description
    :rtype: Series[Any]

    the function implements binary encoding logic into 0/1 ints on cat features
    the mappings are determnistic and must be consistent btn training and serving
    """

    # get unique vals and remove NaN
    vals = list(
        pd.Series(s.dropna().unique()).astype(str)
    )
    valset = set(vals)

    # gender mapping
    if valset == {"Male", "female"}:
        return s.map(
            {
                "Female": 0,
                "Male": 1
            }
        ).astype("Int64")
    
    # Yes/No mapping
    if valset == {"Yes", "No"}:
        return s.map(
            {
                "No": 0,
                "Yes": 1
            }
        ).astype("Int64")
    
    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {
            sorted_vals[0]: 0,
            sorted_vals[1]: 1
        }
        return s.astype(str).map(mapping).astype("Int64")
    
    return s

def build_features(data: pd.DataFrame, target_column: str = "Churn") -> pd.DataFrame:
    """
    Docstring for build_features
    
    :param data: Description
    :type data: pd.DataFrame
    :param target_column: Description
    :type target_column: str
    :return: Description
    :rtype: DataFrame

    applying complete feature engineering pipeline for training data
    """

    data = data.copy()
    print("starting feature engineering on {data.shape[1]} columns.....")

    # feture types
    object_columns = [
        column for column in data.select_dtypes(include=["object"]).columns if column != target_column
    ]
    numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(
        f"found {len(object_columns)} categorical abd {len(numeric_columns)} numerical columns"
    )

    # split file by cardinality
    binary_columns = [
        column for column in object_columns if data[column].dropna().nunique() == 2
    ]
    multi_columns = [
        column for column in object_columns if data[column].dropna().nunique() > 2
    ]

    print(
        f" binary features: {len(binary_columns)} | multi-category features: {len(multi_columns)}"
    )

    # 0/1 determinnistic mappping
    for column in binary_columns:
        original_dtype = data[column].dtype
        data[column] = _map_binary_series(data[column].astype(str))

    # apply binary encoding
    bool_columns = data.select_dtypes(include=["bool"]).columns.tolist()
    if bool_columns:
        data[bool_columns] = data[bool_columns].astype(int)

    # multi category features
    if multi_columns:
        original_shape = data.shape

        # one hot ncode
        data = pd.get_dummies(
            data,
            columns = multi_columns,
            drop_first = True
        )
    
    # convert (Int64) --> standard integers fr xgboost
    for column in binary_columns:
        if pd.api.types.is_integer(data[column]):
            data[column] = data[column].fillna(0).astype(int)
    
    print(
        f"feature building complete: {data.shape[1]} final features"
    )

    return data