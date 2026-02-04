
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