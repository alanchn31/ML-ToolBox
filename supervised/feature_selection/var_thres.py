import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def var_thres(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """
    Remove columns with variance less than 0.1

    Args:
    - df: Input Dataframe
    - threshold: Variance threshold. 
                Columns less than threshold will be removed.

    Returns:
        Dataframe with columns having variance > threshold
    """
    var_thresh = VarianceThreshold(threshold=threshold)
    var_thresh.fit(df)
    return var_thresh, var_thresh.transform(df)


