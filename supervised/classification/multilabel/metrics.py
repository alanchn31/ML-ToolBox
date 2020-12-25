import pandas as pd
import numpy as np
from typing import Union, List


def precision_at_k(y_true: Union[List[float], np.array, pd.Series], 
                   y_pred: Union[List[float], np.array, pd.Series],
                   k: int) -> float:
    """
    Calculates precision for top-k predictions, divided by k

    - y_true: Input series/array with true labels
    - y_pred: Input series/array with predicted labels 

    Returns:
        precision at given value k
    """
    # k should always be >= 1
    if k <= 0:
        return 0
    # Look at only top k predictions
    y_pred = y_pred[:k]
    # Convert predictions to set
    pred_set = set(y_pred)
    # Convert actual values to set
    true_set = set(y_true)
    # Find common values:
    common_values = pred_set.intersection(true_set)
    # Return length of common values over k
    return len(common_values)/len(y_pred)


def average_precision_at_k(y_true: Union[List[float], np.array, pd.Series], 
                           y_pred: Union[List[float], np.array, pd.Series],
                           k: int) -> float:
    """
    Calculates average precision for top-k predictions, divided by k

    - y_true: Input series/array with true labels
    - y_pred: Input series/array with predicted labels 
            (needs to be sorted in desc order, according to prob)

    Returns:
        Average precision at given value k
    """
    # Initialize list of p@k values
    pk_values = []
    # Loop from 1 to k and store p@k
    for i in range(1, k+1):
        pk_values.append(precision_at_k(y_true, y_pred, i))

    if len(pk_values) == 0:
        return 0
    return sum(pk_values) / len(pk_values)
    

def mapk(y_true: Union[List[float], np.array, pd.Series], 
         y_pred: Union[List[float], np.array, pd.Series],
         k: int) -> float:
    """
    Calculates mean average precision for top-k predictions, divided by k

    - y_true: Input series/array with true labels
    - y_pred: Input series/array with predicted labels 
            (needs to be sorted in desc order, according to prob)

    Returns:
        Mean average precision at given value k
    """
    # Initialize list of ap@k values
    apk_values = []
    # Loop over all samples
    for i in range(len(y_true)):
        apk_values.append(average_precision_at_k(y_true[i], y_pred[i], k=k))
    return sum(apk_values)/len(apk_values)