
import pandas as pd
import numpy as np
from tsfresh.feature_extraction import feature_calculators as fc

def generate_ts_signals(x: Union[List[float], np.array, pd.Series]) -> dict:
    """
    Generates time series features from a time series array

    - x: Input array/series

    Returns:
      dict with time series signals generated
    """

    feature_dict = {}
    feature_dict['abs_energy'] = fc.abs_energy(x)
    feature_dict['count_above_mean'] = fc.count_above_mean(x)
    feature_dict['count_below_mean'] = fc.count_below_mean(x)
    feature_dict['mean_abs_change'] = fc.mean_abs_change(x)
    feature_dict['mean_change'] = fc.mean_change(x)
    return feature_dict