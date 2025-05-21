import pandas as pd
import numpy as np

def evaluate_wape(targets, predictions):
    '''
    Returns a WAPE value (Weighted Average Percentage Error)
        Parameters:
            1. targets: A numpy array containing target values
            2. predictions: A numpy array containing forecasts

        Returns:
            wape: WAPE of predictions vs targets
    '''
    resids_summed = np.sum(abs(targets-predictions))
    targets_summed = np.sum(targets)
    if targets_summed == 0:
        return np.nan
    wape = resids_summed/targets_summed
    return wape