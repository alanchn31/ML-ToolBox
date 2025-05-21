from skforecast.preprocessing import RollingFeatures

def eng_rolling_features(predictors: list, window_sizes: list):
    '''
    From: https://skforecast.org/0.14.0/user_guides/window-features-and-custom-features.html#rollingfeatures
    'mean': the mean of the previous n values.
    'std': the standard deviation of the previous n values.
    'min': the minimum of the previous n values.
    'max': the maximum of the previous n values.
    'sum': the sum of the previous n values.
    'median': the median of the previous n values.
    'ratio_min_max': the ratio between the minimum and maximum of the previous n values.
    'coef_variation': the coefficient of variation of the previous n values.
    '''
    pred_expanded = []

    # Expand predictors to match window_size
    # For eg, 2 predictors, mean & median, with window_size 1,7,14
    # Expand predictors to match length of window size [mean,mean,mean,median,median,median]
    # with window_size: [1,7,14,1,7,14]
    for pred in predictors:
        pred_expanded += [pred]*len(window_sizes)
    
    window_size_expanded = window_sizes*len(predictors)

    print(f"predictors: {pred_expanded}")
    print(f"window size: {window_size_expanded}")

    return RollingFeatures(stats=pred_expanded, window_sizes=window_size_expanded)