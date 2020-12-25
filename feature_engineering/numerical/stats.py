import pandas as pd
import numpy as np
from typing import List


def generate_stats_features(df: pd.DataFrame, feature_cols: List[str],
                            label_prefix="feat": str):
    """
    Generates date and time features from a datetime column in a dataframe

    Args:
    - df: Input Dataframe
    - feature_cols: name(s) of numerical column(s) to generate features from
    - new features' label_prefixes (default: feat)

    Returns:
      Dataframe with additional columns of new stat features
    """
    df[f'{label_prefix}_sum'] = df[feature_cols].sum(axis=1)
    df[f'{label_prefix}_mean'] = df[feature_cols].mean(axis=1)
    df[f'{label_prefix}_std'] = df[feature_cols].std(axis=1)
    df[f'{label_prefix}_kurt'] = df[feature_cols].kurtosis(axis=1)
    df[f'{label_prefix}_skew'] = df[feature_cols].skew(axis=1)
    df[f'{label_prefix}_min'] = df[feature_cols].min(axis=1)
    df[f'{label_prefix}_max'] = df[feature_cols].max(axis=1)
    # peak-to-peak (max - min):
    df[f'{label_prefix}_ptp'] = np.ptp(df[feature_cols].values, axis=1)


    
    # Quantile features:
    df[f'{label_prefix}_quantile_5'] = df[feature_cols].quantile(q=0.05, axis=1)
    df[f'{label_prefix}_quantile_95'] = df[feature_cols].quantile(q=0.95, axis=1)
    df[f'{label_prefix}_quantile_99'] = df[feature_cols].quantile(q=0.99, axis=1)
    return df