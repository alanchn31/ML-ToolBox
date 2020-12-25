import pandas as pd


def generate_date_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """
    Generates date and time features from a datetime column in a dataframe

    Args:
    - df: Input Dataframe
    - datetime_col: name of datetime column to generate features from

    Returns:
      Dataframe with additional columns of new date and time features
    """
    df.loc[:, "year"] = df[datetime_col].dt.year
    df.loc[:, "weekofyear"] = df[datetime_col].dt.weekofyear
    df.loc[:, "month"] = df[datetime_col].dt.month
    df.loc[:, "dayofweek"] = df[datetime_col].dt.dayofweek
    df.loc[:, "weekend"] = (df[datetime_col].dt.weekday >= 5).astype(int)
    df.loc[:, 'hour'] = df[datetime_col].dt.hour
    return df


def agg_date_features(df: pd.DataFrame, datetime_col: str, aggs_dict: dict,
                      id_col="": str) -> pd.DataFrame:
    """
    Generates date and time features from a datetime column in a dataframe
    and aggregate based on date and time columns

    Args:
    - df: Input Dataframe
    - datetime_col: name of datetime column to generate features from
    - aggs_dict: dict containing the date/time label -> function to aggregate 
               eg: {"month": ["nunique", "mean"]}

    Returns:
      Dataframe aggregated with additional columns of new date and time features
    """
    df = generate_date_features(df, datetime_col)
    if id_col == "":
        agg_df = df.agg(aggs_dict)
    else:
        agg_df = df.groupby(id_col).agg(aggs_dict)
    agg_df = agg_df.reset_index()
    return agg_df