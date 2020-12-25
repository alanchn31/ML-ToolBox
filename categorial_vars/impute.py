import pandas as pd


def impute_rare_category(df: pd.Dataframe, cat_col: str, threshold: int,
                         rare_label="RARE") -> pd.Dataframe:
    """
    Replace categorical label of a column with "Rare", if its count 
    is less than a threshold. Deals with rare or unknown categories

    df: Input Dataframe
    cat_col: name of categorical column to be imputed
    threshold: count threshold to change label of category to rare_label
    rare_label: label to be imputed for rare category

    Returns:
      Dataframe with categorical column imputed
    """

    df.loc[df[cat_col].value_counts()][df[cat_col]].values < threshold,
           cat_col
    ] = rare_label
    return df