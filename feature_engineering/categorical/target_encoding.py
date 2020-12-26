import copy
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from typing import List

def mean_target_encoding(data: pd.DataFrame, feature_cols: List[str], 
                         target_col: str, nfolds: int = 5) -> pd.DataFrame:
    """
    Replace mean target encoding of categorical columns, with respect to a target column

    - data: Input Dataframe
    - feature_cols: list of names of categorical columns to encode
    - target_col: name of target_col to mean target encode to
    - nfolds: number of folds that dataset is split to

    Returns:
      Dataframe with categorical column(s) encoded
    """
    df = copy.deepcopy(data)
    # Store validation dataframes
    encoded_dfs = []
    for fold in range(nfolds):
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        # Iterate through all categorical cols
        for column in feature_cols:
            mapping_dict = dict(
                df_train.groupby(column)[target_col].mean()
            )
            # column_enc is the new column with mean encoding
            df_valid.loc[:, column + "_enc"] = df_valid[column].map(mapping_dict)
        encoded_dfs.append(df_valid)
    encoded_df = pd.concat(encoded_dfs, axis=0)
    return encoded_df
