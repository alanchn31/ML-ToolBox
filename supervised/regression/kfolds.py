import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


def create_stratified_k_folds(df: pd.Dataframe, target_col: str = "target") -> pd.Dataframe:
    """
    Creates stratified Kfolds for any regression problem

    Args:
        df: Input Dataframe for regression problem

    Returns:
        df: shuffled Dataframe with kfold indicated.
    """
    # Create a new column called kfold and fill with -1
    df["kfold"] = -1
    # Randomize rows of the data:
    df = df.sample(frac=1).reset_index(drop=True)
    # Calculate no. of bins by Sturge's rule
    num_bins = int(np.floor(1 + np.log2(len(df))))
    # Bin targets
    df.loc[:, "bins"] = pd.cut(df[target_col], bins=num_bins, labels=False)
    # Initialize kfold class from model_selection
    kf = StratifiedKFold(n_splits=5)
    # Fill kfold columns using bins column
    for fold, (t_, v_) in enumerate(kf.split(X=df, y=df.bins.values)):
        df.loc[v_, "kfold"] = fold
    df = df.drop("bins", axis=1)
    return df
