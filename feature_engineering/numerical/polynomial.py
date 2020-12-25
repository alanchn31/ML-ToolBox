import pandas as pd
from sklearn import preprocessing
from typing import List


def generate_polynomial_feats(df: pd.DataFrame, feat_cols: List[str], 
                             degree=2: int) -> pd.DataFrame:
    """
    Generates date and time features from a datetime column in a dataframe

    Args:
    - df: Input Dataframe
    - feat_cols: names of numerical columns to generate poly features from
    - degree: degree of polynomial feats to be generated

    Returns:
      Dataframe with additional columns of poly features
    """
    pf = preprocessing.PolynomialFeatures(
        degree=degree,
        interaction_only=False,
        include_bias=False
    )
    for col in feat_cols:
        pf.fit(df[col])
        poly_feats = pf.transform(df[col])
        num_feats = poly_feats.shape[1]
        df_transformed = pd.DataFrame(
            poly_feats,
            columns=[f"{col}_{i}" for i in range(1, num_feats+1)]
        )
        df = pd.concat([df, df_transformed], axis=1)
    return df