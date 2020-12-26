import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


def rfe(X: np.array, y: np.array, n_features_to_select: int = 3):
    """
    Performs recursive feature elimination

    Args:
        - X: Input numpy array (train_data)
        - y: Input numpy array (train_data labels)
        - n_features_to_select: number of features to keep

    Returns:
        - X_transformed: Output numpy array (with columns filtered)
        - y: numpy array (train_data labels)
        - rfe: fitted RFE object
    """
    model = LinearRegression()
    rfe = RFE(
        estimator=model,
        n_features_to_select=n_features_to_select
    )

    rfe.fit(X, y)
    X_transformed = rfe.transform(X)
    return X_transformed, y, rfe