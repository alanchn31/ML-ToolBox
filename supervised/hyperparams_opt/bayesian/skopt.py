import numpy as np
from functools import partial
from sklearn import ensemble, metrics, model_selection
from skopt import gp_minimize, space
from typing import List


def _optimize(params: dict, param_names: List[str], 
              x: np.array, y: np.array,
              model_name: str = "RandomForestClassifier") -> float:
    """
    Takes all args from the search space & training
    features & targets. Then, initializes the models
    by setting the chosen params and runs cross-validation.
    
    Args:
    - params: list of params from gp_minimize
    - param_names: list of param_names. Order is important
    - x: training_data
    - y: labels/targets
    - model_name: name of model to carry out hyperparam opt

    Returns:
    - Negative accuracy after 5 folds
    """

    params = dict(zip(param_names, params))
    # Hardcode to search for model under sklearn.ensemble
    _model = getattr(ensemble, model_name)
    model = _model(**params)

    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []

    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]
        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)

        fold_accuracy = metrics.accuracy_score(
            ytest, preds
        )
        accuracies.append(fold_accuracy)
    return -1 * np.mean(accuracies)


def skopt_optimize(param_space: List, param_names: List[str], 
                   X: np.array, y: np.array, 
                   model_name: str = "RandomForestClassifier") -> dict:
    """
    Defines the optimization function using param_space &
    param_names. Next, call gp_minimize for bayesian optimization
    for minimization of optimization function.
    
    Args:
    - param_space: hyperparam space to search
    - param_names: list of param_names. Order is important
    - X: training_data
    - y: labels/targets
    - model_name: name of model to carry out hyperparam opt
    

    Returns:
    - best_params: dict of best params and result
    """
    optimization_function = partial(
        _optimize,
        param_names=param_names,
        x=X,
        y=y
    )

    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=15,
        n_random_starts=10,
        verbose=10
    )

    best_params = dict(
        zip(
            param_names,
            result.x
        )
    )

    return best_params