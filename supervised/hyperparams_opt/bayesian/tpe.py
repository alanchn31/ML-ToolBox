import numpy as np
from functools import partial
from sklearn import ensemble, metrics, model_selection
from hyperopt import hp, fmin, tpe, Trials
from typing import List


def _optimize(params: dict, 
              x: np.array, y: np.array,
              model_name: str = "RandomForestClassifier") -> float:
    """
    Takes all args from the search space & training
    features & targets. Then, initializes the models
    by setting the chosen params and runs cross-validation.
    
    Args:
    - params: dict of params from hyperopt
    - x: training_data
    - y: labels/targets
    - model_name: name of model to carry out hyperparam opt

    Returns:
    - Negative accuracy after 5 folds
    """

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


def hyopt_tpe_optimize(param_space: dict, X: np.array, y: np.array, 
                       model_name: str = "RandomForestClassifier") -> dict:
    """
    Defines the optimization function using param_space &
    param_names. Next, call gp_minimize for bayesian optimization
    for minimization of optimization function.
    
    Args:
    - param_space: hyperparam space to search
      eg: {
          "max_depth": scope.int(hp.quniform("max_depth", 1, 15, 1)),
          "n_estimators": scope.int(hp.quniform("n_estimators", 100, 1500, 1)),
          "criterion": hp.choice("criterion", ["gini", "entropy"]),
          "max_features": hp.uniform("max_features", 0, 1)
      }
    - X: training_data
    - y: labels/targets
    - model_name: name of model to carry out hyperparam opt
    

    Returns:
    - hopt: dict of best params and result
    """
    optimization_function = partial(
        _optimize,
        x=X,
        y=y
    )

    trials = Trials()

    hopt = fmin(
        fn=optimization_function,
        space=param_space,
        algo=tpe.suggest,
        max_evals=15,
        trials=trials
    )
    return hopt