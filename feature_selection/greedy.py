import numpy as np
import pandas as pd
from sklearn import metrics, linear_model
from typing import Union


class GreedyFeatureSelection:
    """
    Custom greedy feature selection wrapper on
    different greedy selection models from sklearn
    """

    def evaluate_score(self, df: pd.DataFrame, 
                       target_col: str) -> float:
        """
        Evaluates model on data and returns AUC
        fits on entire dataside instead of OOF

        Args:
        - df: input dataframe (must have kfold column)
        - target_col: name of target col

        Returns:
        - Overfitted AUC
        """
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        model = linear_model.LogisticRegression()
        model.fit(X, y)
        predictions = model.predict_proba(X)[:, 1]
        auc = metrics.roc_auc_score(y, predictions)
        return auc
    

    def evaluate_oof_score(self, df: pd.DataFrame, 
                           target_col: str,
                           nfolds=5: int) -> float:
        """
        Evaluates model on data and returns OOF AUC

        Args:
        - df: input dataframe (must have kfold column)
        - target_col: name of target col
        - nfolds: number of folds assigned to train data

        Returns:
        - OOF AUC
        """
        if "kfold" not in X.columns:
            raise Exception("Error, kfold column not found, \
                            please use evaluate_score method instead")
        model = linear_model.LogisticRegression()
        auc = 0
        for fold in range(nfolds):
            train = df.loc[df["kfold"] != fold, :]
            valid = df.loc[df["kfold"] == fold, :]
            train_X = train.drop(columns=["kfold"] + [target_col])
            train_y = train[target_col]
            model.fit(train_X, train_y)
            valid_X = valid.drop(columns=["kfold"] + [target_col])
            valid_y = valid[target_col]
            predictions = model.predict_proba(valid_X)[:, 1]
            auc += metrics.roc_auc_score(valid_y, predictions)/nfolds
        return auc


    def _feature_selection(self, df: pd.DataFrame, 
                           target_col: str, nfolds=-1: int):
        """
        Selects feature based on greedy approach, using AUC

        Args:
        - df: input dataframe (must have kfold column)
        - target_col: name of target col
        - nfolds: number of folds assigned to train data

        Returns:
        - OOF AUC
        """

        good_features = []
        best_scores = []
        num_features = df.shape[1] - 2 if nfolds > 1 else df.shape[1] - 1

        while True:
            # Initialize best feature and score of this loop
            this_feature = None
            best_score = 0

            for feature in range(num_features):
                if feature in good_features:
                    continue
                selected_features = good_features + [feature]
                df_train = df.drop(['kfold'] + [target_col], axis=1, errors='ignore')
                if nfolds > 1:
                    df_train = pd.concat([df_train[:, selected_features], df['kfold']], axis=1)
                    score = self.evaluate_oof_score(df_train, target_col)
                else:
                    score = self.evaluate_oof_sevaluate_score(df_train, target_col)
                if score > best_score:
                    this_feature = feature 
                    best_score = score
                # Add to good feature list and update best scores list
                if this_feature is not None:
                    good_features.append(this_feature)
                    best_scores.append(best_score)

                # If no improvement during previous round, exit while loop
                if len(best_scores) > 2:
                    if best_scores[-1] < best_scores[-2]:
                        break

            return best_scores[:-1], good_features[:-1]
            

    def __call__(self, df: pd.DataFrame, 
                 target_col: str, nfolds=-1: int):
        """
        Call function will call the class on a set of arguments

        Args:
        - df: input dataframe (must have kfold column)
        - target_col: name of target col
        - nfolds: number of folds assigned to train data

        Returns:
        - dataframe with columns filtered according to greedy algorithm
        """
        # select features, returns scores and selected indices
        scores, features = self._feature_selection(df, target_col, nfolds)
        df_tmp = df.drop(['kfold'] + [target_col], axis=1, errors='ignore')
        df = pd.concat([df_tmp[:, features], df[target_col]], axis=1)
        if nfolds > 1:
            df = pd.concat([df, df["kfold"]], axis=1)
        return df