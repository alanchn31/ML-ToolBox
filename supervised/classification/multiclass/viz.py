import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import metrics
from typing import Union


def plot_confusion_matrix(y_true: Union[list, np.array, pd.Series], 
                          y_pred: Union[list, np.array, pd.Series],
                          figsize=(10, 10): tuple,
                          fontsize=20: int):
    """
    Plots confusion matrix as heatmap for a classification problem

    Args:
        y_true: Input series/array with true labels
        y_pred: Input series/array with predicted labels
        figsize: figure size of plot
        fontsize: font size of plot
    """
    cm = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0,
                                 as_cmap=True)
    sns.set(fontscale=2.5)
    sns.heatmap(cm, annot=True, cmap=cmap, cbar=False)
    plt.ylabel('Actual Labels', fontsize=fontsize)
    plt.xlabel('Predicted Labels', fontsize=fontsize)