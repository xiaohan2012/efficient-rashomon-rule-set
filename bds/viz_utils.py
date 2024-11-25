from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_decision_boundary(
    ax,
    clf,
    X: np.ndarray,
    columns: List[str],
    feature_binarizer=None,
    h=0.02,
    cmap=plt.cm.coolwarm,  # step size in the mesh
):
    """
    X: the original feature matrix (without discretization)
    columns: the original feature names (without discretization)
    """
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    data_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=columns)

    if feature_binarizer is not None:
        data_df = feature_binarizer.transform(data_df)

    Z = clf.predict(data_df)

    if isinstance(Z, pd.Series):
        Z = Z.to_numpy()

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.5)
