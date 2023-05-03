import numpy as np
import pandas as pd
from .cache_tree import CacheTree, Node


class EquivalentPointClass:
    """a single class of points all having the same attributes"""

    def __init__(self, this_id, attrs):
        self.id = this_id
        self.attrs = attrs
        self.total_positives = 0
        self.total_negatives = 0
        self.minority_mistakes = 0

    def update(self, label):
        if label == 1:
            self.total_positives += 1
        else:
            self.total_negatives += 1

        self.minority_mistakes = min(
            self.total_negatives, self.total_positives
        )  # recall this is to be normalized


def find_equivalence_classes(X_trn: pd.DataFrame, y_train: np.ndarray):
    """
    Fimd equivalence classes of points having the same attributes but possibly different labels.
    This function is to be used once prior to branch-and-bound execution to exploit the equivalence-points-based bound.

    Parameters
    ----------
    X_trn : pd.DataFrame
       train data attribute matrix


    y_train :  np.ndarray
        labels


    Returns
    -------
    all equivalnce classes of points all_classes
    """

    if isinstance(X_trn, pd.DataFrame):
        X_trn = X_trn.to_numpy()
    
    # find equivalence classes
    all_classes_ids = set()
    all_classes = dict()
    for i, point in enumerate(X_trn):
        attrs = np.where(point == 1)[0]
        attr_str = "-".join(map(str, attrs))
        if attr_str not in all_classes_ids:  # new equivalence class
            all_classes_ids.add(attr_str)
            all_classes[attr_str] = EquivalentPointClass(attr_str, attrs)

        else:  # update existing equivalence class
            all_classes[attr_str].update(y_train[i])

    return all_classes
