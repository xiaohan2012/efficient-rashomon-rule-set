import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class NaiveClassifier(BaseEstimator):
    def __init__(
        self,
        num_components: int = 10,
    ):
        pass

    def fit(self, X_fb: pd.DataFrame, y: np.ndarray):
        assert set(y.unique()) == {0, 1}
        pos_idx, neg_idx = (y == 1).nonzero()[0], (y == 0).nonzero()[0]
