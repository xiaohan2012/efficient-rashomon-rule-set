import pandas as pd
from sklearn import datasets


def load():
    breast_cancer = datasets.load_breast_cancer()
    X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    y = pd.Series(breast_cancer.target)
    return X, y
