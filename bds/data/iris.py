import pandas as pd
from sklearn import datasets


def load():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    return X, y
