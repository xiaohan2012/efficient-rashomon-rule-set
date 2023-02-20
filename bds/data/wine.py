import pandas as pd
from sklearn import datasets


def load():
    wine = datasets.load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target)
    return X, y
