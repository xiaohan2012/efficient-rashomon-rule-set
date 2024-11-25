from typing import Tuple

import numpy as np
import pandas as pd

DATA_TYPE = {
    "age": np.float64,
    "sex": str,
    "cp": str,
    "trestbps": np.float64,
    "chol": np.float64,
    "fbs": str,
    "restecg": str,
    "thalach": np.float64,
    "exang": str,
    "oldpeak": np.float64,
    "slope": str,
    "ca": np.float64,
    "thal": str,
    "num": np.int64,
}

COL_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]
ROWS_TO_SKIP = [88, 167, 193, 267, 288, 303]  # these rows contain invalid values

TARGET_COLUMN = "num"
NEGATIVE_TARGET_VALUE = 0


DATA_URI = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


def load() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(
        DATA_URI,
        header=None,
        delimiter=",",
        engine="python",
        names=COL_NAMES,
        dtype=DATA_TYPE,
        # these rows contain missing values ("?") and row index start from 0
        skiprows=np.array(ROWS_TO_SKIP) - 1,
    )

    y = df[TARGET_COLUMN].apply(lambda v: 0 if v == NEGATIVE_TARGET_VALUE else 1)
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y
