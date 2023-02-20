import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


data_type = {
    "age": float,
    "workclass": str,
    "fnlwgt": float,
    "education": str,
    "education-num": float,
    "marital-status": str,
    "occupation": str,
    "relationship": str,
    "race": str,
    "sex": str,
    "capital-gain": float,
    "capital-loss": float,
    "native-country": str,
    "hours-per-week": float,
    "label": str,
}

col_names = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "label",
]

# data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
data_uri = "~/code/bds/data/adult.data"

TARGET_COLUMN = "label"
POS_VALUE = ">50K"  # Setting positive value of the label for which we train


def load() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    print(f"downloading from {data_uri}")
    df = pd.read_csv(
        data_uri,
        header=None,
        delimiter=", ",
        engine="python",
        names=col_names,
        dtype=data_type,
    )
    ## Comlum names shall not contain whitespace or arithmetic operators (+, -, *, /)
    # We eventually output the rule set in TRXF format, where compound features are supported by parsing an expression string. So simple features like column names of a data frame must not contain these so that they are parsed as a single variable rather than an expression.
    df.columns = df.columns.str.replace("-", "_")

    y = df[TARGET_COLUMN].apply(lambda x: 1 if x == POS_VALUE else 0)
    X = df.drop(columns=[TARGET_COLUMN])

    return X, y
