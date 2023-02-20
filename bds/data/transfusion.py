import pandas as pd

TARGET_COLUMN = "whether he/she donated blood in March 2007"

DATA_TYPE = {
    "Recency (months)": float,
    "Frequency (times)": float,
    "Monetary (c.c. blood)": float,
    "Time (months)": float,
    TARGET_COLUMN: int,
}


DATA_URI = "https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data"


def load():
    df = pd.read_csv(
        DATA_URI,
        delimiter=",",
        engine="python",
        dtype=DATA_TYPE,
    )
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y
