import pandas as pd

TARGET_COLUMN = "win-for-x"

DATA_TYPE = {
    "top-left-square": str,
    "top-middle-square": str,
    "top-right-square": str,
    "middle-left-square": str,
    "middle-middle-square": str,
    "middle-right-square": str,
    "bottom-left-square": str,
    "bottom-middle-square": str,
    "bottom-right-square": str,
    TARGET_COLUMN: str,
}


DATA_URI = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data"
NEGATIVE_TARGET_VALUE = "negative"


def load():
    df = pd.read_csv(
        DATA_URI,
        header=None,
        delimiter=",",
        engine="python",
        dtype=DATA_TYPE,
        names=DATA_TYPE.keys(),
    )

    y = df[TARGET_COLUMN].apply(lambda v: 0 if v == NEGATIVE_TARGET_VALUE else 1)
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y
