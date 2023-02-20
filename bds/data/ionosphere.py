import pandas as pd

NUM_COLS = 34
DATA_URI = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
DATA_TYPE = {'c{}'.format(i): float
             for i in range(NUM_COLS)}

TARGET_COLUMN = 'label'
DATA_TYPE[TARGET_COLUMN] = str

NEGATIVE_TARGET_VALUE = "b"


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

