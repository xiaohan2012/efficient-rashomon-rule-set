import pandas as pd

DATA_TYPE = {
    "age": float,
    "gender": str,
    "total Bilirubin": float,
    "direct Bilirubin": float,
    "Alkphos": float,
    "Sgpt": float,
    "Sgot": float,
    "total protiens": float,
    "Albumin": float,
    "A/G Ratio": float,
    "label": int,
}

TARGET_COLUMN = "label"
NEGATIVE_TARGET_VALUE = 1  # the person is a liver patient

DATA_URI = "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"


def load():
    df = pd.read_csv(
        DATA_URI,
        header=None,
        delimiter=",",
        engine="python",
        names=DATA_TYPE.keys(),
        dtype=DATA_TYPE,
    )

    y = df[TARGET_COLUMN].apply(lambda v: 0 if v == NEGATIVE_TARGET_VALUE else 1)
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y
