import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


data_type = {'age': np.float,
             'workclass': str,
             'fnlwgt': np.float,
             'education': str,
             'education-num': np.float,
             'marital-status': str,
             'occupation': str,
             'relationship': str,
             'race': str,
             'sex': str,
             'capital-gain': np.float,
             'capital-loss': np.float,
             'native-country': str,
             'hours-per-week': np.float,
             'label': str}

col_names = ['age', 'workclass', 'fnlwgt', 'education',
             'education-num', 'marital-status', 'occupation',
             'relationship', 'race', 'sex',
             'capital-gain', 'capital-loss', 'hours-per-week',
             'native-country', 'label']

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
def load(test_size: float=0.2) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    print(f'downloading from {data_url}')
    df = pd.read_csv(data_url,
                     header=None,
                     delimiter=', ',
                     engine='python',
                     names=col_names,
                     dtype=data_type)
    ## Comlum names shall not contain whitespace or arithmetic operators (+, -, *, /)
    # We eventually output the rule set in TRXF format, where compound features are supported by parsing an expression string. So simple features like column names of a data frame must not contain these so that they are parsed as a single variable rather than an expression.
    df.columns = df.columns.str.replace('-', '_')

    TARGET_COLUMN = 'label'
    POS_VALUE = '>50K' # Setting positive value of the label for which we train

    train, test = train_test_split(df, test_size=test_size, random_state=42)
    # Split the data set into 80% training and 20% test set
    print('Training set:')
    print(train[TARGET_COLUMN].value_counts())
    print('Test set:')
    print(test[TARGET_COLUMN].value_counts())

    y_train = train[TARGET_COLUMN].apply(lambda x: 1 if x == POS_VALUE else 0)
    x_train = train.drop(columns=[TARGET_COLUMN])

    y_test = test[TARGET_COLUMN].apply(lambda x: 1 if x == POS_VALUE else 0)
    x_test = test.drop(columns=[TARGET_COLUMN])

    return x_train, y_train, x_test, y_test
