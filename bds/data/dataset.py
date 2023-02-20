from dataclasses import dataclass
from importlib import import_module

from sklearn.model_selection import train_test_split
from aix360.algorithms.rbm import FeatureBinarizer

from ..utils import convert_numerical_columns_to_bool


@dataclass
class Dataset:
    name: str
    train_ratio: float = 0.7
    split_train: bool = False
    validation_ratio: float = 0.2

    random_state: int = 1234

    def reset(self):
        self.X_train, self.X_test, self.X_validation = None, None, None
        self.y_train, self.y_test, self.y_validation = None, None, None

    def load(self):
        # load the data
        dataset = import_module(f'bds.data.{self.name}')
        X, y = dataset.load()

        # split into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=1 - self.train_ratio,
            random_state=self.random_state
        )

        if self.split_train:
            (
                self.X_train,
                self.X_validation,
                self.y_train,
                self.y_validation,
            ) = train_test_split(
                self.X_train,
                self.y_train,
                test_size=self.validation_ratio,
                random_state=self.random_state,
            )

class BinaryDataset(Dataset):
    def load(self):
        super(BinaryDataset, self).load()
        fb = FeatureBinarizer(negations=True)
        
        self.X_train_b = fb.fit_transform(self.X_train)
        self.X_test_b = fb.transform(self.X_test)
        # convert the data types to bool
        list(
            map(convert_numerical_columns_to_bool, [self.X_train_b, self.X_test_b])
        )

        if self.split_train:
            self.X_validation_b = fb.transform(self.X_validation)
            convert_numerical_columns_to_bool(self.X_validation_b)
