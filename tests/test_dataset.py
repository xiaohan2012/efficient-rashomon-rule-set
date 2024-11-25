import numpy as np
import pytest

from bds.data.dataset import BinaryDataset, Dataset

SAMPLE_DATASET_NAMES = ["iris", "heart", "ilpd"]


@pytest.mark.parametrize("dataset_name", SAMPLE_DATASET_NAMES)
def test_BinaryDataset(dataset_name):
    ds = BinaryDataset(name=dataset_name, split_train=True)
    ds.load()

    assert ds.X_train_b.shape[1] == ds.X_validation_b.shape[1] == ds.X_test_b.shape[1]
    assert ds.X_train_b.shape[0] == ds.X_train.shape[0] == ds.y_train.shape[0]
    assert (
        ds.X_validation_b.shape[0]
        == ds.X_validation.shape[0]
        == ds.y_validation.shape[0]
    )
    assert ds.X_test_b.shape[0] == ds.X_test.shape[0] == ds.y_test.shape[0]

    for X in [ds.X_train_b, ds.X_validation_b, ds.X_test_b]:
        for col in X.columns:
            assert X[col].dtype == bool


@pytest.mark.parametrize("dataset_name", SAMPLE_DATASET_NAMES)
def test_Dataset(dataset_name):
    ds = Dataset(name=dataset_name, split_train=True)
    ds.load()

    assert ds.X_train.shape[1] == ds.X_validation.shape[1] == ds.X_test.shape[1]
    assert ds.X_train.shape[0] == ds.y_train.shape[0]
    assert ds.X_validation.shape[0] == ds.y_validation.shape[0]
    assert ds.X_test.shape[0] == ds.y_test.shape[0]

    for y in [ds.y_train, ds.y_test, ds.y_validation]:
        assert isinstance(y, np.ndarray)
