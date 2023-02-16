# fixtures for contrastive pattern set
import pytest
import numpy as np

toy_Dp_data = np.array(
    [[0, 1, 0],
     [1, 1, 0],
     [1, 1, 1],
     [0, 0, 1],
     [1, 0, 0]],
    dtype=int
)

toy_Dn_data = np.array(
    [[1, 0, 1],
     [1, 0, 0],
     [0, 1, 1]
     ],
    dtype=int
)


@pytest.fixture
def toy_Dp():
    return toy_Dp_data


@pytest.fixture
def toy_Dn():
    return toy_Dn_data
