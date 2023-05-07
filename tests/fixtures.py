import numpy as np
import pytest
import gmpy2 as gmp

from gmpy2 import mpz
from bds.sat.min_freq import construct_min_freq_program
from bds.sat.solver import construct_solver
from bds.gf2 import GF
from bds.rule import Rule
from bds.utils import mpz_set_bits

# a very small toy dataset
toy_D_data = np.array([[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0]])

# a larger random dataset
n_examples, n_features = 100, 20
random_D_data = np.array(GF.Random((n_examples, n_features), seed=1234), dtype=int)


@pytest.fixture
def toy_D():
    return toy_D_data


@pytest.fixture
def random_D():
    return random_D_data


@pytest.fixture
def solver():
    return construct_solver()


def get_input_program_by_name(name):
    """either 'toy' or 'random-{k}', where k is some positive integer"""
    if name.startswith("toy"):
        freq = int(name.split("-")[1])
        D = toy_D_data
    elif name.startswith("random"):
        freq = int(name.split("-")[1])
        D = random_D_data
    program, I, T = construct_min_freq_program(D, min_freq_thresh=freq)

    return program, I, T


@pytest.fixture
def y():
    # return mpz_set_bits(mpz(), [2, 4])
    return np.array([0, 0, 1, 0, 1], dtype=bool)


@pytest.fixture
def rules():
    return [
        Rule(
            id=1,
            name="rule-1",
            cardinality=1,
            # truthtable=np.array([0, 1, 0, 1, 0], dtype=bool),
            truthtable=mpz_set_bits(mpz(), [1, 3])
        ),
        Rule(
            id=2,
            name="rule-2",
            cardinality=1,
            # truthtable=np.array([0, 0, 1, 0, 1], dtype=bool),
            truthtable=mpz_set_bits(mpz(), [2, 4])
        ),
        Rule(
            id=3,
            name="rule-3",
            cardinality=1,
            # truthtable=np.array([1, 0, 1, 0, 1], dtype=bool),
            truthtable=mpz_set_bits(mpz(), [0, 2, 4])
        ),
    ]
