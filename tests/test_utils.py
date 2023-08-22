
import gmpy2 as gmp
import numpy as np
import pytest
from gmpy2 import mpz

from bds.rule import Rule
from bds.utils import calculate_obj, calculate_lower_bound, mpz_set_bits


@pytest.fixture
def inputs():
    """
    the rules and labels
    1: 0101010000
    2: 1100101000
    3: 0010100000
    y: 1111000000
    """
    rules = [
        Rule(1, "1", 1, mpz("0b0101010000")),
        Rule(2, "2", 1, mpz("0b1100101000")),
        Rule(3, "3", 1, mpz("0b0010100000")),
    ]
    y_np = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=bool)
    y_mpz = mpz_set_bits(mpz(), y_np.nonzero()[0])
    return rules, y_np, y_mpz



class TestCalculateObj:
    @pytest.mark.parametrize(
        "sol, num_mistakes",
        [
            # 1: 0101010000
            # y: 1111000000
            # FP:     ^
            # FN:^ ^
            ([1], 3),
            # 2: 1100101000
            # y: 1111000000
            # FP:    ^ ^
            # FN:  ^^
            ([2], 4),
            # 3: 0010100000
            # y: 1111000000
            # FP:    ^
            # FN:^^ ^
            ([3], 4),
            # 1: 0101010000
            # 2: 1100101000
            # |: 1101111000
            # y: 1111000000
            # FP:    ^^^
            # FN   ^
            ([1, 2], 4),
            # 1: 0101010000
            # 3: 0010100000
            # |: 0111110000
            # y: 1111000000
            # FP:    ^^
            # FN:^
            ([1, 3], 3),
        ],
    )
    @pytest.mark.parametrize("lmbd", [0.0, 0.1])
    def test_simple(self, sol, num_mistakes, lmbd, inputs):
        """
        the rules and labels
        1: 0101010000
        2: 1100101000
        3: 00101000001
        y: 1111000000
        """
        rules, y_np, y_mpz = inputs
        sol = [0] + sol  # add 0 by convention
        obj = calculate_obj(rules, y_np, y_mpz, sol, lmbd)
        np.testing.assert_allclose(obj, num_mistakes / y_np.shape[0] + len(sol[1:]) * lmbd)


class TestCalculateLowerBound:
    @pytest.mark.parametrize(
        "sol, num_fp",
        [
            # 1: 0101010000
            # y: 1111000000
            # FP:     ^
            # FN:^ ^
            ([1], 1),
            # 2: 1100101000
            # y: 1111000000
            # FP:    ^ ^
            # FN:  ^^
            ([2], 2),
            # 3: 0010100000
            # y: 1111000000
            # FP:    ^
            # FN:^^ ^
            ([3], 1),
            # 1: 0101010000
            # 2: 1100101000
            # |: 1101111000
            # y: 1111000000
            # FP:    ^^^
            # FN   ^
            ([1, 2], 3),
            # 1: 0101010000
            # 3: 0010100000
            # |: 0111110000
            # y: 1111000000
            # FP:    ^^
            # FN:^
            ([1, 3], 2),
        ],
    )
    @pytest.mark.parametrize("lmbd", [0.0])
    def test_simple(self, sol, num_fp, lmbd, inputs):
        rules, y_np, y_mpz = inputs
        sol = [0] + sol  # add 0 by convention
        lb = calculate_lower_bound(rules, y_np, y_mpz, sol, lmbd)
        np.testing.assert_allclose(lb, num_fp / y_np.shape[0] + len(sol) * lmbd)
