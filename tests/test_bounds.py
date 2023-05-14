import pytest
import numpy as np
from gmpy2 import mpz, mpfr

from bds.utils import mpz_set_bits, mpz_clear_bits, randints
from bds.bounds import (
    incremental_update_lb,
    incremental_update_obj,
    prefix_specific_length_upperbound,
)


@pytest.mark.parametrize("seed", randints(5))
@pytest.mark.parametrize("num_fp", np.arange(6))
def test_incremental_update_obj(seed, num_fp):
    np.random.seed(seed)

    num_pts = 10

    # say we capture half of the points
    captured_idx = np.random.permutation(num_pts)[: int(num_pts / 2)]
    v = mpz_set_bits(mpz(), captured_idx)

    y = mpz_clear_bits(v, captured_idx[:num_fp])  # make `num_fp` mistakes
    true_inc_fp = num_fp / mpz(num_pts)
    actual = incremental_update_lb(v, y, mpz(num_pts))
    assert actual == true_inc_fp
    assert isinstance(actual, mpfr)


def test_incremental_update_lb():
    u = mpz_set_bits(mpz(), [1, 2, 5])  # points not captured by prefix
    v = mpz_set_bits(mpz(), [1, 4])  # captured by rule
    f = mpz_set_bits(mpz(), [2, 5])  # not captured by the rule and prefix
    y = mpz_set_bits(mpz(), [1, 2, 4, 5])  # the true labels
    num_pts = mpz(7)
    fn, actual_f = incremental_update_obj(u, v, y, num_pts)

    assert f == actual_f
    assert fn == (mpz(2) / 7)
    assert isinstance(fn, mpfr)


def test_prefix_specific_length_upperbound():
    prefix_lb = 5
    prefix_length = 5
    ub = 10
    lmbd = 0.1
    # ub - prefix_lb = 5
    # 5 / lmbd = 50
    # 5 + 50
    assert 55 == prefix_specific_length_upperbound(prefix_lb, prefix_length, lmbd, ub)

    lmbd = 5.1
    # ub - prefix_lb = 5
    # floor(5 / lmbd) = 0
    # 5 + 0
    assert 5 == prefix_specific_length_upperbound(prefix_lb, prefix_length, lmbd, ub)

    lmbd = 4.9  # floor(5 / lmbd) = 1
    assert 6 == prefix_specific_length_upperbound(prefix_lb, prefix_length, lmbd, ub)
