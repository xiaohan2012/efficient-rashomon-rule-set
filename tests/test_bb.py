import numpy as np
import pytest

from bds.bb import BranchAndBoundNaive, incremental_update_lb, incremental_update_obj
from bds.utils import randints


@pytest.mark.parametrize("seed", randints(5))
@pytest.mark.parametrize("num_fp", np.arange(6))
def test_incremental_update_obj(seed, num_fp):
    np.random.seed(seed)

    arr_len = 10

    # say we capture half of the points
    captured_idx = np.random.permutation(arr_len)[: int(arr_len / 2)]
    v = np.zeros(arr_len, dtype=bool)  # captured array
    v[captured_idx] = 1

    y = v.copy()  # the true labels
    y[captured_idx[:num_fp]] = 0  # make `num_fp` mistakes
    true_inc_fp = num_fp / arr_len
    assert incremental_update_lb(v, y) == true_inc_fp


def test_incremental_update_lb():
    u = np.array([0, 1, 1, 0, 0, 1, 0], dtype=bool)  # not captured by prefix
    v = np.array([0, 1, 0, 0, 1, 0, 0], dtype=bool)  # captured by rule
    f = np.array([0, 0, 1, 0, 0, 1, 0], dtype=bool)  # not captured by the rule + prefix
    y = np.array([0, 1, 1, 0, 1, 1, 0], dtype=bool)  # the true labels
    fn, actual_f = incremental_update_obj(u, v, y)

    assert fn == 2 / 7
    np.testing.assert_allclose(f, actual_f)
    assert actual_f.dtype == bool


class TestBranchAndBoundNaive:
    def test_bb_begin(self):
        BranchAndBoundNaive

    def test__get_captured_for_rule(self):
        pass

    def test_bb_loop(self):
        pass

    def test_run(self):
        pass
