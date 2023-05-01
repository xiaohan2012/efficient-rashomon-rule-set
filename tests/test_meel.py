import pytest
import numpy as np

from bds.gf2 import is_piecewise_linear
from bds.random_hash import generate_h_and_alpha
from bds.utils import randints, bin_array, bin_random
from bds.meel import log_search
from bds.common import EPSILON
from .fixtures import rules, y
from .utils import generate_random_rule_list


class TestLogSearch:
    def generate_random_rules_and_y(self, num_pts, num_rules):
        random_rules = generate_random_rule_list(num_pts, num_rules)
        random_y = bin_random(num_pts)
        return random_rules, random_y

    def check_output(self, m, Y_size, big_cell, Y_size_arr, thresh):
        # big_cell should look like, 1, 1, ...,1, 0, ..., 0, 0
        assert is_piecewise_linear(big_cell)

        assert Y_size == Y_size_arr[m] >= 0  # |Y| should be non-negative

        # the value of big_cell[m] should be consistent with the number of solutions
        if big_cell[m] == 0:  # we found one small enough solution
            # check if the adding first m constraints gives |Y| <= threshold
            assert Y_size < thresh
            # m is at the boundary
            if m > 0:
                assert big_cell[m - 1] == 1
        else:
            assert Y_size >= thresh
            # m is at the boundary
            if m < big_cell.shape[0] - 1:
                assert big_cell[m + 1] == 0

    @pytest.mark.parametrize("ub", np.arange(0, 2, 0.2))
    @pytest.mark.parametrize("m_prev, thresh", [(0, 2), (0, 4), (1, 2), (1, 4)])
    @pytest.mark.parametrize("rand_seed", randints(5))
    def test_on_toy_dataset(self, rules, y, ub, m_prev, thresh, rand_seed):
        n = len(rules)
        m = n - 1

        lmbd = 0.1
        ub = float("inf")
        A, t = generate_h_and_alpha(n, m, seed=rand_seed)
        A, t = bin_array(A), bin_array(t)
        # test statements
        m, Y_size, big_cell, Y_size_arr = log_search(
            rules, y, lmbd, ub, A, t, thresh, m_prev, return_full=True
        )

        self.check_output(m, Y_size, big_cell, Y_size_arr, thresh)

    @pytest.mark.parametrize(
        "ub", [0.5, 0.75]  # using larger ub, e.g., 1.0 tends to run slower
    )
    @pytest.mark.parametrize(
        "m_prev",
        [2, 5, 7],
    )
    @pytest.mark.parametrize("thresh", [5, 10])
    @pytest.mark.parametrize("rand_seed", randints(3))
    def test_on_random_datasets(self, ub, m_prev, thresh, rand_seed):
        # generate the rules and truth label
        num_pts, num_rules = 50, 10
        random_rules, random_y = self.generate_random_rules_and_y(num_pts, num_rules)
        m = num_rules - 1

        A, t = generate_h_and_alpha(num_rules, m, seed=rand_seed)
        A, t = bin_array(A), bin_array(t)

        lmbd = 0.1
        m, Y_size, big_cell, Y_size_arr = log_search(
            random_rules,
            random_y,
            lmbd,
            ub + EPSILON,
            A,
            t,
            thresh,
            m_prev,
            return_full=True,
        )

        self.check_output(m, Y_size, big_cell, Y_size_arr, thresh)

    @pytest.mark.parametrize(
        "ub", [0.6, 0.75]  # using larger ub, e.g., 1.0 tends to run slower
    )
    @pytest.mark.parametrize("thresh", [5, 10])
    @pytest.mark.parametrize("rand_seed", randints(3))
    def test_consistency_on_m(self, ub, thresh, rand_seed):
        """no matter what m_prev is provided, the same m should be given"""
        num_pts, num_rules = 50, 10
        random_rules, random_y = self.generate_random_rules_and_y(num_pts, num_rules)
        m = num_rules - 1

        A, t = generate_h_and_alpha(num_rules, m, seed=rand_seed)
        A, t = bin_array(A), bin_array(t)

        initial_m = int(m / 2)

        lmbd = 0.1
        ref_m, ref_Y_size, ref_big_cell, ref_Y_size_arr = log_search(
            random_rules,
            random_y,
            lmbd,
            ub + EPSILON,
            A,
            t,
            thresh,
            initial_m,
            return_full=True,
        )

        # print("ref_Y_size_arr: ", ref_Y_size_arr)
        for m_prev in range(1, m):
            (
                actual_m,
                actual_Y_size,
                actual_big_cell,
                actual_Y_size_arr,
            ) = log_search(
                random_rules,
                random_y,
                lmbd,
                ub + EPSILON,
                A,
                t,
                thresh,
                m_prev=m_prev,
                return_full=True,
            )
            np.testing.assert_equal(ref_big_cell, actual_big_cell)
            assert ref_m == actual_m
            assert ref_Y_size == actual_Y_size
