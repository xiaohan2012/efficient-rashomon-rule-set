import numpy as np
import pytest

from bds.sat.bounded_weight_sat import (
    BoundedWeightedPatternSATCallback,
    WeightedPatternSATCallback,
    weight_mc_core,
)
from bds.utils import randints

from .fixtures import get_input_program_by_name, solver  # noqa


def uniform_weight(pattern, covered_example):
    return 1


def weight_by_support_size(pattern, covered_example):
    return len(covered_example)


class TestBoundedWeightedPatternSATCallback:
    def setup_method(self, method):
        self.ws = np.array(
            [1, 2, 1, 3, 1, 3, 2]
        )  # weights of patterns, ordered by the time they are found
        self.ws_cumsum = np.cumsum(self.ws)  # [1, 3, 4, 7, 8, 11, 13]

    @pytest.mark.parametrize(
        "weight_func, expected_w_total, expected_w_min, expected_w_max",
        [(uniform_weight, 7.0, 1.0, 1.0), (weight_by_support_size, 13.0, 1.0, 3.0)],
    )
    def test_WeightedPatternSATCallback(
        self, solver, weight_func, expected_w_total, expected_w_min, expected_w_max
    ):
        program, I, T = get_input_program_by_name("toy-1")

        cb = WeightedPatternSATCallback(I, T, weight_func)

        solver.Solve(program, cb)
        assert cb.total_weight == cb.w_total == expected_w_total
        assert cb.w_min == expected_w_min
        assert cb.w_max == expected_w_max

    @pytest.mark.parametrize("pivot", np.arange(1, 7))
    def test_uniform_weight_overflow(self, solver, pivot):
        """uniform weight and overflow w_total"""
        program, I, T = get_input_program_by_name("toy-1")

        r = 1.0
        w_max = 1.0

        cb = BoundedWeightedPatternSATCallback(
            I, T, weight_func=uniform_weight, pivot=pivot, w_max=w_max, r=r
        )

        solver.Solve(program, cb)

        assert cb.w_min == 1.0
        assert cb.overflows_w_total  # the total weight should overflow the pivot
        assert cb.result == (cb.solutions_found, 1.0)
        assert (
            len(cb.solutions_found) == cb.w_total == (cb.pivot + 1) == len(cb.weights)
        )
        assert (np.array(cb.weights) == 1.0).all()

    @pytest.mark.parametrize("pivot", np.arange(7, 10))
    def test_uniform_weight_with_total_weight_nonoverflow(self, solver, pivot):
        """uniform weight and the non-overflow cases"""
        program, I, T = get_input_program_by_name("toy-1")

        r = 1.0
        w_max = 1.0

        cb = BoundedWeightedPatternSATCallback(
            I, T, weight_func=uniform_weight, pivot=pivot, w_max=w_max, r=r
        )

        solver.Solve(program, cb)
        assert cb.overflows_w_total is False

        with pytest.raises(RuntimeError):
            cb.result

    @pytest.mark.parametrize("max_total_weight", np.arange(1, 13))
    def test_nonuniform_and_weight_overflow_and_accurate_r_and_w_max(
        self, solver, max_total_weight
    ):
        """non-uniform weight and overflow w_total

        further, r and w_max being given are accurate
        """
        program, I, T = get_input_program_by_name("toy-1")

        # say r and w_max are accurate
        r = self.ws.max() / self.ws.min()
        w_max = self.ws.max()

        # normalized max total weight
        pivot = max_total_weight / 3.0  # 3.0 = w_min x r
        cb = BoundedWeightedPatternSATCallback(
            I,
            T,
            weight_func=weight_by_support_size,
            pivot=pivot,
            w_max=w_max,
            r=r,
        )

        assert cb.w_min == 1.0

        solver.Solve(program, cb)

        assert cb.w_min == 1.0
        assert cb.w_total == self.ws_cumsum[len(cb.solutions_found) - 1]
        assert cb.overflows_w_total
        assert cb.result == (cb.solutions_found, cb.w_min * cb.r)
        np.testing.assert_array_equal(cb.weights, self.ws[: len(cb.weights)])

    @pytest.mark.parametrize("max_total_weight", np.arange(13, 20))
    def test_nonuniform_and_weight_nonoverflow_and_accurate_r_and_w_max(
        self, solver, max_total_weight
    ):
        """non-uniform weight and non-overflow w_total

        further, r and w_max being given are accurate
        """
        program, I, T = get_input_program_by_name("toy-1")

        # say r and w_max are accurate
        r = self.ws.max() / self.ws.min()
        w_max = self.ws.max()

        # normalized max total weight
        pivot = max_total_weight / 3.0  # 3.0 = w_min x r
        cb = BoundedWeightedPatternSATCallback(
            I,
            T,
            weight_func=weight_by_support_size,
            pivot=pivot,
            w_max=w_max,
            r=r,
        )

        assert cb.w_min == 1.0

        solver.Solve(program, cb)

        assert cb.overflows_w_total is False

        with pytest.raises(RuntimeError):
            cb.result

    @pytest.mark.parametrize("max_total_weight", np.arange(1, 13))
    def test_nonuniform_inaccurate_r_and_w_max_overflow(self, solver, max_total_weight):
        """non-uniform weight and inaccurate r and r_max, with non"""
        program, I, T = get_input_program_by_name("toy-1")

        # say r and w_max are accurate
        r = (self.ws.max() / self.ws.min()) + 1
        w_max = self.ws.max() + 1

        # normalized max total weight
        pivot = (
            max_total_weight / w_max
        )  # the denumerator is (the accurate) w_min x (the inaccurate) r (4 in this case)
        cb = BoundedWeightedPatternSATCallback(
            I,
            T,
            weight_func=weight_by_support_size,
            pivot=pivot,
            w_max=w_max,
            r=r,
        )

        old_w_min = cb.w_min
        assert np.isclose(cb.w_min, w_max / r)

        solver.Solve(program, cb)

        assert cb.w_min == 1.0
        assert cb.w_min <= old_w_min  # w_min never increases

        assert cb.w_total == self.ws_cumsum[len(cb.solutions_found) - 1]
        assert cb.overflows_w_total
        assert cb.result == (cb.solutions_found, cb.w_min * cb.r)
        np.testing.assert_array_equal(cb.weights, self.ws[: len(cb.weights)])

    @pytest.mark.parametrize("max_total_weight", np.arange(13, 20))
    def test_nonuniform_inaccurate_r_and_w_max_nonoverflow(
        self, solver, max_total_weight
    ):
        """non-uniform weight and inaccurate r and r_max"""
        program, I, T = get_input_program_by_name("toy-1")

        # say r and w_max are accurate
        r = (self.ws.max() / self.ws.min()) + 1
        w_max = self.ws.max() + 1

        # normalized max total weight
        pivot = (
            max_total_weight / w_max
        )  # the denumerator is (the accurate) w_min x (the inaccurate) r (4 in this case)
        cb = BoundedWeightedPatternSATCallback(
            I,
            T,
            weight_func=weight_by_support_size,
            pivot=pivot,
            w_max=w_max,
            r=r,
        )

        solver.Solve(program, cb)

        assert cb.overflows_w_total is False

        with pytest.raises(RuntimeError):
            cb.result


class TestWeightMCCore_uniform_weight:
    """this test case assumes the weight is uniform"""

    def check_result(self, c, w_max_new, details):
        """check if the returned result by weight_mc_core is correct"""
        if c is not None:
            # valid result returned
            if details["i"] > 0:
                assert c == (details["w_total"] * np.power(2, details["i"]) / w_max_new)
                assert not details["smallest_cell_too_big"]
                assert not details["cell_size_is_zero"]
            else:
                assert c == 7.0

            if details["i"] == details["n"]:
                assert details["max_iteration_reached"]
            else:
                assert not details["max_iteration_reached"]
        else:
            # invalid result returned
            assert details["max_iteration_reached"]
            assert details["smallest_cell_too_big"] or details["cell_size_is_zero"]

    def get_result(self, pivot, r, w_max, rand_seed):
        program, I, T = get_input_program_by_name("toy-1")

        def make_callback(weight_func, pivot, w_max, r):
            return BoundedWeightedPatternSATCallback(
                I,
                T,
                weight_func=weight_func,
                pivot=pivot,
                w_max=w_max,
                r=r,
            )

        return weight_mc_core(
            program,
            I,
            make_callback,
            uniform_weight,
            pivot=pivot,
            r=r,
            w_max=w_max,
            rand_seed=rand_seed,
            return_details=True,
        )

    @pytest.mark.parametrize("pivot", [7, 8, 9, 10])
    @pytest.mark.parametrize("rand_seed", randints(10))
    def test_accurate_parameters_and_large_pivot(self, pivot, rand_seed):
        """pivot is so large so that the algorithm does not need to partition the solution space"""

        r = 1.0
        w_max = 1.0

        c_expected = 7
        i_expected = 0
        w_total_expected = c_expected

        c, w_max_new, details = self.get_result(pivot, r, w_max, rand_seed)

        assert c == c_expected
        assert w_max_new == 1.0
        assert details["i"] == i_expected
        assert details["w_total"] == w_total_expected

    @pytest.mark.parametrize("pivot", np.arange(2, 7))
    @pytest.mark.parametrize("rand_seed", randints(10))
    def test_with_accurate_parameters_and_nonlarge_pivot(self, pivot, rand_seed):
        """piviot is not that large so that the algorithm needs to partition the solution space"""
        r = 1.0
        w_max = 1.0
        c, w_max_new, details = self.get_result(pivot, r, w_max, rand_seed)
        assert w_max_new == 1.0
        assert 0 < details["i"] <= details["n"]

        self.check_result(c, w_max_new, details)

    @pytest.mark.parametrize("pivot", [1])
    @pytest.mark.parametrize("rand_seed", randints(10))
    def test_with_accurate_parameters_and_too_small_pivot(self, pivot, rand_seed):
        """piviot is too small so that even the smallest cell size cannot be under pivot

        remark:

        due to the randomness of space partitioning, the sampled cell must be exactly 1 solution in order to returna valid size

        - if the number is > 1, then the BoundedWeightSAT overflows the pivot and returns w_total eq 2, and None will be returned
        - if the number is zero, w_total is zero, and None will be returned
        """
        r = 1.0
        w_max = 1.0
        c, w_max_new, details = self.get_result(pivot, r, w_max, rand_seed)

        assert w_max_new == 1.0

        self.check_result(c, w_max_new, details)

    @pytest.mark.parametrize("pivot", np.arange(2, 7))
    @pytest.mark.parametrize("w_max", np.arange(2, 5))
    @pytest.mark.parametrize("rand_seed", randints(5))
    def test_with_inaccurate_w_max(self, pivot, w_max, rand_seed):
        """only w_max is inaccurate"""
        r = 1.0
        c, w_max_new, details = self.get_result(pivot, r, w_max, rand_seed)
        assert w_max_new == 1.0

        self.check_result(c, w_max_new, details)

    @pytest.mark.parametrize("pivot", np.arange(2, 7))
    @pytest.mark.parametrize("r", np.arange(2, 4))
    @pytest.mark.parametrize("w_max", [1.5, 2, 3, 5])
    @pytest.mark.parametrize("rand_seed", randints(5))
    def test_with_inaccurate_r_and_w_max(self, pivot, w_max, r, rand_seed):
        """both r and w_max are inaccurate"""
        c, w_max_new, details = self.get_result(pivot, r, w_max, rand_seed)
        # w_max_new depends on the relation between r and w_max
        # assert c is not None
        w_min_init = w_max / r
        if w_min_init <= 1.0:
            # w_min_init is the smallest ever (w_min is never updated inside BoundedWeightSAT)
            assert w_max_new == w_max
        else:
            # a smaller w_min will be found
            assert w_max_new == r

        self.check_result(c, w_max_new, details)


class TestWeightMCCore_nonuniform_weight:
    """this test case assumes the weight is the support size"""

    def setup_method(self, method):
        # the correct r and w_max
        self.r = 3.0
        self.w_max = 3.0

    def check_result(self, c, w_max_new, details):
        """check if the returned result by weight_mc_core is correct"""
        assert details["i"] <= details["n"]
        if c is not None:
            # valid result returned
            if details["i"] > 0:
                assert c == (details["w_total"] * np.power(2, details["i"]) / w_max_new)
                assert not details["smallest_cell_too_big"]
                assert not details["cell_size_is_zero"]
            else:
                assert c == 13.0

            if details["i"] == details["n"]:
                assert details["max_iteration_reached"]
            else:
                assert not details["max_iteration_reached"]
        else:
            # invalid result returned
            assert details["max_iteration_reached"]
            assert details["smallest_cell_too_big"] or details["cell_size_is_zero"]

    def get_result(self, pivot, r, w_max, rand_seed):
        program, I, T = get_input_program_by_name("toy-1")

        def make_callback(weight_func, pivot, w_max, r):
            return BoundedWeightedPatternSATCallback(
                I,
                T,
                weight_func=weight_func,
                pivot=pivot,
                w_max=w_max,
                r=r,
            )

        return weight_mc_core(
            program,
            I,
            make_callback,
            weight_by_support_size,
            pivot=pivot,
            r=r,
            w_max=w_max,
            rand_seed=rand_seed,
            return_details=True,
        )

    @pytest.mark.parametrize("unnormalized_pivot", [13, 14, 15, 16])
    @pytest.mark.parametrize("rand_seed", randints(10))
    def test_accurate_parameters_and_large_pivot(self, unnormalized_pivot, rand_seed):
        """pivot is so large so that the algorithm does not need to partition the solution space"""

        r = self.r
        w_max = self.w_max

        pivot = unnormalized_pivot / w_max

        c_expected = 13.0
        i_expected = 0
        w_total_expected = c_expected

        c, w_max_new, details = self.get_result(pivot, r, w_max, rand_seed)

        assert c == c_expected
        assert w_max_new == w_max
        assert details["i"] == i_expected
        assert details["w_total"] == w_total_expected

    @pytest.mark.parametrize("unnormalized_pivot", np.arange(2, 13))
    @pytest.mark.parametrize("rand_seed", randints(10))
    def test_with_accurate_parameters_and_nonlarge_pivot(
        self, unnormalized_pivot, rand_seed
    ):
        """piviot is not that large so that the algorithm needs to partition the solution space"""
        r = self.r
        w_max = self.w_max
        pivot = unnormalized_pivot / w_max

        c, w_max_new, details = self.get_result(pivot, r, w_max, rand_seed)

        assert w_max_new == self.w_max
        assert details["i"] > 0  # the space is partitioned

        self.check_result(c, w_max_new, details)

    @pytest.mark.parametrize("unnormalized_pivot", [1])
    @pytest.mark.parametrize("rand_seed", randints(10))
    def test_with_accurate_parameters_and_too_small_pivot(
        self, unnormalized_pivot, rand_seed
    ):
        """piviot is too small so that even the smallest cell size cannot be under pivot"""
        r = self.r
        w_max = self.w_max

        pivot = unnormalized_pivot / w_max
        c, w_max_new, details = self.get_result(pivot, r, w_max, rand_seed)

        assert w_max_new == self.w_max

        self.check_result(c, w_max_new, details)

    @pytest.mark.parametrize("unnormalized_pivot", np.arange(2, 13))
    @pytest.mark.parametrize("w_max", np.arange(4, 7))  # should be at least 3
    @pytest.mark.parametrize("rand_seed", randints(5))
    def test_with_inaccurate_w_max(self, unnormalized_pivot, w_max, rand_seed):
        """only w_max is inaccurate"""
        r = self.r
        pivot = unnormalized_pivot / w_max
        c, w_max_new, details = self.get_result(pivot, r, w_max, rand_seed)

        # note that w_max_new can never be greater than the initial w_max
        # if w_max_new  == self.w_max (=3), it means w_min=1.0 is found
        # otherwise it remains the same as w_max (=4)
        assert w_max_new <= w_max

        self.check_result(c, w_max_new, details)

    @pytest.mark.parametrize("unnormalized_pivot", np.arange(2, 13))
    @pytest.mark.parametrize("r", [3.5, 4, 5])  # at least 3
    @pytest.mark.parametrize("w_max", np.arange(4, 7))
    @pytest.mark.parametrize("rand_seed", randints(5))
    def test_with_inaccurate_r_and_w_max(self, unnormalized_pivot, w_max, r, rand_seed):
        """both r and w_max are inaccurate"""
        pivot = unnormalized_pivot / w_max
        c, w_max_new, details = self.get_result(pivot, r, w_max, rand_seed)
        # assert c is not None

        # w_max_new can never be greater than w_max
        # since w_min never increases, and w_max_new = r * w_min, and w_min is initialized to w_max / r
        assert w_max_new <= w_max

        self.check_result(c, w_max_new, details)
