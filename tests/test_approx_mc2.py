import numpy as np
import pytest

from bds.gf2 import is_piecewise_linear
from bds.random_hash import generate_h_and_alpha
from bds.sat.approx_mc2 import (
    approx_mc2,
    approx_mc2_core,
    get_theoretical_bounds,
    log_sat_search,
)
from bds.sat.bounded_sat import get_xor_constraints
from bds.sat.ground_truth import get_ground_truth_count
from bds.utils import randints

from .fixtures import get_input_program_by_name, solver  # noqa

np.random.seed(123456)


class Test_log_sat_search:
    def check_output(self, m, Y_size, big_cell, Y_size_arr, thresh):
        # big_cell should look like, 1, 1, ...,1, 0, ..., 0, 0
        assert is_piecewise_linear(big_cell)

        assert Y_size == Y_size_arr[m] > 0  # |Y| should be positive

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

    def test_invalid_input(self, solver):
        program, I, T = get_input_program_by_name("toy-1")

        cst_list = []  # no need to create it

        for m_prev in [2, 3, 4]:
            with pytest.raises(ValueError):
                # m_prev is too big
                log_sat_search(
                    program,
                    I,
                    T,
                    cst_list,
                    thresh=2,
                    m_prev=4,
                    solver=solver,
                    return_full=True,
                    verbose=1,
                )

        for thresh in [0, 1]:
            with pytest.raises(ValueError):
                # thresh is too small
                log_sat_search(
                    program,
                    I,
                    T,
                    cst_list,
                    thresh=thresh,
                    m_prev=1,
                    solver=solver,
                    return_full=True,
                    verbose=1,
                )

    @pytest.mark.parametrize("m_prev, thresh", [(0, 2), (0, 4), (1, 2), (1, 4)])
    @pytest.mark.parametrize("rand_seed", randints(5))
    def test_on_toy_dataset(self, solver, m_prev, thresh, rand_seed):
        # - the program
        program, I, T = get_input_program_by_name("toy-1")

        # h and alpha (b) encoded by cst_list
        n = len(I)
        m = n - 1
        A, b = generate_h_and_alpha(n, m, seed=rand_seed)
        cst_list = get_xor_constraints(A, b, I, use_rref=True, verbose=0)

        # test statements
        m, Y_size, big_cell, Y_size_arr = log_sat_search(
            program, I, T, cst_list, thresh, m_prev, solver, return_full=True, verbose=1
        )

        self.check_output(m, Y_size, big_cell, Y_size_arr, thresh)

    @pytest.mark.parametrize("dataset_name", ["random-10", "random-20"])
    @pytest.mark.parametrize("m_prev", [2, 5, 10, 15])
    @pytest.mark.parametrize("thresh", [10, 20, 100])
    @pytest.mark.parametrize("rand_seed", randints(3))
    def test_on_random_dataset(self, solver, dataset_name, m_prev, thresh, rand_seed):
        # - the program
        program, I, T = get_input_program_by_name(dataset_name)

        # h and alpha (b) encoded by cst_list
        n = len(I)
        m = n - 1
        A, b = generate_h_and_alpha(n, m, seed=rand_seed)
        cst_list = get_xor_constraints(A, b, I, use_rref=True, verbose=0)

        # test statements
        m, Y_size, big_cell, Y_size_arr = log_sat_search(
            program, I, T, cst_list, thresh, m_prev, solver, return_full=True, verbose=1
        )

        assert big_cell[m] == 0  # returned m should have a cell size <= threshold
        self.check_output(m, Y_size, big_cell, Y_size_arr, thresh)

    @pytest.mark.parametrize("dataset_name", ["random-10", "random-20"])
    @pytest.mark.parametrize("thresh", [10, 20, 100])
    @pytest.mark.parametrize("rand_seed", randints(1))
    def test_consistency_on_m(self, solver, dataset_name, thresh, rand_seed):
        """no matter what m_prev is provided, the same m should be given"""
        program, I, T = get_input_program_by_name(dataset_name)

        # h and alpha (b) encoded by cst_list
        n = len(I)
        m = n - 1
        A, b = generate_h_and_alpha(n, m, seed=rand_seed)
        cst_list = get_xor_constraints(A, b, I, use_rref=True, verbose=0)

        # test statements
        ref_m, ref_Y_size, ref_big_cell, ref_Y_size_arr = log_sat_search(
            program,
            I,
            T,
            cst_list,
            thresh=thresh,
            m_prev=10,
            solver=solver,
            return_full=True,
            verbose=1,
        )

        # print("ref_Y_size_arr: ", ref_Y_size_arr)
        for m_prev in range(m):
            (
                actual_m,
                actual_Y_size,
                actual_big_cell,
                actual_Y_size_arr,
            ) = log_sat_search(
                program,
                I,
                T,
                cst_list,
                thresh,
                m_prev=m_prev,
                solver=solver,
                return_full=True,
                verbose=1,
            )
            # assert _get_solution_count(program, I, T, cst_list, actual_m, thresh, solver) < thresh
            # print("actual_Y_size_arr: ", actual_Y_size_arr)
            np.testing.assert_equal(ref_big_cell, actual_big_cell)
            assert ref_m == actual_m
            assert ref_Y_size == actual_Y_size


class Test_approx_mc2_core:
    @pytest.mark.parametrize("dataset_name", ["random-10", "random-20"])
    @pytest.mark.parametrize("thresh", [10, 20, 100])
    @pytest.mark.parametrize("rand_seed", randints(3))
    def test_basic(self, dataset_name, thresh, rand_seed):
        """e.g., the code runs and return data with correct types"""
        prev_m = 10
        program, I, T = get_input_program_by_name(dataset_name)
        n_cells, Y_size = approx_mc2_core(
            program,
            I,
            T,
            thresh=thresh,
            prev_n_cells=2**prev_m,  # Q: why not just pass in prev_m directly?
            rand_seed=rand_seed,
            verbose=0,
        )
        assert isinstance(n_cells, int)
        assert Y_size < thresh

    @pytest.mark.skip("thresh should be strictly larger than 1")
    @pytest.mark.parametrize("dataset_name", ["toy-1"])
    @pytest.mark.parametrize(
        "thresh", [1]
    )  # technically, thresh = 1 is not allowed, but I'm not sure how to make the procedure return None
    @pytest.mark.parametrize("rand_seed", randints(5))
    def test_return_None(self, dataset_name, thresh, rand_seed):
        prev_m = 1
        program, I, T = get_input_program_by_name(dataset_name)
        n_cells, Y_size = approx_mc2_core(
            program,
            I,
            T,
            thresh=thresh,
            prev_n_cells=2**prev_m,  # Q: why not just pass in prev_m directly?
            rand_seed=rand_seed,
            verbose=1,
        )
        assert n_cells is None
        assert Y_size is None

    @pytest.mark.parametrize("dataset_name", ["random-10", "random-20"])
    @pytest.mark.parametrize("rand_seed", randints(5))
    def test_monotonicity_wrt_thresh(self, dataset_name, rand_seed):
        # as we increase thresh,
        # Y_size should not decrease
        # and m should not increase
        thresh_list = [10, 20, 30, 40, 50, 100]
        thresh_list

        prev_m = 1
        program, I, T = get_input_program_by_name(dataset_name)

        n_cells_list = []
        Y_size_list = []

        for thresh in thresh_list:
            n_cells, Y_size = approx_mc2_core(
                program,
                I,
                T,
                thresh=thresh,
                prev_n_cells=2**prev_m,  # Q: why not just pass in prev_m directly?
                rand_seed=rand_seed,
                verbose=1,
            )
            Y_size_list.append(Y_size)
            n_cells_list.append(n_cells)

        assert np.all(np.diff(Y_size_list) >= 0)  # monotonically non-decreasing
        assert np.all(np.diff(n_cells_list) <= 0)  # monotonically non-increasing


class Test_approx_mc2:
    @pytest.mark.parametrize("dataset_name", ["random-5", "random-10"])
    @pytest.mark.parametrize("eps", [0.8])
    @pytest.mark.parametrize("delta", [0.8])
    def test_if_estimate_within_bounds(self, dataset_name, eps, delta):
        program, I, T = get_input_program_by_name(dataset_name)

        _, true_count = get_ground_truth_count(program, I, T)
        lb, ub = get_theoretical_bounds(true_count, eps)

        estimate = approx_mc2(
            program,
            I,
            T,
            eps=eps,
            delta=delta,
            verbose=1,
            show_progress=True,
        )

        # strictly speaking, the test is wrong
        # because it does not consider that the assertion holds with probability at least 1 - delta
        # I admit that I'm lazy here
        assert lb <= estimate <= ub
