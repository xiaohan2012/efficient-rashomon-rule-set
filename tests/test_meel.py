import ray
import numpy as np
import pytest
from collections import Counter

from bds.common import EPSILON
from bds.gf2 import is_piecewise_linear
from bds.meel import (
    approx_mc2_core,
    log_search,
    approx_mc2,
    _get_theoretical_bounds,
    UniGen,
)
from bds.random_hash import generate_h_and_alpha
from bds.utils import bin_array, bin_random, bin_zeros, randints
from bds.bb import get_ground_truth_count

from .fixtures import rules, y
from .utils import generate_random_rules_and_y


@pytest.fixture(scope="module")
def ray_fix():
    ray.init(num_cpus=4)
    yield None
    ray.shutdown()


class TestLogSearch:
    def generate_random_input(self, num_rules: int, num_pts: int, rand_seed: int):
        random_rules, random_y = generate_random_rules_and_y(
            num_pts, num_rules, rand_seed=rand_seed
        )
        m = num_rules - 1

        A, b = generate_h_and_alpha(num_rules, m, seed=rand_seed, as_numpy=True)

        return random_rules, random_y, A, b

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
        A, b = generate_h_and_alpha(n, m, seed=rand_seed, as_numpy=True)
        # test statements
        m, Y_size, big_cell, Y_size_arr = log_search(
            rules, y, lmbd, ub, A, b, thresh, m_prev, return_full=True
        )[:4]

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
        random_rules, random_y, A, b = self.generate_random_input(10, 50, 1234)

        lmbd = 0.1
        m, Y_size, big_cell, Y_size_arr = log_search(
            random_rules,
            random_y,
            lmbd,
            ub + EPSILON,
            A,
            b,
            thresh,
            m_prev,
            return_full=True,
        )[:4]

        self.check_output(m, Y_size, big_cell, Y_size_arr, thresh)

    @pytest.mark.parametrize("ub", [0.8])
    @pytest.mark.parametrize("thresh", [25])
    @pytest.mark.parametrize("rand_seed", randints(3))
    def test_on_search_trajectory(self, ub, thresh, rand_seed):
        m_prev = 1
        random_rules, random_y, A, t = self.generate_random_input(20, 50, 1234)

        lmbd = 0.1
        m, Y_size, big_cell, Y_size_arr, search_trajectory = log_search(
            random_rules,
            random_y,
            lmbd,
            ub + EPSILON,
            A,
            t,
            thresh,
            m_prev,
            return_full=True,
        )[:5]

        # check format of the search trajectory
        for tpl in search_trajectory:
            assert len(tpl) == 3
            m, Y_size, t = tpl
            assert isinstance(m, int)
            assert isinstance(Y_size, int)
            assert isinstance(t, int)

        # check the trajectory is logical, meaning the following cases:
        # 1. for a certain m which gives |Y| >= t, (m is too small)
        # all m' > m should not be tried in later search
        # and similarly
        # 2. for a certain m which gives |Y| < t, (m is too large)
        # all m' < m should not be tried in later search
        def extract_later_ms(i: int) -> np.ndarray:
            """extract m values that are searched after the ith iteration"""
            return np.array(list(map(lambda tpl: tpl[0], search_trajectory[i + 1 :])))

        for i, (m, ys, t) in enumerate(search_trajectory):
            if ys < t:  # not enough solutions, m is large, we try smaller m later
                later_ms = extract_later_ms(i)
                np.testing.assert_allclose(later_ms < m, True)
            else:  # m is small, we try larger m later
                later_ms = extract_later_ms(i)
                np.testing.assert_allclose(later_ms > m, True)

    @pytest.mark.parametrize(
        "ub", [0.6, 0.75]  # using larger ub, e.g., 1.0 tends to run slower
    )
    @pytest.mark.parametrize('initial_m', randints(3, vmin=1, vmax=5))
    @pytest.mark.parametrize("thresh", [5, 10])
    @pytest.mark.parametrize("rand_seed", randints(3))
    # @pytest.mark.parametrize(
    #     "ub", [0.75]
    # )
    # @pytest.mark.parametrize('initial_m', [3])
    # @pytest.mark.parametrize("thresh", [10])
    # @pytest.mark.parametrize("rand_seed", [1707139767])
    def test_consistency_on_m(self, ub, initial_m, thresh, rand_seed):
        """no matter which initial m is provided, the same m should be returned"""
        random_rules, random_y, A, b = self.generate_random_input(10, 50, rand_seed)

        m = A.shape[0]

        lmbd = 0.1
        ref_m, ref_Y_size, ref_big_cell, ref_Y_size_arr = log_search(
            random_rules,
            random_y,
            lmbd,
            ub + EPSILON,
            A,
            b,
            thresh,
            initial_m,
            return_full=True,
        )[:4]
        for m_prev in range(1, m):
            (
                actual_m,
                actual_Y_size,
                actual_big_cell,
                actual_Y_size_arr
            ) = log_search(
                random_rules,
                random_y,
                lmbd,
                ub + EPSILON,
                A,
                b,
                thresh,
                m_prev=m_prev,
                return_full=True,
            )[:4]
            print("m_prev: {}".format(m_prev))
            np.testing.assert_equal(ref_big_cell, actual_big_cell)
            assert ref_m == actual_m
            assert ref_Y_size == actual_Y_size


    @pytest.mark.parametrize("m_prev", [2, 3, 4])
    def test_too_large_m_prev(self, m_prev):
        with pytest.raises(ValueError, match="m_prev .* should be smaller than 2"):
            log_search(
                # just pass in a list in order to get the number of rules
                [1, 1],
                [1, 1],
                lmbd=0.1,
                ub=float("inf"),
                A=bin_zeros((2, 2)),
                b=bin_zeros(2),
                thresh=2,
                m_prev=m_prev,
            )

    @pytest.mark.parametrize("thresh", [0, 1])
    def test_too_small_thresh(self, thresh):
        with pytest.raises(ValueError, match="thresh should be at least 1"):
            log_search(
                [1, 1],
                [1, 1],
                lmbd=0.1,
                ub=float("inf"),
                A=bin_zeros((2, 2)),
                b=bin_zeros(2),
                thresh=thresh,
                m_prev=1,
            )


class TestApproxMC2Core:
    @pytest.mark.parametrize(
        "ub",
        # [.5]
        [0.5, 0.75],
    )
    @pytest.mark.parametrize(
        "thresh",
        # [5]
        [5, 10],
    )
    @pytest.mark.parametrize("rand_seed", randints(3))
    def test_basic(self, ub, thresh, rand_seed):
        """e.g., the code runs and return data with correct types"""
        prev_m = 5
        num_pts, num_rules = 100, 10
        random_rules, random_y = generate_random_rules_and_y(
            num_pts, num_rules, rand_seed=1234
        )
        n_cells, Y_size = approx_mc2_core(
            random_rules,
            random_y,
            lmbd=0.1,
            ub=ub,
            thresh=thresh,
            prev_num_cells=2**prev_m,  # Q: why not just pass in prev_m directly?
            rand_seed=rand_seed,
        )
        assert isinstance(n_cells, int)
        assert Y_size <= thresh

    @pytest.mark.parametrize("ub", [0.5, 0.75])
    @pytest.mark.parametrize("rand_seed", randints(3))
    def test_monotonicity(self, ub, rand_seed):
        """as we increase thresh, Y_size should be non-decreasing and m should be non-increasing"""
        thresh_list = np.arange(2, 10, 1, dtype=int)
        prev_m = 1

        random_rules, random_y = generate_random_rules_and_y(100, 10, rand_seed=1234)

        n_cells_list = []
        Y_size_list = []

        for thresh in thresh_list:
            n_cells, Y_size = approx_mc2_core(
                random_rules,
                random_y,
                lmbd=0.1,
                ub=ub,
                thresh=thresh,
                prev_num_cells=2**prev_m,
                rand_seed=rand_seed,
            )
            Y_size_list.append(Y_size)
            n_cells_list.append(n_cells)

        assert np.all(np.diff(Y_size_list) >= 0)  # monotonically non-decreasing
        assert np.all(np.diff(n_cells_list) <= 0)  # monotonically non-increasing

    @pytest.mark.parametrize("ub", [0.5, 0.6, 0.75])
    @pytest.mark.parametrize("thresh", [2, 3])
    def test_return_none(self, ub, thresh):
        prev_m = 1
        num_rules = 10
        random_rules, random_y = generate_random_rules_and_y(
            100, num_rules, rand_seed=1234
        )
        # the constraint system is vacuum
        # thus using all constraints is equivalent to using no constraint at all
        A = bin_zeros((num_rules - 1, num_rules))
        b = bin_zeros((num_rules - 1,))
        n_cells, Y_size = approx_mc2_core(
            random_rules,
            random_y,
            lmbd=0.01,
            ub=ub,
            thresh=thresh,
            prev_num_cells=2**prev_m,
            A=A,
            b=b,
        )
        assert Y_size is None
        assert n_cells is None


class TestApproxMC2:
    @pytest.mark.parametrize("rand_seed", randints(1))
    def test_parallel_execution(self, rand_seed, ray_fix):
        ub = 0.9
        eps = 0.8
        delta = 0.8
        lmbd = 0.1
        num_pts, num_rules = 100, 10
        random_rules, random_y = generate_random_rules_and_y(
            num_pts, num_rules, rand_seed=1234
        )



        estimate_actual = approx_mc2(
            random_rules,
            random_y,
            lmbd=lmbd,
            ub=ub,
            delta=delta,
            eps=eps,
            rand_seed=rand_seed,
            parallel=True,  # using paralle run
        )

        estimate_expected = approx_mc2(
            random_rules,
            random_y,
            lmbd=lmbd,
            ub=ub,
            delta=delta,
            eps=eps,
            rand_seed=rand_seed,
            parallel=False,  # sequential run
        )
        assert estimate_expected == estimate_actual
        
        

    @pytest.mark.parametrize("ub", [0.6, 0.9, 1.0])
    @pytest.mark.parametrize("eps", [0.8, 0.5])
    @pytest.mark.parametrize("delta", [0.8])
    @pytest.mark.parametrize("rand_seed", randints(3))
    def test_if_estimate_within_bounds(self, ub, eps, delta, rand_seed):
        num_pts, num_rules = 100, 10
        random_rules, random_y = generate_random_rules_and_y(
            num_pts, num_rules, rand_seed=1234
        )

        lmbd = 0.1

        true_count = get_ground_truth_count(random_rules, random_y, lmbd, ub)

        est_lb, est_ub = _get_theoretical_bounds(true_count, eps)

        estimate = approx_mc2(
            random_rules,
            random_y,
            lmbd=lmbd,
            ub=ub,
            delta=delta,
            eps=eps,
            rand_seed=rand_seed,
            show_progress=False,
        )

        # the test may fail
        # because it does not consider that the assertion holds with probability at least 1 - delta
        assert est_lb <= estimate <= est_ub


class TestUniGen:
    def create_unigen(self, ub: float, eps: float, rand_seed=None) -> UniGen:
        num_pts, num_rules = 100, 10
        random_rules, random_y = generate_random_rules_and_y(
            num_pts, num_rules, rand_seed=1234
        )

        lmbd = 0.1
        return UniGen(random_rules, random_y, lmbd, ub, eps, rand_seed)

    @pytest.mark.parametrize("eps", [8, 10])
    @pytest.mark.parametrize("ub, sample_directly", [(0.6, True), (0.9, False)])
    @pytest.mark.parametrize("rand_seed", randints(3))
    def test_prepare(self, eps, ub, sample_directly, rand_seed):
        ug = self.create_unigen(ub, eps, rand_seed)

        ug.prepare()
        assert isinstance(ug.presolve_Y_size, int)
        assert isinstance(ug.presolve_Y, list)

        assert ug.sample_directly == sample_directly

        if not sample_directly:
            assert isinstance(ug.C, int)
            assert isinstance(ug.q, int)
        else:
            assert not hasattr(ug, "C")
            assert not hasattr(ug, "q")

    @pytest.mark.parametrize("ub", [0.8])
    @pytest.mark.parametrize("thresh", [10.5, 20.7, 30.0, 40])
    def test_presolve(self, ub, thresh):
        ug = self.create_unigen(ub, eps=2.0)
        Y_size, Y = ug.presolve(thresh)
        assert Y_size <= thresh
        assert isinstance(Y, list)

    @pytest.mark.parametrize("ub", [0.6, 0.9])
    @pytest.mark.parametrize("rand_seed", randints(3))
    def test_sample_once(self, ub, rand_seed):
        ug = self.create_unigen(ub, eps=10.0, rand_seed=rand_seed)
        ug.prepare()
        ret = ug.sample_once()
        assert ret is None or isinstance(ret, tuple)

    @pytest.mark.parametrize("ub", [0.6, 0.9])
    @pytest.mark.parametrize("rand_seed", randints(3))
    def test_sample(self, ub, rand_seed):
        ug = self.create_unigen(ub, eps=10.0, rand_seed=rand_seed)
        ug.prepare()
        samples = ug.sample(10, exclude_none=True)
        assert len(samples) > 0
        for s in samples:
            assert isinstance(s, tuple)

    @pytest.mark.parametrize("rand_seed", randints(3))
    def test_statistical_property(self, rand_seed):
        """warning: this is not a very formal test because it ignores variation in probability estimation"""
        eps = 8  # eps set this way to ensure we go to the else branch
        ug = self.create_unigen(0.9, eps, rand_seed=rand_seed)
        ug.prepare()
        true_count, expected_solutions = get_ground_truth_count(
            ug.rules, ug.y, ug.lmbd, ug.ub, return_sols=True
        )

        n_samples = 500
        samples = ug.sample(n_samples)

        counts = Counter(map(tuple, samples))
        freq = np.array(list(counts.values()))
        proba = freq / n_samples

        # all samples should be a member of the true solution set
        sample_solution_set = set(map(tuple, samples))
        expected_solution_set = set(map(tuple, expected_solutions))

        assert sample_solution_set.issubset(expected_solution_set)

        # probability checking: they should be bounded from both sides
        true_proba = 1 / true_count
        lb, ub = true_proba / (1 + eps), true_proba * (1 + eps)

        # np.testing.assert_allclose(lb <= proba, True)
        # np.testing.assert_allclose(ub >= proba, True)
        assert (lb <= proba).all()
        assert (proba <= ub).all()
