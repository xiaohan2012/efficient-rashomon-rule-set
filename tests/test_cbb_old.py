import numpy as np
import pytest

from bds.cbb import (
    ConstrainedBranchAndBoundNaive,
    check_if_not_unsatisfied,
    check_if_satisfied,
)
from bds.utils import (
    bin_array,
    solutions_to_dict,
    bin_zeros,
    bin_ones,
    get_max_nz_idx_per_row,
    get_indices_and_indptr,
    randints,
)

from bds.random_hash import generate_h_and_alpha
from .fixtures import rules, y
from .utils import (
    assert_dict_allclose,
    generate_random_rules_and_y,
    brute_force_enumeration,
)


class TestConstrainedBranchAndBoundNaive:
    @pytest.mark.skip("because equivalent point bound is disabled for now")
    def test___post_init__(self, rules, y):
        # after cbb is created, equivalent points-related attributes should be available
        cbb = ConstrainedBranchAndBoundNaive(rules, float("inf"), y, 0.1)

        assert isinstance(cbb._equivalent_pts, dict)
        assert isinstance(cbb._pt2rules, list)

    @pytest.mark.parametrize(
        "A, expected",
        [
            # ~A = [[0, 1, 0], [1, 1, 1]]
            ([[1, 0, 1], [0, 0, 0]], {1: [1], 2: [0, 1], 3: [1]}),
            # ~A = [[0, 1, 1], [1, 1, 1]]
            ([[1, 0, 0], [0, 1, 1]], {1: [1], 2: [0], 3: [0]}),
            ([[0, 0, 0], [0, 0, 0]], {1: [0, 1], 2: [0, 1], 3: [0, 1]}),
            ([[1, 1, 1], [1, 1, 1]], {1: [1], 2: [1], 3: [1]}), # A becomes [[1, 1, 1], [0, 0, 0]] after rref
        ],
    )
    def test_attribute_neg_ruleid2cst_idxs(self, rules, y, A, expected):
        A = bin_array(A)
        t = bin_array([0, 0])  # any length-2 binary array is fine
        cbb = ConstrainedBranchAndBoundNaive(rules, float("inf"), y, 0.1)
        cbb.setup_constraint_system(A, t)
        assert_dict_allclose(cbb.neg_ruleid2cst_idxs, expected)

    @pytest.mark.parametrize(
        "t, solvable", [(bin_array([0, 1]), False), (bin_array([0, 0]), True)]
    )
    def test_is_linear_system_solvable(self, rules, y, t, solvable):
        A = bin_array([[1, 0, 1], [0, 0, 0]])
        ub = float("inf")
        lmbd = 0.1
        cbb = ConstrainedBranchAndBoundNaive(rules, ub, y, lmbd)
        cbb.reset(A, t)
        assert cbb.is_linear_system_solvable == solvable

    def test_reset(self, rules, y):
        ub = float("inf")
        lmbd = 0.1
        cbb = ConstrainedBranchAndBoundNaive(rules, ub, y, lmbd)

        A = bin_array([[1, 0, 1], [0, 1, 0]])
        t = bin_array([0, 1])

        cbb.reset(A, t)

        assert cbb.queue.size == 1
        item = cbb.queue.front()
        u, s, z = item[2:]
        for value in [u, s, z]:
            assert isinstance(value, np.ndarray)

        np.testing.assert_allclose(u, 1)
        np.testing.assert_allclose(s, 0)
        np.testing.assert_allclose(z, 0)

    @pytest.mark.parametrize(
        "ub, expected",
        [
            (  # case 1: all satisfied solutions are returned
                float("inf"),
                {(0, 2): 0.1, (0, 1, 2, 3): 0.9},
            ),
            (0.5, {(0, 2): 0.1}),
            (0.01, dict()),
        ],
    )
    def test_varying_ub_case_1(self, rules, y, ub, expected):
        lmbd = 0.1
        cbb = ConstrainedBranchAndBoundNaive(rules, ub, y, lmbd)

        # 3 rules
        # 2 constraints
        A = bin_array([[1, 0, 1], [0, 1, 0]])  # 0  # 1
        # rule-2 has to be selected
        # rule-0 and rule-1 is either both selected or both unselected
        t = bin_array([0, 1])
        res_iter = cbb.run(return_objective=True, A=A, t=t)

        sols = list(res_iter)
        actual = solutions_to_dict(sols)
        assert_dict_allclose(actual, expected)

    @pytest.mark.parametrize(
        "ub, expected",
        [
            (float("inf"), {(0, 3): 0.3, (0, 1): 0.9}),
            (0.5, {(0, 3): 0.3}),
            (0.1, dict()),
        ],
    )
    def test_varying_ub_case_2(self, rules, y, ub, expected):
        lmbd = 0.1
        cbb = ConstrainedBranchAndBoundNaive(rules, ub, y, lmbd)

        A = bin_array([[1, 0, 1], [0, 1, 0]])  # 1  # 0
        # either rule 1 or rule 3 is selected
        # rule 2 cannot be selected
        t = bin_array([1, 0])
        res_iter = cbb.run(return_objective=True, A=A, t=t)

        sols = list(res_iter)
        actual = solutions_to_dict(sols)
        assert_dict_allclose(actual, expected)

    @pytest.mark.parametrize(
        "ub, expected",
        [
            (float("inf"), {(0, 1, 2, 3): 0.9}),
            (0.1, dict()),
        ],
    )
    def test_varying_ub_case_3(self, rules, y, ub, expected):
        lmbd = 0.1
        cbb = ConstrainedBranchAndBoundNaive(rules, ub, y, lmbd)

        A = bin_array([[1, 0, 1], [1, 1, 0]])  # 0  # 0
        # either both rule 1 and rule 3 are selected or neither is selected
        # either both rule 1 and rule 2 are selected or neither is selected
        t = bin_array([0, 0])
        res_iter = cbb.run(return_objective=True, A=A, t=t)
        sols = list(res_iter)
        actual = solutions_to_dict(sols)
        assert_dict_allclose(actual, expected)

    @pytest.mark.parametrize(
        "ub, expected",
        [
            (float("inf"), {(0, 1, 3): 0.8, (0, 2): 0.1}),
            (0.1, {(0, 2): 0.1}),
            (0.01, dict()),
        ],
    )
    def test_varying_ub_case_4(self, rules, y, ub, expected):
        lmbd = 0.1
        cbb = ConstrainedBranchAndBoundNaive(rules, ub, y, lmbd)

        A = bin_array([[1, 0, 1], [1, 1, 0]])  # 0  # 1
        # either both rule 1 and rule 3 are selected or neither is selected
        # either rule 1 or rule 2 is selected
        t = bin_array([0, 1])
        res_iter = cbb.run(return_objective=True, A=A, t=t)
        sols = list(res_iter)

        actual = solutions_to_dict(sols)
        assert_dict_allclose(actual, expected)

    @pytest.mark.parametrize(
        "thresh, count", [(None, 2), (1, 1), (2, 2), (3, 2)]  # total count is returned
    )
    def test_bounded_count(self, rules, y, thresh, count):
        ub = float("inf")
        lmbd = 0.1
        cbb = ConstrainedBranchAndBoundNaive(rules, ub, y, lmbd)

        A = bin_array([[1, 0, 1], [1, 1, 0]])
        t = bin_array([0, 1])

        assert cbb.bounded_count(thresh, A=A, t=t) == count

    @pytest.mark.parametrize(
        "thresh, count", [(None, 2), (1, 1), (2, 2), (3, 2)]  # total count is returned
    )
    def test_bounded_sols(self, rules, y, thresh, count):
        ub = float("inf")
        lmbd = 0.1
        cbb = ConstrainedBranchAndBoundNaive(rules, ub, y, lmbd)

        A = bin_array([[1, 0, 1], [1, 1, 0]])  # 0  # 1
        t = bin_array([0, 1])

        sols = cbb.bounded_sols(thresh, A=A, t=t)
        assert isinstance(sols, list)
        assert len(sols) == count

    @pytest.mark.parametrize("num_rules", [10])
    @pytest.mark.parametrize("num_constraints", [2, 4, 8])
    @pytest.mark.parametrize("lmbd", [0.1])
    @pytest.mark.parametrize("ub", [0.801, 0.501, 0.001])  # float("inf"),  # , 0.01
    @pytest.mark.parametrize("rand_seed", randints(5))
    # @pytest.mark.parametrize("num_rules", [10])
    # @pytest.mark.parametrize("num_constraints", [2])
    # @pytest.mark.parametrize("lmbd", [0.1])
    # @pytest.mark.parametrize("ub", [0.501])  # float("inf"),  # , 0.01
    # @pytest.mark.parametrize("rand_seed", [162140838])    
    def test_solution_correctness(
        self, num_rules, num_constraints, lmbd, ub, rand_seed
    ):
        """the output should be the same as ground truth"""
        rand_rules, rand_y = generate_random_rules_and_y(10, num_rules, rand_seed)
        # for r in rand_rules:
        #     print(f'{r.name}: {bin(r.truthtable)}')
        # print(rand_y[::-1].astype(int))
        cbb = ConstrainedBranchAndBoundNaive(rand_rules, ub, rand_y, lmbd)

        A, t = generate_h_and_alpha(
            num_rules, num_constraints, rand_seed, as_numpy=True
        )
        actual = solutions_to_dict(list(cbb.run(return_objective=True, A=A, t=t)))

        expected = solutions_to_dict(
            list(brute_force_enumeration(rand_rules, rand_y, A, t, ub, lmbd))
        )
        # print(expected)
        # assert set(actual.keys()) == set(expected.keys())
        assert_dict_allclose(actual, expected)
        # print("len(expected): {}".format(len(expected)))
        # print("expected: {}".format(expected))
        # raise ValueError(expected)
