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


class TestCheckIfNotUnsatisfied:
    def test_satisfied_case(self):
        # the parity constraint system
        # we have 3 rules and 2 constraints
        A = bin_array([[0, 1, 0], [1, 0, 1]])
        t = bin_array([1, 0])

        # we add the first rule (note that the rules are 1-indexed)
        # since rule-1 appears in the 2nd constraint
        # we update the parity value for the 2nd constraint to 1
        # however, we cannot evaluate it because the 3rd rule is not considered yet
        j = 1
        u = bin_ones(2)  # all constraints are undecided
        s = bin_zeros(2)
        z = bin_zeros(2)

        max_nz_idx_array = get_max_nz_idx_per_row(A)

        A_indices, A_indptr = get_indices_and_indptr(A)

        up, sp, zp, not_unsatisfied = check_if_not_unsatisfied(
            j, A, t, u, s, z, A_indices, A_indptr, max_nz_idx_array
        )
        assert not_unsatisfied is True
        np.testing.assert_allclose(
            up, bin_ones(2)
        )  # all constraints are still undetermined
        np.testing.assert_allclose(zp, bin_array([0, 1]))
        np.testing.assert_allclose(sp, bin_zeros(2))

        # we add the second rule
        # evaluation of constraint 1 is updated, because it contains only rule-2
        # while the rest remains the same
        j = 2
        # u, z, and s are "inherited" from previous run
        u = bin_ones(2)
        z = bin_array([0, 1])
        s = bin_zeros(2)

        up, sp, zp, not_unsatisfied = check_if_not_unsatisfied(
            j, A, t, u, s, z, A_indices, A_indptr, max_nz_idx_array
        )
        assert not_unsatisfied is True

        np.testing.assert_allclose(up, bin_array([0, 1]))
        np.testing.assert_allclose(zp, bin_ones(2))
        np.testing.assert_allclose(sp, bin_array([1, 0]))

        # we add the third rule
        # all constraints are determined and are satisfied
        j = 3
        u = bin_array([0, 1])
        z = bin_ones(2)
        s = bin_array([1, 0])
        up, sp, zp, not_unsatisfied = check_if_not_unsatisfied(
            j, A, t, u, s, z, A_indices, A_indptr, max_nz_idx_array
        )
        assert not_unsatisfied is True
        np.testing.assert_allclose(up, bin_zeros(2))  # all constraints are determined
        np.testing.assert_allclose(zp, bin_array([1, 0]))
        np.testing.assert_allclose(sp, bin_ones(2))  # all are satisfied

    def test_unsatisfied_case(self):
        # the parity constraint system
        # we have 4 rules and 3 constraints
        A = bin_array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 1, 0]])
        t = bin_array([0, 1, 1])
        # we add the first rule, note that the rules are 1-indexed
        j = 1
        u = bin_ones(3)
        s = bin_zeros(3)
        z = bin_zeros(3)

        max_nz_idx_array = get_max_nz_idx_per_row(A)

        A_indices, A_indptr = get_indices_and_indptr(A)
        up, sp, zp, not_unsatisfied = check_if_not_unsatisfied(
            j, A, t, u, s, z, A_indices, A_indptr, max_nz_idx_array
        )

        # first rule appears in 2nd and 3rd constraint
        # thus we update the 2nd and 3rd rows in z
        np.testing.assert_allclose(up, bin_ones(3))
        np.testing.assert_allclose(zp, bin_array([0, 1, 1]))
        assert not_unsatisfied is True

        # then we add 3rd rule on top of the 1st one
        # only the 2nd constraints is updated
        j = 3
        u = bin_ones(3)
        s = bin_zeros(3)
        z = bin_array([0, 1, 1])

        up, sp, zp, not_unsatisfied = check_if_not_unsatisfied(
            j, A, t, u, s, z, A_indices, A_indptr, max_nz_idx_array
        )

        np.testing.assert_allclose(up, bin_array([1, 0, 1]))
        np.testing.assert_allclose(sp, bin_zeros(3))
        # parity for the the 3rd constarint is not updated because 2nd evaluates to False and checking stops
        np.testing.assert_allclose(zp, bin_array([0, 0, 1]))
        assert not_unsatisfied is False


class TestCheckIfSatisfied:
    @pytest.mark.parametrize(
        "u, s, z, expected_result",
        [
            (bin_ones(2), bin_zeros(2), bin_array([1, 0]), True),
            (bin_array([1, 0]), bin_array([0, 1]), bin_array([1, 0]), True),
            (bin_array([1, 0]), bin_array([0, 0]), bin_array([1, 0]), False),
            (bin_array([0, 0]), bin_array([1, 1]), bin_array([0, 1]), True),
            (bin_array([0, 1]), bin_array([1, 0]), bin_array([1, 1]), False),
            (bin_array([1, 1]), bin_array([0, 0]), bin_array([0, 1]), False),
            (bin_array([0, 0]), bin_array([1, 1]), bin_array([1, 0]), True),
            (bin_array([1, 0]), bin_array([0, 1]), bin_array([0, 0]), False),
        ],
    )
    def test_case(self, u, s, z, expected_result):
        t = bin_array([1, 0])
        assert check_if_satisfied(u, s, z, t) is expected_result


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
