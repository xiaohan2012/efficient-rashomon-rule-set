import numpy as np
import pytest

from bds.cbb import (
    ConstrainedBranchAndBoundNaive,
    check_if_not_unsatisfied,
    check_if_satisfied,
)
from bds.utils import bin_array, solutions_to_dict

from .fixtures import rules, y
from .utils import assert_dict_allclose


class TestCheckIfNotUnsatisfied:
    def test_satisfied_case(self):
        # the parity constraint system
        # we have 4 rules and 3 constraints
        A = bin_array([[0, 1, 0], [1, 0, 1]])
        t = bin_array([1, 0])

        # we add the first rule, note that the rules are 1-indexed
        j = 1
        s = np.array([-1, -1], dtype=int)
        z = bin_array([0, 1])
        sp, zp, not_unsatisfied = check_if_not_unsatisfied(j, A, t, s, z)
        assert not_unsatisfied is True

        # we add the second rule
        j = 2
        s = np.array([1, -1], dtype=int)
        z = bin_array([1, 1])
        sp, zp, not_unsatisfied = check_if_not_unsatisfied(j, A, t, s, z)
        assert not_unsatisfied is True

        # we add the third rule
        j = 3
        s = np.array([1, 1], dtype=int)
        z = bin_array([1, 0])
        sp, zp, not_unsatisfied = check_if_not_unsatisfied(j, A, t, s, z)
        assert not_unsatisfied is True

    def test_unsatisfied_case(self):
        # the parity constraint system
        # we have 4 rules and 3 constraints
        A = bin_array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 1, 0]])
        t = bin_array([0, 1, 1])
        # we add the first rule, note that the rules are 1-indexed
        j = 1
        s = np.array([-1, -1, -1], dtype=int)  # the satisfaction vector is [?, ?, ?]
        z = bin_array(
            [0, 0, 0]
        )  # the parity vector: no rule is added yet, thus all are even (0)
        sp, zp, not_unsatisfied = check_if_not_unsatisfied(j, A, t, s, z)

        # first rule appears in 2nd and 3rd constraint
        # thus we update the 2nd and 3rd rows in z
        expected_zp = bin_array([0, 1, 1])

        # and all constraints are still not unsatisfied
        expected_sp = np.array([-1, -1, -1])

        np.testing.assert_allclose(sp, expected_sp)
        np.testing.assert_allclose(zp, expected_zp)
        assert not_unsatisfied is True

        # then we add 3rd rule on top of the 1st one
        j = 3
        s, z = expected_sp, expected_zp
        sp, zp, not_unsatisfied = check_if_not_unsatisfied(j, A, t, s, z)

        # only the 2nd  constraints is updated
        # the 3rd is not updated because 2nd evaluates to False and checking stops
        expected_zp = bin_array([0, 0, 1])
        # and we can evaluate the 2nd constraint, which is even == odd (False)
        expected_sp = np.array([-1, 0, -1])

        np.testing.assert_allclose(sp, expected_sp)
        np.testing.assert_allclose(
            zp, expected_zp
        )  # [False, False,  True] != [False, False, False]
        assert not_unsatisfied is False


class TestCheckIfSatisfied:
    @pytest.mark.parametrize(
        "s, z, expected_result",
        [
            ([-1, -1], [1, 0], True),
            ([-1, 1], [1, 0], True),
            ([1, 1], [1, 0], True),
            ([0, 1], [1, 0], False),  # 1st constraint is unsatisfied
            ([1, -1], [1, 0], True),
            ([-1, -1], [0, 1], False),
            ([1, -1], [1, 1], False),
            (
                [1, 1],
                [0, 1],
                True,
            ),  # perhaps impossible case: constraints are satisfied but parity vector does not match
            ([-1, 1], [0, 0], False),
        ],
    )
    def test_case(self, s, z, expected_result):
        t = bin_array([1, 0])
        s = np.array(s, dtype=int)
        z = bin_array(z)
        assert check_if_satisfied(s, z, t) is expected_result


class TestConstrainedBranchAndBoundNaive:
    def test_reset(self, rules, y):
        ub = float("inf")
        lmbd = 0.1
        cbb = ConstrainedBranchAndBoundNaive(rules, ub, y, lmbd)

        A = bin_array([[1, 0, 1], [0, 1, 0]])
        t = bin_array([0, 1])

        cbb.reset(A, t)

        assert cbb.queue.size == 1
        item = cbb.queue.front()
        s, z = item[2:]
        np.testing.assert_allclose(s, -1)
        np.testing.assert_allclose(z, 0)

    @pytest.mark.parametrize(
        "ub, expected",
        [
            (
                float("inf"),
                {(0, 2): 0.1, (0, 1, 2, 3): 0.9},
            ),  # all satisfied solutions are returned
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
        res_iter = cbb.run(A, t, return_objective=True)

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
        res_iter = cbb.run(A, t, return_objective=True)

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
        res_iter = cbb.run(A, t, return_objective=True)
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
    def test_varying_ub_case_(self, rules, y, ub, expected):
        lmbd = 0.1
        cbb = ConstrainedBranchAndBoundNaive(rules, ub, y, lmbd)

        A = bin_array([[1, 0, 1], [1, 1, 0]])  # 0  # 1
        # either both rule 1 and rule 3 are selected or neither is selected
        # either rule 1 or rule 2 is selected
        t = bin_array([0, 1])
        res_iter = cbb.run(A, t, return_objective=True)
        sols = list(res_iter)

        actual = solutions_to_dict(sols)
        assert_dict_allclose(actual, expected)
