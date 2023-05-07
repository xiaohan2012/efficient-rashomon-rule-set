import gmpy2 as gmp
import numpy as np
import pytest
from gmpy2 import mpfr, mpz

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
        # we have 3 rules and 2 constraints
        A = bin_array([[0, 1, 0], [1, 0, 1]])
        t = bin_array([1, 0])

        # we add the first rule (note that the rules are 1-indexed)
        # since rule-1 appears in the 2nd constraint
        # we update the parity value for the 2nd constraint to 1
        # however, we cannot evaluate it because the 3rd rule is not considered yet
        j = 1
        u = mpz("0b11")  # all constraints are undecided
        s = mpz()
        z = mpz("0b00")
        # bin_array([0, 1])
        up, sp, zp, not_unsatisfied = check_if_not_unsatisfied(j, A, t, u, s, z)
        assert not_unsatisfied is True
        assert up == mpz("0b11")  # all constraints are still undetermined
        assert zp == mpz("0b10")
        # assert sp == mpz()  # do not test sp because all constraints are not determined, thus sp is not relevant

        # we add the second rule
        # evaluation of constraint 1 is updated, because it contains only rule-2
        # while the rest remains the same
        j = 2
        # u, z, and s are "inheritted" from previous run
        u = mpz("0b11")
        z = mpz("0b10")
        s = mpz()
        up, sp, zp, not_unsatisfied = check_if_not_unsatisfied(j, A, t, u, s, z)
        assert not_unsatisfied is True
        assert up == mpz("0b10")
        assert zp == mpz("0b11")
        assert sp == mpz("0b01")

        # we add the third rule
        # all constraints are determined and are satisfied
        j = 3
        u = mpz("0b10")
        z = mpz("0b11")
        s = mpz("0b01")
        up, sp, zp, not_unsatisfied = check_if_not_unsatisfied(j, A, t, u, s, z)
        assert not_unsatisfied is True
        assert up == mpz("0b00")  # all constraints are determined
        assert zp == mpz("0b01")
        assert sp == mpz("0b11")  # all are satisfied

    def test_unsatisfied_case(self):
        # the parity constraint system
        # we have 4 rules and 3 constraints
        A = bin_array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 1, 0]])
        t = bin_array([0, 1, 1])
        # we add the first rule, note that the rules are 1-indexed
        j = 1
        u = mpz("0b111")
        s = mpz()
        z = mpz()
        up, sp, zp, not_unsatisfied = check_if_not_unsatisfied(j, A, t, u, s, z)

        # first rule appears in 2nd and 3rd constraint
        # thus we update the 2nd and 3rd rows in z
        assert up == mpz("0b111")
        assert zp == mpz("0b110")
        assert not_unsatisfied is True

        # then we add 3rd rule on top of the 1st one
        # only the 2nd constraints is updated
        j = 3
        u = mpz("0b111")
        s = mpz()
        z = mpz("0b110")

        up, sp, zp, not_unsatisfied = check_if_not_unsatisfied(j, A, t, u, s, z)

        assert up == mpz("0b101")
        assert sp == mpz("0b000")
        # parity for the the 3rd constarint is not updated because 2nd evaluates to False and checking stops
        assert zp == mpz("0b100")
        assert not_unsatisfied is False


class TestCheckIfSatisfied:
    @pytest.mark.parametrize(
        "u, s, z, expected_result",
        [
            (mpz("0b11"), mpz("0b00"), mpz("0b01"), True),
            (mpz("0b01"), mpz("0b10"), mpz("0b01"), True),
            (mpz("0b01"), mpz("0b00"), mpz("0b01"), False),
            (mpz("0b00"), mpz("0b11"), mpz("0b10"), True),
            (mpz("0b10"), mpz("0b01"), mpz("0b11"), False),
            (mpz("0b11"), mpz("0b00"), mpz("0b10"), False),
            (mpz("0b00"), mpz("0b11"), mpz("0b01"), True),
            (mpz("0b01"), mpz("0b10"), mpz("0b00"), False)
        ],
    )
    def test_case(self, u, s, z, expected_result):
        t = bin_array([1, 0])
        assert check_if_satisfied(u, s, z, t) is expected_result


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
    def test_varying_ub_case_4(self, rules, y, ub, expected):
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
