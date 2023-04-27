import pytest
import numpy as np

from bds.cbb import check_if_not_unsatisfied, check_if_satisfied
from bds.utils import bin_array


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


class TestCheckIfSatisfied():
    @pytest.mark.parametrize(
        's, z, expected_result',
        [([-1, -1], [1, 0], True),
         ([-1,  1], [1, 0], True),
         ([1, 1], [1, 0], True),
         ([0, 1], [1, 0], False),  # 1st constraint is unsatisfied
         ([1, -1], [1, 0], True),
         ([-1, -1], [0, 1], False),
         ([1, -1], [1, 1], False),
         ([1, 1], [0, 1], True),  # perhaps impossible case: constraints are satisfied but parity vector does not match
         ([-1,  1], [0, 0], False)])
    def test_case(self, s, z, expected_result):
        t = bin_array([1, 0])
        s = np.array(s, dtype=int)
        z = bin_array(z)
        assert check_if_satisfied(s, z, t) is expected_result
