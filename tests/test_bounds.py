import pytest
import numpy as np
from gmpy2 import mpz, mpfr

from bds.utils import mpz_set_bits, mpz_clear_bits, randints, bin_random, bin_array
from bds.rule import Rule
from bds.bounds import (
    incremental_update_lb,
    incremental_update_obj,
    prefix_specific_length_upperbound,
    find_equivalence_points,
    EquivalentPointClass,
    get_equivalent_point_lb,
)


@pytest.mark.parametrize("seed", randints(5))
@pytest.mark.parametrize("num_fp", np.arange(6))
def test_incremental_update_lb(seed, num_fp):
    np.random.seed(seed)

    num_pts = 10

    # say we capture half of the points
    captured_idx = np.random.permutation(num_pts)[: int(num_pts / 2)]
    v = mpz_set_bits(mpz(), captured_idx)

    y = mpz_clear_bits(v, captured_idx[:num_fp])  # make `num_fp` mistakes
    true_inc_fp = num_fp / mpz(num_pts)
    actual = incremental_update_lb(v, y, mpz(num_pts))
    assert actual == true_inc_fp
    assert isinstance(actual, mpfr)


def test_incremental_update_obj():
    u = mpz_set_bits(mpz(), [1, 2, 5])  # points not captured by prefix
    v = mpz_set_bits(mpz(), [1, 4])  # captured by rule
    # not captured by either rule or prefix
    # {2, 3, 5, 6, 7}
    # {2, 3, 5, 6, 7} & {1, 2, 5} -> f = {2, 5}
    f = mpz_set_bits(mpz(), [2, 5])  # not captured by the rule and prefix
    y = mpz_set_bits(mpz(), [1, 2, 4, 5])  # the true labels
    # false negatives: {1, 4}
    num_pts = mpz(7)
    fn, actual_f = incremental_update_obj(u, v, y, num_pts)

    assert f == actual_f, bin(actual_f)
    assert fn == (mpz(2) / 7)
    assert isinstance(fn, mpfr)


def test_prefix_specific_length_upperbound():
    prefix_lb = 5
    prefix_length = 5
    ub = 10
    lmbd = 0.1
    # ub - prefix_lb = 5
    # 5 / lmbd = 50
    # 5 + 50
    assert 55 == prefix_specific_length_upperbound(prefix_lb, prefix_length, lmbd, ub)

    lmbd = 5.1
    # ub - prefix_lb = 5
    # floor(5 / lmbd) = 0
    # 5 + 0
    assert 5 == prefix_specific_length_upperbound(prefix_lb, prefix_length, lmbd, ub)

    lmbd = 4.9  # floor(5 / lmbd) = 1
    assert 6 == prefix_specific_length_upperbound(prefix_lb, prefix_length, lmbd, ub)


class TestEquivalentPointClass:
    def test_basic(self):
        epc = EquivalentPointClass(0)
        data_idx_and_labels = [
            (0, False),
            (1, 0),
            (2, 0),
            (3, True),
            (4, 1),
            (4, 1),
            (4, 1),  # duplicates shouldn't be over-counted
        ]
        for idx, label in data_idx_and_labels:
            epc.update(idx, label)

        assert epc.total_positives == 2
        assert epc.total_negatives == 3
        assert epc.minority_mistakes == 2
        assert epc.data_points == set(range(5))

    @pytest.mark.parametrize("label", [-1, "abc", 10, None])
    def test_invalid_label(self, label):
        epc = EquivalentPointClass(0)
        with pytest.raises(ValueError, match="invalid label.*"):
            epc.update(0, label)


class TestEquivalencePointsLowerBound:
    """test cases for find_equivalence_points and get_equivalence_point_lb"""

    @property
    def rules(self):
        return [
            Rule(id=0, name="rule-0", cardinality=1, truthtable=mpz("0b11000")),
            Rule(id=1, name="rule-1", cardinality=1, truthtable=mpz("0b11110")),
            Rule(id=2, name="rule-2", cardinality=1, truthtable=mpz("0b00111")),
        ]

    @property
    def y(self):
        return bin_array([0, 0, 1, 0, 0])

    def test_find_equivalence_points(self):
        # there are 5 points
        # the equivalent point classes are:
        # {0}, {1, 2}, {3, 4}
        (
            tot_not_captured_error_bound_init,
            pt2rules,
            ep_classes,
        ) = find_equivalence_points(self.y, self.rules)
        eqc_as_tuples = set(
            tuple(sorted(cls.data_points)) for cls in ep_classes.values()
        )
        assert eqc_as_tuples == {(0,), (1, 2), (3, 4)}

        assert tot_not_captured_error_bound_init == 1 / 5  # one minority mistake
        assert pt2rules == [[2], [1, 2], [1, 2], [0, 1], [0, 1]]

    @pytest.mark.parametrize(
        "captured, expected_lb", [(mpz("0b00110"), 1 / 5), (mpz("0b11000"), 0.0)]
    )
    def test_get_equivalent_point_lb(self, captured, expected_lb):
        (
            tot_not_captured_error_bound_init,
            pt2rules,
            ep_classes,
        ) = find_equivalence_points(self.y, self.rules)

        lb = get_equivalent_point_lb(captured, pt2rules, ep_classes)
        np.testing.assert_allclose(lb, expected_lb)
