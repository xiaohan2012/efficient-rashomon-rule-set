import math
import random
import numpy as np
import pytest
from gmpy2 import mpz
from operator import itemgetter

from bds.cbb import ConstrainedBranchAndBound
from bds.gf2 import GF
from bds.random_hash import generate_h_and_alpha
from bds.rule import Rule
from bds.types import RuleSet
from bds.utils import bin_array, randints, solutions_to_dict, powerset, bin_zeros

from .utils import (
    assert_dict_allclose,
    brute_force_enumeration,
    generate_random_rules_and_y,
    is_disjoint,
    normalize_solutions,
)
from .fixtures import rules, y


class UtilityMixin:
    def _create_input_data(self, num_rules, num_constraints, rand_seed):
        """return rules, y, A, b"""
        rand_rules, rand_y = generate_random_rules_and_y(10, num_rules, rand_seed)
        A, b = generate_h_and_alpha(
            num_rules, num_constraints, rand_seed, as_numpy=True
        )
        return rand_rules, rand_y, A, b


class TestParityConstraintRelatedMethods(UtilityMixin):
    def _create_cbb(self):
        num_constraints = 2
        rand_rules, rand_y, A, b = self._create_input_data(10, num_constraints, 1234)
        cbb = ConstrainedBranchAndBound(
            rand_rules, float("inf"), rand_y, lmbd=0.1, reorder_columns=False
        )
        cbb.reset(A=A, b=b)
        return cbb

    def test__ensure_satisfaction_returned_types(self):
        """check the types of the returned data"""
        cbb = self._create_cbb()
        pvts = cbb._ensure_satisfiability(RuleSet({0, 1} - cbb.pivot_rule_idxs))
        assert isinstance(pvts, RuleSet)

    def test__ensure_satisfaction_invalid_input(self):
        """pivot rules are wrongly included in the input"""
        cbb = self._create_cbb()
        with pytest.raises(ValueError, match="prefix should not contain any pivots.*"):
            cbb._ensure_satisfiability(RuleSet([list(cbb.pivot_rule_idxs)[0]]))

    @pytest.mark.parametrize("prefix", random.sample(list(powerset(range(10))), 16))
    def test__ensure_minimal_non_violation_captured_vector_calculation(self, prefix):
        cbb = self._create_cbb()
        input_ruleset = RuleSet(prefix) - RuleSet(cbb.pivot_rule_idxs)
        extension, v, z, s = cbb._ensure_minimal_non_violation(input_ruleset)
        assert v == cbb._lor(RuleSet(extension) + input_ruleset)

    def test__ensure_minimal_non_violation_returned_types(self):
        """check the types of the returned data"""
        cbb = self._create_cbb()
        pvts, v, z, s = cbb._ensure_minimal_non_violation(
            RuleSet({0, 1} - cbb.pivot_rule_idxs)
        )
        assert isinstance(pvts, RuleSet)
        assert isinstance(v, mpz)
        assert isinstance(z, np.ndarray)
        assert z.shape == (cbb.A.shape[0],)
        assert isinstance(s, np.ndarray)
        assert s.shape == (cbb.A.shape[0],)

    def test__ensure_minimal_non_violation_invalid_input(self):
        """pivot rules are wrongly included in the input"""
        cbb = self._create_cbb()
        with pytest.raises(ValueError, match="prefix should not contain any pivots.*"):
            cbb._ensure_minimal_non_violation(RuleSet([list(cbb.pivot_rule_idxs)[0]]))


class TestConstrainedBranchAndBoundMethods(UtilityMixin):
    @property
    def input_rules(self):
        return [
            Rule(0, "rule-1", 1, mpz("0b00101011")),
            Rule(1, "rule-2", 1, mpz("0b00001101")),
            Rule(2, "rule-3", 1, mpz("0b10001011")),
            Rule(3, "rule-4", 1, mpz()),
        ]

    @property
    def input_y(self):
        return bin_array([1, 1, 1, 1, 0, 0, 1, 1])

    def test_init(self):
        cbb = ConstrainedBranchAndBound(
            self.input_rules, float("inf"), self.input_y, 0.1, reorder_columns=False
        )
        assert len(cbb.truthtable_list) == len(self.input_rules)

    @pytest.mark.parametrize(
        "A, b, exp_pivot_rule_idxs, exp_free_rule_idxs, exp_row2pivot_column, exp_B",
        [
            ([[1, 0, 1], [0, 1, 0]], [0, 1], {0, 1}, {2}, [0, 1], [2, -1]),
            ([[1, 0, 1], [0, 0, 1]], [0, 1], {0, 2}, {1}, [0, 2], [-1, -1]),
            ([[1, 0, 1]], [0], {0}, {1, 2}, [0], [2]),
            ([[1, 0, 0], [1, 0, 0], [1, 0, 0]], [0, 0, 0], {0}, {1, 2}, [0], [-1]),
        ],
    )
    def test_setup_constraint_system(
        self,
        A,
        b,
        exp_pivot_rule_idxs,
        exp_free_rule_idxs,
        exp_row2pivot_column,
        exp_B,
    ):
        rand_rules, rand_y = generate_random_rules_and_y(10, 3, 12345)
        cbb = ConstrainedBranchAndBound(
            rand_rules, float("inf"), rand_y, 0.1, reorder_columns=False
        )

        cbb.setup_constraint_system(bin_array(A), bin_array(b))
        assert hasattr(cbb, "A_gf")
        assert hasattr(cbb, "b_gf")
        assert isinstance(cbb.A_gf, GF)
        assert isinstance(cbb.b_gf, GF)

        assert cbb.num_vars == cbb.num_rules == len(A[0])
        assert cbb.num_constraints == len(A)

        assert cbb.pivot_rule_idxs == exp_pivot_rule_idxs
        assert cbb.free_rule_idxs == exp_free_rule_idxs
        np.testing.assert_allclose(cbb.B, exp_B)
        np.testing.assert_allclose(
            cbb.row2pivot_column, np.array(exp_row2pivot_column, dtype=int)
        )
        # the two sets are mutually exclusive and their union covers all idxs
        assert len(exp_pivot_rule_idxs & exp_free_rule_idxs) == 0
        assert len(exp_pivot_rule_idxs | exp_free_rule_idxs) == cbb.num_vars

    @pytest.mark.parametrize(
        "A, b, exp_prefix, exp_lb, exp_u, exp_z, exp_s",
        [
            # rule-1 is included
            # the truthtable  is:  0b00101011
            #                          ^      (FP)
            # the groundtruth is:  0b11001111
            #                        ^^   ^   (FN)
            # FP: 1
            ([[1, 0, 0, 0]], [1], (0,), 1 / 8 + 1 * 0.1, mpz("0b11010100"), [1], [1]),
            # rule-1 and rule-2 are included
            # the truthtable  is:  0b00101111
            #                          ^ (FP)
            # the groundtruth is:  0b11001111
            #                        ^^ (FN)
            # FP: 1
            (
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                [1, 1],
                (0, 1),
                1 / 8 + 2 * 0.1,
                mpz("0b11010000"),
                [1, 1],
                [1, 1],
            ),
            # rule-1, rule-2, and rule-3 are included
            # the truthtable  is:  0b10101111
            #                          ^     (FP)
            # the groundtruth is:  0b11001111
            #                         ^      (FN)
            # FP: 1
            (
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                [1, 1, 1],
                (0, 1, 2),
                1 / 8 + 3 * 0.1,
                mpz("0b01010000"),
                [1, 1, 1],
                [1, 1, 1],
            ),
            # all rules are included
            # the truthtable  is:  0b10101111
            #                          ^     (FP)
            # the groundtruth is:  0b11001111
            #                         ^      (FN)
            # FP: 1
            (
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                [1, 1, 1, 1],
                (0, 1, 2, 3),
                1 / 8 + 4 * 0.1,
                mpz("0b01010000"),
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ),
        ],
    )
    def test_reset_queue(self, A, b, exp_prefix, exp_lb, exp_u, exp_z, exp_s):
        A, b, exp_z, exp_s = map(bin_array, [A, b, exp_z, exp_s])
        lmbd = 0.1

        cbb = ConstrainedBranchAndBound(
            self.input_rules, float("inf"), self.input_y, lmbd, reorder_columns=False
        )
        # cbb.reset(A, b)
        cbb.setup_constraint_system(A, b)
        cbb.reset_status()
        cbb.reset_queue()

        assert cbb.status.queue_size() == 1
        item = cbb.status.pop_from_queue()
        assert len(item) == 5
        (prefix, lb, u, z, s) = item
        assert prefix == exp_prefix
        np.testing.assert_allclose(lb, exp_lb)
        assert bin(u) == bin(exp_u)
        np.testing.assert_allclose(z, exp_z)
        np.testing.assert_allclose(s, exp_s)

    @pytest.mark.parametrize(
        "A, b, expected_sols, expected_obj",
        [
            # rule-1 is included
            # the truthtable  is:  0b00101011
            #                          ^      (FP)
            # the groundtruth is:  0b11001111
            #                        ^^   ^   (FN)
            # TP: 4
            # FP: 1
            # FN: 3
            ([[1, 0, 0, 0]], [1], {0}, 4 / 8 + 1 * 0.1),
            # rule-1 and rule-2 are included
            # the truthtable  is:  0b00101111
            #                          ^ (FP)
            # the groundtruth is:  0b11001111
            #                        ^^ (FN)
            # TP: 4
            # FP: 1
            # FN: 2
            ([[1, 0, 0, 0], [0, 1, 0, 0]], [1, 1], {0, 1}, 3 / 8 + 2 * 0.1),
            # rule-1, rule-2, and rule-3 are included
            # the truthtable  is:  0b10101111
            #                          ^     (FP)
            # the groundtruth is:  0b11001111
            #                         ^      (FN)
            # TP: 4
            # FP: 1
            # FN: 1
            (
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                [1, 1, 1],
                {0, 1, 2},
                2 / 8 + 3 * 0.1,
            ),
            # all rules are included
            # the truthtable  is:  0b10101111
            #                          ^     (FP)
            # the groundtruth is:  0b11001111
            #                         ^      (FN)
            # TP: 4
            # FP: 1
            # FN: 1
            (
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                [1, 1, 1, 1],
                {0, 1, 2, 3},
                2 / 8 + 4 * 0.1,
            ),
        ],
    )
    def test__generate_solution_at_root(self, A, b, expected_sols, expected_obj):
        A = bin_array(A)
        b = bin_array(b)

        lmbd = 0.1
        cbb = ConstrainedBranchAndBound(
            self.input_rules, float("inf"), self.input_y, lmbd
        )
        cbb.reset(A, b)

        sol, obj = list(cbb._generate_solution_at_root(return_objective=True))[0]
        assert sol == tuple(expected_sols)
        np.testing.assert_allclose(float(obj), expected_obj)

        # check the solution set and reserve set in solver_status
        assert cbb.status.solution_set == {sol}
        assert cbb.status.reserve_set == {sol}

    def test__do_reorder_columns(self):
        A = bin_array([[1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        b = bin_zeros(4)

        cbb = ConstrainedBranchAndBound(
            self.input_rules, float("inf"), self.input_y, lmbd=0.1, reorder_columns=True
        )
        cbb.setup_constraint_system(A, b)

        np.testing.assert_allclose(cbb.idx_map_new2old, [0, 2, 3, 1])
        assert cbb.idx_map_old2new == {0: 0, 1: 3, 2: 1, 3: 2}
        np.testing.assert_allclose(cbb.pivot_columns, np.arange(2))

        assert list(range(len(self.input_rules))) == [r.id for r in cbb.rules]

        np.testing.assert_allclose(
            cbb.A, [[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        np.testing.assert_allclose(cbb.A, np.array(cbb.A_gf, dtype=int))


class TestBBNonIncremental:
    """test the branch-and-bound in non-incremental setup"""

    @pytest.mark.parametrize(
        "A, b, expected_sols",
        [
            (
                [[1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 1, 0]],
                [1, 0, 1],
                [(0, 2), (1, 3)],
            ),
            (
                [[1, 0, 0, 1]],
                [1],
                [
                    (0,),
                    (0, 1),
                    (0, 2),
                    (3,),
                    (0, 1, 2),
                    (1, 3),
                    (2, 3),
                    (1, 2, 3),
                ],
            ),
            (
                [[1, 0, 0, 1], [0, 1, 0, 1]],
                [1, 0],
                [
                    (0,),
                    (0, 2),
                    (1, 3),
                    (1, 2, 3),
                ],
            ),
            (
                [[1, 1, 0, 0, 1], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]],
                [0, 0, 1],
                [
                    (3,),
                    (0, 1, 3),
                    (0, 3, 4),
                    (1, 3, 4),
                ],
            ),
        ],
    )
    def test_complete_enumeration_with_infinite_ub(self, A, b, expected_sols):
        A, b = map(bin_array, [A, b])
        rand_rules, rand_y = generate_random_rules_and_y(10, A.shape[1], 12345)
        cbb = ConstrainedBranchAndBound(
            rand_rules, float("inf"), rand_y, 0.1, reorder_columns=False
        )
        sols = cbb.bounded_sols(threshold=None, A=A, b=b)
        assert normalize_solutions(sols) == normalize_solutions(expected_sols)

    @pytest.mark.parametrize(
        "ub, expected",
        [
            (  # case 1: all satisfied solutions are returned
                float("inf"),
                {(1,): 0.1, (0, 1, 2): 0.9},
            ),
            # case 2
            (0.5, {(1,): 0.1}),
            # case 3
            (0.01, dict()),
        ],
    )
    def test_varying_ub_case_1(self, rules, y, ub, expected):
        lmbd = 0.1
        cbb = ConstrainedBranchAndBound(rules, ub, y, lmbd, reorder_columns=False)

        # 3 rules
        # 2 constraints
        A = bin_array([[1, 0, 1], [0, 1, 0]])  # 0  # 1
        # rule-2 has to be selected
        # rule-0 and rule-1 is either both selected or both unselected
        b = bin_array([0, 1])
        res_iter = cbb.run(return_objective=True, A=A, b=b)

        sols = list(res_iter)
        actual = solutions_to_dict(sols)
        assert_dict_allclose(actual, expected)

    @pytest.mark.parametrize(
        "ub, expected",
        [
            (float("inf"), {(2,): 0.3, (0,): 0.9}),
            (0.5, {(2,): 0.3}),
            (0.1, dict()),
        ],
    )
    def test_varying_ub_case_2(self, rules, y, ub, expected):
        lmbd = 0.1
        cbb = ConstrainedBranchAndBound(rules, ub, y, lmbd, reorder_columns=False)

        A = bin_array([[1, 0, 1], [0, 1, 0]])  # 1  # 0
        b = bin_array([1, 0])
        res_iter = cbb.run(return_objective=True, A=A, b=b)

        sols = list(res_iter)
        actual = solutions_to_dict(sols)
        assert_dict_allclose(actual, expected)

    @pytest.mark.parametrize(
        "ub, expected",
        [
            (float("inf"), {(0, 1, 2): 0.9}),
            (0.1, dict()),
        ],
    )
    def test_varying_ub_case_3(self, rules, y, ub, expected):
        lmbd = 0.1
        cbb = ConstrainedBranchAndBound(rules, ub, y, lmbd, reorder_columns=False)

        A = bin_array([[1, 0, 1], [1, 1, 0]])
        b = bin_array([0, 0])
        res_iter = cbb.run(return_objective=True, A=A, b=b)
        sols = list(res_iter)
        actual = solutions_to_dict(sols)
        assert_dict_allclose(actual, expected)

    @pytest.mark.parametrize(
        "ub, expected",
        [
            (float("inf"), {(0, 2): 0.8, (1,): 0.1}),
            (0.1, {(1,): 0.1}),
            (0.01, dict()),
        ],
    )
    def test_varying_ub_case_4(self, rules, y, ub, expected):
        lmbd = 0.1
        cbb = ConstrainedBranchAndBound(rules, ub, y, lmbd, reorder_columns=False)

        A = bin_array([[1, 0, 1], [1, 1, 0]])
        b = bin_array([0, 1])
        res_iter = cbb.run(return_objective=True, A=A, b=b)
        sols = list(res_iter)

        actual = solutions_to_dict(sols)
        assert_dict_allclose(actual, expected)

    @pytest.mark.parametrize(
        "thresh, count", [(None, 2), (1, 1), (2, 2), (3, 2)]  # total count is returned
    )
    def test_bounded_count(self, rules, y, thresh, count):
        ub = float("inf")
        lmbd = 0.1
        cbb = ConstrainedBranchAndBound(rules, ub, y, lmbd, reorder_columns=False)

        A = bin_array([[1, 0, 1], [1, 1, 0]])
        b = bin_array([0, 1])

        assert cbb.bounded_count(thresh, A=A, b=b) == count

    @pytest.mark.parametrize(
        "thresh, count", [(None, 2), (1, 1), (2, 2), (3, 2)]  # total count is returned
    )
    def test_bounded_sols(self, rules, y, thresh, count):
        ub = float("inf")
        lmbd = 0.1
        cbb = ConstrainedBranchAndBound(rules, ub, y, lmbd, reorder_columns=False)

        A = bin_array([[1, 0, 1], [1, 1, 0]])
        b = bin_array([0, 1])

        sols = cbb.bounded_sols(thresh, A=A, b=b)
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
    # @pytest.mark.parametrize("ub", [0.801])  # float("inf"),  # , 0.01
    # @pytest.mark.parametrize("rand_seed", [1320602510])
    def test_solution_correctness(
        self, num_rules, num_constraints, lmbd, ub, rand_seed
    ):
        """the output should be the same as ground truth"""
        rand_rules, rand_y = generate_random_rules_and_y(10, num_rules, rand_seed)
        # for r in rand_rules:
        #     print(f'{r.name}: {bin(r.truthtable)}')
        # print(rand_y[::-1].astype(int))
        cbb = ConstrainedBranchAndBound(
            rand_rules, ub, rand_y, lmbd, reorder_columns=False
        )

        # cbb._print_rules_and_y()
        A, b = generate_h_and_alpha(
            num_rules, num_constraints, rand_seed, as_numpy=True
        )
        actual = solutions_to_dict(list(cbb.run(return_objective=True, A=A, b=b)))

        expected = solutions_to_dict(
            list(brute_force_enumeration(rand_rules, rand_y, A, b, ub, lmbd))
        )
        # print(expected)
        # assert set(actual.keys()) == set(expected.keys())
        assert_dict_allclose(actual, expected)

    @pytest.mark.parametrize("num_rules", [10])
    @pytest.mark.parametrize("num_constraints", [2, 4, 8])
    @pytest.mark.parametrize("lmbd", [0.1])
    @pytest.mark.parametrize(
        "ub", [0.501, 0.801, float("inf")]
    )  # float("inf"),  # , 0.01
    @pytest.mark.parametrize("rand_seed", randints(5))
    def test_column_reordering(self, num_rules, num_constraints, lmbd, ub, rand_seed):
        """+the output with column reordering should be the same as without column reordering+"""
        rand_rules, rand_y = generate_random_rules_and_y(10, num_rules, rand_seed)
        cbb_ref = ConstrainedBranchAndBound(
            rand_rules, ub, rand_y, lmbd, reorder_columns=False
        )
        cbb_test = ConstrainedBranchAndBound(
            rand_rules, ub, rand_y, lmbd, reorder_columns=True
        )

        A, b = generate_h_and_alpha(
            num_rules, num_constraints, rand_seed, as_numpy=True
        )
        expected = solutions_to_dict(list(cbb_ref.run(return_objective=True, A=A, b=b)))
        actual = solutions_to_dict(list(cbb_test.run(return_objective=True, A=A, b=b)))

        assert_dict_allclose(actual, expected)

        # remark: we do not put a threshold here because after reordering the columns
        # rule visiting order is changed, thus the return set of prefixes may differ


class TestBBIncremental(UtilityMixin):
    """test the branch-and-bound in incremental setup"""

    @pytest.mark.parametrize("num_rules", [10, 20])
    @pytest.mark.parametrize("num_constraints", [5, 8])
    @pytest.mark.parametrize("ub", [0.501, 0.801])  # float("inf"),  # , 0.01
    @pytest.mark.parametrize("threshold", randints(2, 2, 100))
    @pytest.mark.parametrize("rand_seed", randints(3))
    @pytest.mark.parametrize("reorder_columns", [True, False])
    def test_recording_of_R_and_S_and_update_of_d_last(
        self, num_rules, num_constraints, ub, threshold, rand_seed, reorder_columns
    ):
        """
        check the following:
        1. S should be the same yeilded solutions and S should be a subset of R as well
        2. d_last is updated
        """
        lmbd = 0.1
        rand_rules, rand_y, A, b = self._create_input_data(
            num_rules, num_constraints, rand_seed
        )

        cbb = ConstrainedBranchAndBound(
            rand_rules, ub, rand_y, lmbd, reorder_columns=reorder_columns
        )

        sols = cbb.bounded_sols(threshold, A=A, b=b)
        assert len(sols) == len(cbb.status.solution_set)
        assert set(sols) == cbb.status.solution_set
        assert cbb.status.solution_set.issubset(cbb.status.reserve_set)
        assert (
            isinstance(cbb.status.last_checked_prefix, RuleSet)
            or cbb.status.last_checked_prefix is None
        )

    def test_reset_with_status_given(self):
        """the status should be set and be the same as the previous run"""
        rand_rules, rand_y, A, b = self._create_input_data(
            num_rules=10, num_constraints=5, rand_seed=None
        )
        # reference CBB solves from scratch
        cbb_prev = ConstrainedBranchAndBound(
            rand_rules, float("inf"), rand_y, lmbd=0.1, reorder_columns=False
        )

        cbb_prev.bounded_sols(5, A=A, b=b)
        # push last checked prefix to to align with what is done in reset method
        cbb_prev._push_last_checked_prefix_to_queue()

        cbb_cur = ConstrainedBranchAndBound(
            rand_rules, float("inf"), rand_y, lmbd=0.1, reorder_columns=False
        )
        cbb_cur.reset(A=A, b=b, solver_status=cbb_prev.status)

        assert (
            cbb_cur.status is not cbb_prev.status
        )  # former status is copied from latter
        assert cbb_cur.status == cbb_prev.status

    @pytest.mark.parametrize("num_rules", [10, 20])
    @pytest.mark.parametrize("num_constraints", [5, 8])
    @pytest.mark.parametrize("lmbd", [0.1])
    @pytest.mark.parametrize("ub", [0.501, 0.801])  # float("inf"),  # , 0.01
    @pytest.mark.parametrize("threshold", randints(3, 1, 100))
    @pytest.mark.parametrize("rand_seed", randints(3))
    @pytest.mark.parametrize("reorder_columns", [False, True])
    def test_continuation_from_previous_run(
        self,
        num_rules,
        num_constraints,
        lmbd,
        ub,
        threshold,
        rand_seed,
        reorder_columns,
    ):
        """check if continuing CBB from a previous run gives the expected behaviour"""
        rand_rules, rand_y, A, b = self._create_input_data(
            num_rules, num_constraints, rand_seed
        )
        cbb_full = ConstrainedBranchAndBound(
            rand_rules, ub, rand_y, lmbd, reorder_columns=reorder_columns
        )
        all_sols_expected = set(cbb_full.bounded_sols(threshold=None, A=A, b=b))

        # reference CBB solves from scratch
        cbb_prev = ConstrainedBranchAndBound(
            rand_rules, ub, rand_y, lmbd, reorder_columns=reorder_columns
        )
        sols_prev = cbb_prev.bounded_sols(threshold, A=A, b=b)

        cbb_cur = ConstrainedBranchAndBound(
            rand_rules, ub, rand_y, lmbd, reorder_columns=reorder_columns
        )
        sols_cur = cbb_cur.bounded_sols(
            threshold=None, A=A, b=b, solver_status=cbb_prev.status
        )  # find the remaining

        assert is_disjoint(
            sols_prev, sols_cur
        ), f"shared entries: {set(sols_prev) & set(sols_cur)}"  # solutions from the two runs should be disjoint

        all_sols_actual = set(sols_cur) | set(sols_prev)
        assert (
            all_sols_actual == cbb_cur.status.solution_set
        )  # solution set of current run should accumulate over previous runs
        assert all_sols_actual == all_sols_expected
        assert cbb_cur.status.solution_set.issubset(
            cbb_cur.status.reserve_set
        )  # reserve set is always a superset of solution set

    def _extract_sols(self, sol_obj_tuples):
        return list(map(itemgetter(0), sol_obj_tuples))

    def _extract_objs(self, sol_obj_tuples):
        return list(map(itemgetter(1), sol_obj_tuples))

    @pytest.mark.parametrize("num_rules", [20])
    @pytest.mark.parametrize("num_constraints", [2, 4, 6])
    @pytest.mark.parametrize("lmbd", [0.1])
    @pytest.mark.parametrize("ub", [0.801])  # float("inf"),  # , 0.01
    @pytest.mark.parametrize("rand_seed", randints(5))
    # @pytest.mark.parametrize("num_rules", [10])
    # @pytest.mark.parametrize("num_constraints", [2])
    # @pytest.mark.parametrize("lmbd", [0.1])
    # @pytest.mark.parametrize("ub", [0.801])  # float("inf"),  # , 0.01
    # @pytest.mark.parametrize("rand_seed", [1320602510])
    @pytest.mark.parametrize("num_continuations", [3, 4, 5])
    @pytest.mark.parametrize("reorder_columns", [True, False])
    def test_continuation_search_the_general_case(
        self,
        num_rules,
        num_constraints,
        lmbd,
        ub,
        rand_seed,
        num_continuations,
        reorder_columns,
    ):
        """the output should be the same as ground truth for more than 1 continuations

        also check the objectives are calculated correctly
        """
        rand_rules, rand_y = generate_random_rules_and_y(10, num_rules, rand_seed)

        # reference CBB solves from scratch
        cbb_ref = ConstrainedBranchAndBound(
            rand_rules, ub, rand_y, lmbd, reorder_columns=reorder_columns
        )

        A, b = generate_h_and_alpha(
            num_rules, num_constraints, rand_seed, as_numpy=True
        )

        sols_with_obj_expected = list(cbb_ref.run(return_objective=True, A=A, b=b))

        num_sols = len(sols_with_obj_expected)
        threshold_per_run = int(math.ceil(num_sols / num_continuations))

        # test CBB solves the problem in "segments"
        # each solver continues from the previous run
        cbb_cur = ConstrainedBranchAndBound(
            rand_rules, ub, rand_y, lmbd, reorder_columns=reorder_columns
        )
        sols_with_obj_actual = cbb_cur.bounded_sols(
            threshold_per_run, return_objective=True, A=A, b=b
        )

        for i in range(num_continuations):
            cbb_next = ConstrainedBranchAndBound(
                rand_rules, ub, rand_y, lmbd, reorder_columns=reorder_columns
            )
            sols_in_this_run = cbb_next.bounded_sols(
                threshold_per_run,
                return_objective=True,
                A=A,
                b=b,
                solver_status=cbb_cur.status,
            )
            # expectation 1:
            # the solutions from each continutation search should be disjoint from the others
            assert is_disjoint(
                self._extract_sols(sols_with_obj_actual),
                self._extract_sols(sols_in_this_run),
            )
            sols_with_obj_actual += sols_in_this_run
            cbb_cur = cbb_next

        # expectation 2: the solutions from continuation search should be the same as solving from scratch
        assert set(self._extract_sols(sols_with_obj_actual)) == set(
            self._extract_sols(sols_with_obj_expected)
        )
        assert set(self._extract_objs(sols_with_obj_actual)) == set(
            self._extract_objs(sols_with_obj_expected)
        )
        assert len(sols_with_obj_actual) == len(sols_with_obj_expected)

        assert (
            set(self._extract_sols(sols_with_obj_expected))
            == cbb_cur.status.solution_set
        )
