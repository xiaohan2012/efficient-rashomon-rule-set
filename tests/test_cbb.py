import numpy as np
import pytest
from gmpy2 import mpz

from bds.cbb import (
    ConstrainedBranchAndBound,
    build_boundary_table,
    count_added_pivots,
    ensure_minimal_no_violation,
    ensure_satisfiability,
)
from bds.gf2 import GF, extended_rref
from bds.random_hash import generate_h_and_alpha
from bds.rule import Rule
from bds.utils import (
    bin_array,
    bin_ones,
    bin_zeros,
    get_indices_and_indptr,
    mpz_set_bits,
    randints,
    solutions_to_dict,
)

from .fixtures import rules, y
from .utils import (
    assert_dict_allclose,
    brute_force_enumeration,
    calculate_obj,
    generate_random_rules_and_y,
    normalize_solutions,
)


@pytest.mark.parametrize(
    "A, expected",
    [
        # case 1: full rank, all -1
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [-1, -1, -1]),
        # case 2: still full rank, all -1
        ([[1, 0, 1], [0, 1, 1], [0, 0, 1]], [-1, -1, -1]),
        # case 3: rank = 2
        # becomes [[1, 0, 1], [0, 1, 1], [0, 0, 0]]
        ([[1, 1, 0], [0, 1, 1], [0, 1, 1]], [2, 2]),
        # case 4: rank = 1
        # becomes [[1, 1, 0], [0, 0, 0], [0, 0, 0]]
        ([[1, 1, 0], [1, 1, 0], [1, 1, 0]], [1]),
        ([[1, 1, 0, 0, 1], [0, 0, 1, 0, 0]], [4, -1]),
        ([[1, 0, 1], [0, 1, 0]], [2, -1]),
        ([[1, 0, 1, 1, 1, 1, 1, 1, 1, 1]], [9]),
    ],
)
def test_build_boundary_table(A, expected):
    A_rref, _, rank, pivot_columns = extended_rref(
        GF(A), GF(np.ones(len(A), dtype=int))
    )
    # print("A_rref: {}".format(A_rref))
    # print("pivot_columns: {}".format(pivot_columns))
    actual = build_boundary_table(bin_array(A_rref), rank, pivot_columns)
    np.testing.assert_allclose(expected, actual)


class TestEnsureMinimalNoViolation:
    @pytest.mark.parametrize(
        "test_name, A, b, j, z, s, exp_rules, exp_zp, exp_sp",
        [
            # the root case -- no rules are added
            (
                "root-case",
                [[1, 0, 1, 0], [0, 1, 0, 1]],
                [1, 0],
                -1,
                [0, 0],
                [0, 0],
                [],
                [0, 0],
                [0, 0],
            ),
            # the root case -- no rules are added
            (
                "root-case",
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                ],
                [1, 1, 1],  # b
                -1,
                [0, 0, 0],  # z
                [0, 0, 0],  # s
                [0, 1, 2],  # all pivots are added
                [1, 1, 1],  # and the sign is flipped only once for each const,
                [1, 1, 1],
            ),
            (
                "t2",
                [
                    [1, 0, 0, 1],  # 3 is bordering
                    [0, 1, 0, 1],  # 3 is bordering
                    [0, 0, 1, 1],  # 3 is bordering, 2 is added because t[2] = 0
                ],
                [1, 1, 0],  # b
                3,
                [0, 0, 0],  # z
                [0, 0, 0],  # s
                [2],
                [1, 1, 0],  # z[0] and z[1] flipped once and z[2] flipped twice,
                [1, 1, 1],
            ),
            (
                "t3",
                [
                    [1, 0, 0, 1],
                    [0, 1, 0, 0],
                ],
                [0, 1],
                3,
                [0, 1],  # z
                [0, 1],  # s, the 2nd constraint is satisfied at the root
                [0],
                [0, 1],
                [1, 1],
            ),
            # the added rule is either at the boundary or over the boundary
            (
                "t4",
                [
                    [1, 0, 0, 1],
                    [0, 1, 1, 0],
                ],
                [0, 1],
                3,
                [0, 0],  # z
                [0, 0],  # s
                [0, 1],  # added rules
                [0, 1],
                [1, 1],
            ),
            # the added rule is either at the boundary or over the boundary
            (
                "t5",
                [
                    [1, 0, 0, 1],
                    [0, 1, 1, 0],
                ],
                [0, 0],
                3,
                [0, 0],  # z
                [0, 0],  # s
                [0],  # added rules
                [0, 0],
                [1, 1],
            ),
            # no const is determined
            # the added rule is both interior and irrelevant
            (
                "t6",
                [
                    [1, 0, 1, 0],  # 1 is irrelevant and interior
                    [0, 0, 0, 1],  # 1 is irrelevant and interior
                ],
                [0, 1],
                1,
                [0, 1],  # z
                [0, 1],  # s
                [],
                [0, 1],
                [0, 1],
            ),
            # no const is determined
            # the added rule is interior and relevant
            (
                "t7",
                [[1, 1, 0, 0, 1], [0, 0, 1, 0, 0]],  # 1 is interior and relevant
                [0, 0],
                1,
                [0, 0],  # z
                [0, 1],  # s
                [],
                [1, 0],
                [0, 1],
            ),
        ],
    )
    def test_basic(self, test_name, A, b, j, z, s, exp_rules, exp_zp, exp_sp):
        A, b, rank, pivot_columns = extended_rref(
            GF(np.array(A, dtype=int)), GF(np.array(b, dtype=int)), verbose=False
        )
        A, b = map(bin_array, (A, b))

        B = build_boundary_table(A, rank, pivot_columns)
        m, n = A.shape

        z = bin_array(z)
        s = bin_array(s)
        actual_rules, actual_zp, actual_sp = ensure_minimal_no_violation(
            j, z, s, A, b, B, pivot_columns
        )
        np.testing.assert_allclose(actual_rules, np.array(exp_rules, dtype=int))
        np.testing.assert_allclose(actual_zp, bin_array(exp_zp))
        np.testing.assert_allclose(actual_sp, bin_array(exp_sp))

    @pytest.mark.parametrize(
        "name, A, b, j1, rules1, zp1, sp1, j2, rules2, zp2, sp2",
        [
            (
                "t1",
                # the case that the added rules are irrelevant
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                [1, 1],
                # the root case
                -1,
                [0, 1],
                [1, 1],
                [1, 1],
                # add rule 2, whcih does not change anything
                2,
                [],
                [1, 1],
                [1, 1],
            ),
            (
                "t2.1",
                [[1, 0, 1, 0], [0, 1, 0, 1]],
                [1, 0],
                # the root case, no rules are added
                -1,
                [],
                [0, 0],
                [0, 0],
                # add 3, which determines both constraints
                # rule 1 and 2 are added
                3,
                [0, 1],
                [1, 0],
                [1, 1],
            ),
            (
                "t2.2",
                [[1, 0, 1, 0], [0, 1, 0, 1]],
                [0, 0],
                # the root case, no rules are added
                -1,
                [],
                [0, 0],
                [0, 0],
                # add 3, which determines both constraints
                # rule 1 and 2 are added
                3,
                [1],
                [0, 0],
                [1, 1],
            ),
            (
                "t3.1",
                [[1, 0, 1, 0], [0, 1, 0, 1]],
                [1, 0],
                # the root case, no rules are added
                -1,
                [],
                [0, 0],
                [0, 0],
                # add 2, which determines C1
                # rule 1 is added
                2,
                [],
                [1, 0],
                [1, 0],
            ),
            (
                "t3.2",
                [[1, 0, 1, 0], [0, 1, 0, 1]],
                [0, 0],
                # the root case, no rules are added
                -1,
                [],
                [0, 0],
                [0, 0],
                # add 2, which determines C1
                # rule 1 is added
                2,
                [0],
                [0, 0],
                [1, 0],
            ),
        ],
    )
    def test_multiple_calls(
        self, name, A, b, j1, rules1, zp1, sp1, j2, rules2, zp2, sp2
    ):
        """test calling ensure_no_violation multiple times"""

        A, b, rank, pivot_columns = extended_rref(
            GF(np.array(A, dtype=int)), GF(np.array(b, dtype=int)), verbose=False
        )
        A, t = map(bin_array, (A, b))

        B = build_boundary_table(A, rank, pivot_columns)
        m, n = A.shape

        z = bin_zeros(m)
        s = bin_zeros(m)
        actual_rules1, actual_zp1, actual_sp1 = ensure_minimal_no_violation(
            j1, z, s, A, b, B, pivot_columns
        )

        np.testing.assert_allclose(actual_rules1, rules1)
        np.testing.assert_allclose(actual_zp1, zp1)
        np.testing.assert_allclose(actual_sp1, sp1)

        actual_rules2, actual_zp2, actual_sp2 = ensure_minimal_no_violation(
            j2, actual_zp1, actual_sp1, A, b, B, pivot_columns
        )

        np.testing.assert_allclose(actual_rules2, rules2)
        np.testing.assert_allclose(actual_zp2, zp2)
        np.testing.assert_allclose(actual_sp2, sp2)  # all constraints are satisfied


class TestCountAddedPivots:
    @pytest.mark.parametrize(
        "j, A, b, z, exp_count",
        [
            (1, [[1, 0, 0]], [1], [0], 1),
            (1, [[1, 0, 0]], [1], [1], 0),
            (1, [[1, 1, 0]], [0], [0], 1),
            (1, [[1, 1, 0]], [1], [0], 0),
        ],
    )
    def test_basic(self, j, A, b, z, exp_count):
        A, b, z = map(bin_array, [A, b, z])
        assert count_added_pivots(j, A, b, z) == exp_count


class TestEnsureSatisfiability:
    @pytest.mark.parametrize(
        "name, A, b, z, s, j, expected_rules",
        [
            (
                # the root case
                "root-case",
                [[1, 0, 1, 1], [0, 1, 0, 0]],
                [1, 1],  # b
                [0, 0],  # z
                [0, 0],  # s
                -1,
                [0, 1],
            ),
            (
                # the root case
                "root-case",
                [[1, 0, 1, 1], [0, 1, 0, 0]],
                [1, 0],  # b
                [0, 0],  # z
                [0, 0],  # s
                -1,
                [0],
            ),
            (
                "root-case-affected-by-rref",
                [[1, 0, 1, 1], [0, 0, 1, 0]],
                [0, 1],  # b
                [0, 0],  # z
                [0, 0],  # s
                -1,
                [0, 2],  # 0 is added because t becomes [1, 1] due to rref
            ),
            (
                "t1.1",
                [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]],  # A
                [1, 1, 0],  # b
                [0, 0, 0],  # z
                [0, 0, 0],  # s
                3,
                [2],
            ),
            (
                "t1.2",
                [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]],
                [1, 1, 0],  # b
                [1, 1, 0],  # z
                [0, 0, 0],  # s
                3,
                [0, 1, 2],
            ),
            (
                "t1.3",
                [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]],
                [1, 1, 0],  # b
                [1, 1, 0],  # z
                [0, 1, 1],  # s
                3,
                [0],
            ),
            (
                "t1.4",
                [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]],
                [1, 1, 0],  # b
                [1, 0, 1],  # z
                [0, 0, 0],  # s
                3,
                [0],
            ),
            (
                "t1.5",
                [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]],
                [1, 1, 0],  # b
                [1, 0, 1],  # z
                [1, 0, 0],  # s
                3,
                [],
            ),
            (
                "t2.1",
                [[1, 0, 1, 1], [0, 1, 0, 0]],
                [0, 1],  # b
                [0, 0],  # z
                [0, 0],  # s
                2,
                [0, 1],  # added rules
            ),
            (
                "t2.2",
                [[1, 0, 1, 1], [0, 1, 0, 0]],
                [0, 1],  # b
                [0, 1],  # z
                [0, 0],  # s
                2,
                [0],  # added rules
            ),
            (
                "t2.3",
                [[1, 0, 1, 1], [0, 1, 0, 0]],
                [0, 1],  # b
                [0, 1],  # z
                [1, 1],  # s
                2,
                [],  # added rules
            ),
            (
                "t3",
                [[1, 0, 1, 1], [0, 0, 0, 0]],
                [0, 0],
                [0, 0],
                [0, 0],
                3,
                [0],
            ),
        ],
    )
    def test_basic(self, name, A, b, z, s, j, expected_rules):
        A = bin_array(A)
        b = bin_array(b)

        A, b, rank, row2pivot_column = extended_rref(
            GF(A.astype(int)), GF(b.astype(int))
        )

        m, n = A.shape

        z = bin_array(z)
        s = bin_array(s)
        actual_rules = ensure_satisfiability(
            j, rank, z, s, A, b, row2pivot_column  # A_indices, A_indptr,
        )
        np.testing.assert_allclose(actual_rules, np.array(expected_rules, dtype=int))


class TestConstrainedBranchAndBoundMethods:
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
    def test_generate_solution_at_root(self, A, b, expected_sols, expected_obj):
        A = bin_array(A)
        b = bin_array(b)

        lmbd = 0.1
        cbb = ConstrainedBranchAndBound(
            self.input_rules, float("inf"), self.input_y, lmbd
        )
        cbb.reset(A, b)

        sol, obj = list(cbb.generate_solution_at_root(return_objective=True))[0]
        assert sol == tuple(expected_sols)
        np.testing.assert_allclose(float(obj), expected_obj)

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
        cbb = ConstrainedBranchAndBound(rand_rules, float("inf"), rand_y, 0.1, reorder_columns=False)

        cbb.setup_constraint_system(bin_array(A), bin_array(b))

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
        cbb.setup_constraint_system(A, b)
        cbb.reset_queue()
        assert cbb.queue.size == 1
        item = cbb.queue.pop()
        assert len(item) == 5
        (prefix, lb, u, z, s) = item
        assert prefix == exp_prefix
        np.testing.assert_allclose(lb, exp_lb)
        assert bin(u) == bin(exp_u)
        np.testing.assert_allclose(z, exp_z)
        np.testing.assert_allclose(s, exp_s)


class TestConstrainedBranchAndBoundEnd2End:
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
        cbb = ConstrainedBranchAndBound(rand_rules, float("inf"), rand_y, 0.1, reorder_columns=False)
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
        cbb = ConstrainedBranchAndBound(rand_rules, ub, rand_y, lmbd, reorder_columns=False)

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
    @pytest.mark.parametrize("ub", [0.801, 0.501, 0.001])  # float("inf"),  # , 0.01
    @pytest.mark.parametrize("rand_seed", randints(5))
    def test_column_reodering(self, num_rules, num_constraints, lmbd, ub, rand_seed):
        """the output with column reordering should be the same as without column reordering"""
        rand_rules, rand_y = generate_random_rules_and_y(10, num_rules, rand_seed)
        cbb_ref = ConstrainedBranchAndBound(rand_rules, ub, rand_y, lmbd, reorder_columns=False)
        cbb_test = ConstrainedBranchAndBound(rand_rules, ub, rand_y, lmbd, reorder_columns=True)

        A, b = generate_h_and_alpha(
            num_rules, num_constraints, rand_seed, as_numpy=True
        )
        expected = solutions_to_dict(list(cbb_ref.run(return_objective=True, A=A, b=b)))
        actual = solutions_to_dict(list(cbb_test.run(return_objective=True, A=A, b=b)))

        assert_dict_allclose(actual, expected)
