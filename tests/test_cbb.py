import gmpy2 as gmp
import numpy as np
import pytest
from gmpy2 import mpfr, mpz

from bds.cbb import (
    ConstrainedBranchAndBound,
    build_boundary_table,
    count_added_pivots,
    ensure_minimal_no_violation,
    ensure_satisfiability,
)
from bds.gf2 import GF, extended_rref
from bds.random_hash import generate_h_and_alpha
from bds.rule import Rule, lor_of_truthtable
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


class TestEnsureSatisfiability:
    @pytest.mark.skip(
        "skipped because numba (nonpython) does not support raise ValueError"
    )
    @pytest.mark.parametrize(
        "A, t, j",
        [
            ([[1, 0, 0, 0]], [0], 1),
            ([[1, 0, 1, 1], [0, 1, 0, 0]], [0, 1], 2),
            ([[1, 0, 1, 1], [0, 1, 0, 0]], [0, 1], 1),
        ],
    )
    def test_trying_to_set_pivot_variable(self, A, t, j):
        """adding pivot rule should not be allowed"""
        A = bin_array(A).astype(int)
        t = bin_array(t).astype(int)

        A, t, rank, row2pivot_column = extended_rref(A, t)
        A_indices, A_indptr = get_indices_and_indptr(A)
        max_nz_idx_array = get_max_nz_idx_per_row(A)
        m, n = A.shape

        j = 1

        with pytest.raises(
            ValueError, match=f"cannot set pivot variable of column {j-1}"
        ):
            ensure_satisfiability(
                j,
                rank,
                bin_zeros(m),
                t,
                A_indices,
                A_indptr,
                max_nz_idx_array,
                row2pivot_column,
            )

    @pytest.mark.parametrize(
        "name, A, t, j, expected_rules",
        [
            (
                # rule 4 added
                # x1 + x4 = 1 -> x1 = 0
                # x2 + x4 = 1 -> x2 = 0
                # x3 + x4 = 0 -> x3 = 1
                "case-1",
                [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]],
                [1, 1, 0],
                4,
                [3],
            ),
            (
                # rule 3 added
                # x1 + x3 + x4 = 0 -> x1 = 1
                # x2 + x3 = 1 -> x2 = 0
                "case-2",
                [[1, 0, 1, 1], [0, 1, 1, 0]],
                [0, 1],
                3,
                [1],
            ),
            (
                # j = 3
                # x1 + x3 + x4 = 0 -> x1 = 1
                # x2 = 1
                "case-3",
                [[1, 0, 1, 1], [0, 1, 0, 0]],
                [0, 1],
                3,
                [1, 2],
            ),
            (
                # j = 4
                # x1 + x3 + x4 = 0 -> x1 = 1
                # x2 = 1
                "case-4",
                [[1, 0, 1, 1], [0, 1, 0, 0]],
                [0, 1],
                4,
                [1, 2],
            ),
            # (
            #     # adding rule 0 (the default rule)
            #     # x1 = x2 = 1
            #     "case-5",
            #     [[1, 0, 1, 1], [0, 1, 0, 0]],
            #     [1, 1],
            #     0,
            #     [1, 2],
            # ),
            # (
            #     # j = 0 (adding the default rule)
            #     # x1 + x4 = 1 (due to rref) -> x1 = 1
            #     # x3 = 1
            #     "case-6",
            #     [[1, 0, 1, 1], [0, 0, 1, 0]],
            #     [0, 1],
            #     0,
            #     [1, 3],  # 1 is added because t becomes [1, 1] due to rref
            # ),
            # (
            #     # adding default rule
            #     # x1 + x3 + x4 = 0
            #     # x3 = 1
            #     "case-7",
            #     [[1, 0, 1, 1], [0, 0, 1, 0]],
            #     [1, 1],
            #     0,
            #     [3],
            # ),
            (
                # adding 4
                # x1 + x3 + x4 = 0 -> x1 = 1
                "case-5",
                [[1, 0, 1, 1], [0, 0, 0, 0]],
                [0, 0],
                4,
                [1],
            ),
        ],
    )
    def test_basic(self, name, A, t, j, expected_rules):
        """j is not the maximum index for the 1st constraint"""
        A = bin_array(A)
        t = bin_array(t)

        A, t, rank, row2pivot_column = extended_rref(
            GF(A.astype(int)), GF(t.astype(int))
        )

        A_indices, A_indptr = get_indices_and_indptr(A)
        max_nz_idx_array = get_max_nz_idx_per_row(A)
        m, n = A.shape

        z = bin_zeros(m)
        rules = ensure_satisfiability(
            j, rank, z, t, A, max_nz_idx_array, row2pivot_column  # A_indices, A_indptr,
        )
        np.testing.assert_allclose(rules, np.array(expected_rules, dtype=int))

        # take a free ride and test count_added_pivots as well
        assert count_added_pivots(j, A, t, z) == len(rules)


class TestConstrainedBranchAndBoundMethods:
    @pytest.mark.parametrize(
        "A, t, expected_sols, expected_obj",
        [
            # rule-1 is included
            # the truthtable  is:  0b00101011
            #                          ^      (FP)
            # the groundtruth is:  0b11001111
            #                        ^^   ^   (FN)
            # TP: 4
            # FP: 1
            # FN: 3
            ([[1, 0, 0, 0]], [1], {0, 1}, 4 / 8 + 1 * 0.1),
            # rule-1 and rule-2 are included
            # the truthtable  is:  0b00101111
            #                          ^ (FP)
            # the groundtruth is:  0b11001111
            #                        ^^ (FN)
            # TP: 4
            # FP: 1
            # FN: 2
            ([[1, 0, 0, 0], [0, 1, 0, 0]], [1, 1], {0, 1, 2}, 3 / 8 + 2 * 0.1),
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
                {0, 1, 2, 3},
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
                {0, 1, 2, 3, 4},
                2 / 8 + 4 * 0.1,
            ),
        ],
    )
    def test_generate_solution_at_root(self, A, t, expected_sols, expected_obj):
        A = bin_array(A)
        t = bin_array(t)

        # we have 8 points
        rules = [
            Rule(1, "rule-1", 1, mpz("0b00101011")),
            Rule(2, "rule-2", 1, mpz("0b00001101")),
            Rule(3, "rule-3", 1, mpz("0b10001011")),
            Rule(4, "rule-4", 1, mpz()),
        ]
        y = bin_array([1, 1, 1, 1, 0, 0, 1, 1])
        lmbd = 0.1

        cbb = ConstrainedBranchAndBound(rules, float("inf"), y, lmbd)
        cbb.reset(A, t)

        sol, obj = list(cbb.generate_solution_at_root(return_objective=True))[0]
        assert sol == expected_sols
        np.testing.assert_allclose(float(obj), expected_obj)

    @pytest.mark.parametrize(
        "A, t, exp_pivot_rule_idxs, exp_free_rule_idxs, exp_row2pivot_column, exp_border_rule_idxs",
        [
            ([[1, 0, 1], [0, 1, 0]], [0, 1], {1, 2}, {3}, [0, 1], {2, 3}),
            ([[1, 0, 1], [0, 0, 1]], [0, 1], {1, 3}, {2}, [0, 2], {1, 3}),
            ([[1, 0, 1]], [0], {1}, {2, 3}, [0], {3}),
            ([[1, 0, 0], [1, 0, 0], [1, 0, 0]], [0, 0, 0], {1}, {2, 3}, [0], {1}),
        ],
    )
    def test_setup_constraint_system(
        self,
        A,
        t,
        exp_pivot_rule_idxs,
        exp_free_rule_idxs,
        exp_row2pivot_column,
        exp_border_rule_idxs,
    ):
        rules, y = generate_random_rules_and_y(10, 3, 12345)
        cbb = ConstrainedBranchAndBound(rules, float("inf"), y, 0.1)

        cbb.setup_constraint_system(bin_array(A), bin_array(t))

        assert cbb.num_vars == 3

        assert cbb.pivot_rule_idxs == exp_pivot_rule_idxs
        assert cbb.free_rule_idxs == exp_free_rule_idxs
        assert cbb.border_rule_idxs == exp_border_rule_idxs
        np.testing.assert_allclose(
            cbb.row2pivot_column, np.array(exp_row2pivot_column, dtype=int)
        )
        # the two sets are mutually exclusive and their union covers all idxs
        assert len(exp_pivot_rule_idxs & exp_free_rule_idxs) == 0
        assert len(exp_pivot_rule_idxs | exp_free_rule_idxs) == 3

    def test__create_new_node_and_add_to_tree(self):
        rules, y = generate_random_rules_and_y(10, 5, 12345)
        cbb = ConstrainedBranchAndBound(rules, float("inf"), y, 0.1)
        cbb.reset_tree()
        assert cbb.tree.num_nodes == 1

        child = cbb._create_new_node_and_add_to_tree(
            rules[2],
            lb=mpfr(),
            obj=mpfr(),
            captured=mpz(),
            parent_node=cbb.tree.root,
            pivot_rule_idxs_to_add=[1, 2],  # add rule-1 and rule-2 as pivot
        )
        assert cbb.tree.num_nodes == 2  # tree is updated
        assert child.pivot_rule_ids == [1, 2]

        grandchild = cbb._create_new_node_and_add_to_tree(
            rules[4],
            lb=mpfr(),
            obj=mpfr(),
            captured=mpz(),
            parent_node=child,
            pivot_rule_idxs_to_add=[3],  # add rule-3 as pivot
        )
        assert cbb.tree.num_nodes == 3  # tree is updated
        grandchild.pivot_rule_ids == [4]

        # depth should be correct
        # parent should be correct
        assert child.depth == 1
        assert child.parent == cbb.tree.root

        assert grandchild.depth == 2
        assert grandchild.parent == child

        # add an already-added node just return the added node
        grandchild_same = cbb._create_new_node_and_add_to_tree(
            rules[4], lb=mpfr(), obj=mpfr(), captured=mpz(), parent_node=child
        )
        assert grandchild_same == grandchild


class TestConstrainedBranchAndBoundEnd2End:
    @pytest.mark.parametrize(
        "A, t, expected_sols",
        [
            (
                [[1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 1, 0]],
                [1, 0, 1],
                [{0, 1, 3}, {0, 2, 4}],
            ),
            (
                [[1, 0, 0, 1]],
                [1],
                [
                    {0, 1},
                    {0, 1, 2},
                    {0, 1, 3},
                    {0, 4},
                    {0, 1, 2, 3},
                    {0, 2, 4},
                    {0, 3, 4},
                    {0, 2, 3, 4},
                ],
            ),
            (
                [[1, 0, 0, 1], [0, 1, 0, 1]],
                [1, 0],
                [
                    {0, 1},
                    {0, 1, 3},
                    {0, 2, 4},
                    {0, 2, 3, 4},
                ],
            ),
        ],
    )
    def test_complete_enumeration_with_infinite_ub(self, A, t, expected_sols):
        A, t = bin_array(A), bin_array(t)
        rand_rules, rand_y = generate_random_rules_and_y(10, A.shape[1], 12345)
        cbb = ConstrainedBranchAndBound(rand_rules, float("inf"), rand_y, 0.1)
        sols = cbb.bounded_sols(threshold=None, A=A, t=t)
        assert set(map(tuple, sols)) == set(map(tuple, expected_sols))

    @pytest.mark.skip(
        "skipped because if cbb is correct, testing cbb_v2 against cbb (shown above) is enough"
    )
    @pytest.mark.parametrize("num_rules", [10])
    @pytest.mark.parametrize("num_constraints", [8])
    @pytest.mark.parametrize("lmbd", [0.1])
    @pytest.mark.parametrize("ub", [1.0])  # float("inf"),  # , 0.01
    @pytest.mark.parametrize("rand_seed", [1859619716])
    def test_corretness(self, num_rules, num_constraints, lmbd, ub, rand_seed):
        """the output should be the same as ground truth"""
        rand_rules, rand_y = generate_random_rules_and_y(10, num_rules, rand_seed)

        cbb = ConstrainedBranchAndBound(rand_rules, ub, rand_y, lmbd)

        A, t = generate_h_and_alpha(
            num_rules, num_constraints, rand_seed, as_numpy=True
        )
        actual = solutions_to_dict(list(cbb.run(return_objective=True, A=A, t=t)))

        expected = solutions_to_dict(
            list(brute_force_enumeration(rand_rules, rand_y, cbb.A, cbb.t, ub, lmbd))
        )

        # print("A:\n {}".format(cbb.A.astype(int)))
        # print("t:\n {}".format(cbb.t.astype(int)))

        # print("actual: {}".format(actual))
        # print("expected: {}".format(expected))
        # print('obj((0, 2, 6, 10)):', calculate_obj(cbb.rules, cbb.y_np, cbb.y_mpz, (0, 2, 6, 10), lmbd))
        assert set(actual.keys()) == set(expected.keys())
        assert_dict_allclose(actual, expected)
        # raise
