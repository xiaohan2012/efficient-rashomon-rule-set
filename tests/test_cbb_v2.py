import gmpy2 as gmp
import numpy as np
import pytest
from gmpy2 import mpfr, mpz

from bds.cbb import ConstrainedBranchAndBoundNaive
from bds.cbb_v2 import (
    ConstrainedBranchAndBound,
    count_added_pivots,
    ensure_no_violation,
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
    get_max_nz_idx_per_row,
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


class TestEnsureNoViolation:
    @pytest.mark.skip(
        "skipped because numba (nonpython) does not support raise ValueError"
    )
    def test_trying_to_set_pivot_variable(self):
        A = np.array([[1, 0, 0, 0]], dtype=bool)
        t = np.array([0], dtype=bool)

        A_indices, A_indptr = get_indices_and_indptr(A)
        max_nz_idx_array = get_max_nz_idx_per_row(A)
        row2pivot_column = np.array([0], dtype=int)
        m, n = A.shape

        j = 1
        # adding rule-1 should not be allowed
        # because rule-1 is a pivot varable
        z = bin_zeros(m)
        with pytest.raises(ValueError, match="cannot set pivot variable of column 0"):
            rules, zp = ensure_no_violation(
                j, z, t, A_indices, A_indptr, max_nz_idx_array, row2pivot_column
            )

    @pytest.mark.parametrize(
        "test_name, A, t, j, z, exp_rules, exp_zp, exp_sp",
        [
            # j=4
            (
                "t1",
                [
                    [1, 0, 0, 1],  # 4 is bordering
                    [0, 1, 0, 1],  # 4 is bordering
                    [0, 0, 1, 1],  # 4 is bordering, 3 is added because t[2] = 0
                ],
                [1, 1, 0],
                4,
                [0, 0, 0],  # z
                [3],
                [1, 1, 0],  # z[0] and z[1] flipped once and z[2] flipped twice,
                [1, 1, 1],
            ),
            # all const are affected but j=4 is not bordering rule for any
            (
                "t2",
                [
                    [1, 0, 0, 0],  # 4 is not relevant but exceeds bordering index
                    [0, 1, 0, 0],  # 4 is not relevant but exceeds bordering index
                    [0, 0, 1, 0],  # 4 is not relevant but exceeds bordering index
                ],
                [1, 1, 1],
                4,
                [0, 0, 0],  # z
                [1, 2, 3],  # all are added
                [1, 1, 1],  # and the sign is flipped only once for each const,
                [1, 1, 1],
            ),
            #  all consts are affected but j=4 is bordering for one but not for the other
            (
                "t3",
                [
                    [1, 0, 0, 1],  # 4 is bordering
                    [0, 1, 0, 0],  # 4 is irrelevant but exceeds the bordering index
                ],
                [0, 1],
                4,
                [0, 0],  # z
                [1, 2],
                [0, 1],
                [1, 1],
            ),
            # only C1 is affected
            (
                "t4",
                [
                    [1, 0, 1, 0],  # 3 is bordering, add 1 because t[0] = 0
                    [0, 0, 0, 1],  # 3 is irrelevant and is interior, do not add 4
                ],
                [0, 1],
                3,
                [0, 0],  # z
                [1],
                [0, 0],
                [1, 0],
            ),
            # no const is determined
            # the added rule is both interior and irrelevant
            (
                "t5",
                [
                    [1, 0, 1, 0],  # 2 is irrelevant and interior
                    [0, 0, 0, 1],  # 2 is irrelevant and interior
                ],
                [0, 1],
                2,
                [0, 0],  # z
                [],
                [0, 0],
                [0, 0],
            ),
            # no const is determined
            # the added rule is interior and relevant
            (
                "t6",
                [[1, 1, 0, 0, 1], [0, 0, 1, 0, 0]],  # 2 is interior and relevant
                [0, 0],
                2,
                [0, 0],  # z
                [],
                [1, 0],
                [0, 0],
            ),
            # adding 3, which is exterior and irelevant
            # however, no pivot rules are added because z = t already
            (
                "t7",
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                [1, 1],
                3,
                [1, 1],  # z
                [],
                [1, 1],
                [1, 1],
            ),
        ],
    )
    def test_basic(self, test_name, A, t, j, z, exp_rules, exp_zp, exp_sp):
        A, t, _, pivot_columns = extended_rref(
            GF(np.array(A, dtype=int)), GF(np.array(t, dtype=int)), verbose=False
        )
        A, t = map(bin_array, (A, t))
        # A_indices, A_indptr = get_indices_and_indptr(A)
        max_nz_idx_array = get_max_nz_idx_per_row(A)
        m, n = A.shape

        z = bin_array(z)
        s = bin_zeros(m)
        actual_rules, actual_zp, actual_sp = ensure_no_violation(
            j, z, s, t, A, max_nz_idx_array, pivot_columns
        )
        np.testing.assert_allclose(actual_rules, np.array(exp_rules, dtype=int))
        np.testing.assert_allclose(actual_zp, bin_array(exp_zp))
        np.testing.assert_allclose(actual_sp, bin_array(exp_sp))

    @pytest.mark.parametrize(
        "name, A, t, j1, rules1, zp1, sp1, j2, rules2, zp2, sp2",
        [
            (
                "t1",
                # the case that the added rules are irrelevant
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                [1, 1],
                # add rule 3
                3,
                [1, 2],
                [1, 1],
                [1, 1],
                # add rule 3, whcih does not change anything
                3,
                [],
                [1, 1],
                [1, 1],
            ),
            (
                "t2",
                [[1, 0, 1, 0], [0, 1, 0, 1]],
                [1, 0],
                # add 3, which determines C1 only
                # but rule 1 is not added
                3,
                [],
                [1, 0],
                [1, 0],
                # add 4, which determines C2
                # rule 2 is added
                4,
                [2],
                [1, 0],
                [1, 1],
            ),
        ],
    )
    def test_multiple_calls(
        self, name, A, t, j1, rules1, zp1, sp1, j2, rules2, zp2, sp2
    ):
        """test calling ensure_no_violation multiple times"""

        A, t, _, pivot_columns = extended_rref(
            GF(np.array(A, dtype=int)), GF(np.array(t, dtype=int)), verbose=False
        )
        A, t = map(bin_array, (A, t))

        max_nz_idx_array = get_max_nz_idx_per_row(A)
        m, n = A.shape

        z = bin_zeros(m)
        s = bin_zeros(m)
        actual_rules1, actual_zp1, actual_sp1 = ensure_no_violation(
            j1, z, s, t, A, max_nz_idx_array, pivot_columns
        )

        np.testing.assert_allclose(actual_rules1, rules1)
        np.testing.assert_allclose(actual_zp1, zp1)
        np.testing.assert_allclose(actual_sp1, sp1)

        actual_rules2, actual_zp2, actual_sp2 = ensure_no_violation(
            j2, actual_zp1, actual_sp1, t, A, max_nz_idx_array, pivot_columns
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

    @pytest.mark.parametrize("ub", [float("inf"), 0.5, 0.01])
    def test_complete_enumeration_and_alignment_with_cbb_on_toy_data(
        self, rules, y, ub
    ):
        """the output should be the same as cbb"""
        lmbd = 0.1
        cbb = ConstrainedBranchAndBoundNaive(rules, ub, y, lmbd)
        cbb_v2 = ConstrainedBranchAndBound(rules, ub, y, lmbd)

        A = bin_array([[1, 0, 1], [0, 1, 0]])
        t = bin_array([0, 1])
        expected_sols = solutions_to_dict(
            list(cbb.run(return_objective=True, A=A, t=t))
        )

        actual_sols = solutions_to_dict(
            list(cbb_v2.run(return_objective=True, A=A, t=t))
        )

        assert_dict_allclose(actual_sols, expected_sols)

    @pytest.mark.parametrize("num_rules", [10, 15, 20])
    @pytest.mark.parametrize("num_constraints", [2, 5, 8])
    @pytest.mark.parametrize("lmbd", [0.1])
    @pytest.mark.parametrize("ub", [0.5001, 0.2001, 0.0001])  # float("inf"),  # , 0.01
    @pytest.mark.parametrize("rand_seed", randints(10))
    def test_complete_enumeration_and_alignment_with_cbb_on_random_dataset(
        self, num_rules, num_constraints, lmbd, ub, rand_seed
    ):
        """the output should be the same as cbb"""
        rand_rules, rand_y = generate_random_rules_and_y(10, num_rules, rand_seed)
        print("rand_y: {}".format("".join(map(str, rand_y[::-1].astype(int)))))
        for r in rand_rules:
            print(f"{r.name:>20}: {bin(r.truthtable):>20}")

        cbb = ConstrainedBranchAndBoundNaive(rand_rules, ub, rand_y, lmbd)
        cbb_v2 = ConstrainedBranchAndBound(rand_rules, ub, rand_y, lmbd)

        A, t = generate_h_and_alpha(
            num_rules, num_constraints, rand_seed, as_numpy=True
        )
        expected_sols = solutions_to_dict(
            list(cbb.run(return_objective=True, A=A, t=t))
        )

        actual_sols = solutions_to_dict(
            list(cbb_v2.run(return_objective=True, A=A, t=t))
        )
        print("cbb_v2.A.astype(int):\n {}".format(cbb_v2.A.astype(int)))
        print("cbb_v2.t.astype(int):\n {}".format(cbb_v2.t.astype(int)))

        print("len(actual_sols): {}".format(len(actual_sols)))
        print("len(expected_sols): {}".format(len(expected_sols)))
        print("actual_sols: {}".format(actual_sols))
        print("expected_sols: {}".format(expected_sols))
        print("-" * 10 + "sol(v1) - sol(v2)" + "-" * 10)
        for sol in set(expected_sols.keys()) - set(actual_sols.keys()):
            print(
                "obj({}): {}".format(
                    sol,
                    calculate_obj(cbb_v2.rules, cbb_v2.y_np, cbb_v2.y_mpz, sol, lmbd),
                )
            )

        print("-" * 10 + "sol(v2) - sol(v1)" + "-" * 10)
        for sol in set(actual_sols.keys()) - set(expected_sols.keys()):
            print(
                "obj({}): {}".format(
                    sol,
                    calculate_obj(cbb_v2.rules, cbb_v2.y_np, cbb_v2.y_mpz, sol, lmbd),
                )
            )

        actual_keys = set(actual_sols.keys())
        expected_keys = set(expected_sols.keys())
        assert actual_keys == expected_keys
        assert_dict_allclose(actual_sols, expected_sols)

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
