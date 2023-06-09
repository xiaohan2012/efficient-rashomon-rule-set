import pytest
import numpy as np
import gmpy2 as gmp

from gmpy2 import mpz, mpfr

from bds.gf2 import extended_rref, GF
from bds.utils import (
    bin_zeros,
    bin_array,
    bin_ones,
    get_indices_and_indptr,
    get_max_nz_idx_per_row,
    solutions_to_dict,
    mpz_set_bits,
    randints,
)
from bds.cbb import ConstrainedBranchAndBoundNaive
from bds.cbb_v2 import (
    update_pivot_variables,
    assign_pivot_variables,
    ConstrainedBranchAndBound,
)
from bds.rule import Rule, lor_of_truthtable
from bds.random_hash import generate_h_and_alpha
from .utils import (
    generate_random_rules_and_y,
    assert_dict_allclose,
    brute_force_enumeration,
)
from .fixtures import rules, y


class TestUpdatePivotVariables:
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
            rules, zp = update_pivot_variables(
                j, z, t, A_indices, A_indptr, max_nz_idx_array, row2pivot_column
            )

    def test_basic_1(self):
        A = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=bool)
        t = np.array([1, 1, 0], dtype=bool)

        A_indices, A_indptr = get_indices_and_indptr(A)
        max_nz_idx_array = get_max_nz_idx_per_row(A)
        row2pivot_column = np.array([0, 1, 2], dtype=int)
        m, n = A.shape

        j = 4
        z = bin_zeros(m)
        rules, zp = update_pivot_variables(
            j, z, t, A_indices, A_indptr, max_nz_idx_array, row2pivot_column
        )
        assert rules == {3}
        np.testing.assert_allclose(zp, bin_array([1, 1, 0]))

    def test_basic_2(self):
        A = np.array([[1, 0, 1, 0], [0, 0, 0, 1]], dtype=bool)
        t = np.array([0, 1], dtype=bool)

        A_indices, A_indptr = get_indices_and_indptr(A)
        max_nz_idx_array = get_max_nz_idx_per_row(A)
        row2pivot_column = np.array([0, 3], dtype=int)
        m, n = A.shape

        j = 3
        z = bin_zeros(m)
        rules, zp = update_pivot_variables(
            j, z, t, A_indices, A_indptr, max_nz_idx_array, row2pivot_column
        )
        assert rules == {1}
        np.testing.assert_allclose(zp, bin_array([0, 0]))

        j = 2
        z = bin_zeros(m)
        rules, zp = update_pivot_variables(
            j, z, t, A_indices, A_indptr, max_nz_idx_array, row2pivot_column
        )
        assert rules == set()
        np.testing.assert_allclose(zp, bin_array([0, 0]))

        # cannot set rule-4 because it is pivot
        with pytest.raises(ValueError, match="cannot set pivot variable of column 3"):
            j = 4
            rules, zp = update_pivot_variables(
                j, z, t, A_indices, A_indptr, max_nz_idx_array, row2pivot_column
            )

    @pytest.mark.parametrize(
        "j, z, z_expected",
        [
            (2, [0, 1], [0, 1]),
            (2, [1, 0], [1, 0]),
            (3, [0, 1], [0, 1]),  # 1 and 3 are selected
            (3, [1, 0], [1, 0]),  # so the parity states vector should be the same
            (3, [0, 0], [0, 0]),
        ],
    )
    def test_updated_parity_states(self, j, z, z_expected):
        A = np.array([[1, 0, 1, 0], [0, 0, 0, 1]], dtype=bool)
        t = np.array([0, 1], dtype=bool)

        A_indices, A_indptr = get_indices_and_indptr(A)
        max_nz_idx_array = get_max_nz_idx_per_row(A)
        row2pivot_column = np.array([0, 3], dtype=int)
        m, n = A.shape

        j = 2
        z = bin_array(z)
        rules, zp = update_pivot_variables(
            j, z, t, A_indices, A_indptr, max_nz_idx_array, row2pivot_column
        )
        np.testing.assert_allclose(zp, bin_array(z_expected))


class TestAssignPivotVariables:
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

        A_indices, A_indptr = get_indices_and_indptr(A)
        max_nz_idx_array = get_max_nz_idx_per_row(A)
        A, t, rank, row2pivot_column = extended_rref(A, t)
        m, n = A.shape

        j = 1

        with pytest.raises(
            ValueError, match=f"cannot set pivot variable of column {j-1}"
        ):
            assign_pivot_variables(
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
                # update_pivot_variables determines all constraints
                # therefore assign_pivot_variables changes nothing
                "case-1",
                [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]],
                [1, 1, 0],
                4,
                set(),
            ),
            (
                # j = 3
                # update_pivot_variables determines the 2nd constraint and x2 = 0
                # assign_pivot_variables determines the 1st constraint, setting x4 = 0
                # and further sets x1 = 1
                "case-2",
                [[1, 0, 1, 1], [0, 1, 1, 0]],
                [0, 1],
                3,
                {1},
            ),
            (
                # j = 3
                # update_pivot_variables determines no constraint
                # assign_pivot_variables determines the 1st and 2nd constraint
                # for the 1st constraint: x1 = 1
                # for the 2nd constraint: x2 = 1
                "case-3",
                [[1, 0, 1, 1], [0, 1, 0, 0]],
                [0, 1],
                3,
                {1, 2},
            ),
            (
                # j = 4
                # update_pivot_variables determines the 1st constraint: x1 = 1
                # assign_pivot_variables determines the 2nd constraint: x2 = 1
                "case-4",
                [[1, 0, 1, 1], [0, 1, 0, 0]],
                [0, 1],
                4,
                {2},
            ),
            (
                # j = 0 (adding the default rule)
                # all free rules are excludeda
                # update_pivot_variables does nothing
                # assign_pivot_variables determines both constraints:
                # x1 = x2 = 1
                "case-5",
                [[1, 0, 1, 1], [0, 1, 0, 0]],
                [1, 1],
                0,
                {1, 2},
            ),
            (
                # j = 0 (adding the default rule)
                # all free rules are excluded
                # update_pivot_variables does nothing
                # assign_pivot_variables determines both constraints:
                # x3 = 1
                "case-6",
                [[1, 0, 1, 1], [0, 0, 1, 0]],
                [0, 1],
                0,
                {1, 3},  # 1 is added because t becomes [1, 1] due to rref
            ),
            (
                # j = 0 (adding the default rule)
                # all free rules are excluded
                # update_pivot_variables does nothing
                # assign_pivot_variables determines both constraints:
                # x3 = 1
                "case-7",
                [[1, 0, 1, 1], [0, 0, 1, 0]],
                [1, 1],
                0,
                {3},  # 1 is added because t becomes [0, 1] due to rref
            ),
            (
                # (the last row has no pivot variable, thus should not be checked
                # j = 4
                # update_pivot_variables sets x1=1
                # assign_pivot_variables does nothing
                # x1 = x3 = 1
                "case-8",
                [[1, 0, 1, 1], [0, 0, 0, 0]],
                [0, 0],
                4,
                set(),
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

        _, z = update_pivot_variables(
            j, bin_zeros(m), t, A_indices, A_indptr, max_nz_idx_array, row2pivot_column
        )

        rules = assign_pivot_variables(
            j, rank, z, t, A_indices, A_indptr, max_nz_idx_array, row2pivot_column
        )
        assert rules == expected_rules


class TestConstrainedBranchAndBound:
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

    @pytest.mark.parametrize(
        "A, t, exp_pivot_rule_idxs, exp_free_rule_idxs, exp_row2pivot_column",
        [
            ([[1, 0, 1], [0, 1, 0]], [0, 1], {1, 2}, {3}, [0, 1]),
            ([[1, 0, 1], [0, 0, 1]], [0, 1], {1, 3}, {2}, [0, 2]),
            ([[1, 0, 1]], [0], {1}, {2, 3}, [0]),
        ],
    )
    def test_setup_constraint_system(
        self, A, t, exp_pivot_rule_idxs, exp_free_rule_idxs, exp_row2pivot_column
    ):
        rules, y = generate_random_rules_and_y(10, 3, 12345)
        cbb = ConstrainedBranchAndBound(rules, float("inf"), y, 0.1)

        cbb.setup_constraint_system(bin_array(A), bin_array(t))

        assert cbb.num_vars == 3

        assert cbb.pivot_rule_idxs == exp_pivot_rule_idxs
        assert cbb.free_rule_idxs == exp_free_rule_idxs
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
            pivot_rules_to_add=rules[:2],  # add rule-1 and rule-2 as pivot
        )
        assert cbb.tree.num_nodes == 2  # tree is updated
        assert child.pivot_rule_ids == [1, 2]

        grandchild = cbb._create_new_node_and_add_to_tree(
            rules[4],
            lb=mpfr(),
            obj=mpfr(),
            captured=mpz(),
            parent_node=child,
            pivot_rules_to_add=[rules[3]],  # add rule-4 as pivot
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

    # @pytest.mark.parametrize("num_rules", [10, 15])
    # @pytest.mark.parametrize("num_constraints", [2, 5, 8])
    # @pytest.mark.parametrize("lmbd", [0.1])
    # @pytest.mark.parametrize("ub", [0.5001, 0.2001, 0.0001])  # float("inf"),  # , 0.01
    # @pytest.mark.parametrize("rand_seed", randints(10))
    @pytest.mark.parametrize("num_rules", [10])
    @pytest.mark.parametrize("num_constraints", [8])
    @pytest.mark.parametrize("lmbd", [0.1])
    @pytest.mark.parametrize("ub", [0.5001])  # float("inf"),  # , 0.01
    @pytest.mark.parametrize("rand_seed", [1859619716])
    def test_complete_enumeration_and_alignment_with_cbb_on_random_dataset(
        self, num_rules, num_constraints, lmbd, ub, rand_seed
    ):
        """the output should be the same as cbb"""
        rand_rules, rand_y = generate_random_rules_and_y(10, num_rules, rand_seed)
        # print("rand_y: {}".format(rand_y[::-1].astype(int)))
        # for r in rand_rules:
        #     print(f"{r.name}: {bin(r.truthtable)}")

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

        # actual_keys = set(actual_sols.keys())
        # expected_keys = set(expected_sols.keys())
        # assert actual_keys == expected_keys
        assert_dict_allclose(actual_sols, expected_sols)

    @pytest.mark.skip("skipped because if cbb is correct, testing cbb_v2 against cbb (shown above) is enough")
    @pytest.mark.parametrize("num_rules", [10])
    @pytest.mark.parametrize("num_constraints", [2, 4, 8])
    @pytest.mark.parametrize("lmbd", [0.1])
    @pytest.mark.parametrize("ub", [0.5001])  # float("inf"),  # , 0.01
    @pytest.mark.parametrize("rand_seed", [895595566])
    def test_corretness(self, num_rules, num_constraints, lmbd, ub, rand_seed):
        """the output should be the same as ground truth"""
        rand_rules, rand_y = generate_random_rules_and_y(10, num_rules, rand_seed)

        cbb = ConstrainedBranchAndBound(rand_rules, ub, rand_y, lmbd)

        A, t = generate_h_and_alpha(
            num_rules, num_constraints, rand_seed, as_numpy=True
        )
        actual = solutions_to_dict(list(cbb.run(return_objective=True, A=A, t=t)))

        expected = solutions_to_dict(
            list(brute_force_enumeration(rand_rules, rand_y, A, t, ub, lmbd))
        )

        assert set(actual.keys()) == set(expected.keys())
        assert_dict_allclose(actual, expected)
