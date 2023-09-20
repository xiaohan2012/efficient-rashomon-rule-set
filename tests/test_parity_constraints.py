import pytest
import numpy as np
from bds.gf2 import GF, extended_rref
from bds.utils import bin_array, bin_zeros
from bds.parity_constraints import (
    build_boundary_table,
    ensure_minimal_no_violation,
    ensure_satisfiability,
    count_added_pivots
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
