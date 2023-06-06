import pytest
import numpy as np
from bds.utils import (
    bin_zeros,
    bin_array,
    bin_ones,
    get_indices_and_indptr,
    get_max_nz_idx_per_row,
)
from bds.cbb_v2 import update_pivot_variables


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
        np.testing.assert_allclose(zp, bin_array([1, 0]))

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
