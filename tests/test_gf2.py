import numpy as np
import pytest

from bds.gf2 import GF, extended_rref
from bds.gf2 import eye as gf2_eye
from bds.gf2 import fix_variables_to_one, negate_all, num_of_solutions
from bds.random_hash import generate_h_and_alpha
from bds.utils import bin_array

np.random.seed(12345)


class TestExtendedRref:
    @pytest.mark.parametrize(
        "b",
        [
            GF([[[0, 0], [1, 1]]]),  # 3D
            GF([[[0], [0], [0], [0]]]),  # 2D, shape mismatch
            GF([0, 0, 1, 1]),  # 1D, too long
        ],
    )
    def test_invalid_inputs(self, b):
        A = GF([[0, 1, 0], [1, 1, 0], [1, 0, 1]])

        with pytest.raises(ValueError):
            extended_rref(A, b)

    @pytest.mark.parametrize(
        "A, b, exp_A, exp_b, exp_p, pivots",
        [
            # expected reduction process and per-step output
            # 1. at column 0
            #    swap row 0 and row 1
            #    A:
            #     [1, 1, 0],
            #     [0, 1, 0],
            #     [1, 0, 1]
            #    b:
            #     [0]
            #     [1]
            #     [0]
            #    -- no need to force 1 on the pivot
            # 2. at column 1
            #   pivot is at row 1
            #   no need to swap
            #   force above and below to be zero
            #     A:
            #     [1, 0, 0],
            #     [0, 1, 0],
            #     [0, 0, 1]
            #     b:
            #     [1]
            #     [1]
            #     [1]
            # 3. at column 2
            #   pivot is at row 2
            #   no need to swap
            #   no need to force above and below to be zero
            # done with p = 3
            (
                GF([[0, 1, 0], [1, 1, 0], [1, 0, 1]]),  # A
                GF([1, 0, 0]),  # b
                gf2_eye(3),  # expected A
                GF([1, 1, 1]),  # expected b
                3,  # rank
                np.asarray([0, 1, 2]),  # pivots
            ),
            # ---------------------------
            # 1. at column 0
            #    swap row 0 and row 1
            #    A:
            #    [1, 1]
            #    [0, 0]
            #    b:
            #     [0]
            #     [1]
            # 2. at column 1
            #    there is no pivot
            # done
            # we return with p = 1 and A is eye matrix
            (
                GF([[0, 0], [1, 1]]),
                GF([1, 0]),
                GF([[1, 1], [0, 0]]),
                GF([0, 1]),
                1,
                np.asarray([0]),
            ),
            # ---------------------------
            # 1. at column 0
            #    swap row 0 and row 1
            #    A:
            #    [1, 0, 1]
            #    [0, 0, 1]
            #    b:
            #     [0]
            #     [1]
            # 2. at column 1
            #    there is no pivot
            # 3. at column 2
            #    substract row 0 by row 1
            #    A:
            #    [1, 0, 0]
            #    [0, 0, 1]
            #    b:
            #     [1]
            #     [1]
            # done with p = 2
            (
                GF(
                    [
                        [0, 0, 1],
                        [1, 0, 1],
                    ]
                ),
                GF([1, 0]),
                GF([[1, 0, 0], [0, 0, 1]]),
                GF([1, 1]),
                2,
                np.asarray([0, 2]),
            ),
            # ---------------------------
            # 1. at column 0
            #    swap row 0 and row 1
            #    A:
            #     [1, 0]
            #     [0, 0]
            #     [0, 1]
            #    b:
            #    [0]
            #    [1]
            #    [0]
            # 2. at column 1
            #   swap row 1 and row 2
            #   A:
            #     [1, 0]
            #     [0, 1]
            #     [0, 0]
            #   b:
            #    [0]
            #    [0]
            #    [1]
            # done with p = 2
            (
                GF([[0, 0], [1, 0], [0, 1]]),
                GF([1, 0, 0]),
                GF([[1, 0], [0, 1], [0, 0]]),
                GF([0, 0, 1]),
                2,
                np.asarray([0, 1]),
            ),
        ],
    )
    def test_valid_inputs(self, A, b, exp_A, exp_b, exp_p, pivots):
        A_rref, b_rref, p, actual_pivots = extended_rref(A, b, verbose=False)
        assert (A_rref == exp_A).all()
        assert (b_rref == exp_b).all()
        assert p == exp_p
        assert p == len(actual_pivots)
        np.testing.assert_allclose(actual_pivots, pivots)

    def test_np_input(self):
        A, b = generate_h_and_alpha(10, 9, 42, as_numpy=False)
        expected_A, expected_b, expected_p, expected_pivots = extended_rref(
            A, b, verbose=False
        )
        actual_A, actual_b, actual_p, actual_pivots = extended_rref(
            np.asarray(A, dtype=bool), np.asarray(b, dtype=bool), verbose=False
        )

        np.testing.assert_allclose(bin_array(expected_A), bin_array(actual_A))
        np.testing.assert_allclose(bin_array(expected_b), bin_array(actual_b))
        np.testing.assert_allclose(expected_p, actual_p)
        np.testing.assert_allclose(expected_pivots, actual_pivots)


@pytest.mark.parametrize(
    "which, expected_t",
    [
        ([0], [0, 1, 0, 0]),
        ([1], [1, 0, 0, 0]),
        ([4], [0, 0, 1, 0]),
        ([0, 1, 3], [0, 1, 0, 0]),
        ([0, 1, 4], [1, 1, 1, 0]),
    ],
)
def test_fix_variables_to_one(which, expected_t):
    R = GF([[1, 0, 0, 0, 1], [0, 1, 0, 1, 1], [0, 0, 1, 0, 1], [0, 0, 0, 0, 0]])
    t = GF([1, 1, 0, 0])
    Rp, tp = fix_variables_to_one(R, t, which)
    assert Rp.shape[1] == (R.shape[1] - len(which))
    np.testing.assert_allclose(np.asarray(tp), expected_t)


@pytest.mark.parametrize(
    "A, b, expected",
    [
        (GF([[1, 0], [0, 0]]), GF([0, 0]), 2),
        (GF([[1, 0], [0, 0]]), GF([0, 1]), 0),
        (GF([[1, 1], [0, 1]]), GF([1, 1]), 1),
    ],
)
def test_num_of_solutions(A, b, expected):
    assert num_of_solutions(A, b) == expected


@pytest.mark.parametrize(
    "A, expected",
    [
        (np.array([0, 0, 1], dtype=int), GF([1, 1, 0])),
        (np.array([1, 1, 1], dtype=int), GF([0, 0, 0])),
        (np.array([0, 0, 0], dtype=int), GF([1, 1, 1])),
        (GF([0, 0, 1]), GF([1, 1, 0])),
        (GF([1, 1, 1]), GF([0, 0, 0])),
        (GF([0, 0, 0]), GF([1, 1, 1])),
        (np.array([[1, 1, 1], [1, 1, 1]], dtype=int), GF([[0, 0, 0], [0, 0, 0]])),
    ],
)
def test_negate_all(A, expected):
    assert isinstance(expected, GF)
    np.testing.assert_allclose(
        np.array(negate_all(A), dtype=int), np.array(expected, dtype=int)
    )
