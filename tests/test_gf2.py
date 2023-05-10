import pytest
import numpy as np
from bds import gf2
from bds.gf2 import extended_rref, GF, eye as gf2_eye
from bds.utils import randints

np.random.seed(12345)


class TestExtendedRref:
    @pytest.mark.parametrize('b', [
        GF([[[0, 0], [1, 1]]]), # 3D
        GF([[[0], [0], [0], [0]]]), # 2D, shape mismatch
        GF([0, 0, 1, 1]), # 1D, too long
    ])
    def test_invalid_inputs(self, b):
        A = GF([[0, 1, 0], [1, 1, 0], [1, 0, 1]])
        
        with pytest.raises(ValueError):
            extended_rref(A, b)
        
        

    @pytest.mark.parametrize(
        "A, b, exp_A, exp_b, exp_p",
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
            (GF([[0, 0], [1, 1]]), GF([1, 0]), GF([[1, 1], [0, 0]]), GF([0, 1]), 1),
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
            ),
        ],
    )
    def test_valid_inputs(self, A, b, exp_A, exp_b, exp_p):
        A_rref, b_rref, p = extended_rref(A, b, verbose=False)
        assert (A_rref == exp_A).all()
        assert (b_rref == exp_b).all()
        assert p == exp_p
