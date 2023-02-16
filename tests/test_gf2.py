import pytest
import numpy as np
from bds import gf2
from bds.gf2 import rref_of_triu, GF, rref
from bds.utils import randints

np.random.seed(12345)


class Test_rref_triu:
    def test_toy(self):
        U = GF([[1, 0, 0, 0, 1], [0, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]])
        actual_rref_U, actual_E = rref_of_triu(U, return_E=True)

        # checking rref matrix
        expected_rref_U = GF(
            np.array(
                [[1, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]]
            )
        )

        np.testing.assert_equal(actual_rref_U, expected_rref_U)

        # checking elimination matrix
        # subtract row 3 from row 1
        E1 = GF(np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]))
        # substract row 2 from row 1
        E2 = GF(np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

        # the overall elimination matrix is the product between the individual elimination matrices
        expected_E = E2 @ E1
        np.testing.assert_equal(actual_E, expected_E)

    def test_eye_like(self):
        rank = 5
        I = gf2.eye(rank)
        R, E = rref_of_triu(I)
        np.testing.assert_equal(R, I)
        np.testing.assert_equal(E, I)

        # input is a fat eye-like matrix
        # of shape rank x (2 rank)
        fat_I = GF(
            np.hstack((np.eye(rank, dtype=int), np.zeros((rank, rank), dtype=int)))
        )
        R, E = rref_of_triu(fat_I)
        np.testing.assert_equal(R, fat_I)
        np.testing.assert_equal(E, I)

        # input is a thin eye-like matrix
        # of shape (2 rank) x rank
        thin_I = GF(
            np.vstack((np.eye(rank, dtype=int), np.zeros((rank, rank), dtype=int)))
        )
        R, E = rref_of_triu(thin_I)
        np.testing.assert_equal(R, thin_I)
        np.testing.assert_equal(E, gf2.eye(rank * 2))

    @pytest.mark.parametrize("rank", [3, 10, 25])
    def test_all_ones(self, rank):
        # generate a full-rank uppertriangular matrix
        U = GF(np.triu(np.ones((rank, rank), dtype=int)))
        rref, E = rref_of_triu(U, return_E=True, verbose=True)
        np.testing.assert_equal(gf2.eye(rank), rref)

    @pytest.mark.parametrize("seed", randints(5))
    def test_if_satsifying_structural_requirements(self, seed):
        """for a rref, at each pivot, the entries above should be zero"""
        rank = 10

        # generate a full-rank uppertriangular matrix
        A = GF.Random((rank, rank), seed=4321)
        U = np.triu(A)
        for i in range(rank):
            U[i, i] = 1  # make it full rank
        U = GF(U)
        rref = rref_of_triu(U, return_E=False, verbose=True)

        np.testing.assert_equal(gf2.eye(rank), rref)


class Test_rref:
    def test_toy_1(self):
        A = GF([[1, 0, 0, 0, 1], [0, 1, 1, 1, 1], [1, 1, 1, 0, 1], [0, 1, 0, 1, 1]])

        b = GF([1, 0, 0, 1])

        R, d = rref(A, b)

        expected_R = GF(
            [[1, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]]
        )
        expected_d = GF([1, 0, 1, 1])

        np.testing.assert_equal(R, expected_R)
        np.testing.assert_equal(d, expected_d)

    def test_toy_2(self):
        A = GF([[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]])

        expected_R = GF([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # try different b values
        b_and_d = [
            (GF([1, 1, 1, 1]), GF([0, 0, 0, 1])),
            (GF([1, 1, 1, 0]), GF([0, 0, 1, 0])),
            (GF([1, 1, 0, 0]), GF([0, 1, 0, 0])),
        ]

        for b, expected_d in b_and_d:
            R, d = rref(A, b)
            np.testing.assert_equal(R, expected_R)
            np.testing.assert_equal(d, expected_d)

    def test_eye_like(self):
        rank = 5
        I = gf2.eye(rank)
        R = rref(I)
        np.testing.assert_equal(R, I)

        # input is a fat eye-like matrix
        # of shape rank x (2 rank)
        fat_I = GF(
            np.hstack((np.eye(rank, dtype=int), np.zeros((rank, rank), dtype=int)))
        )
        R = rref(fat_I)
        np.testing.assert_equal(R, fat_I)

        # ignore the thin case since L is not defined
        # input is a thin eye-like matrix
        # of shape (2 rank) x rank
