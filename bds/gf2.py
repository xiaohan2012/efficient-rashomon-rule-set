# util functions for operations uedr GF(2)
import numpy as np
import galois
from typing import Tuple, Union, Optional

GF = galois.GF(2)


def eye(rank: int) -> GF:
    return GF(np.eye(rank, dtype=int))


def rref_of_triu(
    U: GF, return_E: bool = True, verbose: int = -1
) -> Union[GF, Tuple[GF, GF]]:
    """find the reduced row echelon form of an upper triangular matrix U in GF(2)

    return the elimination matrix if return_E is True
    """
    assert np.allclose(np.asarray(U), np.triu(U)), "the matrix is not upper triangular"

    U_acc = U.copy()

    nrow, ncol = U.shape
    # E_list = [] # list of elementary matrices
    E_acc = eye(nrow)  # accumulated elimination matrix
    for i in range(nrow - 1, -1, -1):
        if verbose > 0:
            print("at row", i)
        # find pivot -- the left most non-zero entry
        pivot_pos = -1
        for j in range(0, ncol):
            if U[i, j] != 0:
                pivot_pos = j
                break

        if not (0 <= pivot_pos < ncol):
            if verbose > 0:
                print("the row is all zero")
            continue

        if verbose > 0:
            print("pivot_pos", pivot_pos)

        # look upward
        for k in range(i - 1, -1, -1):
            if verbose > 0:
                print("checking row ", k)
            if U[k, pivot_pos] != 0:
                # we need to perform row elimination
                E = eye(nrow)

                assert i != k
                # subtract row i from row k
                if verbose > 0:
                    print("subtract row {} from {}".format(i, k))
                E[k, i] = 1
                E_acc = E @ E_acc

                if verbose > 0:
                    print("E:\n {}".format(E))

                    U_acc = E @ U_acc
                    print("U_acc:", U_acc)
    rref = E_acc @ U
    if not return_E:
        return rref
    else:
        return rref, E_acc


def rref(A: GF, b: Optional[GF] = None) -> Tuple[GF, GF]:
    """given a linear system Ax=b or a matrix A in GF2

    calculate the row reduced echolon form of A (and the corresponding vector of b if b is given)
    """
    # do Gaussian elimination, PA = LU
    # P: the permutation matrix
    # L: the inverse of row elimination matrix
    # U: the upper triangular matrix of A
    P, L, U = A.plu_decompose()

    # get rref of upper triangular matrix
    R, E = rref_of_triu(U, return_E=True, verbose=-1)

    if b is not None:
        # find c
        # PAx = Pb -> LUx = Pb = Lc
        c = np.linalg.solve(L, P @ b)  # solve Lc = Pb

        # replay the backward elimination on c
        d = E @ c

        return R, d
    else:
        return R


def is_piecewise_linear(arr: GF):
    """
    check if the arr in GF2 is piecewise linear, starting from 1

    examples:

    >> assert not is_piecewise_linear([0, 0, 1, 1])
    >> assert not is_piecewise_linear([1, 1, 0, 1])

    >> assert is_piecewise_linear([1, 1, 0, 0])
    >> assert is_piecewise_linear([1, 1, 1, 1])
    >> assert is_piecewise_linear([0, 0, 0, 0])
    """
    arr = np.array(arr, dtype=int)
    unique_elems = set(np.unique(arr))
    assert len(unique_elems - {0, 1}) == 0, 'extra values {}'.format(unique_elems - {0, 1})
    try:
        idx_of_1st_zero = next(i for i, el in enumerate(arr) if el == 0)
    except StopIteration:  # all 1s
        return True

    arr_mirror = arr.copy()
    arr_mirror.fill(0)
    arr_mirror[idx_of_1st_zero:] = 1

    return ((arr + arr_mirror) == 1).all()
