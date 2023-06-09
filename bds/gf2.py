# util functions for operations uedr GF(2)
import numpy as np
import galois
from typing import Tuple, Union, Optional, List

GF = galois.GF(2)


def extended_rref(A: GF, b: GF, verbose: bool = False) -> Tuple[GF, GF, int, np.ndarray]:
    """
    given a 2D matrix A and a vector b, both in GF2,
    obtain the reduced row echelon form of a matrix A and repeat the reduction process on a column vector b
    return:

    - the transformed A and b
    - the rank of A
    - the indices of the pivot columns
    """
    if b.ndim == 1:
        b = b.reshape(-1, 1)  # transform it to column vector

    if (not b.ndim == 2) or b.shape[0] != A.shape[0] or b.shape[1] != 1:
        raise ValueError(
            f"b (of shape {b.shape}) should be a 2D column vector of the same number of rows as A"
        )

    if not A.ndim == 2:
        raise ValueError(
            f"Only 2-D matrices can be converted to reduced row echelon form, not {A.ndim}-D."
        )

    ncols = A.shape[1]
    A_rre = A.copy()
    b_rre = b.copy()
    p = 0  # The pivot
    pivot_columns = []

    if verbose:
        print(f"A:\n{A_rre}")
        print(f"b:\n{b_rre}")
    for j in range(ncols):
        if verbose:
            print(f"p={p}")
        # Find a pivot in column `j` at or below row `p`
        idxs = np.nonzero(A_rre[p:, j])[0]
        if idxs.size == 0:
            continue
        i = p + idxs[0]  # Row with a pivot
        if verbose:
            print(f"checking column {j}")
            print(f"pivot is at row {i}")

        # Swap row `p` and `i`. The pivot is now located at row `p`.
        A_rre[[p, i], :] = A_rre[[i, p], :]
        b_rre[[p, i], :] = b_rre[[i, p], :]
        if p != i and verbose:
            print(f"swap row {p} and {i}")
            print("A:")
            print(f"{A_rre}")
            print("b:")
            print(f"{b_rre}")

        # the pivot of row p is the first non-zero column index)
        pivot_columns.append(A_rre[p, :].nonzero()[0][0])

        idxs = np.nonzero(A_rre[:, j])[0].tolist()
        idxs.remove(p)
        A_rre[idxs, :] -= A_rre[p, :]
        b_rre[idxs, :] -= b_rre[p, :]

        if len(idxs) > 0 and verbose:
            print("Force zeros above and below the pivot")
            print(f"substract row {p} to row {idxs}")
            print("A:")
            print(A_rre)
            print("b:")
            print(b_rre)

        p += 1
        if p == A_rre.shape[0]:
            break

    return A_rre, b_rre.flatten(), p, np.asarray(pivot_columns, dtype=int)


def eye(rank: int) -> GF:
    return GF(np.eye(rank, dtype=int))


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
    assert len(unique_elems - {0, 1}) == 0, "extra values {}".format(
        unique_elems - {0, 1}
    )
    try:
        idx_of_1st_zero = next(i for i, el in enumerate(arr) if el == 0)
    except StopIteration:  # all 1s
        return True

    arr_mirror = arr.copy()
    arr_mirror.fill(0)
    arr_mirror[idx_of_1st_zero:] = 1

    return ((arr + arr_mirror) == 1).all()


def num_of_solutions(A: GF, b: GF) -> int:
    """return the number of solutions to a linear system Ax=b in GF2"""
    R, t, rank = extended_rref(A, b)
    if not (t[rank:] == 0).all():
        # the system is not solvable
        return 0
    else:
        return np.power(2, A.shape[1] - rank)


def fix_variables_to_one(A: GF, b: GF, which: List[int]) -> Tuple[GF, GF]:
    """fix the value of all variables in `which` to 1 for a linear system Ax=b,
    return the updated linear system with those fixed variables removed"""
    # identify the rows in b whose values should change
    row_occurences = A[:, which].nonzero()[0]

    # get those rows with odd number of occurences
    # these rows in b should be updated
    row_idxs, freq = np.unique(row_occurences, return_counts=True)
    row_idxs_to_update = row_idxs[
        (freq % 2) == 1
    ]  # select rows with odd number of occurences

    # update the rows in b
    bp = b.copy()
    bp[row_idxs_to_update] = GF(1) - bp[row_idxs_to_update]  # flip the value

    # remove those columns from A
    Ap = np.delete(A, which, axis=1)

    return Ap, bp
