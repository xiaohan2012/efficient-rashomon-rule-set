import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from .gf2 import GF
from numba import jit
from .types import RuleSet
from .utils import bin_zeros


def build_boundary_table(
    A: np.ndarray, rank: int, pivot_columns: np.ndarray
) -> np.ndarray:
    """for a 2D matrix A, compute the maximum non-zero non-pivot index per row, if it does not exist, use -1"""
    assert isinstance(A, np.ndarray)
    assert A.ndim == 2
    Ap = A.copy()
    result = []
    for i in range(rank):
        Ap[i, pivot_columns[i]] = 0
        if Ap[i, :].sum() == 0:
            result.append(-1)
        else:
            result.append((Ap[i, :] > 0).nonzero()[0].max())
    return np.array(result, dtype=int)


@jit(nopython=True, cache=True)
def inc_ensure_minimal_no_violation(
    j: int,
    z: np.ndarray,
    s: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    B: np.ndarray,
    row2pivot_column: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    incrementally ensure minimal non violation

    upon adding rule j to the current prefix (represented  by `z` and `s`),
    add a set of pivot rules to ensure that the new prefix is minimally non-violating

    return:

    - selected pivot rule indices after adding the jth rule to the current prefix
    - the satisfiability vector
    - the updated parity states vector

    note that the rule index must correspond to a non-pivot column

    the parity states vector `s` and satisfiability vector `z` for the current prefix are provided for incremental computation

    row2pivot_column is the mapping from row id to the corresponding pivot column

    for performance reasons, the following data structures are given:

    - max_nz_idx_array: the array of largest non-zero idx per constraint
    """
    zp: np.ndarray = z.copy()
    sp: np.ndarray = s.copy()

    selected_rules = np.empty(A.shape[1], np.int_)
    num_rules_selected = 0
    for i in range(A.shape[0]):
        if j == -1:
            # the initial case, where no rules are added
            if j == B[i]:
                sp[i] = 1
                if b[i] == 1:
                    selected_rules[num_rules_selected] = row2pivot_column[i]
                    num_rules_selected += 1
                    zp[i] = not zp[i]
            continue
        if sp[i] == 0:
            # constraint i is not satisfied yet
            if j >= B[i]:
                # j is exterior
                # the corresponding pivot rule maybe added
                sp[i] = 1
                if A[i][j]:
                    #  j is relevant
                    if b[i] == zp[i]:
                        selected_rules[num_rules_selected] = row2pivot_column[i]
                        num_rules_selected += 1
                    else:
                        zp[i] = not zp[i]
                elif b[i] != zp[i]:
                    # j is irrelevant
                    selected_rules[num_rules_selected] = row2pivot_column[i]
                    num_rules_selected += 1
                    zp[i] = not zp[i]
            elif A[i][j]:
                # j is interior and relevant
                zp[i] = not zp[i]
    return selected_rules[:num_rules_selected], zp, sp


@jit(nopython=True, cache=True)
def count_added_pivots(j: int, A: np.ndarray, b: np.ndarray, z: np.ndarray) -> int:
    """count the number of pivot rules to add in order to satisfy Ax=b

    which is equivalent to counting the number of constraints that satisfies either of the conditions below:

    - A[i][j] is False and (z[i] == b[i]) is False
    - A[i][j] is True and (z[i] == b[i]) is True

    which is equivalent to counting the number entries in A[:, j] == (z == b) that are True
    """
    assert j >= 0
    return (A[:, j] == (z == b)).sum()


@jit(nopython=True, cache=True)
def inc_ensure_satisfiability(
    j: int,
    rank: int,
    z: np.ndarray,
    s: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    row2pivot_column: np.ndarray,
) -> np.ndarray:
    """
    return the pivot variables that are assigned to 1s if no rules with index larger than j are added further.

    j must corresponds to a free variable, meaning:

    1. it is not the default one (j=0)
    2. and it does not correspond to any pivot variable
    """
    selected_rules = np.empty(A.shape[1], np.int_)
    num_rules_selected = 0

    for i in range(rank):  # loop up to rank
        if j == -1:
            if b[i] == 1:
                selected_rules[num_rules_selected] = row2pivot_column[i]
                num_rules_selected += 1
        elif s[i] == 0:
            if ((A[i][j] == 0) and (z[i] != b[i])) or (
                (A[i][j] == 1) and (z[i] == b[i])
            ):
                # case 1: the rule is irrelevant
                # case 2: the rule is relevant
                selected_rules[num_rules_selected] = row2pivot_column[i]
                num_rules_selected += 1
    return selected_rules[:num_rules_selected]


# @jit(nopython=True, cache=True)
def ensure_minimal_non_violation(
    prefix: RuleSet,
    A: np.ndarray,
    b: np.ndarray,
    B: np.ndarray,
    row2pivot_column: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ensure minimal non violation for a prefix w.r.t. parity constraint system Ax=b

    the following information is provided for incremental computation:
    A and b define the parity constraint system
    B: the boundary table
    row2pivot_column: mapping from row index to the index of pivot column

    returns:

    1. the set of pivot rules being added
    2. parity status vector
    3. satisfiability status vector
    """
    m = A.shape[0]
    z = bin_zeros(m)
    s = bin_zeros(m)

    # at root
    all_rules_added, z, s = inc_ensure_minimal_no_violation(
        -1, z, s, A, b, B, row2pivot_column
    )
    for j in prefix:
        rules_added, z, s = inc_ensure_minimal_no_violation(
            j, z, s, A, b, B, row2pivot_column
        )
        all_rules_added = np.concatenate((all_rules_added, rules_added))
    return all_rules_added, z, s


def ensure_satisfiability(
    prefix: RuleSet,
    A_gf: GF,
    b_gf: GF,
    row2pivot_column: np.ndarray,
) -> np.ndarray:
    """ensure satisfaction for a prefix w.r.t. parity constraint system Ax=b

    the following information is provided for incremental computation:
    A and b define the parity constraint system
    B: the boundary table
    row2pivot_column: mapping from row index to the index of pivot column

    returns the array of pivot rules being added
    """
    x = GF.Zeros(A_gf.shape[1])
    x[list(prefix)] = True
    Ax = A_gf @ x
    return row2pivot_column[np.asarray(b_gf - Ax).nonzero()[0]]
