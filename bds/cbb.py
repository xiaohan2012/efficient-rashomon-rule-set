# constrained branch-and-bounch
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from .utils import assert_binary_array


def check_if_not_unsatisfied(
    j: int, A: np.ndarray, t: np.ndarray, s: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    given:

    j: the index of the rule to be inserted (we assume rules are 1-indexed, i.e., the first rule's index is 1)
    A: the constraint matrix
    t: target parity vector
    s: the satisfication vector of a given prefix, 0 means 'unsatisfied', 1 means 'satisfied', and -1 means "undetermined"
    z: parity states vector of a given preifx, 0 means 'even' and 1 means 'odd'

    (note that A and t determines the parity constraint system)

    return:

    - the updated satisfaction vector after inserting the jth rule into the prefix
    - the updated parity constraint
    - whether the constraint system is still not unsatisfied
    """
    # print(f"==== checking the parity system ===== ")
    assert_binary_array(z)
    assert_binary_array(t)
    assert s.shape == z.shape == t.shape

    sp, zp = s.copy(), z.copy()
    num_constraints, num_variables = A.shape
    for i in range(num_constraints):
        if s[i] == -1:  # s[i] == ?
            # print(f"constraint {i+1} is undetermined")
            if A[i, j - 1]:  # j-1 because we assume rule index is 1-indexed
                # print(f"rule {j} is active in this constraint")
                # print(f"parity value from {zp[i]} to {np.invert(zp[i])}")
                zp[i] = np.invert(zp[i])  # flip the sign
                max_nz_idx = A[i].nonzero()[0].max()  # TODO: cache this information
                if j == (max_nz_idx + 1):  # we can evaluate this constraint
                    # print(f"we can evaluate this constraint")
                    if zp[i] == t[i]:
                        # this constraint evaluates to tue, but we need to consider remaining constraints
                        # print(f"and it is satisfied")
                        sp[i] = 1
                    else:
                        # this constraint evaluates to false, thus the system evaluates to false
                        # print(f"and it is unsatisfied")
                        sp[i] = 0
                        return sp, zp, False
    return sp, zp, True


def check_if_satisfied(s: np.ndarray, z: np.ndarray, t: np.ndarray) -> bool:
    """check if yielding the current prefix as solution satisfies the parity constraint

    the calculation is based on incremental results from previous parity constraint checks
    """
    assert_binary_array(z)
    assert_binary_array(t)

    assert s.shape == z.shape == t.shape

    num_constraints = z.shape[0]
    for i in range(num_constraints):
        if s[i] == 0:  # failed this constraint
            return False
        elif (s[i] == -1) and (z[i] != t[i]):  # undecided
            return False
    return True
