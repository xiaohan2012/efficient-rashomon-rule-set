import numpy as np
import itertools
from logzero import logger
from typing import Optional, List, Dict, Tuple, Union
from .rule import Rule
from .utils import fill_array_from, fill_array_until
from .cbb import ConstrainedBranchAndBoundNaive


def log_search(
    rules: List[Rule],
    y: np.ndarray,
    lmbd: float,
    ub: float,
    A: np.ndarray,
    t: np.ndarray,
    thresh: int,
    m_prev: int,
    return_full: Optional[bool] = False,
) -> Union[Tuple[int, int], Tuple[int, int, np.ndarray, np.ndarray]]:
    """
    for a random XOR constraint system,
    find the correct number of constraints (m) such that the number of solutions under the constraint sub-system is right below `thresh`

    or m is the smallest number of constraints such that the solution number is below `thresh`]

    note that as m grows, the solution number decreases

    rules: the list of candidate rules
    y: the label of each training point
    A and t: the parity constraint system
    thresh: the maximum value of cell size
    m_prev: previous value of m (where 2^m is supposed to be an estimate of the number of partitions)
    return_full:
      is True if returning all information relevant to the call,
      otherwise return the m value and the corresponding solution set size
    """
    # TODO: cache the number of solutions and return the one corresponding to the m
    if thresh <= 1:
        raise ValueError("thresh should be greater than 1")

    num_vars = A.shape[1]
    lo, hi = 0, num_vars - 1
    m = m_prev

    # meaining of entry value:
    # | value | meaning                                 |
    # |-------+-----------------------------------------|
    # |    -1 | not decided yet                         |
    # |     0 | cell is too large, i.e., m is too small |
    # |     1 | cell is too small, i.e., m is too large |
    big_cell = np.empty(num_vars - 1, dtype=int)

    # storing |Y| corr. to different m values
    Y_size_arr = np.empty(num_vars - 1, dtype=int)
    Y_size_arr.fill(-1)

    if m >= num_vars - 1:
        raise ValueError(f"m ({m}) should be smaller than {num_vars - 1}")

    big_cell.fill(-1)  # -1 means not initialized

    cbb = ConstrainedBranchAndBoundNaive(rules, ub, y, lmbd)

    # i = 0
    while True:
        # i += 1
        # if i <= 4:
        #     break

        logger.debug(f"current m = {m}")
        logger.debug("big_cell: {}".format(big_cell))
        # solve the problem with all constraints
        A_sub, t_sub = A[:m], t[:m]
        sol_iter = cbb.run(A_sub, t_sub)

        # obtain only the first `thresh` solutions in the random cell
        Y_bounded = itertools.islice(sol_iter, thresh)
        Y_bounded_size = len(list(Y_bounded))
        Y_size_arr[m] = Y_bounded_size

        if Y_bounded_size >= thresh:
            logger.debug(f"|Y| >= thresh ({Y_bounded_size} >= {thresh})")

            if m == num_vars - 2:
                # Q: this is a "failure" case by def of the algorithm, why?
                big_cell[m] = 1
                # m = m + 1  # assuming m (which is -1) is big enough (producing small enough partitionings)
                fill_array_until(big_cell, m, 1)
                logger.debug(f"m is as large as it can be, return {m} (m)")
                break
            elif big_cell[m + 1] == 0:
                big_cell[m] = 1
                fill_array_until(big_cell, m - 1, 1)
                logger.debug(
                    f"big_cell[{m+1}]={big_cell[m+1]}, return {m+1} (m+1)"
                )
                m = m + 1
                break

            fill_array_until(big_cell, m, 1)

            lo = m
            if np.abs(m - m_prev) < 3:
                m += 1
            elif 2 * m < num_vars:
                m *= 2
            else:
                m = int((hi + m) / 2)
        else:
            logger.debug(f"|Y| < thresh ({Y_bounded_size} < {thresh})")
            if m == 0:
                big_cell[m] = 0
                logger.debug(f"m is as small as it can be, thus {m}")
                break
            elif big_cell[m - 1] == 1:  # assuming m > 0
                logger.debug(
                    f"big_cell[{m-1}]={big_cell[m-1]}, return {m}"
                )
                big_cell[m] = 0
                fill_array_from(big_cell, m + 1, 0)
                break

            fill_array_from(big_cell, m, 0)

            hi = m
            if np.abs(m - m_prev) < 3:
                m -= 1
            else:
                m = int((m + lo) / 2)
        logger.debug("-" * 10)

    logger.debug(f"big_cell: {big_cell}")
    logger.debug(f"Y_size_arr: {Y_size_arr}")

    if return_full:
        return m, Y_size_arr[m], big_cell, Y_size_arr
    else:
        return m, Y_size_arr[m]
