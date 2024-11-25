import math
from typing import List, Optional, Tuple, Union

import numpy as np
from logzero import logger
from ortools.sat.python import cp_model
from tqdm import tqdm

from ..common import ConstraintInfo, CPVarList, Program, Solver
from ..random_hash import generate_h_and_alpha
from .bounded_sat import (
    BoundedPatternSATCallback,
    add_constraints_to_program,
    get_xor_constraints,
)
from .solver import construct_solver
from .utils import copy_cpmodel


def log_sat_search(
    program: Program,
    I: CPVarList,
    T: CPVarList,
    cst_list: List[ConstraintInfo],
    thresh: int,
    m_prev: int,
    solver: Solver,
    return_full: Optional[bool] = False,
    verbose: Optional[int] = 0,
) -> Union[Tuple[int, int], Tuple[int, int, np.ndarray, np.ndarray]]:
    """
    for a random XOR constraint system,
    find the correct number of constraints (m) such that the number of solutions under the constraint sub-system is right below `thresh`

    or m is the smallest number of constraints such that the solution number is below `thresh`]

    note that as m grows, the solution number decreases

    program: the pattern mining-based SAT program
    I: the list of feature variables
    T: the list of transactiion variables
    cst_list: a list of XOR constraints
    thresh: the maximum value of cell size
    m_prev: previous value of m (where 2^m is supposed to be a good estimate of the number of partitions)
    solver: the SAT solver to use
    return_full:
      is True if returning all information relevant to the call,
      otherwise return the m value and the corresponding |Y|
    """
    # TODO: cache the number of solutions and return the one corresponding to the m
    if thresh <= 1:
        raise ValueError("thresh should be greater than 1")

    lo, hi = 0, len(I) - 1
    m = m_prev

    big_cell = np.empty(len(I) - 1, dtype=int)

    # storing |Y| corr. to different m values
    Y_size_arr = np.empty(len(I) - 1, dtype=int)
    Y_size_arr.fill(-1)

    if m >= len(I) - 1:
        raise ValueError(f"m ({m}) should be smaller than {len(I) - 1}")

    big_cell.fill(-1)  # -1 means not initialized

    # Han: I shouldn't make any assumption on big_cell[0] (which could be 0) and big_cell[1] (which could be 1)
    # big_cell[0] = 1  # = 1 means the corresponding m produces too many solutions
    # # Q: why is it so? that the m in big_cell is too fine
    # big_cell[-1] = 0  # = 0 means the corresponding m produces too few solutions

    def fill_until(arr, idx, val):
        """fill the values in arr from index 0 to idx with value `val`"""
        for i in range(0, idx + 1):
            arr[i] = val

    def fill_from(arr, idx, val):
        """fill the values in arr from index idx until the end with value `val`"""
        for i in range(idx, len(arr)):  # typo in paper |S| -> |S| - 1
            arr[i] = val

    while True:
        if verbose > 0:
            print(f"current m = {m}")
        program_cp = cp_model.CpModel()
        program_cp.CopyFrom(program)

        # add first m constraints
        if verbose > 0:
            print(f"adding the first {m} XOR constraints")
        add_constraints_to_program(program_cp, cst_list[:m])

        cb = BoundedPatternSATCallback(
            I,
            T,
            limit=thresh,
            verbose=verbose,
        )

        solver.Solve(program_cp, cb)

        Y_size = cb.solution_count
        Y_size_arr[m] = Y_size

        if Y_size >= thresh:
            if verbose > 0:
                print(f"|Y| >= thresh ({Y_size} >= {thresh})")

            if m == len(I) - 2:
                # Q: this is a "failure" case by def of the algorithm, why?
                big_cell[m] = 1
                # m = m + 1  # assuming m (which is -1) is big enough (producing small enough partitionings)
                fill_until(big_cell, m, 1)
                if verbose > 0:
                    print(f"m is as large as it can be, return {m} (m)")
                break
            elif big_cell[m + 1] == 0:
                big_cell[m] = 1
                fill_until(big_cell, m - 1, 1)
                if verbose > 0:
                    print(
                        f"big_cell[m+1]={big_cell[m+1]} (which is 0), return {m+1} (m+1)"
                    )
                m = m + 1
                break

            fill_until(big_cell, m, 1)

            lo = m
            if np.abs(m - m_prev) < 3:
                m += 1
            elif 2 * m < len(I):
                m *= 2
            else:
                # m = (hi + m) / 2
                m = int((hi + m) / 2)
                # m = int(math.ceil((hi + m) / 2))
        else:
            if verbose > 0:
                print(f"|Y| < thresh ({Y_size} < {thresh})")
            if big_cell[m - 1] == 1:
                if verbose > 0:
                    print(f"big_cell[m-1]={big_cell[m-1]} (which is 1), return {m} (m)")
                big_cell[m] = 0
                fill_from(big_cell, m + 1, 0)
                break

            fill_from(big_cell, m, 0)

            hi = m
            if np.abs(m - m_prev) < 3:
                m -= 1
            else:
                # m = (m + lo) / 2
                m = int((m + lo) / 2)
        if verbose > 0:
            print("-" * 10)

    if verbose > 0:
        print("big_cell: ", big_cell)
        print("Y_size_arr: ", Y_size_arr)

    if return_full:
        return m, Y_size_arr[m], big_cell, Y_size_arr
    else:
        return m, Y_size_arr[m]


def approx_mc2_core(
    program: Program,
    I: CPVarList,
    T: CPVarList,
    thresh: float,
    prev_n_cells: int,
    rand_seed: int,
    verbose: int,
    use_rref: bool = False,
) -> Optional[Tuple[int, int]]:
    """count the number of solutions in a random cell of the solution space

    the "location" of the random cell is determined by random XOR/parity constraints
    """
    if use_rref:
        logger.warning(
            "use rref seems to give wrong results and make SAT solving even slower, are you sure?"
        )

    n = len(I)
    m = n - 1
    A, b = generate_h_and_alpha(n, m, seed=rand_seed)
    cst_list = get_xor_constraints(A, b, I, use_rref=use_rref, verbose=0)

    program_cp = copy_cpmodel(program)
    add_constraints_to_program(program_cp, cst_list)
    cb = BoundedPatternSATCallback(I, T, limit=thresh, verbose=0)

    solver = construct_solver()
    solver.Solve(program_cp, cb)
    Y_size = cb.solution_count

    if verbose > 0:
        print(
            "adding all XOR constraints: Y_size = {} and thresh = {}".format(
                Y_size, thresh
            )
        )

    if Y_size >= thresh:
        if verbose > 0:
            print("invalid input, return None")
        # the cell is too big, producing too many solutions
        return None, None
    else:
        if verbose > 0:
            print("goto search")
        m_prev = int(np.log2(prev_n_cells))
        m, Y_size = log_sat_search(
            program,
            I,
            T,
            cst_list,
            thresh,
            m_prev,
            solver,
            return_full=False,
            verbose=verbose,
        )
        return (2**m, Y_size)


def calculate_thresh(eps):
    return 1 + 9.84 * (1 + eps / (1 + eps)) * np.power(1 + 1 / eps, 2)


def calculate_t(delta):
    """t is the number of calls to approx_mc2_core"""
    # assert 0 < delta < 1
    return 17 * np.log2(3 / delta)


def approx_mc2(
    program: Program,
    I: CPVarList,
    T: CPVarList,
    eps: float = 0.2,
    delta: float = 0.5,
    verbose: int = 0,
    show_progress: bool = True,
) -> float:
    """the ApproxMC2 algorithm"""
    solver = construct_solver()

    thresh = calculate_thresh(eps)
    if verbose > 0:
        print(f"eps = {eps} gives thresh = {thresh:.2f}")

    prev_n_cells = 2  # m = 1

    program_cp = copy_cpmodel(program)

    cb = BoundedPatternSATCallback(I, T, thresh)

    solver.Solve(program_cp, cb)
    Y_size = cb.solution_count
    if verbose > 0:
        print(f"initial solving with such thresh gives |Y|={Y_size}")
    if Y_size < thresh:
        if verbose > 0:
            print(f"terminate since Y_size < thresh: {Y_size} < {thresh:.2f}")
    else:
        max_num_calls = int(math.ceil(calculate_t(delta)))

        if verbose > 0:
            print(f"max. num. of calls to ApproxMC2Core: {max_num_calls}")

        estimates = []

        iter_obj = range(max_num_calls)
        if show_progress:
            iter_obj = tqdm(iter_obj)
        for _ in iter_obj:
            # TODO: it can be parallelized
            n_cells, n_sols = approx_mc2_core(
                program,
                I,
                T,
                thresh,
                prev_n_cells=prev_n_cells,
                rand_seed=None,
                verbose=verbose,
                use_rref=False,
            )
            prev_n_cells = n_cells
            if n_cells is not None:
                estimates.append(n_cells * n_sols)
            else:
                if verbose > 0:
                    print("failed iteration")
        final_estimate = np.median(estimates)

        if verbose > 0:
            print(f"final estimate: {final_estimate}")

    return final_estimate


def get_theoretical_bounds(ground_truth: int, eps: float) -> Tuple[float, float]:
    r"""given the true count ground_truth, return the lower bound and upper bound of the estimate assuming \epsilon = eps"""
    return ground_truth / (1 + eps), ground_truth * (1 + eps)
