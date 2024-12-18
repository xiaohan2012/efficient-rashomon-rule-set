import logging
import math
import random
from typing import List, Optional, Set, Tuple, Union

import gmpy2 as gmp
import numpy as np
import psutil
import ray
from contexttimer import Timer
from logzero import logger
from tqdm import tqdm

from .bb import BranchAndBoundNaive
from .cbb import ConstrainedBranchAndBound
from .random_hash import generate_h_and_alpha
from .ray_pbar import RayProgressBar
from .rule import Rule
from .utils import (
    assert_binary_array,
    fill_array_from,
    fill_array_until,
    int_ceil,
    int_floor,
    randints,
)

# logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)


def _check_log_search_trajectory(search_trajectory: List[Tuple[int, int, int]]):
    """check if the search trajectory of the log_search procedure is logical
    raise an error if it is not

    a trajectory consists of:

    - m: num. of constraints considered
    - |Y|: number of feasible solutions under m constraints
    - t: the threshold on the number of solutions

    the trajectory is logical is

    1. for each m which gives |Y| >= t, (m is too small)
      - all m' > m should not be tried in later search
    2. for each m which gives |Y| < t, (m is too large)
      - all m' < m should not be tried in later search
    """

    def _extract_later_ms(i: int) -> np.ndarray:
        """extract m values that are searched after the ith iteration"""
        return np.array(list(map(lambda tpl: tpl[0], search_trajectory[i + 1 :])))

    for i, (m, ys, t) in enumerate(search_trajectory):
        if ys < t:  # not enough solutions, m is large, we try smaller m later
            later_ms = _extract_later_ms(i)
            np.testing.assert_allclose(later_ms < m, True)
        else:  # m is small, we try larger m later
            later_ms = _extract_later_ms(i)
            np.testing.assert_allclose(later_ms > m, True)


# @profile
def log_search(
    rules: List[Rule],
    y: np.ndarray,
    lmbd: float,
    ub: float,
    A: np.ndarray,
    b: np.ndarray,
    thresh: int,
    m_prev: int,
    return_full: Optional[bool] = False,
) -> Union[Tuple[int, int], Tuple[int, int, np.ndarray, np.ndarray]]:
    """
    for a random XOR constraint system,
    find the correct number of constraints (m) such that the number of solutions under the constraint sub-system is right below `thresh`

    or m is the smallest number of constraints such that the solution number is below `thresh`

    note that as m grows, the solution number decreases

    rules: the list of candidate rules
    y: the label of each training point
    lmbd: model complexity parameter
    ub: upper bound on the objective value
    A and t: the parity constraint system
    thresh: the maximum value of cell size
    m_prev: previous value of m (where 2^m is supposed to be an estimate of the number of partitions)
    return_full:
      is True if returning all information relevant to the call,
      otherwise return the m value and the corresponding solution set size
    """
    # print("lmbd={}, ub={}, thresh={}, m_prev={}".format(lmbd, ub, thresh, m_prev))
    logger.debug(f"calling log_search with m_prev={m_prev} and thresh={thresh}")
    # TODO: cache the number of solutions and return the one corresponding to the m
    if thresh <= 1:
        raise ValueError("thresh should be at least 1")

    num_vars = A.shape[1]
    lo, hi = 0, num_vars - 1

    if m_prev >= num_vars:
        raise ValueError(f"m_prev ({m_prev}) should be smaller than {num_vars}")

    cur_m = m_prev

    # meaining of entry value in big_cell
    # | value | meaning                         |
    # |-------+---------------------------------|
    # |    -1 | not decided yet                 |
    # |     0 | cell is small, i.e., m is large |
    # |     1 | cell is large, i.e., m is small |
    # the final big_cell should look like
    #        m: low   ->   high
    # big_cell: 1 1 1 1 0 0 0 0
    #                 ^ m corresponding to this entry is used
    big_cell = np.empty(num_vars - 1, dtype=int)

    # storing |Y| corr. to different m values
    Y_size_arr = np.empty(num_vars - 1, dtype=int)
    Y_size_arr.fill(-1)

    big_cell.fill(-1)  # -1 means not initialized

    # cbb = ConstrainedBranchAndBoundNaive(rules, ub, y, lmbd)

    # latest_solver_status: SolverStatus = None
    # latest_usable_m: int = None
    # we store the list of m values that are tried
    # as well as the solution size and threshold
    search_trajectory = []

    time_cost_info = []

    # cbb = ConstrainedBranchAndBound(rules, ub, y, lmbd, reorder_columns=True)
    while True:
        logger.debug(
            "---- solve m = {}----".format(
                cur_m,
                # f"(based on {latest_usable_m})" if latest_usable_m else "from scratch",
            )
        )

        # obtain only the first `thresh` solutions in the random cell
        with Timer() as timer:
            # create a new instance of ICBB
            # to avoid performing column re-ordering multi times on the same ICBB instance
            cbb = ConstrainedBranchAndBound(rules, ub, y, lmbd, reorder_columns=True)
            Y_size = cbb.bounded_count(
                thresh,
                A=A[:cur_m],
                b=b[:cur_m],
                # , solver_status=latest_solver_status
            )
            # logger.debug(f"number of popped items: {cbb.status.queue.popped_count}")
            # logger.debug(f"number of pushed items: {cbb.status.queue.pushed_count}")

            logger.debug(f"solving takes {timer.elapsed:.2f} secs")
            time_cost_info.append(
                {
                    "m": cur_m,
                    "elapsed": timer.elapsed,
                    "popped_count": cbb.status.queue.popped_count,
                    "pushed_count": cbb.status.queue.pushed_count,
                }
            )

        Y_size_arr[cur_m] = Y_size

        search_trajectory.append((cur_m, Y_size, thresh))

        if Y_size >= thresh:
            # cell is large
            # in other words, not enough constraints, we increase m
            logger.debug(f"|Y| >= thresh ({Y_size} >= {thresh})")

            if cur_m == num_vars - 2:
                # Q: this is a "failure" case by def of the algorithm, why?
                big_cell[cur_m] = 1
                # m = m + 1  # assuming m (which is -1) is big enough (producing small enough partitionings)
                fill_array_until(big_cell, cur_m, 1)
                logger.debug(f"m is as large as it can be, return {cur_m} (m)")
                break
            elif big_cell[cur_m + 1] == 0:
                big_cell[cur_m] = 1
                fill_array_until(big_cell, cur_m - 1, 1)
                logger.debug(
                    f"big_cell[{cur_m+1}]={big_cell[cur_m+1]}, return {cur_m+1}"
                )
                cur_m = cur_m + 1
                # print("m (to return): {}".format(m))
                break

            fill_array_until(big_cell, cur_m, 1)

            lo = cur_m

            # # we only update the checkpoint when search lower bound is updated
            # logger.debug(f"using the solver status for m = {cur_m} as the latest")
            # latest_solver_status = cbb.status
            # latest_usable_m = cur_m

            if np.abs(cur_m - m_prev) < 3:
                cur_m += 1
            elif (2 * cur_m < num_vars) and big_cell[
                2 * cur_m
            ] == -1:  # 2 * m must be unexplored
                cur_m *= 2
            else:
                cur_m = int((hi + cur_m) / 2)
        else:
            # too many constraints, we decrease m
            logger.debug(f"|Y| < thresh ({Y_size} < {thresh})")
            if cur_m == 0:
                big_cell[cur_m] = 0
                logger.debug(f"m is as small as it can be, thus {cur_m}")
                break
            elif big_cell[cur_m - 1] == 1:  # assuming m > 0
                logger.debug(f"big_cell[{cur_m-1}]={big_cell[cur_m-1]}, return {cur_m}")
                big_cell[cur_m] = 0
                fill_array_from(big_cell, cur_m + 1, 0)
                break

            fill_array_from(big_cell, cur_m, 0)

            hi = cur_m
            if np.abs(cur_m - m_prev) < 3:
                cur_m -= 1
            else:
                cur_m = int((cur_m + lo) / 2)

        # logger.debug("big_cell: {}".format(big_cell))
        # logger.debug("Y_size_arr: {}".format(Y_size_arr))
        # logger.debug(f"lo: {lo}")
        # logger.debug(f"hi: {hi}")
        # logger.debug("\n")
    # logger.debug(f"big_cell: {big_cell}")
    # logger.debug(f"Y_size_arr: {Y_size_arr}")

    # to make sure that the search trajectory is logical
    _check_log_search_trajectory(search_trajectory)

    # print("time_cost_info: ")
    # for cur_m, etime in time_cost_info:
    #     print("|{}|{}|".format(cur_m, etime))
    if return_full:
        return (
            cur_m,
            Y_size_arr[cur_m],
            big_cell,
            Y_size_arr,
            search_trajectory,
            time_cost_info,
        )
    else:
        return cur_m, Y_size_arr[cur_m]


# @profile
def approx_mc2_core(
    rules: List[Rule],
    y: np.ndarray,
    lmbd: float,
    ub: float,
    thresh: int,
    prev_num_cells: int,
    rand_seed: Optional[int] = None,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
) -> Optional[Tuple[int, int]]:
    """
    a wrapper of log_search, which counts the number of solutions in a random cell of the solution space

    the "location" of the random cell is determined by random XOR/parity constraints

    rules: the list of candidate rules
    y: the label of each training point
    lmbd: model complexity parameter
    ub: upper bound on the objective value
    thresh: the maximum value of cell size
    prev_num_cells: previous an estimate of the number of partitions
    rand_seed: random seed

    the constraint system (specified by A and t) can be optinally provided, but usually for debugging purposes

    returns:

    - the total number of cells
    - the cell size which is closest to but no larger than thresh
    """
    assert_binary_array(y)
    # generate random constraints
    num_vars = len(rules)
    num_constraints = num_vars - 1
    if A is None or b is None:
        A, b = generate_h_and_alpha(
            num_vars, num_constraints, seed=rand_seed, as_numpy=True
        )

    # try to find at most thresh solutions using all constraints
    cbb = ConstrainedBranchAndBound(rules, ub, y, lmbd)
    logger.debug(f"initial solving under {A.shape[0]} constraints")
    with Timer() as timer:
        Y_size = cbb.bounded_count(thresh, A=A, b=b)
        logger.debug(f"solving takes {timer.elapsed:.2f} secs")

    # print("rand_seed: {}".format(rand_seed))
    if Y_size >= thresh:
        logger.debug(f"with |Y| {Y_size} >= {thresh}, therefore return None")
        return None, None
    else:
        logger.debug(
            f"with |Y| {Y_size} < {thresh}, therefore call log_search to find the appropriate number of constraints"
        )
        m_prev = int(np.log2(prev_num_cells))
        m, Y_size = log_search(
            rules, y, lmbd, ub, A, b, thresh, m_prev, return_full=False
        )

        return (int(np.power(2, m)), Y_size)


def _get_theoretical_bounds(ground_truth: int, eps: float) -> Tuple[float, float]:
    """given the true count ground_truth, return the lower bound and upper bound of the count estimate based on the accuracy parameter `eps`
    for debugging and testing purpose
    """
    return ground_truth / (1 + eps), ground_truth * (1 + eps)


def _calculate_thresh(eps: float) -> float:
    """calculate the cell size threshold using the accuracy parameter `eps`"""
    return 1 + 9.84 * (1 + eps / (1 + eps)) * np.power(1 + 1 / eps, 2)


def _calculate_t(delta: float) -> float:
    """calculate the number of calls to approx_mc2_core using the estimation confidence parameter"""
    assert 0 < delta < 1
    return 17 * np.log2(3 / delta)


def _check_input(rules: List[Rule], y: np.ndarray):
    for r in rules:
        assert (
            gmp.popcount(r.truthtable) <= y.shape[0]
        ), "number of captured points should be at most total number of points, however {} > {}".format(
            gmp.popcount(r.truthtable), y.shape[0]
        )


def approx_mc2(
    rules: List[Rule],
    y: np.ndarray,
    *,
    lmbd: float,
    ub: float,
    delta: float,
    eps: float,
    parallel: Optional[bool] = False,
    log_level: Optional[int] = logging.INFO,
    rand_seed: Optional[int] = None,
    show_progress: Optional[bool] = True,
    ncpus_per_job: Optional[int] = 2,
) -> int:
    """return an estimate of th number of feasible solutions to a decision set learning problem

    the learning problem is encoded by:

    rules: a list of rules
    y: the label vector
    lmbd: the model complexity parameter
    ub: the upper bound on the objective function

    the estimation quality is controlled by:

    delta: the estimation confidence parameter
    eps: the accuracy parameter

    show_progress: True if a progress bar is to be displayed
    parallel: True if run in paralle mode,
              note that in this case, the previous values of m cannot be reused by subsequent runs of approx_mc2_core
    log_level: the log level to use inside the Python process (only applicable in parallel mode)
    """
    _check_input(rules, y)

    logger.debug(f"calling approx_mc2 with eps = {eps:.2f} and delta={delta:.2f}")

    thresh = _calculate_thresh(eps)
    logger.debug(f"thresh = {thresh:.2f}")

    prev_num_cells = 2  # m = 1

    bb = BranchAndBoundNaive(rules, ub, y, lmbd)

    thresh_floor = int_floor(thresh)  # round down to integer

    # take at most thresh_floor solutions
    with Timer() as timer:
        Y_bounded_size = bb.bounded_count(thresh_floor)
        logger.debug(
            f"BB solving (thresh={thresh_floor}) takes {timer.elapsed:.2f} secs\nand gave {Y_bounded_size} solutions"
        )

    if Y_bounded_size < thresh_floor:
        logger.debug(
            f"exit since number of solutions {Y_bounded_size} < {thresh_floor}"
        )
        final_estimate = Y_bounded_size
    else:
        max_num_calls = int_floor(_calculate_t(delta))

        logger.debug(f"maximum number of calls to ApproxMC2Core: {max_num_calls}")

        np.random.seed(rand_seed)
        rand_seed_pool = randints(max_num_calls)

        if not parallel:
            estimates = []

            iter_obj = tqdm(
                range(max_num_calls) if show_progress else range(max_num_calls)
            )

            for trial_idx, rand_seed_next in zip(iter_obj, rand_seed_pool):
                with Timer() as timer:
                    num_cells, num_sols = approx_mc2_core(
                        rules,
                        y,
                        lmbd,
                        ub,
                        thresh_floor,
                        prev_num_cells=prev_num_cells,
                        rand_seed=rand_seed_next,
                    )
                    logger.debug(f"running approx_mc2_core takes {timer.elapsed:.2f}s")
                prev_num_cells = num_cells
                if num_cells is not None:
                    logger.debug(f"num_cells: {num_cells}, num_sols: {num_sols}")
                    estimates.append(num_cells * num_sols)
                else:
                    logger.debug("one esimtation failed")
        else:
            print(f"ray.available_resources(): {ray.available_resources()}")
            if "CPU" in ray.available_resources():
                num_available_cpus = int(ray.available_resources()["CPU"])
            else:
                num_available_cpus = psutil.cpu_count()
                logger.warning(
                    "Fall back to psutil.cpu_count() to detect the number of CPUs. Cannot use ray.available_resources()."
                )

            logger.info(f"number of available CPUs: {num_available_cpus}")

            num_cpus_per_job = ncpus_per_job
            num_jobs_in_first_round = int(num_available_cpus / num_cpus_per_job)

            @ray.remote(num_cpus=num_cpus_per_job)
            def approx_mc2_core_wrapper(log_level, *args, **kwargs):
                # reset the loglevel since the function runs in a separate process
                logger.setLevel(log_level)
                with Timer() as timer:
                    num_cells, num_sols = approx_mc2_core(*args, **kwargs)
                    logger.debug(f"running approx_mc2_core takes {timer.elapsed:.2f}s")
                if num_cells is not None:
                    return (num_cells, num_sols)
                else:
                    return None

            # we do two rounds of parallel execution
            # the first round uses prev_num_cells = 2
            # the second round uses prev_m_cells of the first round

            # the 1st round executes k jobs, where k = the number of available CPUs / 2
            promise_1st_round = [
                approx_mc2_core_wrapper.remote(
                    log_level,
                    rules,
                    y,
                    lmbd,
                    ub,
                    thresh_floor,
                    prev_num_cells,
                    # int(2**13),
                    seed,
                )
                for seed in rand_seed_pool[:num_jobs_in_first_round]
            ]

            logger.info(
                f"doing 1st round of parallel execution of {len(rand_seed_pool[:num_jobs_in_first_round])} jobs"
            )
            RayProgressBar.show(promise_1st_round)
            results_1st_round = ray.get(promise_1st_round)
            results_1st_round = list(filter(None, results_1st_round))

            # do the 2nd round, which reuses the values of prev_num_cells obtained from the 1st round
            # and finishes the remaining jobs
            prev_num_cells_1st_round = [num_cell for num_cell, _ in results_1st_round]
            promise_2nd_round = [
                approx_mc2_core_wrapper.remote(
                    log_level,
                    rules,
                    y,
                    lmbd,
                    ub,
                    thresh_floor,
                    # take a random prev_num_cells value from results in  the 1st round
                    prev_num_cells=random.choice(prev_num_cells_1st_round),
                    rand_seed=seed,
                )
                for seed in rand_seed_pool[num_jobs_in_first_round:]
            ]
            logger.info(
                f"doing 2nd round of parallel execution of {len(rand_seed_pool[num_jobs_in_first_round:])} jobs"
            )
            RayProgressBar.show(promise_2nd_round)
            results_2nd_round = ray.get(promise_2nd_round)
            results = list(filter(None, results_2nd_round)) + results_1st_round
            estimates = [num_cells * num_sols for num_cells, num_sols in results]

        final_estimate = np.median(estimates)

        logger.debug(f"final estimate: {final_estimate}")

    return int(final_estimate)


class UniGen:
    """an implementation of the UniGen algorithm for sampling decision sets"""

    def __init__(
        self,
        rules: List[Rule],
        y: np.ndarray,
        lmbd: float,
        ub: float,
        eps: float,
        rand_seed: Optional[int],
    ):
        """
        rules: the list of candidate rules
        y: the label of each training point
        lmbd: model complexity parameter
        ub: upper bound on the objective value
        eps: the epsilon parameter that controls the closeness between the sampled distribution and uniform distribution
            the smaller eps (less error), the higher sampling accuracy and slower sampling speed
        rand_seed: the random seed
        """
        assert_binary_array(y)

        self.rules = rules
        self.y = y
        self.lmbd = lmbd
        self.ub = ub
        self.eps = eps
        self.rand_seed = rand_seed

        self.num_vars = len(self.rules)

    def _create_bb(self):
        """create the branch-and-bound solvers for both complete and constrained enumeration"""
        return BranchAndBoundNaive(self.rules, self.ub, self.y, self.lmbd)

    def _create_cbb(self):
        return ConstrainedBranchAndBound(
            self.rules, self.ub, self.y, self.lmbd, reorder_columns=True
        )

    def _find_kappa(self, eps: float) -> float:
        """given eps, find kappa using binary search"""
        if eps < 1.71:
            raise ValueError(f"eps must be at least 1.71, but is {eps}")

        def get_eps(kappa) -> float:
            return (1 + kappa) * (2.23 + 0.48 / np.power((1 - kappa), 2)) - 1

        lo, hi = 0.0, 1.0
        while True:
            kappa = np.mean([lo, hi])
            cur_eps = get_eps(kappa)
            if np.abs(cur_eps - eps) <= 1e-15:
                return kappa
            elif cur_eps > eps:
                hi = kappa
            else:
                lo = kappa

    def _compute_kappa_and_pivot(self, eps: float) -> Tuple[float, float]:
        def get_pivot(kappa: float) -> float:
            return math.ceil(3 * np.sqrt(np.e) * np.power(1 + 1 / kappa, 2))

        kappa = self._find_kappa(eps)
        pivot = get_pivot(kappa)
        return (kappa, pivot)

    def presolve(self, thresh: float) -> Tuple[int, List[Set[int]]]:
        """enumerate at most thresh solutions
        returns the number of found solutions and the found solutions (a list of decision sets)
        """
        thresh_floor = int_floor(thresh)  # round down to integer

        # take at most thresh_floor solutions
        bb = self._create_bb()
        Y = bb.bounded_sols(thresh_floor)
        Y_size = len(Y)
        return Y_size, Y

    def prepare(self, parallel: bool = True, show_approx_mc2_progress: bool = False):
        """
        presolve the problem with upperbound and estimate the number of feasible solutions if needed
        """
        kappa, pivot = self._compute_kappa_and_pivot(self.eps)
        logger.debug(f"eps = {self.eps} -> pivot = {pivot}, kappa = {kappa:.5f}")

        self.hi_thresh = int_ceil(1 + (1 + kappa) * pivot)
        self.lo_thresh = int_floor(pivot / (1 + kappa))

        logger.debug(f"(lo_thresh, hi_thresh) = {self.lo_thresh, self.hi_thresh}")

        self.hi_thresh_rounded = int_floor(self.hi_thresh)
        self.presolve_Y_size, self.presolve_Y = self.presolve(self.hi_thresh)

        self.sample_directly = self.presolve_Y_size < self.hi_thresh
        if not self.sample_directly:
            logger.info(
                f"|Y| {self.presolve_Y_size} >= {self.hi_thresh}, thus calling approx_mc2"
            )
            self.C = approx_mc2(
                self.rules,
                self.y,
                lmbd=self.lmbd,
                ub=self.ub,
                delta=0.8,
                eps=0.8,
                rand_seed=self.rand_seed,
                parallel=parallel,
                show_progress=show_approx_mc2_progress,
            )
            self.q = int_ceil(np.log2(self.C) + np.log2(1.8) - np.log2(pivot))
            logger.debug(f"esimated C = {self.C}")
            logger.debug(f"q = {self.q}")
        else:
            logger.info(
                f"|Y| {self.presolve_Y_size} < {self.hi_thresh}, thus sample directly"
            )

    def sample_once(self) -> Optional[Set[int]]:
        """sample one feasible solution from the solution space"""
        if self.sample_directly:
            logger.debug(
                f"sample directly from presolve_Y (of size {self.presolve_Y_size})"
            )
            return random.sample(self.presolve_Y, 1)[0]
        else:
            m = self.num_vars - 1

            A, b = generate_h_and_alpha(
                self.num_vars,
                m,
                seed=None,  # TODO: set the seed to control randomness
                as_numpy=True,
            )

            logger.debug(f"searching in the range [{max(0, self.q-4)}, {self.q}]")

            solver_status = None
            success = False
            for m in range(max(0, self.q - 4), self.q + 1):
                logger.debug(f"current i = {m}")
                A_sub, b_sub = A[:m], b[:m]

                # obtain only the first `thresh` solutions in the random cell
                cbb = self._create_cbb()
                Y = cbb.bounded_sols(
                    self.hi_thresh_rounded,
                    A=A_sub,
                    b=b_sub,
                    solver_status=solver_status,
                )
                solver_status = cbb.status
                Y_size = len(Y)

                if self.lo_thresh <= Y_size <= self.hi_thresh:
                    logger.debug(
                        f"m={m} gives lt <= |Y| <= ht: {self.lo_thresh} <= {Y_size} <= {self.hi_thresh}"
                    )
                    success = True
                    break
            if success:
                return random.sample(Y, 1)[0]
            else:
                return None

    def sample(
        self, k: int, exclude_none: Optional[bool] = True, show_progress: bool = True
    ) -> List[Optional[set]]:
        """take k samples"""
        iter_obj = range(k)
        if show_progress:
            iter_obj = tqdm(iter_obj)
        raw_samples = [self.sample_once() for _ in iter_obj]
        if exclude_none:
            return list(filter(None, raw_samples))
        return raw_samples
