import random
import ray
import numpy as np
import math
from logzero import logger
from typing import Callable, Optional
from tqdm import tqdm

from .weight_mc import weight_mc
from .utils import copy_cpmodel
from .solver import construct_solver
from .bounded_weight_sat import BoundedWeightedPatternSATCallback
from .bounded_sat import get_xor_constraints, add_constraints_to_program

from ..utils import int_ceil
from ..random_hash import generate_h_and_alpha
from ..common import CPVarList, Program, Solver
from ..ray_pbar import RayProgressBar


def get_eps(kappa):
    return (1 + kappa) * (7.55 + 0.29 / np.power((1 - kappa), 2)) - 1


def find_kappa(eps: float) -> float:
    """given eps, find kappa using binary search"""
    min_eps = 6.84
    if eps <= min_eps:
        raise ValueError(f"eps must be at least {min_eps}, but is {eps}")

    lo, hi = 0.0, 1.0
    while True:
        kappa = np.mean([lo, hi])
        cur_eps = get_eps(kappa)
        if np.abs(cur_eps - eps) <= 1e-13:
            return kappa
        elif cur_eps > eps:
            hi = kappa
        else:
            lo = kappa


def get_pivot(kappa: float) -> float:
    return math.ceil(4.03 * np.power(1 + 1 / kappa, 2))


def compute_kappa_and_pivot(eps: float):
    kappa = find_kappa(eps)
    pivot = get_pivot(kappa)
    return kappa, pivot


def _sample_once(
    program: Program,
    S: CPVarList,
    make_callback: Callable,
    q: int,
    weight_func: Callable,
    pivot: float,
    hi_thresh: float,
    lo_thresh: float,
    w_max: float,
    r: float,
    rand_seed: Optional[int] = None,
    solver: Optional[Solver] = None,
    verbose: bool = True,
):
    def log_debug(msg):
        if verbose:
            logger.debug(msg)

    if solver is None:
        solver = construct_solver()

    i = max(q - 4, 0)
    log_debug(f"initialize i={i}")

    n = len(S)
    A, b = generate_h_and_alpha(n, q, seed=rand_seed)

    cst_list = get_xor_constraints(A, b, S)

    w_max = w_max  # use the w_max obtained during pre-solve
    while True:
        i += 1
        log_debug(f"i={i}")

        # get a new callback
        cb = make_callback(weight_func=weight_func, pivot=hi_thresh, w_max=w_max, r=r)

        cb.w_max = w_max
        cb.reset()

        # solve the modified program
        program_cp = copy_cpmodel(program)
        add_constraints_to_program(program_cp, cst_list[:i])
        solver.Solve(program_cp, cb)

        w_max = cb.new_w_max  # update w_max
        normalized_w_total = cb.w_total / w_max

        within_bound = lo_thresh <= normalized_w_total <= hi_thresh
        max_iter_reached = (i == q)
        if within_bound or max_iter_reached:
            if within_bound:
                log_debug(
                    f"within bound: {lo_thresh} <= {normalized_w_total} <= {hi_thresh}"
                )
            if max_iter_reached:
                log_debug(f"i = q = {i}")
            break
    if not within_bound:
        log_debug("sampling failed because w_total is out of bound")
        log_debug(f"normalized_w_total = {normalized_w_total}")
        log_debug(f"hi_thresh = {hi_thresh}")
        log_debug(f"lo_thresh = {lo_thresh}")
        return None
    else:
        log_debug("sampling succeeded")
        idx_arr = np.arange(len(cb.solutions_found))
        idx = random.choices(idx_arr, weights=cb.weights)[0]
        sample = cb.solutions_found[idx]
        weight = cb.weights[idx]
        return sample, weight


class WeightGen:
    def __init__(
        self,
        weight_func: Callable,
        r: float,
        eps: float,
        verbose: bool = False,
        parallel: bool = True,
    ):
        """
        r: an upperbound on the tilt
        """
        self.solver = construct_solver()
        self.verbose = verbose
        self.weight_func = weight_func
        self.parallel = parallel

        self.r = r
        self.eps = eps

        self.make_callback = None  # value to be set in self.prepare

    def presolve(
        self, program: Program, S: CPVarList, pivot: float, w_max: float, r: float
    ):
        self.log_info(f"pre-solve the program with pivot={pivot}, w_max={w_max}, r={r}")
        cb = self.make_callback(self.weight_func, pivot, w_max, r)

        program_cp = copy_cpmodel(program)
        self.solver.Solve(program_cp, cb)

        self.log_info("pre-solving done")

        return cb

    def log_debug(self, msg):
        if self.verbose:
            logger.debug(msg)

    def log_info(self, msg):
        if self.verbose:
            logger.info(msg)

    def prepare(self, program: Program, S: CPVarList, make_callback: Callable):
        """prepare for sampling, e.g., computes a few numbers related to the actual sampling"""
        self.program, self.S = program, S
        self.make_callback = make_callback

        w_max = 1.0
        kappa, pivot = compute_kappa_and_pivot(self.eps)
        if self.verbose > 0:
            self.log_info(f"eps = {self.eps}")
            self.log_info(f"pivot = {pivot}, kappa = {kappa:.5f}")

        self.pivot = pivot

        sqrt2 = np.sqrt(2)
        self.hi_thresh = 1 + sqrt2 * (1 + kappa) * pivot
        self.lo_thresh = pivot / (sqrt2 * (1 + kappa))
        self.log_info(f"(lo_thresh, hi_thresh) = {self.lo_thresh, self.hi_thresh}")

        self.presolve_cb = self.presolve(program, S, pivot, w_max, self.r)

        if not self.presolve_cb.overflows_w_total:
            self.sample_directly = True
            self.log_info(
                "total weight of solutions is smaller than pivot, we can sample directly"
            )
        else:
            self.sample_directly = False
            self.log_info(
                "total weight of solutions is larger than pivot, we need to sample from the partitioned space"
            )
            self.C, self.w_max = weight_mc(
                program,
                S,
                self.make_callback,
                epsilon=0.8,
                delta=0.2,
                r=self.r,
                weight_func=self.weight_func,
                show_progress=True,
                solver=self.solver,
                parallel=self.parallel,
                verbose=self.verbose,
            )
            self.log_info(f"estimated total weight: {self.C}, and w_max: {self.w_max}")
            self.q = int_ceil(
                np.log2(self.C) - np.log2(self.w_max) + np.log2(1.8) - np.log2(pivot)
            )
            self.log_debug(f"upper bound of i: q = {self.q}")

    def sample_once(self, return_weight=True, rand_seed=None):
        """take one sample"""
        if self.sample_directly:
            # sample = random.choices(self.presolve_cb.solutions_found, weights=self.presolve_cb.weights)[0]
            idx_arr = np.arange(len(self.presolve_cb.solutions_found))
            idx = random.choices(idx_arr, weights=self.presolve_cb.weights)[0]
            sample = self.presolve_cb.solutions_found[idx]
            weight = self.presolve_cb.weights[idx]
        else:
            # print("self.pivot: ", self.pivot)
            sample, weight = _sample_once(
                self.program,
                self.S,
                self.make_callback,
                self.q,
                self.weight_func,
                self.pivot,
                self.hi_thresh,
                self.lo_thresh,
                self.w_max,
                self.r,
                rand_seed=rand_seed,
                solver=self.solver,
                verbose=self.verbose,
            )

        if return_weight:
            if sample is not None:
                return sample, float(weight)
            else:
                return None
        else:
            return sample

    def sample(
        self,
        k: int,
        return_weight=False,
        show_progress: bool = False,
    ):
        """take k samples"""
        iter_obj = range(k)

        if not self.sample_directly and self.parallel:
            # if sampling by partitioning and in parallel mode, we call _sample_once to sample, instead of the method sample_once
            # because ray.remote can only take in a function or a class
            _sample_once_job = ray.remote(_sample_once)
            promise = [
                _sample_once_job.remote(
                    self.program,
                    self.S,
                    self.make_callback,
                    self.q,
                    self.weight_func,
                    self.pivot,
                    self.hi_thresh,
                    self.lo_thresh,
                    self.w_max,
                    self.r,
                    rand_seed=None,
                    verbose=self.verbose,
                )
                for _ in iter_obj
            ]

            RayProgressBar.show(promise)
            ret = ray.get(promise)
            ret = list(filter(None, ret))
            if not return_weight:
                return [p for p, _ in ret]
            return ret
        else:
            # serial mode
            if show_progress:
                iter_obj = tqdm(iter_obj)
            ret = [self.sample_once(return_weight=True) for _ in iter_obj]

            ret = list(filter(None, ret))
            return ret
