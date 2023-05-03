import numpy as np
import math
import random
from tqdm import tqdm

from .utils import copy_cpmodel
from ..utils import int_ceil, int_floor
from .solver import construct_solver
from .approx_mc2 import approx_mc2

from ..random_hash import generate_h_and_alpha
from .bounded_sat import (
    get_xor_constraints,
    add_constraints_to_program,
    BoundedPatternSATCallback,
)


def get_eps(kappa):
    return (1 + kappa) * (2.23 + 0.48 / np.power((1 - kappa), 2)) - 1


def find_kappa(eps: float) -> float:
    """given eps, find kappa using binary search"""
    if eps < 1.71:
        raise ValueError(f"eps must be at least 1.71, but is {eps}")

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


def get_pivot(kappa: float) -> float:
    return math.ceil(3 * np.sqrt(np.e) * np.power(1 + 1 / kappa, 2))


def compute_kappa_and_pivot(eps: float):
    kappa = find_kappa(eps)
    pivot = get_pivot(kappa)
    return (kappa, pivot)


class UniGen:
    def __init__(self, verbose: int = 0):
        self.solver = construct_solver()
        self.verbose = verbose

    def presolve(self, program, I, T, thresh):
        program_cp = copy_cpmodel(program)
        cb = BoundedPatternSATCallback(I, T, thresh)

        self.solver.Solve(program_cp, cb)
        return cb

    def prepare(self, program, I, T, eps: float):
        """prepare for sampling, e.g., computes a few numbers related to the actual sampling"""
        self.program, self.I, self.T = program, I, T

        kappa, pivot = compute_kappa_and_pivot(eps)
        if self.verbose > 0:
            print(f"pivot = {pivot}, kappa = {kappa:.5f}")

        self.hi_thresh = int_ceil(
            1 + (1 + kappa) * pivot
        )  # can we round it? is it safe to round it?
        self.lo_thresh = int_floor(pivot / (1 + kappa))

        if self.verbose > 0:
            print(f"(lo_thresh, hi_thresh) = {self.lo_thresh, self.hi_thresh}")

        self.presolve_callback = self.presolve(program, I, T, self.hi_thresh)

        if self.verbose > 0:
            print(f"presolve gives |Y| = {self.presolve_callback.solution_count}")

        if self.presolve_callback.solution_count >= self.hi_thresh:
            # we will not return immediately
3            # so we estimate the number of solutions
            if self.verbose > 0:
                print(f"which is >= hi_thresh: {self.hi_thresh}")

            self.C = approx_mc2(
                program, I, T, eps=0.8, delta=0.8, verbose=0, show_progress=False
            )
            self.q = int_ceil(np.log2(self.C) + np.log2(1.8) - np.log2(pivot))

            if self.verbose > 0:
                print(f"count estimate: {self.C}")
                print(f"q: {self.q}")

    def sample_once(self):
        """take one sample"""
        if self.presolve_callback.solution_count < self.hi_thresh:
            if self.verbose > 0:
                print(f"which is < hi_thresh: {self.hi_thresh}")
                print("so terminate")
            return random.sample(self.presolve_callback.solutions_found, 1)[0]
        else:
            n = len(self.I)
            m = n - 1
            A, b = generate_h_and_alpha(n, m, seed=None)
            cst_list = get_xor_constraints(A, b, self.I, use_rref=False, verbose=0)

            if self.verbose > 0:
                print(f"searching i in [{max(0, self.q-4)}..{self.q}]")

            success = False
            for i in range(max(0, self.q - 4), self.q + 1):
                program_cp = copy_cpmodel(self.program)
                add_constraints_to_program(program_cp, cst_list[:i])
                cb = BoundedPatternSATCallback(
                    self.I, self.T, limit=self.hi_thresh, verbose=0
                )

                solver = construct_solver()
                solver.Solve(program_cp, cb)
                Y_size = cb.solution_count

                if self.lo_thresh <= Y_size <= self.hi_thresh:
                    if self.verbose > 0:
                        print(
                            f"i={i} gives lo_thresh <= Y_size <= hi_thresh ({self.lo_thresh} <= {Y_size} <= {self.hi_thresh})"
                        )
                    success = True
                    break
            if success:
                return random.sample(cb.solutions_found, 1)[0]
            else:
                return None

    def sample(self, k: int):
        """take k samples"""
        return [self.sample_once() for _ in tqdm(range(k))]
