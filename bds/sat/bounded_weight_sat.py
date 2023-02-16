import numpy as np

from ortools.sat.python import cp_model
from typing import List, Tuple, Any, Union, Callable, Optional
from logzero import logger

from ..common import CPVar, CPVarList, Program
from .utils import copy_cpmodel
from .solver import construct_solver
from ..random_hash import generate_h_and_alpha
from .bounded_sat import get_xor_constraints, add_constraints_to_program

PATTERN_LOG_LEVEL = 5


class WeightedPatternSATCallback(cp_model.CpSolverSolutionCallback):
    """a class which collects all satisfying patterns and stores information such as total weight w.r.t. to a weight function"""

    def __init__(
        self,
        pattern_variables: CPVarList,
        coverage_variables: CPVarList,
        weight_func: Callable,  # thr weight function
        save_stat: bool = False,
    ):
        """pattern_variables: one for each feature, whether the feature is selected or not
        coverage_variables: one for each transaction, whether the transaction is covered or not

        """
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__pattern_variabels = pattern_variables
        self.__coverage_variables = coverage_variables
        self.weight_func = weight_func
        self.save_stat = save_stat

        self.reset()

    def reset(self):
        self.Y = []
        self.__solution_stat = {}
        self.w_total = 0
        self.w_max = 0
        self.w_min = np.Inf

    def log_pattern(self, pattern, covered_examples, pattern_weight):
        logger.log(
            level=PATTERN_LOG_LEVEL,
            msg=f"|supp|={len(covered_examples)}, w={pattern_weight}, supp={covered_examples}, y={pattern}",
        )

        if self.save_stat:
            self.__solution_stat[pattern] = {
                "|supp|": len(covered_examples),
                "supp": covered_examples,
                "w": pattern_weight,
            }

    def on_solution_callback(self):
        y = tuple(
            i for i, v in enumerate(self.__pattern_variabels) if self.Value(v) == 1
        )
        covered_examples = tuple(
            t for t, v in enumerate(self.__coverage_variables) if self.Value(v) == 1
        )

        pattern_weight = self.weight_func(y, covered_examples)

        self.w_max = max(self.w_max, pattern_weight)
        self.w_min = min(self.w_min, pattern_weight)

        assert pattern_weight >= 0, "non-negative weights are assumed"

        self.Y.append(y)
        self.w_total += pattern_weight

        self.log_pattern(y, covered_examples, pattern_weight)

    @property
    def total_weight(self):
        return self.w_total

    @property
    def solutions_found(self):
        return self.Y

    @property
    def solution_stat(self):
        return self.__solution_stat


class BoundedWeightedPatternSATCallback(cp_model.CpSolverSolutionCallback):
    """
    the BoundedWeightSAT algorithm, which collects a number of patterns (the solutions in a SAT program) with total weight at most `pivot`
    """

    def __init__(
        self,
        pattern_variables: CPVarList,
        coverage_variables: CPVarList,
        weight_func: Callable,  # thr weight function
        pivot: float,  # the total weight upper bound (normalized)
        w_max: float,  # estimation of maximum of pattern weight
        r: float,  # upperbound of the tilt parameter
        save_stat: bool = True,
    ):
        """pattern_variables: one for each feature, whether the feature is selected or not
        coverage_variables: one for each transaction, whether the transaction is covered or not

        if limit < 0, all solutions are printed
        """
        assert (
            pivot > 0
        ), "pivot should be positive (how can cell sizes be non-positive?)"

        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__pattern_variabels = pattern_variables
        self.__coverage_variables = coverage_variables
        self.weight_func = weight_func
        self.save_stat = save_stat

        self.pivot = pivot
        self.w_max = w_max
        self.r = r
        self.reset()

    def reset(self):
        self.Y = []
        self.weights = []

        self.__solution_stat = {}

        self.w_total = 0
        self.w_min = self.w_max / self.r

        # a flag which indicates if the search procedure exits because the w_total overflows the limit
        # being False also means we cannot find any more solutions
        self.overflows_w_total = False
        logger.debug(f"reset: w_min={self.w_min}")

    def log_pattern(self, pattern, covered_examples, pattern_weight):
        logger.log(
            level=PATTERN_LOG_LEVEL,
            msg=f"|supp|={len(covered_examples)}, w={pattern_weight}, supp={covered_examples}, y={pattern}",
        )

        if self.save_stat:
            self.__solution_stat[pattern] = {
                "|supp|": len(covered_examples),
                "supp": covered_examples,
                "w": pattern_weight,
            }

    def on_solution_callback(self):
        y = tuple(
            i for i, v in enumerate(self.__pattern_variabels) if self.Value(v) == 1
        )
        covered_examples = tuple(
            t for t, v in enumerate(self.__coverage_variables) if self.Value(v) == 1
        )

        pattern_weight = self.weight_func(y, covered_examples)

        assert pattern_weight >= 0, "non-negative weights are assumed"

        self.Y.append(y)
        self.weights.append(pattern_weight)

        self.w_total += pattern_weight
        self.w_min = min(self.w_min, pattern_weight)

        self.log_pattern(y, covered_examples, pattern_weight)

        w_total_normalized = self.w_total / self.w_min / self.r
        if w_total_normalized > self.pivot:
            logger.debug(
                f"BoundedWeightPatternSATCallback: search stops after overflowing pivot: {w_total_normalized} > {self.pivot}, where the former = {self.w_total} (w_total) / {self.w_min} (w_min) / {self.r} (r)"
            )
            self.overflows_w_total = True

            self.StopSearch()

    @property
    def new_w_max(self):
        return self.w_min * self.r

    @property
    def result(self):
        if self.overflows_w_total:
            return (self.Y, self.new_w_max)
        else:
            raise RuntimeError(
                "no result is returned since self.overflows_w_total is False"
            )

    @property
    def solutions_found(self):
        return self.Y

    @property
    def solution_stat(self):
        return self.__solution_stat


def weight_mc_core(
    program: Program,
    S: CPVarList,
    make_callback: Callable,
    # I: CPVarList,
    # T: CPVarList,
    weight_func: Callable,
    pivot: float,
    r: float,
    w_max: float,
    solver: Optional = None,
    return_details: bool = False,
    rand_seed: int = 123,
    verbose: bool = False,
):
    """
    program: the constrained program to be solved
    S: the list of independent variables
    make_callback: a function that returns a CP solving callback
        the function takes weight_func, pivot, w_max, and r as inputs
    """
    n = len(S)
    if solver is None:
        solver = construct_solver()

    # pre-solve it
    if verbose:
        logger.debug(f"given w_max={w_max}, pivot={pivot}")

    cb = make_callback(weight_func=weight_func, pivot=pivot, w_max=w_max, r=r)

    if verbose:
        logger.debug("--- i=0: pre-solving ---")

    solver.Solve(program, cb)

    # the total weight of solutions is small enough, no need to partition the space
    if cb.w_total / cb.new_w_max <= pivot:
        if verbose:
            logger.debug(
                f"cb.w_total / cb.new_w_max <= pivot: {cb.w_total} / {cb.new_w_max} <= {pivot}"
            )
            logger.debug("exit immediately after pre-solving")
        res = (cb.w_total, cb.new_w_max)
        details = {
            "i": 0,
            "n": n,
            "w_total": cb.w_total,
            "max_iteration_reached": False,
        }
        if return_details:
            res += (details,)
            return res
        else:
            return res

    if verbose:
        logger.debug("calling WeightMCCore")
    # gradually decrease the cell size by introducing more constraints
    # until the cell size is close but **under** pivot
    for i in range(1, n + 1):
        if verbose:
            logger.debug(f"--- i={i} ---")
        # add random XOR constraints to the program
        A, b = generate_h_and_alpha(n, i, seed=rand_seed)

        # logger.debug('A.shape {}'.format(A.shape))
        # logger.debug('b.shape {}'.format(b.shape))
        cst_list = get_xor_constraints(A, b, S)

        program_cp = copy_cpmodel(program)
        add_constraints_to_program(program_cp, cst_list)

        # call BoundedWeightSAT on the modified program
        cb = make_callback(weight_func=weight_func, pivot=pivot, w_max=w_max, r=r)

        solver.Solve(program_cp, cb)

        w_max = cb.new_w_max
        w_total = cb.w_total

        if 0 < w_total / w_max <= pivot:
            if verbose:
                logger.debug(
                    f"found the cell size that is small enough: i={i}, w_total={w_total}, w_max={w_max}"
                )
            break

    if w_total / w_max > pivot or w_total == 0.0:
        if verbose:
            logger.debug("invalid sampled cell size, due to:")
        if w_total / w_max > pivot:
            if verbose:
                logger.debug(
                    "the smallest sampled cell size is bigger the pivot: {} > {}".format(
                        w_total / w_max, pivot
                    )
                )

        if w_total == 0.0:
            if verbose:
                logger.debug("w_total is zero (no solution found in the sampled cell)")

        res = (None, w_max)
    else:
        res = (w_total * np.power(2, i) / w_max, w_max)

    if verbose:
        logger.debug(f"return {res}")

    details = {
        "i": i,
        "n": n,
        "w_total": w_total,
        "smallest_cell_too_big": w_total / w_max > pivot,
        "cell_size_is_zero": w_total == 0.0,
        "max_iteration_reached": i == n,
    }

    if return_details:
        res += (details,)
        return res
    else:
        return res
