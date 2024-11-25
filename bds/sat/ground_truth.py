from typing import Callable

from .bounded_sat import BoundedPatternSATCallback
from .bounded_weight_sat import WeightedPatternSATCallback
from .solver import construct_solver
from .utils import copy_cpmodel


def get_ground_truth_count(program, I, T, verbose: int = 0):
    program_cp = copy_cpmodel(program)
    solver = construct_solver()
    cb = BoundedPatternSATCallback(I, T, -1, verbose=verbose)
    solver.Solve(program_cp, cb)
    return cb, cb.solution_count


def get_ground_truth_total_weight(program, I, T, weight_func: Callable):
    program_cp = copy_cpmodel(program)
    solver = construct_solver()
    cb = WeightedPatternSATCallback(I, T, weight_func=weight_func, save_stat=True)
    solver.Solve(program_cp, cb)
    return cb, cb.total_weight
