import numpy as np
import os
import pickle as pkl
import json
import math
import tempfile
from ortools.sat.python import cp_model
from common import Program


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__solutions = []

    def on_solution_callback(self):
        self.__solution_count += 1
        sol = []
        for v in self.__variables:
            sol.append(self.Value(v))
            print("%s = %i" % (v, self.Value(v)), end=", ")
        print()
        self.__solutions.append(tuple(sol))

    @property
    def solution_count(self):
        return self.__solution_count

    @property
    def solutions(self):
        return self.__solutions


class PatternSolutionPrinter(cp_model.CpSolverSolutionCallback):
    """print pattern solutions"""

    def __init__(self, pattern_variables, coverage_variables):
        """pattern_variables: one for each feature, whether the feature is selected or not
        coverage_variables: one for each transaction, whether the transaction is covered or not
        """
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__pattern_variables = pattern_variables
        self.__coverage_variables = coverage_variables
        self.__solution_count = 0

    def reset(self):
        self.__solution_count = 0
        self.__solutions = []

    def on_solution_callback(self):
        self.__solution_count += 1
        pattern = tuple(
            i for i, v in enumerate(self.__pattern_variables) if self.Value(v) == 1
        )
        self.__solutions.append(pattern)

        covered_examples = tuple(
            t for t, v in enumerate(self.__coverage_variables) if self.Value(v) == 1
        )
        print(
            f"pattern: {pattern}, frequency: {len(covered_examples)}, covered examples: {covered_examples}",
            end=" ",
        )
        print()

    @property
    def solution_count(self):
        return self.__solution_count

    @property
    def solutions_found(self):
        return self.__solutions




def randints(num, vmin=0, vmax=100000):
    return np.random.randint(vmin, vmax, num)


def copy_cpmodel(program: Program):
    program_cp = cp_model.CpModel()
    program_cp.CopyFrom(program)
    return program_cp


def int_floor(num: float):
    return int(math.floor(num))


def int_ceil(num: float):
    return int(math.ceil(num))


def flatten(stuff):
    """flatten an array"""
    return np.asarray(stuff).flatten()


def get_tempdir(prefix=None, suffix=None, dir=None):
    if dir is not None:
        makedir(dir, usedir=False)
    return tempfile.mkdtemp(prefix=prefix, suffix=suffix, dir=dir)


def makedir(d, usedir=True):
    if usedir:
        d = os.path.dirname(d)

    if not os.path.exists(d):
        os.makedirs(d)


def save_pickle(obj, path):
    return pkl.dump(obj, open(path, "wb"))


def save_file(string, path):
    with open(path, "w") as f:
        f.write(string)


def load_pickle(path):
    return pkl.load(open(path, "rb"))


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)
