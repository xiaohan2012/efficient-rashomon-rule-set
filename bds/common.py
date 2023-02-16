import ortools
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import BoundedLinearExpression

from matplotlib import pyplot as plt

from typing import List, Tuple, Union

# variable type of constraint programing in ortools
CPVar = ortools.sat.python.cp_model.IntVar
CPVarList = List[CPVar]
CPVarList2D = List[List[CPVar]]
Program = cp_model.CpModel

ConstraintInfo = Tuple[str, Union[BoundedLinearExpression, List[Union[CPVar, int]]]]

Solver = cp_model.CpSolver

# output of the solver
Pattern = Tuple[int]
PatternSet = Tuple[Pattern]

# plotting related
CMAP = plt.cm.coolwarm

PATTERN_LOG_LEVEL = 5
