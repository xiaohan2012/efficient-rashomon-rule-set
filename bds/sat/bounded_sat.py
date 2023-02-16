from ortools.sat.python import cp_model
from typing import List, Tuple, Any, Union

from ..gf2 import GF, rref
from ..common import CPVar, Program, BoundedLinearExpression, ConstraintInfo


class BoundedPatternSATCallback(cp_model.CpSolverSolutionCallback):
    """a class which collects at most a number of patterns (the solutions in a SAT program)"""

    def __init__(
        self,
        pattern_variables: List[CPVar],
        coverage_variables: List[CPVar],
        limit: int,
        verbose: int = 0,
        save_stat: bool = True,
    ):
        """pattern_variables: one for each feature, whether the feature is selected or not
        coverage_variables: one for each transaction, whether the transaction is covered or not

        if limit < 0, all solutions are printed
        if verbose > 0: stopping info is printed
        if verbose > 1: each found solution is printed
        """
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__pattern_variabels = pattern_variables
        self.__coverage_variables = coverage_variables
        self.__solution_limit = limit
        self.verbose = verbose
        self.save_stat = save_stat

        self.reset()

    def reset(self):
        self.__solution_count = 0
        self.__solutions = []
        self.__solution_stat = {}

    def on_solution_callback(self):
        self.__solution_count += 1

        pattern = tuple(
            i for i, v in enumerate(self.__pattern_variabels) if self.Value(v) == 1
        )
        self.__solutions.append(pattern)

        covered_examples = tuple(
            t for t, v in enumerate(self.__coverage_variables) if self.Value(v) == 1
        )
        if self.verbose > 1:
            print(
                f"pattern: {pattern}, frequency: {len(covered_examples)}, covered examples: {covered_examples}",
                end=" ",
            )

            if self.save_stat:
                self.__solution_stat[pattern] = {
                    "frequency": len(covered_examples),
                    "covered_examples": covered_examples,
                }
            print()

        if (
                self.__solution_limit >= 0
                and self.__solution_count >= self.__solution_limit
        ):
            if self.verbose > 0:
                print("BoundedPatternSATCallback: stop search after finding %i solutions" % self.__solution_limit)
            self.StopSearch()

    @property
    def solution_count(self):
        return self.__solution_count

    @property
    def solutions_found(self):
        return self.__solutions

    @property
    def solution_stat(self):
        return self.__solution_stat



def get_xor_constraints(A: GF, b: GF, var_list: List[CPVar], use_rref: bool = True, verbose: int = 0) -> List[ConstraintInfo]:
    """
    get XOR constraints applied to the variables specified in var_list

    the constraints are specified by:

    A: the right-hand side 2D matrix in GF2
    b: the assignment vector in GF2

    assignment of the variables (denoted by a column vector x) should satisfy the constraints Ax = b

    if use_rref is True, A is converted in its reduced row echelon form (rref), the constraints should look simpler than its counterpart

    return a list of constraints, where each constraint, corresponding to row of A,  is represented by a tuple of:

    - the constraint type (str): 'eq' or 'xor'
    - the constraint data: either an equality constraint or the list of variables for XOR constraints

    there are 4 types of rows in A:

    1. the row contains only 1 variable (therefore it is an assignment)
    2. the b value is 1, we just pass in the non-zero columns
    3. the b value is 0, we pass in another 1, in addition to the non-zero columns
    4. the row is empty (all zeros) -- no constraint to add
    """
    if use_rref:
        A, b = rref(A, b)
        
    constraints = []
    n = A.shape[0]

    for j in range(n):
        Aj = A[j, :]
        bj = int(b[j])

        # get the index of active variables
        nz_idx = Aj.nonzero()[0]
        nnz = len(nz_idx)  # the number of active variables

        if nnz == 0:  # no constraint to add
            continue
        elif nnz == 1:  # add assignment constraint
            if verbose > 0:
                print(f"new constraint: vars[{nz_idx[0]}] == {bj}")
            constraints.append(("eq", var_list[nz_idx[0]] == bj))
        else:
            var_subset = [var_list[idx] for idx in nz_idx]
            if bj == 1:
                if verbose > 0:
                    print(f"new constraint: sum of vars at {nz_idx} = 1")
            else:
                assert bj == 0
                if verbose > 0:
                    print(f"new constraint: sum of vars at {nz_idx} = 0")
                var_subset.append(1)

            constraints.append(("xor", var_subset))
    return constraints


def add_constraints_to_program(
    program: cp_model.CpModel, constraints: List[ConstraintInfo]
):
    """add constraints to given a program  **in place**
    the type of constraint is limited to 'eq' or 'xor'
    """
    for ctype, cst in constraints:
        if ctype == "eq":
            program.Add(cst)
        elif ctype == "xor":
            program.AddBoolXOr(cst)
        else:
            raise ValueError(f"{ctype}")
    return program
