from ortools.sat.python import cp_model


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
