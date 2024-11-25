from ortools.sat.python import cp_model

from bds import gf2
from bds.common import BoundedLinearExpression, CPVar
from bds.sat.bounded_sat import (
    BoundedPatternSATCallback,
    add_constraints_to_program,
    get_xor_constraints,
)
from bds.sat.printers import VarArraySolutionPrinter

from .fixtures import get_input_program_by_name, solver  # noqa


def test_basic(solver):
    program, I, T = get_input_program_by_name("toy-1")

    for limit in range(1, 5):
        cb = BoundedPatternSATCallback(I, T, limit)

        solver.Solve(program, cb)

        assert cb.solution_count == limit
        cb.reset()


class Test_get_xor_constraints:
    def test_all_eq(self):
        # all constraints are equality constraints
        num_vars = 5
        A = gf2.eye(num_vars)
        b = gf2.GF([1] * num_vars)

        model = cp_model.CpModel()
        variables = [model.NewBoolVar(f"V_{i}") for i in range(num_vars)]

        cst = get_xor_constraints(A, b, variables)

        assert len(cst) == A.shape[0]
        for i in range(num_vars):
            assert cst[i][0] == "eq"
            expr = cst[i][1]

            isinstance(expr, BoundedLinearExpression)
            assert expr.Expression() == variables[i]
            assert expr.Bounds() == [1, 1]

    def test_empty_rows_and_rref(self):
        A = gf2.GF(
            [[1, 0, 0], [1, 0, 0], [1, 0, 0]]  # should be ignored  # should be ignored
        )
        b = gf2.GF([0, 0, 0])

        model = cp_model.CpModel()
        variables = [model.NewBoolVar(f"V_{i}") for i in range(A.shape[1])]

        cst = get_xor_constraints(A, b, variables, use_rref=True)

        assert len(cst) == 1

    def test_mixed(self):
        # both eq and xor are present
        A = gf2.GF(
            [
                [1, 0, 0, 1, 1, 0],  # first xor
                [0, 1, 1, 0, 0, 0],  # second xor
                [0, 0, 0, 0, 0, 1],  # eq
            ]
        )
        b = gf2.GF([0, 1, 0])

        model = cp_model.CpModel()
        variables = [model.NewBoolVar(f"V_{i}") for i in range(A.shape[1])]

        cst = get_xor_constraints(A, b, variables)

        assert len(cst) == A.shape[0]

        # 1st constraint
        c0 = cst[0]
        assert c0[0] == "xor"
        assert len(c0[1]) == 4
        assert c0[1][-1] == 1
        for v in c0[1][:3]:
            assert isinstance(v, CPVar)

        c1 = cst[1]
        assert c1[0] == "xor"
        assert len(c1[1]) == 2
        for v in c1[1]:
            assert isinstance(v, CPVar)

        # 3rd constraint
        c2 = cst[2]
        assert c2[0] == "eq"
        isinstance(c2[1], BoundedLinearExpression)
        assert c2[1].Expression() == variables[5]
        assert c2[1].Bounds() == [0, 0]


class Test_add_constraints_to_program:
    def test_all_eq(self, solver):
        # all constraints are equality constraints
        num_vars = 5
        A = gf2.eye(num_vars)
        b = gf2.GF([1] * num_vars)

        program = cp_model.CpModel()
        variables = [program.NewBoolVar(f"V_{i}") for i in range(num_vars)]

        add_constraints_to_program(program, get_xor_constraints(A, b, variables))

        cb = VarArraySolutionPrinter(variables)
        solver.Solve(program, cb)

        assert cb.solution_count == 1
        assert cb.solutions[0] == (1,) * num_vars

    def test_mixed(self, solver):
        # constraints have both xor and eq
        A = gf2.GF([[1, 1, 0], [0, 0, 1]])  # first xor  # eq
        b = gf2.GF([0, 1])

        program = cp_model.CpModel()
        variables = [program.NewBoolVar(f"V_{i}") for i in range(A.shape[1])]

        add_constraints_to_program(program, get_xor_constraints(A, b, variables))

        cb = VarArraySolutionPrinter(variables)
        solver.Solve(program, cb)

        assert cb.solution_count == 2
        assert set(cb.solutions) == {(1, 1, 1), (0, 0, 1)}


class TestSolvingUnderXORConstraints:
    def test_case_1(self, solver):
        program, I, T = get_input_program_by_name("toy-1")

        # the constraints imply the following:
        # I_3 should be 1
        # I_1 and I_2 are either both present or both absent

        A = gf2.GF([[1, 1, 0], [0, 0, 1], [0, 0, 1]])

        b = gf2.GF([0, 1, 1])

        cst_list = get_xor_constraints(A, b, I, use_rref=True)
        add_constraints_to_program(program, cst_list)

        cb = BoundedPatternSATCallback(I, T, limit=-1, verbose=2)

        solver.Solve(program, cb)
        cb.solution_count == 2
        set(cb.solutions_found) == {(0, 1, 2), (2)}

    def test_case_2(self, solver):
        program, I, T = get_input_program_by_name("toy-1")

        # the constraints imply the following:
        # I_3 should be 1
        # either I_1 or I_2 is present
        A = gf2.GF([[1, 1, 0], [0, 0, 1], [0, 0, 1]])
        b = gf2.GF([1, 1, 1])

        cst_list = get_xor_constraints(A, b, I, use_rref=True)
        add_constraints_to_program(program, cst_list)

        cb = BoundedPatternSATCallback(I, T, limit=-1, verbose=2)

        solver.Solve(program, cb)
        cb.solution_count == 2
        set(cb.solutions_found) == {(0, 2), (1, 2)}
