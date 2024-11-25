from ortools.sat.python import cp_model

from ..common import Program


def copy_cpmodel(program: Program):
    program_cp = cp_model.CpModel()
    program_cp.CopyFrom(program)
    return program_cp
