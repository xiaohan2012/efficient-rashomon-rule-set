# SAT programs with minimum frequency constraints

import numpy as np
from ortools.sat.python import cp_model
from typing import Tuple

from ..common import CPVarList
from .freq import construct_program as construct_program_base


def construct_min_freq_program(
        D: np.ndarray, min_freq_thresh: int,
) -> Tuple[cp_model.CpModel, CPVarList, CPVarList]:
    return construct_program_base(D, min_freq_thresh, '>=')
