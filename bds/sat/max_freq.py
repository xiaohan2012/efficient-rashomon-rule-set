# SAT programs with maximum frequency constraints

import numpy as np
from ortools.sat.python import cp_model
from typing import Tuple

from ..common import CPVarList
from .freq import construct_program as construct_program_base


def construct_max_freq_program(
        D: np.ndarray, max_freq_thresh: int,
) -> Tuple[cp_model.CpModel, CPVarList, CPVarList]:
    return construct_program_base(D, max_freq_thresh, '<=')
