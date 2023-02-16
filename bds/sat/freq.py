# SAT programs with frequency-related constraints
import numpy as np
from scipy import sparse as sp
from ortools.sat.python import cp_model
from typing import Tuple, List

from ..common import CPVarList


def construct_program(
        D: np.ndarray, freq_thresh: int,
        which: str,
) -> Tuple[cp_model.CpModel, CPVarList, CPVarList]:
    """
    given D the data matrix of shape num points x num features

    `which` specifies the operator to compare pattern frequency against freq_thresh
            which can be either ">=" or "<="

    returns a tuple of:

    - the formula/program
    - the set of feature variables (which is an independent set of the formula)
    - the set of data example variables
    """
    if which not in ('>=', '<='):
        raise ValueError(f'operation "{which}" is not supported')
    # ------- problem definition ----------
    # Create the CP-SAT model.
    num_pts, num_feats = D.shape

    model = cp_model.CpModel()

    # Declare our two primary variables.
    T = [model.NewBoolVar(f"T_{t}") for t in range(num_pts)]  # T for examples
    I = [model.NewBoolVar(f"I_{i}") for i in range(num_feats)]  # I for features

    # constraints to add
    # 1. coverage constraints
    for t in range(num_pts):
        absent_features = (D[t] == 0).nonzero()[0]
        num_matched_absent_features = sum(I[i] for i in absent_features)
        # add the following constraint:
        # T[t] == 1 iff num_matched_absent_features == 0
        model.Add(num_matched_absent_features == 0).OnlyEnforceIf(T[t])
        model.Add(num_matched_absent_features > 0).OnlyEnforceIf(T[t].Not())

    # 2. frequency constraint
    for i in range(num_feats):
        examples_with_feat_i = D[:, i].nonzero()[0]
        num_covered_examples = sum(T[t] for t in examples_with_feat_i)
        if which == '>=':
            model.Add(num_covered_examples >= freq_thresh).OnlyEnforceIf(I[i])
        else:
            model.Add(num_covered_examples <= freq_thresh).OnlyEnforceIf(I[i])

    # 3. the pattern is not empty
    # we can upperbound the pattern length as well
    model.Add(sum(I[i] for i in range(num_feats)) > 0)

    # Search for I values in increasing order.
    model.AddDecisionStrategy(I, cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE)

    return model, I, T
