import numpy as np
import gmpy2 as gmp
import pandas as pd

from gmpy2 import mpz, mpfr

from typing import Tuple, Optional, List, Iterable
from .cache_tree import CacheTree, Node
from .bounds_utils import *


#### TODO: test and eventually integrate within the bb.py loop() calling the function below
#### TODO2: (optionally) add more bounds


def incremental_update_lb(v: mpz, y: mpz, num_pts: mpz) -> mpfr:
    """
    return the incremental false positive fraction for a given rule

    v: a bit vector indicating which points are captured
    y: a bit vector of the true labels
    num_pts: length of the bit vectors, which is the total number of points
    """
    n = gmp.popcount(v)  # number of predicted positive
    w = v & y  # true positives

    t = gmp.popcount(w)
    return (n - t) / num_pts


def incremental_update_obj(u: mpz, v: mpz, y: mpz, num_pts: mpz) -> Tuple[mpfr, mpz]:
    """
    return the incremental false negative fraction for a rule set (prefix + current rule)
    and the indicator vector of false negatives

    u: points not captured by the prefix
    v: points captured by the current rule (in the context of the prefix)
    y: true labels
    num_pts: the total number
    """
    f = u & (~v)  # points not captured by both prefix and the rule
    g = f & y  # false negatives
    return gmp.popcount(g) / num_pts, f


def rule_set_size_bound_with_default(
    parent_node: Node, lmbd: float, current_optimal_objective: float, alpha: float
):
    """
    Simple pruning condition according to solely rule-set size.
    Important: this pruning condition assumes the minority class is the class of positives (labelled with 1). Otherwise,
    re-labelling is needed to use this bound in this form

    Parameters
    ----------
    parent_node : Node

    lmbd : float
       penalty parameter

    current_optimal_objective: current optimal set objective

    alpha: Rashomon set confidence parameter

    Returns
    -------
    bool
        true:  prune all the children of parent_node , false: do not prune
    """

    ruleset_size = len(
        parent_node.get_ruleset_ids()
    )  # this should be the number of rules each set contains
    return ruleset_size > ((min(current_optimal_objective, 0.5) + alpha) / lmbd - 1)


def rule_set_size_bound(
    parent_node: Node, lmbd: float, current_optimal_objective: float, alpha: float
):
    """
    Simple pruning condition according to solely rule-set size.
    This pruning condition does not assume that the minority class is the class of positives and can always be used.

    Parameters
    ----------
    parent_node : Node

    lmbd : float
       penalty parameter

    current_optimal_objective: float
        current optimal set objective

    alpha: float
        Rashomon set confidence parameter

    Returns
    -------
    bool
        true: prune all the children of parent_node , false: do not prune
    """

    ruleset_size = len(
        parent_node.get_ruleset_ids()
    )  # this should be the number of rules each set contains
    return ruleset_size > ((current_optimal_objective + alpha) / lmbd - 1)


def equivalent_points_bounds(
    lb: float,
    lmbd: float,
    current_optimal_objective: float,
    alpha: float,
    not_captured: np.ndarray,
    X: np.ndarray,
    all_classes: dict,
):
    """
    Pruning condition according to hierarchical lower bound in the Rashomon set formulation

    Parameters
    ----------
    lb : float
       incrementally computed lower bound

    lmbd: float
        penalty parameter

    current_optimal_objective: float
          current optimal set objective

    alpha: float
          Rashomon set confidence parameter

    not_captured: np.ndarray/ bool
        data point not covered (nor by parent node neither by the extension with current child)

    X: np.ndarray
        training set attributes

    all_classes: dict
          all classes of euqivalent points / should be computed prior to b&b execution


    Returns
    -------
    bool
        true: prune all the children of parent_node , false: do not prune
    """

    #  comput minimum error in the uncovered/ not captured part due to th equivalence classes
    tot_not_captured_error_bound = 0
    for not_captured_data_point in not_captured:
        attrs = np.where(X[not_captured_data_point] == 1)[0]
        attr_str = "-".join(map(str, attrs))
        tot_not_captured_error_bound += all_classes[attr_str].minority_mistakes
    tot_not_captured_error_bound = (
        tot_not_captured_error_bound / X.shape[0]
    )  # normalize as usual for mistakes

    return lb + tot_not_captured_error_bound > (current_optimal_objective + alpha)


def prefix_specific_length_upperbound(
    prefix_lb: float, prefix_length: int, lmbd: float, ub: float
):
    """
    for a prefix `d` with lb=`prefix_lb` and length=`prefix_length`,
    determine the upper bound of the sizes of rulesets that extend from `d`

    Parameters
    ----------
    lb: lower bound of the objective on the current prefix

    perfix_length: length of the current prefix

    lmbd: the complexity penalty parameter

    ub: the maximum objective function value for a solution to be considered in the Rashomon set

    Returns
    -------
    the upper bound of the length of its extensions
    """
    return (prefix_length + np.floor((ub - prefix_lb) / lmbd))
