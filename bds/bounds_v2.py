import numpy as np
from gmpy2 import mpz

from .bounds_utils import *
from .cache_tree import Node
from .utils import mpz2bag

#### TODO: test and eventually integrate within the bb.py loop() calling the function below
#### TODO2: (optionally) add more bounds

"""
Here problem formulation is f(S^*) < \alpha, i.e., no  Rashomon set is explictitly considered
"""


def rule_set_size_bound_with_default(parent_node: Node, lmbd: float, alpha: float):
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

    # ruleset_size includes also "0" which is the root and not an actual rule

    return ruleset_size > (alpha / lmbd)


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

    # ruleset includes the default rule, which should be excluded from the following calculation
    K = prefix_length - 1  # parent rule set size

    return prefix_length > (K + np.ceil((ub - prefix_lb) / lmbd))


def update_equivalent_lower_bound(
    captured: mpz, data_points2rules: dict, all_classes: dict
):
    """
    captured: points captured by a rule and not by its parents
    data_points2rules: pre-computed dictionary (essentially inverted index for rule truthtables)
    all_classses: pre-computed equivalence classes

    """

    tot_update_bound = 0
    added = set()
    newly_captured_points = mpz2bag(captured)  # newly captured
    for cap in newly_captured_points:
        n = mpz_set_bits(gmp.mpz(), data_points2rules[cap])
        if n not in added:
            tot_update_bound += all_classes[n].minority_mistakes
            added.add(n)

    tot_update_bound = tot_update_bound / len(
        data_points2rules
    )  # normalize as usual for mistakes

    return tot_update_bound


def equivalent_points_bounds(lb: float, equivalent_lower_bound: float, alpha: float):
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

    data_points2rules : dict

    Returns
    -------
    bool
        true: prune all the children of parent_node , false: do not prune
    """

    return (lb + equivalent_lower_bound) > alpha
