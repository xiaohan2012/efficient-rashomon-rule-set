import numpy as np
import gmpy2 as gmp

from gmpy2 import mpz, mpfr

from typing import Tuple, Optional, List, Iterable, Dict
from .cache_tree import Node
from .rule import Rule
from .utils import mpz2bag, mpz_set_bits


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
    return prefix_length + np.floor((ub - prefix_lb) / lmbd)


class EquivalentPointClass:
    """a single class of points captured by the same set of rules"""

    def __init__(self, this_id: int):
        self.id = this_id
        self.total_positives = 0
        self.total_negatives = 0
        self.data_points = (
            set()
        )  # we also keep track of the points so we do not need to access rules anymore

    @property
    def valid_label_values(self):
        return {0, 1, True, False}

    def update(self, idx: int, label: int):
        """add one point of index idx with label to the equivalent point class"""
        if label not in self.valid_label_values:
            raise ValueError(
                f"invalid label {label}, use one of the following: {self.valid_label_values}"
            )

        if idx in self.data_points:
            # idx is added alreaady
            return
        
        self.data_points.add(idx)

        if label == 1 or label is True:
            self.total_positives += 1
        else:
            self.total_negatives += 1

    @property
    def minority_mistakes(self):
        return min(
            self.total_negatives, self.total_positives
        )  # recall this is to be normalized


def find_equivalence_points(
    y_train: np.ndarray, rules: List[Rule]
) -> Tuple[float, Dict, Dict[int, EquivalentPointClass]]:
    """
    Fimd equivalence classes of points having the same attributes but possibly different labels.
    This function is to be used once prior to branch-and-bound execution to exploit the equivalence-points-based bound.

    Parameters
    ----------
    y_train :  np.ndarray
        labels

    rules: all rules

    Returns
    -------
    all equivalent point classes
    """
    assert isinstance(y_train, np.ndarray), type(y_train)
    ep_classes = dict()  # equivalent point classes

    n_pts = len(y_train)
    
    # find equivalence classes
    pt2rules = [[] for _ in range(n_pts)]
    for rule in rules:
        covered = mpz2bag(rule.truthtable)
        for cov in covered:
            try:
                pt2rules[cov].append(rule.id)
            except IndexError:
                print("cov: {}".format(cov))
                raise IndexError

    for i in range(n_pts):
        n = mpz_set_bits(gmp.mpz(), pt2rules[i])
        if n not in ep_classes:
            ep_classes[n] = EquivalentPointClass(n)
        ep_classes[n].update(i, y_train[i])

    # also compute the equivalent lower bound for the root node
    tot_not_captured_error_bound_init = 0
    for equi_class in ep_classes.keys():
        tot_not_captured_error_bound_init += ep_classes[equi_class].minority_mistakes

    tot_not_captured_error_bound_init = (
        tot_not_captured_error_bound_init / n_pts
    )  # normalize as usual for mistakes

    return tot_not_captured_error_bound_init, pt2rules, ep_classes

def get_equivalent_point_lb(
    captured: mpz, pt2rules: Dict[int, List[int]], ep_classes: Dict[int, EquivalentPointClass]
) -> float:
    """
    get the equivalent point lower bound for a rule
    
    captured: indicator vector of points captured by the rule (and not by its parents)
    pt2rules: mapping from a point index to indices of rules that capture it (essentially inverted index for rule truthtables)
    ep_classses: pre-computed equivalence point classes
    """

    tot_update_bound = 0
    added = set()
    newly_captured_points = mpz2bag(captured)  # newly captured
    for cap in newly_captured_points:
        n = mpz_set_bits(gmp.mpz(), pt2rules[cap])
        if n not in added:
            tot_update_bound += ep_classes[n].minority_mistakes
            added.add(n)

    tot_update_bound = tot_update_bound / len(
        pt2rules
    )  # normalize as usual for mistakes

    return tot_update_bound
