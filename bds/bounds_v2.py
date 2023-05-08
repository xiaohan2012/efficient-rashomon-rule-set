import numpy as np
import pandas as pd
from .cache_tree import CacheTree, Node
from .bounds_utils import *

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
    
    return ruleset_size > ((alpha/lmbd) - 1) # the -1 is because of the extra rule moving from the parent to its children


def equivalent_points_bounds(
    lb: float,
    lmbd: float,
    alpha: float,
    not_captured: mpz,
    X: np.array,
    data_points2rules: dict,
    all_classes: dict
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

    data_points2rules : dict 

    Returns
    -------
    bool
        true: prune all the children of parent_node , false: do not prune
    """



    
    #  comput minimum error in the uncovered/ not captured part due to th equivalence classes
    tot_not_captured_error_bound = 0
    added = set() 
    not_captured_points = not_captured.nonzero()[0]
    for not_captured_data_point in not_captured_points: 
       
            n = mpz_set_bits(gmp.mpz(), data_points2rules[not_captured_data_point]) 
            if n not in added: # this to make sure that we add only once for each not captured class
                tot_not_captured_error_bound += all_classes[n].minority_mistakes
                added.add(n)
        
    tot_not_captured_error_bound = (
        tot_not_captured_error_bound / X.shape[0]
    )  # normalize as usual for mistakes

    return lb + tot_not_captured_error_bound > alpha
