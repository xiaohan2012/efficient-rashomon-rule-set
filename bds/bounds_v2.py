import numpy as np
import pandas as pd 
from .cache_tree import CacheTree, Node
from .bounds_utils import * 

#### TODO: test and eventually integrate within the bb.py loop() calling the function below 
#### TODO2: (optionally) add more bounds 

''' 
Here problem formulation is f(S^*) \leq \alpha, i.e., no  Rashomon set is explictitly considered 
'''

def rule_set_size_bound_with_default(parent_node: Node, lmbd: float, alpha: float): 
    '''
    
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
    '''
    
    ruleset_size = len( parent_node.get_ruleset_ids() ) #this should be the number of rules each set contains 
    return ruleset_size > ( alpha /lmbd - 1) 
    



def equivalent_points_bounds(lb: float, lmbd: float, alpha: float, 
                             not_captured: np.ndarray, X: np.ndarray, all_classes: dict): 
    '''
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
    '''

    #  comput minimum error in the uncovered/ not captured part due to th equivalence classes 
    tot_not_captured_error_bound = 0 
    for not_captured_data_point in not_captured: 
        attrs = np.where(X[not_captured_data_point] == 1)[0] 
        attr_str = "-".join(map(str, attrs))
        tot_not_captured_error_bound +=  all_classes[attr_str].minority_mistakes
    tot_not_captured_error_bound = tot_not_captured_error_bound / X.shape[0] # normalize as usual for mistakes 
        
    
    
    return lb + tot_not_captured_error_bound > alpha 


