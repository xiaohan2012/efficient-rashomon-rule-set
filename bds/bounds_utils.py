import numpy as np
import pandas as pd
from .cache_tree import CacheTree, Node
from gmpy2 import mpz, mpfr
import gmpy2 as gmp 
from .utils import mpz_set_bits, mpz2bag
from typing import List
from .rule import Rule



class EquivalentPointClass:
    """a single class of points all having the same attributes"""

    def __init__(self, this_id):
        self.id = this_id
        self.total_positives = 0
        self.total_negatives = 0
        self.data_points = set()  # we also keep track of the points so we do not need to access rules anymore

    def update(self, idx, label):
        self.data_points.add(idx) 
        if label == 1:
            self.total_positives += 1
        else:
            self.total_negatives += 1

        self.minority_mistakes = min(
            self.total_negatives, self.total_positives
        )  # recall this is to be normalized


def find_equivalence_classes(y_train: np.ndarray, rules: List[Rule]):
    """ 
    Fimd equivalence classes of points having the same attributes but possibly different labels.
    This function is to be used once prior to branch-and-bound execution to exploit the equivalence-points-based bound.

    Parameters
    ----------
    X_trn : pd.DataFrame
       train data attribute matrix


    y_train :  np.ndarray
        labels

    rules: all rules 

    Returns
    -------
    all equivalnce classes of points all_classes
    """

    
    

    all_classes = dict()
    
    nPnt = len(y_train)
    
    print(nPnt)
    
    # find equivalence classes
    data_points2rules = [[] for _ in range(nPnt)] 
    
   
    for rule in rules: 
        covered = mpz2bag(rule.truthtable)  
        for cov in covered: 
            data_points2rules[cov].append(rule.id)            
            
            
    for i in range(nPnt): 
        n = mpz_set_bits(gmp.mpz(), data_points2rules[i]) 
        if n not in all_classes: 
            all_classes[n] = EquivalentPointClass(n)
        all_classes[n].update(i,y_train[i])
        
        
        
    # also compute the equivalent lower bound for the root node 
    tot_not_captured_error_bound_init = 0
    for equi_class in all_classes.keys():     
        tot_not_captured_error_bound_init += all_classes[equi_class].minority_mistakes
        
    tot_not_captured_error_bound_init = (
         tot_not_captured_error_bound_init / nPnt
    )  # normalize as usual for mistakes

    return tot_not_captured_error_bound_init, data_points2rules, all_classes
