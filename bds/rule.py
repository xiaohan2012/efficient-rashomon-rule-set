import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass

@dataclass
class Rule:
    """a rule"""
    name: str  # string representation of the rule, e.g., attr1 == 0 AND attr2 < 10.
    cardinality: int  # number of conditions in the rule
    ids: np.ndarray  # the indices of samples that are 1 in the truthtable
    truthtable: np.ndarray  # whether a sample evaluates to true for this rule, 1 bit per sample,

    @property
    def support(self):
        return self.ids.shape[0]

    
@dataclass
class RuleEntry:
    """a rule inside a rule set"""
    rule_id: int
    n_captured: int # Number of 1's in bit vector.
    captured: np.ndarray # a binary array indicating whether a sample is captured by he rule

@dataclass
class RuleSet:
    rule_entries: List[RuleEntry]

    @property
    def n_rules(self):
        return len(self.rule_entries)
    
    def __iter__(self):
        return (ent for ent in self.rule_entries)
