import numpy as np
import gmpy2 as gmp
from gmpy2 import mpz
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
from functools import reduce
from .utils import bin_zeros, assert_binary_array, mpz_set_bits


@dataclass
class Rule:
    """a rule"""

    id: int
    name: str  # string representation of the rule, e.g., attr1 == 0 AND attr2 < 10.
    cardinality: int  # number of conditions in the rule
    truthtable: mpz  # whether a sample evaluates to true for this rule, 1 bit per sample,

    def __post_init__(self):
        """if parent is not None, 'bind' self and parent by upting children and depth accordingly"""
        assert isinstance(self.truthtable, mpz)

    @property
    def support(self):
        return gmp.popcount(self.truthtable)

    @classmethod
    def random(cls, id, num_pts: int, random_seed=None) -> "Rule":
        if random_seed is not None:
            np.random.seed(random_seed)

        rand_array = np.random.randint(0, 2, num_pts, dtype=bool)
        truthtable = mpz_set_bits(mpz(), rand_array.nonzero()[0])
        return Rule(
            id=id, name=f"random-rule-{id}", cardinality=1, truthtable=truthtable
        )

    def __eq__(self, other: "Rule") -> bool:
        assert isinstance(other, Rule)

        return (
            (self.id == other.id)
            and (self.cardinality == other.cardinality)
            and (self.name == other.name)
            and (self.truthtable == other.truthtable)
        )


def lor_of_truthtable(rules: List[Rule]) -> mpz:
    """take the logical OR of rules' truth tables"""
    bit_vec_list = [r.truthtable for r in rules]
    return reduce(lambda x, y: x | y, bit_vec_list, mpz())


# @dataclass
# class RuleEntry:
#     """a rule inside a rule set"""

#     rule_id: int
#     n_captured: int  # Number of 1's in bit vector.
#     captured: np.ndarray  # a binary array indicating whether a sample is captured by he rule


# @dataclass
# class RuleSet:
#     rule_entries: List[RuleEntry]

#     @property
#     def n_rules(self):
#         return len(self.rule_entries)

#     def __iter__(self):
#         return (ent for ent in self.rule_entries)


# # void rule_vand(VECTOR, VECTOR, VECTOR, int, int *);
# # void rule_vandnot(VECTOR, VECTOR, VECTOR, int, int *);
# # void rule_vor(VECTOR, VECTOR, VECTOR, int, int *);
# # void rule_not(VECTOR, VECTOR, int, int *);
