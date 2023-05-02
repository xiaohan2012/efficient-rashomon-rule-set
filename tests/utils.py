import numpy as np
from typing import Dict, Any
from numbers import Number
from bds.rule import Rule
from bds.utils import bin_random


def assert_dict_allclose(actual: Dict[Any, Number], expected: [Any, Number]):
    """check the equivalence of two dicts, assuming the value field of both dict are numeric"""
    assert set(actual.keys()) == (expected.keys())

    for k in actual.keys():
        np.testing.assert_allclose(actual[k], expected[k])


def generate_random_rule_list(num_pts: int, num_rules: int):
    """generate a list of `num_rules` random rules on a dataset with num_pts"""
    return [Rule.random(id=i + 1, num_pts=num_pts) for i in range(num_rules)]


def generate_random_rules_and_y(num_pts, num_rules):
    """generate a list of random rules and a random label vector"""
    random_rules = generate_random_rule_list(num_pts, num_rules)
    random_y = bin_random(num_pts)
    return random_rules, random_y
