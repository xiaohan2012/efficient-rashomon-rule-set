import numpy as np
from typing import Dict, Any
from numbers import Number
from bds.rule import Rule
from bds.utils import bin_random, randints


def assert_dict_allclose(actual: Dict[Any, Number], expected: [Any, Number]):
    """check the equivalence of two dicts, assuming the value field of both dict are numeric"""
    assert set(actual.keys()) == set(
        expected.keys()
    ), f"{set(actual.keys())} != {set(expected.keys())}"

    for k in actual.keys():
        np.testing.assert_allclose(float(actual[k]), float(expected[k]))


def generate_random_rule_list(num_pts: int, num_rules: int, rand_seed: int = None):
    """generate a list of `num_rules` random rules on a dataset with num_pts"""
    np.random.seed(1234)
    rand_seeds = randints(num_rules)
    return [
        Rule.random(id=i + 1, num_pts=num_pts, random_seed=seed)
        for i, seed in zip(range(num_rules), rand_seeds)
    ]


def generate_random_rules_and_y(num_pts, num_rules, rand_seed: int = None):
    """generate a list of random rules and a random label vector"""
    random_rules = generate_random_rule_list(num_pts, num_rules, rand_seed)
    random_y = bin_random(num_pts)
    return random_rules, random_y


def assert_close_mpfr(v1, v2):
    np.testing.assert_allclose(float(v1), float(v2))
