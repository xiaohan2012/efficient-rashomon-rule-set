import numpy as np

from .rule import Rule
from .utils import bin_random, randints


def generate_random_rule_list(num_pts: int, num_rules: int, rand_seed: int = None):
    """generate a list of `num_rules` random rules on a dataset with num_pts"""
    np.random.seed(1234)
    rand_seeds = randints(num_rules)
    return [
        Rule.random(id=i, num_pts=num_pts, random_seed=seed)
        for i, seed in zip(range(num_rules), rand_seeds)
    ]


def generate_random_rules_and_y(num_pts, num_rules, rand_seed: int = None):
    """generate a list of random rules and a random label vector"""
    random_rules = generate_random_rule_list(num_pts, num_rules, rand_seed)
    random_y = bin_random(num_pts)
    return random_rules, random_y
