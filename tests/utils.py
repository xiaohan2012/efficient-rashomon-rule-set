import numpy as np

import gmpy2 as gmp
from gmpy2 import mpz

from typing import Dict, Any, List, Iterable, Tuple, Set
from numbers import Number
from itertools import combinations

from bds.rule import Rule, lor_of_truthtable
from bds.utils import bin_random, randints, mpz_set_bits, calculate_obj
from bds.gf2 import GF


def assert_dict_allclose(actual: Dict[Any, Number], expected: [Any, Number]):
    """check the equivalence of two dicts, assuming the value field of both dict are numeric"""
    actual_keys = set(actual.keys())
    expected_keys = set(expected.keys())
    assert actual_keys == expected_keys, f"{actual_keys} != {expected_keys}"

    for k in actual.keys():
        np.testing.assert_allclose(
            np.array(actual[k], dtype=float),
            np.array(expected[k], dtype=float),
            err_msg=k,
        )


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


def assert_close_mpfr(v1, v2):
    np.testing.assert_allclose(float(v1), float(v2))


def brute_force_enumeration(
    rules: List, y: np.ndarray, A: np.ndarray, b: np.ndarray, ub: float, lmbd: float
) -> Iterable[Tuple[Tuple[int], float]]:
    """enumerate all feasible solutions in a brute-force way"""
    A_gf, b_gf = GF(A.astype(int)), GF(b.astype(int))
    y_mpz = mpz_set_bits(mpz(), y.nonzero()[0])
    num_rules = len(rules)
    all_rule_idxs = np.arange(num_rules)
    for size in range(1, num_rules + 1):
        for sol in combinations(all_rule_idxs, size):
            sol_vect = np.zeros(num_rules, dtype=int)
            sol_vect[np.asarray(sol)] = 1
            prod = A_gf @ GF(sol_vect)
            # if tuple(sorted(sol)) == (0, 3, 4):
            #     print("A_gf:\n {}".format(A_gf))
            #     print("b_gf:\n {}".format(b_gf))
            #     print("sol: {}".format(sol))
            #     print("prod == b_gf: {}".format(prod == b_gf))
            if (prod == b_gf).all():
                obj = calculate_obj(rules, y, y_mpz, sol, lmbd)
                # print("obj: {}".format(obj))
                # and obj is upper boudned by ub
                if sol in {(0, 2, 4, 9), (0, 2, 4, 8), (0, 4, 8, 9)}:
                    print("sol: {}".format(sol))
                    print("obj: {}".format(obj))
                if obj <= ub:
                    yield (tuple(sorted(sol)), obj)


def normalize_solutions(sols: List[Set[int]]) -> Set[Tuple[int]]:
    return set(map(tuple, map(sorted, sols)))
