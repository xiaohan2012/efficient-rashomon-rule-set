import numpy as np

import gmpy2 as gmp
from gmpy2 import mpz

from typing import Dict, Any, List, Iterable, Tuple
from numbers import Number
from itertools import combinations

from bds.rule import Rule, lor_of_truthtable
from bds.utils import bin_random, randints, mpz_set_bits
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
            err_msg=k
        )


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


def calculate_obj(
    rules: List[Rule], y_np: np.ndarray, y_mpz: mpz, sol: Tuple[int], lmbd: float
) -> float:
    """calcuclate the objective for a given decision rule set (indicated by `sol`)
    by convention, `sol` is sorted and `0` appears first
    """
    # print("sol: {}".format(sol))
    ds_rules = [rules[i - 1] for i in sol[1:]]
    # print("ds_rules: {}".format(ds_rules))
    pred = lor_of_truthtable(ds_rules)
    # print("bin(pred): {}".format(bin(pred)))
    # print("bin(y_mpz): {}".format(bin(y_mpz)))
    num_mistakes = gmp.popcount(y_mpz ^ pred)
    print("num_mistakes: {}".format(num_mistakes))
    obj = len(sol[1:]) * lmbd + num_mistakes / y_np.shape[0]
    return float(obj)


def brute_force_enumeration(
    rules: List, y: np.ndarray, A: np.ndarray, b: np.ndarray, ub: float, lmbd: float
) -> Iterable[Tuple[Tuple[int], float]]:
    """enumerate all feasible solutions in a brute-force way"""
    A_gf, b_gf = GF(A.astype(int)), GF(b.astype(int))
    y_mpz = mpz_set_bits(mpz(), y.nonzero()[0])
    num_rules = len(rules)
    population = np.arange(1, num_rules + 1)
    for size in range(1, num_rules + 1):
        for sol in combinations(population, size):
            sol_arr = np.zeros(num_rules, dtype=int)
            sol_arr[np.asarray(sol) - 1] = 1
            prod = A_gf @ GF(sol_arr)
            # Ax=b is satisfied                
            if (prod == b_gf).all():
                sol = (0, ) + sol
                obj = calculate_obj(rules, y, y_mpz, sol, lmbd)
                # and obj is upper boudned by ub
                if obj <= ub:
                    yield (tuple(sorted(sol)), obj)
