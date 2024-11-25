from itertools import chain, combinations
from numbers import Number
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np
from gmpy2 import mpz

from bds.gf2 import GF
from bds.rule_utils import generate_random_rules_and_y  # noqa
from bds.utils import calculate_obj, mpz_set_bits


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
                    print(f"sol: {sol}")
                    print(f"obj: {obj}")
                if obj <= ub:
                    yield (tuple(sorted(sol)), obj)


def normalize_solutions(sols: List[Set[int]]) -> Set[Tuple[int]]:
    return set(map(tuple, map(sorted, sols)))


def is_disjoint(left: Iterable, right: Iterable) -> bool:
    """determines if two iteratables  after converting to sets are disjoint"""
    return len(set(left) & set(right)) == 0


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
