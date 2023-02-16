from .gf2 import GF
from typing import Tuple, Optional


def generate_h(n: int, m: int, seed: Optional[int] = None) -> GF:
    """
    generate a random XOR-based hash function, parametrized by a m x (n+1) matrix in GF2

    n: input dim
    m: output dim
    seed: random seed
    """
    assert n > 0
    assert m > 0
    return GF.Random((m, n + 1), seed=seed)


def generate_alpha(m: int, seed: Optional[int] = None) -> GF:
    """generate a random m-dimensional partition vector in GF2"""
    assert m > 0
    return GF.Random((m,), seed=seed)


def generate_h_and_alpha(n: int, m: int, seed: Optional[int] = None) -> Tuple[GF, GF]:
    """
    generate a random XOR-based hash function, parametrized by a m x n matrix in GF2

    and a random m-dimensional partition vector in GF2
    """
    A = generate_h(n, m, seed=seed)
    alpha = generate_alpha(m, seed=seed)

    A_new = A[:, 1:]  # remove the first column

    alpha_new = alpha - A[:, 0]

    return A_new, alpha_new
