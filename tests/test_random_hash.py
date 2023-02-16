import pytest
import numpy as np
from bds.random_hash import generate_h, generate_alpha, generate_h_and_alpha
from bds.gf2 import GF
from bds.utils import randints

class TestSimple:
    def test_generate_alpha(self):
        m = 10
        alpha = generate_alpha(m)
        assert type(alpha) == GF
        assert alpha.shape == (m, )

    def test_generate_h(self):
        n = 50
        m = 20
        A = generate_h(n, m)
        assert type(A) == GF
        assert A.shape == (m, n+1)

    @pytest.mark.parametrize('seed', randints(1))
    def test_generate_h_and_alpha(self, seed):
        n = 50
        m = 20
        A, alpha = generate_h_and_alpha(n, m, seed=seed)

        A_ref = generate_h(n, m, seed=seed)
        alpha_ref = generate_alpha(m, seed=seed)

        assert type(A) == GF
        assert type(alpha) == GF
        assert A.shape == (m, n)
        assert alpha.shape == (m, )

        np.testing.assert_equal(A_ref[:, 1:], A)
        np.testing.assert_equal(alpha_ref - A_ref[:, 0], alpha)
