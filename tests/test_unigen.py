import pytest
import numpy as np

from collections import Counter
from bds.sat.unigen import find_kappa, get_eps, UniGen
from bds.sat.ground_truth import get_ground_truth_count

from .fixtures import get_input_program_by_name


class Test_find_kappa:
    def test_basic(self):
        for expected_kappa in np.arange(0, 1, 0.01):
            actual_kappa = find_kappa(get_eps(expected_kappa))
            assert np.isclose(expected_kappa, actual_kappa)

    def test_corner_cases(self):
        """eps should >= 1.71"""
        for eps in [0.1, 0.5, 1.70]:
            with pytest.raises(ValueError):
                find_kappa(eps)


@pytest.fixture
def ug():
    return UniGen()


class TestUniGen:
    @pytest.mark.parametrize(
        "eps", [2.0, 5.0]
    )  # the eps values are chosen so that they test both branches of the if-else statement
    def test_sample_once_runnable(self, ug, eps):
        """here we only test the sampler runs"""
        prog, I, T = get_input_program_by_name("random-25")
        ug.prepare(prog, I, T, eps)
        ret = ug.sample_once()
        assert isinstance(ret, tuple)

    @pytest.mark.skip(
        "a test which may fail due to sampling variation. it is written here for history tracking"
    )
    def test_statistical_property(self, ug):
        """warning: this is not a very formal test because it ignores variation in probability estimation"""
        prog, I, T = get_input_program_by_name("random-25")
        eps = 3.05  # eps set this way to ensure we go to the else branch
        ug.prepare(prog, I, T, eps)

        truth_cb, true_count = get_ground_truth_count(prog, I, T)

        n_samples = 250
        samples = ug.sample(n_samples)

        counts = Counter(samples)
        freq = np.array(list(counts.values()))
        proba = freq / n_samples

        # all samples should be a member of the true solution set
        assert set(samples).issubset(set(truth_cb.solutions_found))

        # probability checking: they should be bounded from both sides
        true_proba = 1 / true_count
        lb, ub = true_proba / (1 + eps), true_proba * (1 + eps)

        assert (lb <= proba).all()
        assert (proba <= ub).all()
