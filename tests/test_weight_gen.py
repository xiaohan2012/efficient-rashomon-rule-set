import pytest
import operator
import numpy as np

from collections import Counter
from bds.sat.weight_gen import find_kappa, get_eps, WeightGen
from bds.sat.ground_truth import get_ground_truth_total_weight
from bds.sat.bounded_weight_sat import BoundedWeightedPatternSATCallback

from .test_bounded_weight_sat import uniform_weight, weight_by_support_size
from .fixtures import get_input_program_by_name

class Test_find_kappa:
    @pytest.mark.parametrize('expected_kappa', np.arange(0.01, 1, 0.01))
    def test_basic(self, expected_kappa):
        actual_kappa = find_kappa(get_eps(expected_kappa))
        assert np.isclose(expected_kappa, actual_kappa)

    @pytest.mark.parametrize('eps', [1, 5, 6.84])
    def test_wrong_input(self, eps):
        """eps should > 6.84"""
        with pytest.raises(ValueError):
            find_kappa(eps)

                

MIN_FREQ = 25
@pytest.fixture
def wg():
    return WeightGen(
        weight_func=weight_by_support_size,
        # r=3,
        r=100 / MIN_FREQ,
        verbose=True,
        eps=10
    )

@pytest.fixture
def truth():
    program, I, T = get_input_program_by_name(f'random-{MIN_FREQ}')
    cb, true_weight = get_ground_truth_total_weight(program, I, T, weight_func=weight_by_support_size)
    return cb, true_weight

class TestWeightGen:

    def setup_method(self, method):
        self.prog, self.I, self.T = get_input_program_by_name(f'random-{MIN_FREQ}')

        def make_callback(weight_func, pivot, w_max, r):
            return BoundedWeightedPatternSATCallback(
                self.I,
                self.T,
                weight_func=weight_func,
                pivot=pivot,
                w_max=w_max,
                r=r,
            )

        self.make_callback = make_callback
        
    @pytest.mark.parametrize('eps', [
        # the eps are chosen so that they test both branches of the if-else statement
        7,  # eps is small, resulting in smaller pivot than, total weight therefore we sample directly
        8, # eps is large, resulting in larger pivot than total weight, therefore we sampled from the partitioned space
    ])  
    def test_sample_once_runnable(self, wg, truth, eps):
        """here we only test the sampler runs"""
        _, true_weight = truth

        wg.eps = eps  # override eps
        wg.prepare(self.prog, self.I, self.make_callback)

        # check the if-else branch
        if wg.pivot > true_weight:
            assert wg.sample_directly
        else:
            assert not wg.sample_directly
            
        pattern, weight = wg.sample_once(return_weight=True)

        # the returned pattern is other a tuple (pattern) or None
        assert isinstance(pattern, tuple) or pattern is None
        assert isinstance(weight, float) or weight is None

    @pytest.mark.skip("a test which may fail due to sampling variation. it is written here for history tracking")
    def test_statistical_property(self, wg, truth):
        cb, true_weight = truth

        # get ground truth
        expected_proba_dict = {}
        for pattern, data in cb.solution_stat.items():
            expected_proba_dict[pattern] = data['w'] / true_weight
    
        expected_proba = np.array([expected_proba_dict[p] for p in sorted(expected_proba_dict.keys())])

        # get the sampler
        program, I, T = get_input_program_by_name(f'random-{MIN_FREQ}')
        
        eps = 8.0  # we sample approximately from the partitioned space
        wg.eps = eps
        wg.prepare(program, I, T)

        # sample
        num_samples = 1000
        patterns = wg.sample(num_samples, exclude_none=True, show_progress=False)

        counts = Counter(patterns)
        freq = np.array(list(sorted(counts.values())))
        actual_proba = np.array([counts[p] / num_samples for p in sorted(expected_proba_dict.keys())])

        # compare with the bounds
        lb, ub = expected_proba / (1 + eps), expected_proba * (1 + eps)
        np.testing.assert_array_compare(operator.__le__, lb, actual_proba)
        np.testing.assert_array_compare(operator.__le__, actual_proba, ub)
