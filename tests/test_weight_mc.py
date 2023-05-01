import pytest
import logzero


from bds.sat.weight_mc import weight_mc
from bds.sat.ground_truth import get_ground_truth_total_weight
from bds.sat.approx_mc2 import get_theoretical_bounds
from bds.sat.bounded_weight_sat import BoundedWeightedPatternSATCallback

from .fixtures import get_input_program_by_name, random_D_data
from .test_bounded_weight_sat import uniform_weight, weight_by_support_size


logzero.loglevel(
    logzero.INFO
)  # to avoid printing the debug info, will speed up the tests


class TestWeightMc_unweighted:
    @pytest.mark.parametrize("dataset_name", ["random-30", "random-25"])
    @pytest.mark.parametrize("eps", [0.1])
    @pytest.mark.parametrize("delta", [0.8])
    def test_if_estimate_within_bounds(self, dataset_name, eps, delta):
        program, I, T = get_input_program_by_name(dataset_name)

        def make_callback(weight_func, pivot, w_max, r):
            return BoundedWeightedPatternSATCallback(
                I,
                T,
                weight_func=weight_func,
                pivot=pivot,
                w_max=w_max,
                r=r,
            )

        _, true_weight = get_ground_truth_total_weight(
            program, I, T, weight_func=uniform_weight
        )
        lb, ub = get_theoretical_bounds(true_weight, eps)

        estimate, _ = weight_mc(
            program,
            I,
            make_callback,
            epsilon=eps,
            delta=delta,
            r=1.0,
            weight_func=uniform_weight,
            show_progress=False,
        )

        # strictly speaking, the test is wrong
        # because it does not consider that the assertion holds with probability at least 1 - delta
        # I admit that I'm lazy here
        assert lb <= estimate <= ub


class TestWeightMc_weighted:
    @pytest.mark.parametrize("min_freq", [25, 30])
    @pytest.mark.parametrize("eps", [0.1])
    @pytest.mark.parametrize("delta", [0.8])
    def test_if_estimate_within_bounds(self, min_freq, eps, delta):
        dataset_name = f"random-{min_freq}"
        max_n_pts = random_D_data.shape[0]

        r = max_n_pts / min_freq
        program, I, T = get_input_program_by_name(dataset_name)

        def make_callback(weight_func, pivot, w_max, r):
            return BoundedWeightedPatternSATCallback(
                I,
                T,
                weight_func=weight_func,
                pivot=pivot,
                w_max=w_max,
                r=r,
            )

        _, truth = get_ground_truth_total_weight(
            program, I, T, weight_func=weight_by_support_size
        )
        lb, ub = get_theoretical_bounds(truth, eps)

        estimate, _ = weight_mc(
            program,
            I,
            make_callback,
            epsilon=eps,
            delta=delta,
            r=r,
            weight_func=weight_by_support_size,
            show_progress=False,
        )

        # strictly speaking, the test may fail
        # because it does not consider that the assertion holds with probability at least 1 - delta
        # I admit that I'm lazy here
        assert lb <= estimate <= ub
