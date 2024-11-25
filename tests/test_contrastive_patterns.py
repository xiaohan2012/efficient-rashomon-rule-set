from itertools import chain

import numpy as np
import pytest

from bds.sat.contrastive_patterns import (
    BoundedWeightSATCallback,
    ContrastPatternSolutionPrinter,
    construct_program,
)
from bds.sat.ground_truth import get_ground_truth_count
from bds.sat.max_freq import construct_max_freq_program
from bds.sat.min_freq import construct_min_freq_program

from .cset_fixtures import toy_Dn, toy_Dp  # noqa
from .fixtures import solver  # noqa


class TestProgramConstruction:
    def get_reference_output(self, Dp, Dn, min_pos_freq, max_neg_freq):
        min_freq_prog, I, T = construct_min_freq_program(Dp, min_pos_freq)
        cb_min_freq, _ = get_ground_truth_count(min_freq_prog, I, T, verbose=0)

        max_freq_prog, I, T = construct_max_freq_program(Dn, max_neg_freq)
        cb_max_freq, _ = get_ground_truth_count(max_freq_prog, I, T, verbose=2)

        return set(cb_max_freq.solutions_found).intersection(
            set(cb_min_freq.solutions_found)
        )

    @pytest.mark.parametrize(
        "min_pos_freq, max_neg_freq", [(2, 1), (0, 3), (2, 3), (1, 4), (10, 0)]
    )
    def test_num_patterns_1(self, toy_Dp, toy_Dn, solver, min_pos_freq, max_neg_freq):
        """the case num_patterns = 1"""
        prog, I, Tp, Tn = construct_program(
            toy_Dp,
            toy_Dn,
            num_patterns=1,
            min_pos_freq=min_pos_freq,
            max_neg_freq=max_neg_freq,
        )

        cb = ContrastPatternSolutionPrinter(I, Tp, Tn)
        solver.Solve(prog, cb)

        actual = set()
        for patterns in cb.solutions_found:
            assert len(patterns) == 1
            actual.add(next(iter(patterns)))

        expected = self.get_reference_output(toy_Dp, toy_Dn, min_pos_freq, max_neg_freq)

        assert actual == expected

    @pytest.mark.parametrize(
        "min_pos_freq, max_neg_freq", [(3, 2), (3, 1), (1, 2), (4, 4), (10, 0)]
    )
    def test_num_patterns_2(self, toy_Dp, toy_Dn, solver, min_pos_freq, max_neg_freq):
        prog, I, Tp, Tn = construct_program(
            toy_Dp,
            toy_Dn,
            num_patterns=2,
            min_pos_freq=min_pos_freq,
            max_neg_freq=max_neg_freq,
        )

        cb = ContrastPatternSolutionPrinter(I, Tp, Tn)
        solver.Solve(prog, cb)

        if min_pos_freq >= 10:
            assert cb.solution_count == 0
        else:
            for row in cb.solution_stat:
                assert row["stat"]["pos_freq"] >= min_pos_freq
                assert row["stat"]["neg_freq"] <= max_neg_freq

    @pytest.mark.parametrize(
        "min_pos_freq, max_neg_freq",
        [
            (3, 1),
            (1, 2),  # this fail, (0, 2), (1, 2) has equal TP, FP and lengths
            (2, 1),
            (4, 4),
        ],
    )
    def test_succinctness(self, toy_Dp, toy_Dn, solver, min_pos_freq, max_neg_freq):
        """redundant pattern sets should be removed"""
        num_patterns = 2
        prog, I, Tp, Tn = construct_program(
            toy_Dp,
            toy_Dn,
            num_patterns=num_patterns,
            min_pos_freq=min_pos_freq,
            max_neg_freq=max_neg_freq,
        )

        cb = ContrastPatternSolutionPrinter(I, Tp, Tn)
        solver.Solve(prog, cb)

        # no redundant patterns in each solution e.g., ((1, )) should not appear
        for patterns in cb.solutions_found:
            assert len(set(patterns)) == num_patterns

        # no redundant solutions
        num_solutions = cb.solution_count
        num_unique_solutions = len(set(cb.solutions_found))
        assert num_unique_solutions == num_solutions

    @pytest.mark.parametrize("min_pos_freq, max_neg_freq", [(3, 1), (1, 3), (0, 6)])
    @pytest.mark.parametrize(
        "feature_group_with_max_cardinality", [((0, 1), 1), ((0, 1, 2), 1), ((0, 2), 1)]
    )
    def test_succinctness(
        self,
        toy_Dp,
        toy_Dn,
        solver,
        min_pos_freq,
        max_neg_freq,
        feature_group_with_max_cardinality,
    ):
        """redundant pattern sets should be removed"""
        num_patterns = 2
        prog, I, Tp, Tn = construct_program(
            toy_Dp,
            toy_Dn,
            num_patterns=num_patterns,
            min_pos_freq=min_pos_freq,
            max_neg_freq=max_neg_freq,
            feature_groups_with_max_cardinality=[feature_group_with_max_cardinality],
        )

        cb = ContrastPatternSolutionPrinter(I, Tp, Tn)
        solver.Solve(prog, cb)
        for patterns in cb.solutions_found:
            for p in patterns:
                feature_group = feature_group_with_max_cardinality[0]
                # features in feature_group cannot be present at the same time in p
                assert not set(feature_group).issubset(set(p))


def uniform_weight(*args, **kwargs):
    return 1


def weight_by_pos_support_size(pattern, pos_covered_examples, neg_covered_examples):
    return len(pos_covered_examples)


class TestBoundedWeightSATCallback:
    def setup_method(self, method):
        pass

    def get_inputs(self, Dp, Dn, k=1, min_pos_freq=0, max_neg_freq=3):
        prog, I, Tp, Tn = construct_program(
            Dp,
            Dn,
            num_patterns=k,
            min_pos_freq=min_pos_freq,
            max_neg_freq=max_neg_freq,
        )
        return prog, I, Tp, Tn

    @pytest.mark.parametrize(
        "pivot, overflow",
        chain(zip(np.arange(1, 7), [True] * 6), zip(np.arange(7, 12), [False] * 5)),
    )
    def test_uniform_weight(self, pivot, overflow, solver, toy_Dp, toy_Dn):
        prog, I, Tp, Tn = self.get_inputs(toy_Dp, toy_Dn)
        toy_Dp.shape[1]
        cb = BoundedWeightSATCallback(
            I,
            Tp,
            Tn,
            weight_func=uniform_weight,
            pivot=pivot,
            w_max=1.0,
            r=1.0,
        )
        solver.Solve(prog, cb)

        assert cb.w_min == 1.0
        assert (
            cb.overflows_w_total == overflow
        )  # the total weight should overflow the pivot
        assert len(cb.solutions_found) == cb.w_total == len(cb.weights)

        if overflow:
            assert cb.w_total == (cb.pivot + 1)

        assert (np.array(cb.weights) == 1.0).all()

    @pytest.mark.parametrize(
        "pivot, overflow",
        chain(zip(np.arange(1, 13), [True] * 12), zip(np.arange(13, 16), [False] * 3)),
    )
    def test_weight_by_pos_covereage(self, pivot, overflow, solver, toy_Dp, toy_Dn):
        ws = np.array([1, 2, 1, 3, 1, 3, 2])
        ws_cumsum = np.cumsum(ws)

        prog, I, Tp, Tn = self.get_inputs(toy_Dp, toy_Dn)
        toy_Dp.shape[1]
        cb = BoundedWeightSATCallback(
            I,
            Tp,
            Tn,
            weight_func=weight_by_pos_support_size,
            pivot=pivot,
            w_max=1.0,
            r=1.0,
        )
        solver.Solve(prog, cb)

        assert cb.w_min == 1.0
        assert cb.w_total == ws_cumsum[len(cb.solutions_found) - 1]
        assert cb.overflows_w_total == overflow

        np.testing.assert_array_equal(cb.weights, ws[: len(cb.weights)])
