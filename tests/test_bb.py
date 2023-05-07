import numpy as np
import pytest
import gmpy2 as gmp
from gmpy2 import mpz, mpfr

from bds.bb import BranchAndBoundNaive, incremental_update_lb, incremental_update_obj
from bds.common import EPSILON
from bds.utils import (
    bin_array,
    mpz_all_ones,
    mpz_clear_bits,
    mpz_set_bits,
    randints,
    solutions_to_dict,
)

from .fixtures import rules, y
from .utils import assert_dict_allclose, assert_close_mpfr


@pytest.mark.parametrize("seed", randints(5))
@pytest.mark.parametrize("num_fp", np.arange(6))
def test_incremental_update_obj(seed, num_fp):
    np.random.seed(seed)

    num_pts = 10

    # say we capture half of the points
    captured_idx = np.random.permutation(num_pts)[: int(num_pts / 2)]
    v = mpz_set_bits(mpz(), captured_idx)

    y = mpz_clear_bits(v, captured_idx[:num_fp])  # make `num_fp` mistakes
    true_inc_fp = num_fp / mpz(num_pts)
    actual = incremental_update_lb(v, y, mpz(num_pts))
    assert actual == true_inc_fp
    assert isinstance(actual, mpfr)


def test_incremental_update_lb():
    u = mpz_set_bits(mpz(), [1, 2, 5])  # points not captured by prefix
    v = mpz_set_bits(mpz(), [1, 4])  # captured by rule
    f = mpz_set_bits(mpz(), [2, 5])  # not captured by the rule and prefix
    y = mpz_set_bits(mpz(), [1, 2, 4, 5])  # the true labels
    num_pts = mpz(7)
    fn, actual_f = incremental_update_obj(u, v, y, num_pts)

    assert f == actual_f
    assert fn == (mpz(2) / 7)
    assert isinstance(fn, mpfr)


class TestBranchAndBoundNaive:
    def create_bb_object(self, rules, y, lmbd=0.1):
        return BranchAndBoundNaive(rules, ub=10, y=y, lmbd=lmbd)

    def test_bb_init(self, rules, y):
        bb = self.create_bb_object(rules, y)

        assert bb.num_train_pts == mpz(5)
        assert isinstance(bb.y, mpz)

        assert bb.default_rule_fnr == 0.4
        assert bb.rules == rules

    def test_prepare(self, rules, y):
        bb = self.create_bb_object(rules, y)
        bb.reset()

        # front and tree root are both accessible
        # and refer to the same node
        node, not_captured = bb.queue.front()
        assert node == bb.tree.root
        assert not_captured == mpz_all_ones(y.shape[0])
        assert gmp.popcount(not_captured) == bb.num_train_pts

    @pytest.mark.parametrize("lmbd", np.random.random(10))
    def test_loop_1st_infinite_ub(self, rules, y, lmbd):
        # we consider search one level down with the lower bound being infinite
        # thus, all singleton rulesets should be enumerated
        ub = float("inf")

        lmbd = mpfr(lmbd)
        # the first iteration of the branch and bound
        bb = BranchAndBoundNaive(rules, ub=ub, y=y, lmbd=lmbd)
        bb.reset()

        node, not_captured = bb.queue.pop()

        iter_obj = bb._loop(node, not_captured)
        feasible_ruleset_ids = list(iter_obj)  # evoke the generator

        assert len(feasible_ruleset_ids) == 3  # all singleton rulesets are feasible

        assert bb.queue.size == 3
        node_1, not_captured_1 = bb.queue.pop()
        node_2, not_captured_2 = bb.queue.pop()
        node_3, not_captured_3 = bb.queue.pop()

        for n in [node_1, node_2, node_3]:
            assert isinstance(n.objective, mpfr)
            assert isinstance(n.lower_bound, mpfr)

        assert node_1.rule_id == 2
        assert node_2.rule_id == 3
        assert node_3.rule_id == 1

        assert node_1.lower_bound == lmbd
        assert node_2.lower_bound == (mpfr(1) / 5 + lmbd)
        assert node_3.lower_bound == (mpfr(2) / 5 + lmbd)

        assert_close_mpfr(node_1.objective, lmbd)
        assert_close_mpfr(node_2.objective, (mpfr(1) / 5 + lmbd))
        assert_close_mpfr(node_3.objective, (mpfr(4) / 5 + lmbd))

        assert not_captured_1 == mpz("0b01011")
        assert not_captured_2 == mpz("0b01010")
        assert not_captured_3 == mpz("0b10101")

    @pytest.mark.parametrize(
        "ub, num_feasible_solutions, queue_size",
        [
            (0.1 + EPSILON, 1, 1),
            (1 / 5 + 0.1 + EPSILON, 2, 2),
            (2 / 5 + 0.1 + EPSILON, 2, 3),
            (4 / 5 + 0.1 + EPSILON, 3, 3),
        ],
    )
    def test_loop_1st_varying_ub(
        self, rules, y, ub, num_feasible_solutions, queue_size
    ):
        # we assume lmbd is fixed and try different upper bo
        lmbd = 0.1
        bb = BranchAndBoundNaive(rules, ub=ub, y=y, lmbd=lmbd)
        bb.reset()

        node, not_captured = bb.queue.pop()

        iter_obj = bb._loop(node, not_captured)
        feasible_ruleset_ids = list(iter_obj)  # evoke the generator

        assert len(feasible_ruleset_ids) == num_feasible_solutions

        assert bb.queue.size == queue_size

    def test_run_infinite_ub(self, rules, y):
        # we run the branch-and-bound with infinite upper bounds
        # all solutions should be returned as feasible
        ub = float("inf")
        lmbd = 0.1
        bb = BranchAndBoundNaive(rules, ub=ub, y=y, lmbd=lmbd)
        feasible_solutions = list(bb.run())

        assert len(feasible_solutions) == 7  #

        all_feasible_solutions_ordered_by_appearance = [
            {0, 1},
            {0, 2},
            {0, 3},
            {0, 2, 3},
            {0, 1, 2},
            {0, 1, 3},
            {0, 1, 2, 3},
        ]
        # the order of yielded solutions should be exactly the same
        assert feasible_solutions == all_feasible_solutions_ordered_by_appearance

    @pytest.mark.parametrize(
        "ub, num_feasible_solutions",
        [
            (0.1, 1),
            (0.3, 2),
            (0.4, 3),
            (0.6, 4),
            (0.8, 5),
            (0.9, 7),
        ],
    )
    def test_run_with_varying_ub(self, rules, y, ub, num_feasible_solutions):
        """
        assuming the lmbd is 0.1, the rulesets being sorted by the objective values is as follows

        | rule set     | objective |
        |--------------+-----------|
        | {0, 2}       |        .1 |
        | {0, 3}       |        .3 |
        | {0, 2, 3}    |        .4 |
        | {0, 1, 2}    |        .6 |
        | {0, 1, 3}    |        .8 |
        | {0, 1, 2, 3} |        .9 |
        | {0, 1}       |        .9 |
        """
        lmbd = 0.1
        bb = BranchAndBoundNaive(rules, ub=ub + EPSILON, y=y, lmbd=lmbd)
        feasible_solutions = list(bb.run())

        feasible_solutions = set(map(tuple, feasible_solutions))

        all_feasible_solutions_sorted_by_objective = [
            {0, 2},
            {0, 3},
            {0, 2, 3},
            {0, 1, 2},
            {0, 1, 3},
            {0, 1, 2, 3},
            {0, 1},
        ]

        # we compare by set not list since the yielding order may not be the same as ordering by objective
        assert feasible_solutions == set(
            map(
                tuple,
                all_feasible_solutions_sorted_by_objective[:num_feasible_solutions],
            )
        )

    def test_check_objective_calculation(self, rules, y):
        lmbd = 0.1
        ub = float("inf")
        bb = BranchAndBoundNaive(rules, ub=ub, y=y, lmbd=lmbd)
        feasible_solutions = list(bb.run(return_objective=True))

        actual = solutions_to_dict(feasible_solutions)
        expected = {
            (0, 2): 0.1,
            (0, 3): 0.3,
            (0, 2, 3): 0.4,
            (0, 1, 2): 0.6,
            (0, 1, 3): 0.8,
            (0, 1, 2, 3): 0.9,
            (0, 1): 0.9,
        }

        # compare the two dict
        assert_dict_allclose(actual, expected)
