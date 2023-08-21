import numpy as np
import pytest
import gmpy2 as gmp
from gmpy2 import mpz, mpfr

from bds.bb import BranchAndBoundNaive
from bds.rule import Rule
from bds.common import EPSILON
from bds.utils import (
    bin_random,
    mpz_all_ones,
    solutions_to_dict,
)

from .fixtures import rules, y
from .utils import assert_dict_allclose, assert_close_mpfr


class TestBranchAndBoundNaive:
    def create_bb_object(self, rules, y, lmbd=0.1):
        return BranchAndBoundNaive(rules, ub=10, y=y, lmbd=lmbd)

    @pytest.mark.parametrize(
        "invalid_rules",
        [
            [Rule.random(1, 10), Rule.random(0, 10)],  # wrong ordering of rule ids
            [Rule.random(1, 10), Rule.random(2, 10)],  # invalid start index
        ],
    )
    def test_bb_init_invalid_rule_ids(self, invalid_rules):
        with pytest.raises(AssertionError):
            rand_y = bin_random(10)
            self.create_bb_object(invalid_rules, rand_y)

    def test_bb_init(self, rules, y):
        bb = self.create_bb_object(rules, y)

        assert bb.num_train_pts == mpz(5)
        assert isinstance(bb.y_mpz, mpz)
        assert isinstance(bb.y_np, np.ndarray)

        assert bb.default_rule_fnr == 0.4
        assert bb.rules == rules

    def test_prepare(self, rules, y):
        bb = self.create_bb_object(rules, y)
        bb.reset()

        # front and tree root are both accessible
        # and refer to the same node
        prefix, lb, not_captured = bb.queue.front()
        assert prefix == tuple()
        assert lb == 0.0
        assert not_captured == mpz_all_ones(y.shape[0])
        assert gmp.popcount(not_captured) == bb.num_train_pts

    @pytest.mark.parametrize("lmbd", np.random.random(10))
    def test_loop_1st_infinite_ub(self, rules, y, lmbd):
        """
        y : 10100
        r1: 01010
        r2: 10100
        r3: 10101

        y:  10100
        r1: 01010
        fp:  ^ ^
        fn: ^ ^

        y : 10100
        r2: 10100
        fp:
        fn:

        y : 10100
        r3: 10101
        fp:     ^
        fn:
        """
        # we consider search one level down with the lower bound being infinite
        # thus, all singleton rulesets should be enumerated
        ub = float("inf")

        # the first iteration of the branch and bound
        bb = BranchAndBoundNaive(rules, ub=ub, y=y, lmbd=lmbd)
        bb.reset()

        prefix, lb, not_captured = bb.queue.pop()

        iter_obj = bb._loop(prefix, lb, not_captured, return_objective=True)
        feasible_ruleset_ids = list(iter_obj)  # evoke the generator

        assert len(feasible_ruleset_ids) == 3  # all singleton rulesets are feasible

        assert bb.queue.size == 3
        prefix_1, lb_1, not_captured_1 = bb.queue.pop()
        prefix_2, lb_2, not_captured_2 = bb.queue.pop()
        prefix_3, lb_3, not_captured_3 = bb.queue.pop()

        # the order of nodes in the queue in determined by lb
        assert prefix_1 == (1,)
        assert prefix_2 == (2,)
        assert prefix_3 == (0,)

        assert lb_1 == lmbd
        assert lb_2 == (mpfr(1) / 5 + lmbd)
        assert lb_3 == (mpfr(2) / 5 + lmbd)

        # the yielded order is determined by the lexicographical order
        assert feasible_ruleset_ids[0][0] == (0,)
        np.testing.assert_allclose(feasible_ruleset_ids[0][1], 4 / 5 + lmbd)
        assert feasible_ruleset_ids[1][0] == (1,)
        np.testing.assert_allclose(feasible_ruleset_ids[1][1], lmbd)
        assert feasible_ruleset_ids[2][0] == (2,)
        np.testing.assert_allclose(feasible_ruleset_ids[2][1], (1 / 5 + lmbd))

        assert not_captured_1 == mpz("0b01011")
        assert not_captured_2 == mpz("0b01010")
        assert not_captured_3 == mpz("0b10101")

    @pytest.mark.skip("due the addition of look-ahead bound")
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

        d, lb, u = bb.queue.pop()

        iter_obj = bb._loop(d, lb, u)
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
            (0, ),
            (1, ),
            (2, ),
            (1, 2, ),
            (0, 1, ),
            (0, 2, ),
            (0, 1, 2, ),
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

        | rule set  | objective |
        |-----------+-----------|
        | (1)       |        .1 |
        | (2)       |        .3 |
        | (1, 2)    |        .4 |
        | (0, 1)    |        .6 |
        | (0, 2)    |        .8 |
        | (0, 1, 2) |        .9 |
        | (0)       |        .9 |
        """
        lmbd = 0.1
        bb = BranchAndBoundNaive(rules, ub=ub + EPSILON, y=y, lmbd=lmbd)
        feasible_solutions = list(bb.run())

        feasible_solutions = set(map(tuple, feasible_solutions))

        all_feasible_solutions_sorted_by_objective = [
            (1,),
            (2,),
            (1, 2,),
            (0, 1,),
            (0, 2,),
            (0, 1, 2),
            (0,),
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
            (1, ): 0.1,
            (2, ): 0.3,
            (1, 2): 0.4,
            (0, 1): 0.6,
            (0, 2): 0.8,
            (0, 1, 2,): 0.9,
            (0, ): 0.9,
        }

        # compare the two dict
        assert_dict_allclose(actual, expected)
