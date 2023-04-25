import numpy as np
import pytest

from bds.bb import BranchAndBoundNaive, incremental_update_lb, incremental_update_obj
from bds.common import EPSILON
from bds.rule import Rule
from bds.utils import bin_array, randints


@pytest.mark.parametrize("seed", randints(5))
@pytest.mark.parametrize("num_fp", np.arange(6))
def test_incremental_update_obj(seed, num_fp):
    np.random.seed(seed)

    arr_len = 10

    # say we capture half of the points
    captured_idx = np.random.permutation(arr_len)[: int(arr_len / 2)]
    v = np.zeros(arr_len, dtype=bool)  # captured array
    v[captured_idx] = 1

    y = v.copy()  # the true labels
    y[captured_idx[:num_fp]] = 0  # make `num_fp` mistakes
    true_inc_fp = num_fp / arr_len
    assert incremental_update_lb(v, y) == true_inc_fp


def test_incremental_update_lb():
    u = np.array([0, 1, 1, 0, 0, 1, 0], dtype=bool)  # points not captured by prefix
    v = np.array([0, 1, 0, 0, 1, 0, 0], dtype=bool)  # captured by rule
    f = np.array(
        [0, 0, 1, 0, 0, 1, 0], dtype=bool
    )  # not captured by the rule and prefix
    y = np.array([0, 1, 1, 0, 1, 1, 0], dtype=bool)  # the true labels
    fn, actual_f = incremental_update_obj(u, v, y)

    assert fn == 2 / 7
    np.testing.assert_allclose(f, actual_f)
    assert actual_f.dtype == bool


@pytest.fixture
def y():
    return np.array([0, 0, 1, 0, 1], dtype=bool)


@pytest.fixture
def rules(y):
    return [
        Rule(
            id=1,
            name="rule-1",
            cardinality=1,
            truthtable=np.array([0, 1, 0, 1, 0], dtype=bool),
            ids=np.array([1, 3]),
        ),
        Rule(
            id=2,
            name="rule-2",
            cardinality=1,
            truthtable=np.array([0, 0, 1, 0, 1], dtype=bool),
            ids=np.array([2, 4]),
        ),
        Rule(
            id=3,
            name="rule-3",
            cardinality=1,
            truthtable=np.array([1, 0, 1, 0, 1], dtype=bool),
            ids=np.array([0, 2, 4]),
        ),
    ]


class TestBranchAndBoundNaive:
    def create_bb_object(self, rules, y, lmbd=0.1):
        return BranchAndBoundNaive(rules, ub=10, y=y, lmbd=lmbd)

    def test_bb_init(self, rules, y):
        bb = self.create_bb_object(rules, y)

        assert bb.num_train_pts == 5
        np.testing.assert_allclose(bb.default_rule_fnr, 0.4)
        assert bb.rules == rules

    def test_prepare(self, rules, y):
        bb = self.create_bb_object(rules, y)
        bb.prepare()

        # front and tree root are both accessible
        # and refer to the same node
        node, not_captured = bb.queue.front()
        assert node == bb.tree.root
        assert (not_captured == 1).all()

    @pytest.mark.parametrize("lmbd", np.random.random(10))
    def test_loop_1st_infinite_ub(self, rules, y, lmbd):
        # we consider search one level down with the lower bound being infinite
        # thus, all singleton rulesets should be enumerated
        ub = float("inf")

        # the first iteration of the branch and bound
        bb = BranchAndBoundNaive(rules, ub=ub, y=y, lmbd=lmbd)
        bb.prepare()

        node, not_captured = bb.queue.pop()

        iter_obj = bb.loop(node, not_captured)
        feasible_ruleset_ids = list(iter_obj)  # evoke the generator

        assert len(feasible_ruleset_ids) == 3  # all singleton rulesets are feasible

        assert bb.queue.size == 3
        node_1, not_captured_1 = bb.queue.pop()
        node_2, not_captured_2 = bb.queue.pop()
        node_3, not_captured_3 = bb.queue.pop()

        assert node_1.rule_id == 2
        assert node_2.rule_id == 3
        assert node_3.rule_id == 1

        np.testing.assert_allclose(node_1.lower_bound, lmbd)
        np.testing.assert_allclose(node_2.lower_bound, 1 / 5 + lmbd)
        np.testing.assert_allclose(node_3.lower_bound, 2 / 5 + lmbd)

        np.testing.assert_allclose(node_1.objective, lmbd)
        np.testing.assert_allclose(node_2.objective, 1 / 5 + lmbd)
        np.testing.assert_allclose(node_3.objective, 4 / 5 + lmbd)

        np.testing.assert_allclose(not_captured_1, bin_array([1, 1, 0, 1, 0]))
        np.testing.assert_allclose(not_captured_2, bin_array([0, 1, 0, 1, 0]))
        np.testing.assert_allclose(not_captured_3, bin_array([1, 0, 1, 0, 1]))

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
        bb.prepare()

        node, not_captured = bb.queue.pop()

        iter_obj = bb.loop(node, not_captured)
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

        actual = dict(map(lambda tpl: (tuple(tpl[0]), tpl[1]), feasible_solutions))
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
        # due to numerical instability, we use allclose to check
        assert set(actual.keys()) == set(expected.keys())
        for k in actual.keys():
            np.testing.assert_allclose(actual[k], expected[k])
