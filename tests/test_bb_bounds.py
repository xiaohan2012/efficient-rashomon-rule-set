import numpy as np
import pytest
from bds.bb import  BranchAndBoundV1, incremental_update_lb, incremental_update_obj
from bds.common import EPSILON
from bds.utils import bin_array, randints, solutions_to_dict
from .fixtures import rules, y, toy_D
from .utils import assert_dict_allclose
from bds.bounds_utils import *
from bds.bounds_v2 import * 


class TestBranchAndBoundV1:
    def create_bb_object(self, rules, y, lmbd=0.1):
        return BranchAndBoundV1(rules, ub=10, y=y, lmbd=lmbd)

    def test_bb_init(self, rules, y):
        bb = self.create_bb_object(rules, y)

        assert bb.num_train_pts == 5
        np.testing.assert_allclose(bb.default_rule_fnr, 0.4)
        assert bb.rules == rules

    def test_prepare(self, rules, y):
        bb = self.create_bb_object(rules, y)
        bb.reset()
        # front and tree root are both accessible
        # and refer to the same node
        node, not_captured = bb.queue.front()
        assert node == bb.tree.root
        assert (not_captured == 1).all()

    @pytest.mark.parametrize("lmbd", np.random.random(10))
    def test_loop_1st_infinite_ub(self, rules, y, toy_D, lmbd):
        # we consider search one level down with the lower bound being infinite
        # thus, all singleton rulesets should be enumerated
        ub = float("inf")

        # the first iteration of the branch and bound
        bb = BranchAndBoundV1(rules, ub=ub, y=y, lmbd=lmbd)
        bb.reset()

        node, not_captured = bb.queue.pop()

        print(toy_D)

        equivalence_classes = find_equivalence_classes(toy_D, y)
        iter_obj = bb._loop(node, not_captured, toy_D, equivalence_classes)
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
        self, rules, y, toy_D, ub, num_feasible_solutions, queue_size
    ):
        # we assume lmbd is fixed and try different upper bo
        lmbd = 0.1
        bb = BranchAndBoundV1(rules, ub=ub, y=y, lmbd=lmbd)
        bb.reset()

        node, not_captured = bb.queue.pop()

        equivalence_classes = find_equivalence_classes(toy_D, y)
        iter_obj = bb._loop(node, not_captured, toy_D, equivalence_classes)
        feasible_ruleset_ids = list(iter_obj)  # evoke the generator

        #assert len(feasible_ruleset_ids) == num_feasible_solutions

        #assert bb.queue.size == queue_size

    def test_run_infinite_ub(self, rules, y, toy_D):
        # we run the branch-and-bound with infinite upper bounds
        # all solutions should be returned as feasible
        ub = float("inf")
        lmbd = 0.1
        bb = BranchAndBoundV1(rules, ub=ub, y=y, lmbd=lmbd)
        feasible_solutions = list(bb.run(X_trn = toy_D))

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


        
    def test_equivalence_classes(self, rules, y, toy_D):
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
        ub = 0.5 
        bb = BranchAndBoundV1(rules, ub=ub + EPSILON, y=y, lmbd=lmbd)        
        equivalence_classes = find_equivalence_classes(toy_D, y)
        
        
        assert len(equivalence_classes) == 5
        
        assert "1" in equivalence_classes.keys() 
        
        assert "0-1" in equivalence_classes.keys() 
        
        assert "1-2" not in equivalence_classes.keys() 
        
        
        assert equivalence_classes["0-1"].minority_mistakes == 0 

        assert equivalence_classes["0-1"].total_positives == 0 
    
        assert equivalence_classes["0-1"].total_negatives == 1 
        
        
        toy_D_new = np.array([[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
        
        y_new = np.array([0, 0, 1, 0, 1, 1], dtype=bool)

        #bb = BranchAndBoundV1(rules, ub=ub + EPSILON, y=y_new, lmbd=lmbd)        
        equivalence_classes = find_equivalence_classes(toy_D_new, y_new)
        
        assert len(equivalence_classes) == 5
        
        assert equivalence_classes["1"].minority_mistakes == 1

        assert equivalence_classes["1"].total_positives == 1 
    
        assert equivalence_classes["1"].total_negatives == 1 
        
        
    
    
    
    def test_check_objective_calculation(self, rules, y, toy_D):
        lmbd = 0.1
        ub = float("inf")
        bb = BranchAndBoundV1(rules, ub=ub, y=y, lmbd=lmbd)
        feasible_solutions = list(bb.run(X_trn = toy_D, return_objective=True))

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
