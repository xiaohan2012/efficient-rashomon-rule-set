import numpy as np
import pytest
from bds.bb import  BranchAndBoundV1, BranchAndBoundV0, incremental_update_lb, incremental_update_obj
from bds.cbb import  ConstrainedBranchAndBoundV1
from bds.common import EPSILON
from bds.utils import bin_array, randints, solutions_to_dict
from .fixtures import rules, y , rulesForBounds1, yForBounds1 ,  rulesForBounds2, yForBounds2
from .utils import assert_dict_allclose, assert_close_mpfr
from bds.bounds_utils import *
from bds.bounds_v2 import * 
from bds.utils import (
    bin_array,
    bin_random,
    mpz_all_ones,
    mpz_clear_bits,
    mpz_set_bits,
    randints,
    solutions_to_dict,
)

class TestBranchAndBoundV1:
    def create_bb_object(self, rules, y, lmbd=0.1):
        return BranchAndBoundV1(rules, ub=10, y=y, lmbd=lmbd)

    def test_bb_init(self, rules, y):
        bb = self.create_bb_object(rules, y)

        assert bb.num_train_pts == mpz(5)
        assert isinstance(bb.y, mpz)

        assert bb.default_rule_fnr == 0.4
        assert bb.rules == rules

    def test_prepare(self,  rules, y):
        bb = self.create_bb_object(rules, y)
        
        first_lb, data_points2rules, equivalence_classes = find_equivalence_classes(y, rules)
        
        bb.reset(first_lb)

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
        
        first_elb, data_points2rules, equivalence_classes = find_equivalence_classes(y, rules)
        
        
        bb = BranchAndBoundV1(rules, ub=ub, y=y, lmbd=lmbd)
        bb.reset(first_elb)

        node, not_captured = bb.queue.pop()

        iter_obj = bb._loop(node, not_captured, data_points2rules, equivalence_classes)
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
        
       

    
    def test_run_infinite_ub(self, rules, y):
        # we run the branch-and-bound with infinite upper bounds
        # all solutions should be returned as feasible
        ub = float("inf")
        lmbd = 0.1
        bb = BranchAndBoundV1(rules, ub=ub, y=y, lmbd=lmbd)
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


        
    
    def test_equivalence_classes(self, rules, y):
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
        first_elb, data_points2rules, equivalence_classes = find_equivalence_classes( y, rules)
        
        
        assert len(equivalence_classes) == 3
        
        #assert 0 in data_points2rules
        assert data_points2rules[1] == [1]
        assert data_points2rules[2] == [2,3]
        
    
        n = mpz_set_bits(gmp.mpz(), [1]) 
        assert equivalence_classes[n].minority_mistakes == 0 

        n = mpz_set_bits(gmp.mpz(), [2,3])  # 2 4 covered
        assert equivalence_classes[n].total_positives == 2 
    
        assert equivalence_classes[n].total_negatives == 0
        
        assert equivalence_classes[n].minority_mistakes == 0 
        
        
        
    
        
    
    def test_V0(self, rulesForBounds1, yForBounds1):
        
        
        
        
        lmbd = 0.1
        ub = float("inf")
        bb = BranchAndBoundV0(rulesForBounds1, ub=ub, y=yForBounds1, lmbd=lmbd)
        feasible_solutions = list(bb.run(return_objective=True))
        actual = solutions_to_dict(feasible_solutions)


        expected = {(0, 1): mpfr('0.3'),
         (0, 2): mpfr('0.7'),
         (0, 3): mpfr('0.9'),
         (0, 4): mpfr('0.9'),
         (0, 1, 2): mpfr('0.8'),
         (0, 1, 3): mpfr('1.0'),
         (0, 1, 4): mpfr('1.0'),
         (0, 2, 3): mpfr('1.0'),
         (0, 2, 4): mpfr('1.0'),
         (0, 3, 4): mpfr('1.0'),
         (0, 2, 3, 4): mpfr('1.1'),
         (0, 1, 2, 3): mpfr('1.1'),
         (0, 1, 2, 4): mpfr('1.1'),
         (0, 1, 3, 4): mpfr('1.1'),
         (0, 1, 2, 3, 4): mpfr('1.2')}
        
        assert_dict_allclose(actual, expected)
    

        
    def test_V1(self, rulesForBounds1, yForBounds1):
        
        
        lmbd = 0.1
        ub = float("inf")
        bb = BranchAndBoundV1(rulesForBounds1, ub=ub, y=yForBounds1, lmbd=lmbd)
        feasible_solutions = list(bb.run(return_objective=True))
        actual = solutions_to_dict(feasible_solutions)


        expected = {(0, 1): mpfr('0.3'),
         (0, 2): mpfr('0.7'),
         (0, 3): mpfr('0.9'),
         (0, 4): mpfr('0.9'),
         (0, 1, 2): mpfr('0.8'),
         (0, 1, 3): mpfr('1.0'),
         (0, 1, 4): mpfr('1.0'),
         (0, 2, 3): mpfr('1.0'),
         (0, 2, 4): mpfr('1.0'),
         (0, 3, 4): mpfr('1.0'),
         (0, 2, 3, 4): mpfr('1.1'),
         (0, 1, 2, 3): mpfr('1.1'),
         (0, 1, 2, 4): mpfr('1.1'),
         (0, 1, 3, 4): mpfr('1.1'),
         (0, 1, 2, 3, 4): mpfr('1.2')}
        
        assert_dict_allclose(actual, expected)
    


    def test_Equi(self, rulesForBounds1, yForBounds1):
         
         
         lmbd = 0.1
         ub = float("inf")
         bb = BranchAndBoundV1(rulesForBounds1, ub=ub, y=yForBounds1, lmbd=lmbd)
         first_elb, data_points2rules, equivalence_classes = find_equivalence_classes(yForBounds1, rulesForBounds1)
           
         first_elb == 1 / 5 
     

         
    
    def test_V1(self, rulesForBounds2, yForBounds2):
        
        
        lmbd = 0.1
        ub = 0.7
        bb = BranchAndBoundV1(rulesForBounds2, ub=ub, y=yForBounds2, lmbd=lmbd)
        feasible_solutions = list(bb.run(return_objective=True))
        actual = solutions_to_dict(feasible_solutions)


        expected =  {(0, 1): mpfr('0.6833333'),
                     (0, 2): mpfr('0.6833333'),
                     (0, 3): mpfr('0.6833333'),
                     (0, 4): mpfr('0.6833333'),
                     (0, 3, 4): mpfr('0.7')}
        assert_dict_allclose(actual, expected)
    
    
        
        
    def test_check_objective_calculation(self, rules, y):
        lmbd = 0.1
        ub = float("inf")
        bb = BranchAndBoundV1(rules, ub=ub, y=y, lmbd=lmbd)
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