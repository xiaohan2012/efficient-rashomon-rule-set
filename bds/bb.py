import numpy as np
import gmpy2 as gmp
import itertools

from logzero import logger
from gmpy2 import mpz, mpfr
from typing import Tuple, Optional, List, Iterable

from .cache_tree import CacheTree, Node
from .queue import Queue
from .rule import Rule
from .utils import (
    bin_ones,
    assert_binary_array,
    mpz_set_bits,
    mpz_all_ones,
    count_iter,
)
from .bounds import incremental_update_lb, incremental_update_obj

# logger.setLevel(logging.INFO)
from .bounds_utils import *
from .bounds_v2 import rule_set_size_bound_with_default, equivalent_points_bounds, update_equivalent_lower_bound


class BranchAndBoundGeneric:
    """a generic class of branch-and-bound algorithm for decision set enumeration"""

    def __init__(self, rules: List[Rule], ub: float, y: np.ndarray, lmbd: float):
        """
        rules: a list of candidate rules
        ub: the upper bound on objective value of any ruleset to be returned
        y: the ground truth label represented as a numpy.ndarray
        lmbd: the parameter that controls regularization strength
        """
        assert_binary_array(y)
        self.y_bool = y
        self.rules = rules
        self.ub = ub
        self.y = mpz_set_bits(mpz(), y.nonzero()[0])  # convert y from np.array to mpz
        self.lmbd = mpfr(lmbd)

        logger.debug(f"calling {self.__class__.__name__} with ub={ub}, lmbd={lmbd}")

        self.num_train_pts = mpz(y.shape[0])

        # false negative rate of the default rule = fraction of positives
        self.default_rule_fnr = mpz(gmp.popcount(self.y)) / self.num_train_pts

        self._check_rule_ids()

    def _check_rule_ids(self):
        """check the rule ids are consecutive integers starting from 1"""
        rule_ids = np.array([r.id for r in self.rules])
        np.testing.assert_allclose(rule_ids, np.arange(1, 1 + len(rule_ids)))

    def reset_tree(self):
        raise NotImplementedError()

    def reset_queue(self):
        raise NotImplementedError()

    def reset(self):
        self.reset_tree()
        self.reset_queue()

    def _captured_by_rule(self, rule: Rule, parent_not_captured: mpz):
        """return the captured array for the rule in the context of parent"""
        return parent_not_captured & rule.truthtable

    # @profile
    def run(self, *args, return_objective=False):
        self.reset(*args)

        while not self.queue.is_empty:
            queue_item = self.queue.pop()
            yield from self._loop(*queue_item, return_objective=return_objective)

    def _loop(self, *args, **kwargs):
        "the inner loop, corresponding to the evaluation of one item in the queue"
        raise NotImplementedError()

    def _bounded_sols_iter(self, threshold: Optional[int] = None, *args) -> Iterable:
        """return an iterable of at most `threshold` feasible solutions
        if threshold is None, return all
        """
        Y = self.run(*args)
        if threshold is not None:
            Y = itertools.islice(Y, threshold)
        return Y

    # @profile
    def bounded_count(self, threshold: Optional[int] = None, *args) -> int:
        """return min(|Y|, threshold), where Y is the set of feasible solutions"""
        # lst = list(self._bounded_sols_iter(threshold, *args))
        # return len(lst)
        return count_iter(self._bounded_sols_iter(threshold, *args))

    def bounded_sols(self, threshold: Optional[int] = None, *args) -> List:
        """return at most threshold feasible solutions"""
        return list(self._bounded_sols_iter(threshold, *args))


class BranchAndBoundNaive(BranchAndBoundGeneric):
    """an implementation of the branch and bound algorithm for enumerating good decision sets.

    only hierarchical lower bound is used for pruning
    """

    def reset_tree(self):
        self.tree = CacheTree()
        root = Node.make_root(self.default_rule_fnr, self.num_train_pts)

        # add the root
        self.tree.add_node(root)

    def _not_captured_by_default_rule(self):
        """return the vector of not captured by default rule
        the dafault rule captures nothing
        """
        return mpz_all_ones(self.num_train_pts)

    def reset_queue(self):
        self.queue: Queue = Queue()

        not_captured = self._not_captured_by_default_rule()

        item = (self.tree.root, not_captured)
        self.queue.push(item, key=0)

    def _incremental_update_lb(self, v: mpz, y: np.ndarray) -> mpfr:
        return incremental_update_lb(v, y, self.num_train_pts)

    def _incremental_update_obj(self, u: mpz, v: mpz) -> Tuple[mpfr, mpz]:
        return incremental_update_obj(u, v, self.y, self.num_train_pts)

    def _loop(
        self, parent_node: Node, parent_not_captured: mpz, return_objective=False
    ):
        """
        check one node in the search tree, update the queue, and yield feasible solution if exists

        parent_node: the current node/prefix to check
        parent_not_captured: postives not captured by the current prefix
        return_objective: True if return the objective of the evaluated node
        """
        parent_lb = parent_node.lower_bound
        for rule in self.rules[parent_node.rule_id:]:
            captured = self._captured_by_rule(rule, parent_not_captured)
            lb = (
                parent_lb
                + self._incremental_update_lb(captured, self.y)
                + self.lmbd
            )
            if lb <= self.ub:
                fn_fraction, not_captured = self._incremental_update_obj(
                    parent_not_captured, captured
                )
                obj = lb + fn_fraction

                child_node = Node(
                    rule_id=rule.id,
                    lower_bound=lb,
                    objective=obj,
                    num_captured=gmp.popcount(captured),
                )

                self.tree.add_node(child_node, parent_node)

                self.queue.push(
                    (child_node, not_captured),
                    key=child_node.lower_bound,  # TODO: consider other types of prioritization
                )
                if obj <= self.ub:
                    ruleset = child_node.get_ruleset_ids()
                    # logger.debug(
                    #     f"yield rule set {ruleset}: {child_node.objective:.4f} (obj) <= {self.ub:.4f} (ub)"
                    # )
                    if return_objective:
                        yield (ruleset, child_node.objective)
                    else:
                        yield ruleset


class BranchAndBoundV1(BranchAndBoundGeneric):
    """an implementation of the branch and bound algorithm for enumerating good decision sets.

    bounds used: hierarchical lower bound, bounds on the total number of rules, bounds on equivalent points
    """
    
    def reset(self, first_equivalent_lower_bound):
        self.reset_tree(first_equivalent_lower_bound)
        self.reset_queue()

    def reset_tree(self, first_equivalent_lower_bound):
        self.tree = CacheTree()
        root = Node.make_root(self.default_rule_fnr, self.num_train_pts)
        root.equivalent_lower_bound = first_equivalent_lower_bound

        # add the root
        self.tree.add_node(root)

    def _not_captured_by_default_rule(self):
        """return the vector of not captured by default rule
        the dafault rule captures nothing
        """
        return mpz_all_ones(self.num_train_pts)

    def reset_queue(self):
        self.queue: Queue = Queue()

        not_captured = self._not_captured_by_default_rule()

        item = (self.tree.root, not_captured)
        self.queue.push(item, key=0)

    def _incremental_update_lb(self, v: mpz, y: np.ndarray) -> mpfr:
        return incremental_update_lb(v, y, self.num_train_pts)

    def _incremental_update_obj(self, u: mpz, v: mpz) -> Tuple[mpfr, mpz]:
        return incremental_update_obj(u, v, self.y, self.num_train_pts)


    # override method from the base class
    def run(self, *args, return_objective=False):
        tot_not_captured_error_bound_init, data_points2rules, equivalence_classes = find_equivalence_classes(self.y_bool, self.rules)
        self.equivalence_classes = equivalence_classes # for testing
        self.reset(*args, first_equivalent_lower_bound = tot_not_captured_error_bound_init)
        while not self.queue.is_empty:
            queue_item = self.queue.pop()
            yield from self._loop(
                *queue_item,
                data_points2rules,
                equivalence_classes,
                return_objective=return_objective,
            )



    def _loop(
        self,
        parent_node: Node,
        parent_not_captured: mpz,
        data_points2rules: dict,
        equivalence_classes: dict,
        return_objective=False
        ):
        """
        check one node in the search tree, update the queue, and yield feasible solution if exists

        parent_node: the current node/prefix to check
        parent_not_captured: postives not captured by the current prefix
        return_objective: True if return the objective of the evaluated node
        """
        parent_lb = parent_node.lower_bound

        for rule in self.rules[parent_node.rule_id:]:
            # logger.debug(f"considering rule {rule.id}")
            captured = self._captured_by_rule(rule, parent_not_captured)

            
            #if lb <= self.ub:
            flag_rule_set_size = rule_set_size_bound_with_default(
                parent_node, self.lmbd, self.ub
            )  # i
            if not flag_rule_set_size: # this is fast to compute so we check it first 
                
                lb = (
                    parent_lb
                    + self._incremental_update_lb(captured, self.y)
                    + self.lmbd
                )
                
                parent_equivalent_lower_bound = parent_node.equivalent_lower_bound

                equivalent_lower_bound = parent_equivalent_lower_bound - update_equivalent_lower_bound(captured, data_points2rules, equivalence_classes) 
                # just checking 
                #print("rule " + str(rule)) 
                ##print("parent rules " + str(parent_node.get_ruleset_ids())) 
                #print("parent_equivalent_lower_bound " + str(parent_equivalent_lower_bound)) 
                #print("equivalent lower bound after update " + str(equivalent_lower_bound))
                #print("lower bound " + str(lb))
                #print("parent lower bound " + str(parent_lb)) 
                flag_equivalent_classes = equivalent_points_bounds(
                    lb,
                    equivalent_lower_bound, 
                    self.ub
                )  # if true, we prune
                
                if not flag_equivalent_classes: 
                        
                    fn_fraction, not_captured = self._incremental_update_obj(
                        parent_not_captured, captured
                    )
                    obj = lb + fn_fraction

                    child_node = Node(
                        rule_id=rule.id,
                        lower_bound=lb,
                        equivalent_lower_bound=equivalent_lower_bound,
                        objective=obj,
                        num_captured=gmp.popcount(captured),
                    )

                    self.tree.add_node(child_node, parent_node)

                    self.queue.push(
                        (child_node, not_captured),
                        key=child_node.lower_bound,  # TODO: consider other types of prioritization
                    )
                    if obj <= self.ub:
                        ruleset = child_node.get_ruleset_ids()
                        # logger.debug(
                        #     f"yield rule set {ruleset}: {child_node.objective:.4f} (obj) <= {self.ub:.4f} (ub)"
                        # )
                        if return_objective:
                            yield (ruleset, child_node.objective)
                        else:
                            yield ruleset
                            
                            
                            



class BranchAndBoundV0(BranchAndBoundGeneric):
    
    
    """
    V0 only uses the bound on rule set size , probably mostly for testing 
    
    an implementation of the branch and bound algorithm for enumerating good decision sets.

    bounds used: hierarchical lower bound, bounds on the total number of rules, bounds on equivalent points
    """

    def reset_tree(self):
        self.tree = CacheTree()
        root = Node.make_root(self.default_rule_fnr, self.num_train_pts)

        # add the root
        self.tree.add_node(root)

    def _not_captured_by_default_rule(self):
        """return the vector of not captured by default rule
        the dafault rule captures nothing
        """
        return mpz_all_ones(self.num_train_pts)

    def reset_queue(self):
        self.queue: Queue = Queue()

        not_captured = self._not_captured_by_default_rule()

        item = (self.tree.root, not_captured)
        self.queue.push(item, key=0)

    def _incremental_update_lb(self, v: mpz, y: np.ndarray) -> mpfr:
        return incremental_update_lb(v, y, self.num_train_pts)

    def _incremental_update_obj(self, u: mpz, v: mpz) -> Tuple[mpfr, mpz]:
        return incremental_update_obj(u, v, self.y, self.num_train_pts)




    def _loop(
        self,
        parent_node: Node,
        parent_not_captured: mpz,
        return_objective=False
        ):
        """
        check one node in the search tree, update the queue, and yield feasible solution if exists

        parent_node: the current node/prefix to check
        parent_not_captured: postives not captured by the current prefix
        return_objective: True if return the objective of the evaluated node
        """
        parent_lb = parent_node.lower_bound

        for rule in self.rules[parent_node.rule_id:]:
            # logger.debug(f"considering rule {rule.id}")
            captured = captured = self._captured_by_rule(rule, parent_not_captured)

            flag_rule_set_size = rule_set_size_bound_with_default(
                parent_node, self.lmbd, self.ub
            )  # i
            if not flag_rule_set_size: # this is super fast to compute so we check it first 
                
                lb = (
                    parent_lb
                    + self._incremental_update_lb(captured, self.y)
                    + self.lmbd
                )
                
                
                fn_fraction, not_captured = self._incremental_update_obj(
                    parent_not_captured, captured
                )
                obj = lb + fn_fraction

                child_node = Node(
                    rule_id=rule.id,
                    lower_bound=lb,
                    objective=obj,
                    num_captured=gmp.popcount(captured),
                )

                self.tree.add_node(child_node, parent_node)

                self.queue.push(
                    (child_node, not_captured),
                    key=child_node.lower_bound,  # TODO: consider other types of prioritization
                )
                if obj <= self.ub:
                    ruleset = child_node.get_ruleset_ids()
                    # logger.debug(
                    #     f"yield rule set {ruleset}: {child_node.objective:.4f} (obj) <= {self.ub:.4f} (ub)"
                    # )
                    if return_objective:
                        yield (ruleset, child_node.objective)
                    else:
                        yield ruleset





class BranchAndBoundV2(BranchAndBoundGeneric):
    
    
    """
    V2 uses HLB and RSSB
    
    an implementation of the branch and bound algorithm for enumerating good decision sets.

    bounds used: hierarchical lower bound, bounds on the total number of rules, bounds on equivalent points
    """

    def reset_tree(self):
        self.tree = CacheTree()
        root = Node.make_root(self.default_rule_fnr, self.num_train_pts)

        # add the root
        self.tree.add_node(root)

    def _not_captured_by_default_rule(self):
        """return the vector of not captured by default rule
        the dafault rule captures nothing
        """
        return mpz_all_ones(self.num_train_pts)

    def reset_queue(self):
        self.queue: Queue = Queue()

        not_captured = self._not_captured_by_default_rule()

        item = (self.tree.root, not_captured)
        self.queue.push(item, key=0)

    def _incremental_update_lb(self, v: mpz, y: np.ndarray) -> mpfr:
        return incremental_update_lb(v, y, self.num_train_pts)

    def _incremental_update_obj(self, u: mpz, v: mpz) -> Tuple[mpfr, mpz]:
        return incremental_update_obj(u, v, self.y, self.num_train_pts)




    def _loop(
        self,
        parent_node: Node,
        parent_not_captured: mpz,
        return_objective=False
        ):
        """
        check one node in the search tree, update the queue, and yield feasible solution if exists

        parent_node: the current node/prefix to check
        parent_not_captured: postives not captured by the current prefix
        return_objective: True if return the objective of the evaluated node
        """
        parent_lb = parent_node.lower_bound

        for rule in self.rules[parent_node.rule_id:]:
            # logger.debug(f"considering rule {rule.id}")
            captured = captured = self._captured_by_rule(rule, parent_not_captured)

            #if lb <= self.ub:
            flag_rule_set_size = rule_set_size_bound_with_default(
                parent_node, self.lmbd, self.ub
            )  # i
            if not flag_rule_set_size: # this is super fast to compute so we check it first 
                
                lb = (
                    parent_lb
                    + self._incremental_update_lb(captured, self.y)
                    + self.lmbd
                )
                
                if lb <= self.ub:
                    fn_fraction, not_captured = self._incremental_update_obj(
                        parent_not_captured, captured
                    )
                    obj = lb + fn_fraction
    
                    child_node = Node(
                        rule_id=rule.id,
                        lower_bound=lb,
                        objective=obj,
                        num_captured=gmp.popcount(captured),
                    )
    
                    self.tree.add_node(child_node, parent_node)
    
                    self.queue.push(
                        (child_node, not_captured),
                        key=child_node.lower_bound,  # TODO: consider other types of prioritization
                    )
                    if obj <= self.ub:
                        ruleset = child_node.get_ruleset_ids()
                        # logger.debug(
                        #     f"yield rule set {ruleset}: {child_node.objective:.4f} (obj) <= {self.ub:.4f} (ub)"
                        # )
                        if return_objective:
                            yield (ruleset, child_node.objective)
                        else:
                            yield ruleset









def get_ground_truth_count(
    rules: List[Rule],
    y: np.ndarray,
    lmbd: float,
    ub: float,
    return_sols: Optional[bool] = False,
) -> int:
    """solve the complete enumeration problem and return the number of feasible solutions, and optionally the solutions themselves"""
    bb = BranchAndBoundNaive(rules, ub, y, lmbd)
    sols = list(bb.run())
    if return_sols:
        return len(sols), sols
    else:
        return len(sols)
