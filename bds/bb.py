import logging
from typing import Tuple, Optional, List
import numpy as np
from logzero import logger

from .cache_tree import CacheTree, Node
from .queue import Queue
from .rule import Rule
from .utils import bin_ones, assert_binary_array, debug2

# logger.setLevel(logging.INFO)
from .bounds_utils import *
from .bounds_v2 import rule_set_size_bound_with_default, equivalent_points_bounds


def incremental_update_lb(v: np.ndarray, y: np.ndarray):
    """
    return the incremental false positive fraction for a given rule

    v: points captured by the rule
    y: true labels
    """
    assert_binary_array(v)
    assert_binary_array(y)
    n = v.sum()  # number of predicted positive
    w = np.logical_and(v, y)  # true positives
    t = w.sum()
    return (n - t) / v.shape[0]  # fraction of false positive


def incremental_update_obj(
    u: np.ndarray, v: np.ndarray, y: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    return the incremental false negative fraction for a rule set (prefix + current rule)
    and the indicator vector of false negatives

    u: points not captured by the prefix
    v: points captured by the current rule (in the context of the prefix)
    y: true labels
    """
    assert_binary_array(u)
    assert_binary_array(y)
    assert_binary_array(v)
    f = np.logical_and(
        u, np.bitwise_not(v)
    )  # points not captured by both prefix and the rule
    g = np.logical_and(f, y)  # false negatives
    return g.sum() / y.shape[0], f


class BranchAndBoundGeneric:
    """generic class of branch-and-bound algorithm for decision set enumeration"""

    def __init__(self, rules: List[Rule], ub: float, y: np.ndarray, lmbd: float):
        """
        rules: a list of candidate rules
        ub: the upper bound on objective value of any ruleset to be returned
        y: the ground truth label
        lmbd: the parameter that controls regularization strength
        """
        assert_binary_array(y)

        self.rules = rules
        self.ub = ub
        self.y = y
        self.lmbd = lmbd

        debug2(f"calling branch-and-bound with ub={ub}, lmbd={lmbd}")

        self.num_train_pts = y.shape[0]

        # false negative rate of the default rule = fraction of positives
        self.default_rule_fnr = y.sum() / self.num_train_pts

    def reset_tree(self):
        raise NotImplementedError()

    def reset_queue(self):
        raise NotImplementedError()

    def reset(self):
        self.reset_tree()
        self.reset_queue()

    def _captured_by_rule(self, rule: Rule, parent_not_captured: np.ndarray):
        """return the captured array for the rule in the context of parent"""
        return np.logical_and(parent_not_captured, rule.truthtable)

    def run(self, *args, return_objective=False):
        self.reset(*args)

        while not self.queue.is_empty:
            queue_item = self.queue.pop()
            yield from self._loop(*queue_item, return_objective=return_objective)

    def _loop(self, *args, **kwargs):
        "the inner loop, corresponding to the evaluation of one item in the queue"
        raise NotImplementedError()


class BranchAndBoundNaive(BranchAndBoundGeneric):
    """an implementation of the branch and bound algorithm for enumerating good decision sets.

    only hierarchical lower bound is used for pruning
    """

    def reset_tree(self):
        self.tree = CacheTree()
        root = Node.make_root(self.default_rule_fnr, self.num_train_pts)

        # add the root
        self.tree.add_node(root)

    def reset_queue(self):
        self.queue: Queue = Queue()
        not_captured = bin_ones(self.y.shape)  # the dafault rule captures nothing

        item = (self.tree.root, not_captured)
        self.queue.push(item, key=0)

    def _loop(
        self, parent_node: Node, parent_not_captured: np.ndarray, return_objective=False
    ):
        """
        check one node in the search tree, update the queue, and yield feasible solution if exists

        parent_node: the current node/prefix to check
        parent_not_captured: postives not captured by the current prefix
        return_objective: True if return the objective of the evaluated node
        """
        parent_lb = parent_node.lower_bound
        for rule in self.rules:
            if rule.id > parent_node.rule_id:
                # logger.debug(f"considering rule {rule.id}")
                captured = self._captured_by_rule(rule, parent_not_captured)

                lb = parent_lb + incremental_update_lb(captured, self.y) + self.lmbd

                if lb <= self.ub:
                    fn_fraction, not_captured = incremental_update_obj(
                        parent_not_captured, captured, self.y
                    )
                    obj = lb + fn_fraction

                    child_node = Node(
                        rule_id=rule.id,
                        lower_bound=lb,
                        objective=obj,
                        num_captured=captured.sum(),
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

    def reset_tree(self):
        self.tree = CacheTree()
        root = Node.make_root(self.default_rule_fnr, self.num_train_pts)

        # add the root
        self.tree.add_node(root)

    def reset_queue(self):
        self.queue: Queue = Queue()
        not_captured = bin_ones(self.y.shape)  # the dafault rule captures nothing

        item = (self.tree.root, not_captured)
        self.queue.push(item, key=0)

    # override method from the base class
    def run(self, *args, X_trn, return_objective=False):
        self.reset(*args)
        equivalence_classes = find_equivalence_classes(X_trn, self.y)
        while not self.queue.is_empty:
            queue_item = self.queue.pop()
            yield from self._loop(
                *queue_item,
                X_trn,
                equivalence_classes,
                return_objective=return_objective,
            )

    def _loop(
        self,
        parent_node: Node,
        parent_not_captured: np.ndarray,
        X_trn: np.array,
        equivalence_classes: dict,
        return_objective=False,
    ):
        """
        check one node in the search tree, update the queue, and yield feasible solution if exists

        parent_node: the current node/prefix to check
        parent_not_captured: postives not captured by the current prefix
        return_objective: True if return the objective of the evaluated node
        """
        parent_lb = parent_node.lower_bound

        for rule in self.rules:
            if rule.id > parent_node.rule_id:
                # logger.debug(f"considering rule {rule.id}")
                captured = self._captured_by_rule(rule, parent_not_captured)

                lb = parent_lb + incremental_update_lb(captured, self.y) + self.lmbd

                flag_rule_set_size = rule_set_size_bound_with_default(
                    parent_node, self.lmbd, self.ub
                )  # if true, we prune

                flag_equivalent_classes = equivalent_points_bounds(
                    lb,
                    self.lmbd,
                    self.ub,
                    parent_not_captured,
                    X_trn,
                    equivalence_classes,
                )  # if true, we prune

                if (
                    lb <= self.ub
                    and not flag_rule_set_size
                    and not flag_equivalent_classes
                ):
                    fn_fraction, not_captured = incremental_update_obj(
                        parent_not_captured, captured, self.y
                    )
                    obj = lb + fn_fraction

                    child_node = Node(
                        rule_id=rule.id,
                        lower_bound=lb,
                        objective=obj,
                        num_captured=captured.sum(),
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
    bb = BranchAndBoundNaive(rules, ub, y, lmbd)
    sols = list(bb.run())
    if return_sols:
        return len(sols), sols
    else:
        return len(sols)
