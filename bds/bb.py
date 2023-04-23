from typing import Dict, Optional, Tuple, Union
from typing import List
import numpy as np

from .cache_tree import CacheTree, Node
from .queue import Queue
from .rule import Rule


def _assert_binary_array(arr):
    assert arr.dtype == bool


def incremental_update_lb(v: np.ndarray, y: np.ndarray):
    """
    return the incremental false positive rate for a given rule

    v: points captured by the rule
    y: true labels
    """
    _assert_binary_array(v)
    _assert_binary_array(y)
    n = v.sum()  # number of predicted positive
    w = np.logical_and(v, y)  # false positives
    t = w.sum()
    return (n - t) / v.shape[0]  # false positive rate


def incremental_update_obj(
    u: np.ndarray, v: np.ndarray, y: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    return the incremental false negative rate for a rule set (prefix + current rule)
    and the indicator vector of false negatives

    u: points not captured by the prefix
    v: points captured by the current rule (in the context of the prefix)
    y: true labels
    """
    _assert_binary_array(u)
    _assert_binary_array(y)
    _assert_binary_array(v)
    f = np.logical_and(
        u, np.bitwise_not(v)
    )  # positives not captured by both prefix and the rule
    g = np.logical_and(f, y)  # false negatives
    return g.sum() / y.shape[0], f


class BranchAndBoundNaive:
    """an implementation of the branch and bound algorithm for enumerating good decision sets"""

    def __init__(self, rules: List[Rule], ub: float, y: np.ndarray):
        """
        rules: a list of candidate rules
        ub: the upper bound on objective value of any ruleset to be returned
        y: the ground truth label
        """
        self.rules = rules
        self.ub = ub
        self.y = y

    def bb_begin(self):
        """prepare for the branch and bound, e.g., initialize the queue and the cache tree"""
        self.tree = CacheTree()

        self.tree.add_node(Node.(), parent=None)

        self.queue = Queue()
        not_captured = None  # TODO: what does empty rule not capture?
        self.queue.push((self.tree.root, not_captured))

    def run(self):
        self.bb_begin()

        while not self.queue.empty():
            cur_node, not_captured = self.queue.pop()
            yield from self.bb_loop(
                cur_node,
                not_captured,
                self.tree,
                self.queue,
                self.rules,
                self.ub,
                self.y,
            )

    def _get_captured_for_rule(self, j: int, parent_not_captured: np.ndarray):
        """return the captured array for the jth rule in the context of parent"""
        return np.logical_and(parent_not_captured, self.rules[j].truthtable)

    def bb_loop(
        self,
        parent_node: Node,
        parent_not_captured: np.ndarray,
    ):
        """
        check one node in the search tree and search one level down

        parent_node: the current node/prefix to evaluate on
        parent_not_captured: postives not captured by the current prefix
        """
        parent_lb = parent_node.lower_bound
        for j, rule in enumerate(self.rules):
            if j > parent_node.rule_id:
                captured = self._get_captured_for_rule(j, parent_not_captured)

                lb = parent_lb + incremental_update_lb(captured, self.y)

                if lb <= self.ub:
                    fn_rate, not_captured = incremental_update_obj(
                        parent_not_captured, captured, self.y
                    )
                    obj = lb + fn_rate  # TODO: what about the regularization term?

                    child_node = Node(
                        node_id=self.tree.num_nodes,
                        lower_bound=lb,
                        objective=obj,
                        depth=parent_node.depth + 1,
                        num_captured=captured.sum(),
                    )

                    self.tree.add_node(child_node, parent_node)

                    self.queue.push((child_node, not_captured))

                    if obj <= self.ub:
                        yield child_node.as_ruleset()
