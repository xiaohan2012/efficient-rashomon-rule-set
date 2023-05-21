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
from .bounds import (
    incremental_update_lb,
    incremental_update_obj,
    prefix_specific_length_upperbound,
)

# logger.setLevel(logging.INFO)
from .bounds_utils import *
from .bounds_v2 import (
    rule_set_size_bound_with_default,
    equivalent_points_bounds,
    update_equivalent_lower_bound,
)


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
        self.y_np = y  # np.ndarray version of y
        self.rules = rules
        self.ub = ub
        self.y_mpz = mpz_set_bits(
            mpz(), y.nonzero()[0]
        )  # convert y from np.array to mpz
        self.lmbd = mpfr(lmbd)

        logger.debug(f"calling {self.__class__.__name__} with ub={ub}, lmbd={lmbd}")

        self.num_train_pts = mpz(y.shape[0])

        # false negative rate of the default rule = fraction of positives
        self.default_rule_fnr = mpz(gmp.popcount(self.y_mpz)) / self.num_train_pts

        self._check_rule_ids()

        self.__post_init__()

    def __post_init__(self):
        """hook function to be called after __init__ is called"""
        pass

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

        self.reset()

    # @profile
    def generate(self, return_objective=False) -> Iterable:
        while not self.queue.is_empty:
            queue_item = self.queue.pop()
            yield from self._loop(*queue_item, return_objective=return_objective)

    # @profile
    def run(self, return_objective=False, **kwargs) -> Iterable:
        self.reset(**kwargs)
        yield from self.generate(return_objective=return_objective)

    def _loop(self, *args, **kwargs):
        "the inner loop, corresponding to the evaluation of one item in the queue"
        raise NotImplementedError()

    def _bounded_sols_iter(self, threshold: Optional[int] = None, **kwargs) -> Iterable:
        """return an iterable of at most `threshold` feasible solutions
        if threshold is None, return all
        """
        Y = self.run(**kwargs)
        if threshold is not None:
            Y = itertools.islice(Y, threshold)
        return Y

    # @profile
    def bounded_count(self, threshold: Optional[int] = None, **kwargs) -> int:
        """return min(|Y|, threshold), where Y is the set of feasible solutions"""
        return count_iter(self._bounded_sols_iter(threshold, **kwargs))

    def bounded_sols(self, threshold: Optional[int] = None, **kwargs) -> List:
        """return at most threshold feasible solutions"""
        return list(self._bounded_sols_iter(threshold, **kwargs))


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

    def _create_new_node_and_add_to_tree(
        self, rule: Rule, lb: mpfr, obj: mpfr, captured: mpz, parent_node: Node
    ) -> Node:
        """create a node using information provided by rule, lb, obj, and captured
        and add it as a child of parent"""
        child_node = Node(
            rule_id=rule.id,
            lower_bound=lb,
            objective=obj,
            num_captured=gmp.popcount(captured),
        )

        self.tree.add_node(child_node, parent_node)
        return child_node

    def _incremental_update_lb(self, v: mpz, y: np.ndarray) -> mpfr:
        return incremental_update_lb(v, y, self.num_train_pts)

    def _incremental_update_obj(self, u: mpz, v: mpz) -> Tuple[mpfr, mpz]:
        return incremental_update_obj(u, v, self.y_mpz, self.num_train_pts)

    # @profile
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
        length_ub = prefix_specific_length_upperbound(
            parent_lb, parent_node.num_rules, self.lmbd, self.ub
        )

        for rule in self.rules[parent_node.rule_id :]:
            # prune by ruleset length
            if (parent_node.num_rules + 1) > length_ub:
                continue

            captured = self._captured_by_rule(rule, parent_not_captured)
            lb = (
                parent_lb
                + self._incremental_update_lb(captured, self.y_mpz)
                + self.lmbd
            )
            if lb <= self.ub:
                fn_fraction, not_captured = self._incremental_update_obj(
                    parent_not_captured, captured
                )
                obj = lb + fn_fraction

                child_node = self._create_new_node_and_add_to_tree(
                    rule, lb, obj, captured, parent_node
                )
                # apply look-ahead bound
                lb = child_node.lower_bound + self.lmbd

                if lb <= self.ub:
                    self.queue.push(
                        (child_node, not_captured),
                        key=child_node.lower_bound,  # the choice of key shouldn't matter for complete enumeration
                    )
                if obj <= self.ub:
                    ruleset = child_node.get_ruleset_ids()
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
