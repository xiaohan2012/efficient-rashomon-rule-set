import itertools
import math
import random
from typing import Iterable, List, Optional, Tuple

import gmpy2 as gmp
import numpy as np
from gmpy2 import mpfr, mpz
from logzero import logger

from .bounds import (
    incremental_update_lb,
    incremental_update_obj,
    prefix_specific_length_upperbound,
)

# logger.setLevel(logging.INFO)
from .bounds_utils import *
from .queue import Queue
from .rule import Rule
from .SampleTreeUtils import CacheTreeST, NodeST
from .utils import assert_binary_array, count_iter, mpz_all_ones, mpz_set_bits


class BranchAndBoundGeneric:
    """a generic class of branch-and-bound algorithm for decision set enumeration"""

    def __init__(
        self, rules: List[Rule], ub: float, y: np.ndarray, lmbd: float, l: float, k: int
    ):
        """
        rules: a list of candidate rules
        ub: the upper bound on objective value of any ruleset to be returned
        y: the ground truth label represented as a numpy.ndarray
        lmbd: the parameter that controls regularization strength
        """
        assert_binary_array(y)
        self.y_np = y  # np.ndarray version of y
        self.l = l
        self.k = k
        self.rules = rules
        self.ub = ub
        self.y_mpz = mpz_set_bits(
            mpz(), y.nonzero()[0]
        )  # convert y from np.array to mpz
        self.lmbd = lmbd
        self.n = len(rules)

        logger.debug(f"calling {self.__class__.__name__} with ub={ub}, lmbd={lmbd}")

        self.num_train_pts = mpz(y.shape[0])

        # false negative rate of the default rule = fraction of positives
        self.default_rule_fnr = mpz(gmp.popcount(self.y_mpz)) / self.num_train_pts

        self.num_prefix_evaluations = 0
        self._check_rule_ids()

        self.__post_init__()
        # self.not_captured_dict = dict()

    def __post_init__(self):
        """hook function to be called after __init__ is called"""

    def _check_rule_ids(self):
        """check the rule ids are consecutive integers starting from 1"""
        rule_ids = np.array([r.id for r in self.rules])
        np.testing.assert_allclose(rule_ids, np.arange(1, 1 + len(rule_ids)))

    def reset_tree(self):
        raise NotImplementedError()

    def reset_queue(self):
        raise NotImplementedError()

    def reset(self):
        # logger.debug("initializing search tree and priority queue")
        self.reset_tree()
        self.reset_queue()

    def compute_union(self, vectors):
        result = vectors[0]  # Initialize result with the first vector
        for vector in vectors[1:]:
            result = result | vector  # Perform bitwise OR operation
        return result

    def _captured_by_rule(self, rule: Rule, parent_not_captured: mpz):
        """return the captured array for the rule in the context of parent"""
        return parent_not_captured & rule.truthtable

    def _captured_by_rules(self, rules: list, parent_not_captured: mpz):
        return parent_not_captured & self.compute_union(
            [rule.truthtable for rule in rules]
        )

    # @profile
    def generateST(self, return_objective=False) -> Iterable:
        while not self.queue.is_empty:
            queue_item = self.queue.pop()
            yield from self._loopST(*queue_item, return_objective=return_objective)

    def runST(self, return_objective=False, **kwargs) -> Iterable:
        self.reset_tree()
        # self.reset(**kwargs) # ??

        not_captured_root = self._not_captured_by_default_rule()
        pseudosolutions = [
            ({}, (self.tree._root, not_captured_root), self.tree._root.objective)
        ]  # empty -

        L = math.ceil(self.n / self.l)

        self.current_length = 0

        for h in range(L):
            pseudosolutions = self.generate_single_level(pseudosolutions)
            self.current_length += self.l
            # if len(pseudosolutions) == 0: # hit the length bound
            #    break

        last_level_pseudosolutions = [
            pseudosol for pseudosol in pseudosolutions if pseudosol[-1] <= self.ub
        ]
        return last_level_pseudosolutions

    def generate_single_level(self, pseudosolutions) -> Iterable:
        n_samples = min(self.k, len(pseudosolutions))
        sampled_items = random.sample(pseudosolutions, n_samples)

        all_solutions = sampled_items  # we start with the current ones as well corresponding to set everything else to zero

        self.visited_rule_sets = set()

        for j in range(n_samples):
            #
            #
            this_sample = sampled_items[j]

            # if this_sample!=None:

            # print("this sample " + str(this_sample))

            self.reset_queue_arbitrary_node(this_sample[1])
            self.reset_tree_arbitrary_node(this_sample[1][0])
            # all_solutions.extend( list(self._loop( *this_sample , current_length, return_objective=False) ) )

            all_solutions.extend(list(self.generateST()))

            # print(len(all_solutions))

        # print(all_solutions)
        return all_solutions

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
        self.tree = CacheTreeST()
        root = NodeST.make_root(self.default_rule_fnr, self.num_train_pts)
        #  root.rule_id = "0"

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

    def reset_queue_arbitrary_node(self, tup):
        self.queue: Queue = Queue()

        self.queue.push(tup, key=0)

    def reset_tree_arbitrary_node(self, nod):
        self.tree = CacheTreeST()
        # root = NodeST.make_root(self.default_rule_fnr, self.num_train_pts)
        # root.rule_id = "0"
        # add the root
        self.tree.add_node(nod)

    def _create_new_node_and_add_to_tree(
        self, rule: Rule, lb: mpfr, obj: mpfr, captured: mpz, parent_node: Node
    ) -> Node:
        """create a node using information provided by rule, lb, obj, and captured
        and add it as a child of parent"""
        if rule.id not in parent_node.children:
            child_node = Node(
                rule_id=rule.id,
                lower_bound=lb,
                objective=obj,
                num_captured=gmp.popcount(captured),
                pivot_rule_ids=[],
            )
            self.tree.add_node(child_node, parent_node)
            return child_node
        else:
            return parent_node.children[rule.id]

    def _incremental_update_lb(self, v: mpz, y: np.ndarray) -> float:
        return incremental_update_lb(v, y, self.num_train_pts)

    def _incremental_update_obj(self, u: mpz, v: mpz) -> Tuple[float, mpz]:
        return incremental_update_obj(u, v, self.y_mpz, self.num_train_pts)

    # @profile
    def _loopST(
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

        for rule in self.rules[self.current_length : (self.current_length + self.l)]:
            # prune by ruleset length
            # print("rule " + str(rule))
            if (parent_node.num_rules + 1) > length_ub:
                # print("here!")
                continue

            # print( " a " + str(frozenset(parent_node.get_ruleset_ids().union({rule.id}))) )
            # print(self.visited_rule_sets)

            if (
                frozenset(parent_node.get_ruleset_ids().union({rule.id}))
                not in self.visited_rule_sets
            ):
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
                    lookahead_lb = child_node.lower_bound + self.lmbd

                    if lookahead_lb <= self.ub:
                        self.queue.push(
                            (child_node, not_captured),
                            key=child_node.lower_bound,  # the choice of key shouldn't matter for complete enumeration
                        )
                    # if obj <= self.ub:
                    ruleset = child_node.get_ruleset_ids()
                    # print("ruleset " + str(ruleset))
                    # if return_objective:
                    #    yield (ruleset, child_node.objective)
                    # else:
                    self.visited_rule_sets.add(frozenset(ruleset))

                    yield (ruleset, (child_node, not_captured), obj)


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
