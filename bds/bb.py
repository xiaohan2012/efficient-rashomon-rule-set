import itertools
from copy import deepcopy
from typing import Iterable, List, Optional, Tuple, Union

import gmpy2 as gmp
import numpy as np
from gmpy2 import mpz
from logzero import logger

from .bounds import (
    incremental_update_lb,
    incremental_update_obj,
    prefix_specific_length_upperbound,
)
from .rule import Rule
from .solver_status import SolverStatus
from .types import RuleSet
from .utils import (
    assert_binary_array,
    calculate_lower_bound,
    calculate_obj,
    count_iter,
    mpz_all_ones,
    mpz_set_bits,
)

# logger.setLevel(logging.INFO)

Prefix = Tuple[int]


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
        self.y_np = y.copy()  # np representation of y
        self.rules = deepcopy(rules)
        self.ub = ub
        self.y_mpz = mpz_set_bits(
            mpz(), y.nonzero()[0]
        )  # convert y from np.array to mpz
        self.lmbd = lmbd

        logger.debug(f"calling {self.__class__.__name__} with ub={ub}, lmbd={lmbd}")

        self.num_train_pts = mpz(y.shape[0])

        # false negative rate of the default rule = fraction of positives
        self.default_rule_fnr = mpz(gmp.popcount(self.y_mpz)) / self.num_train_pts

        # counters for performance monitorning
        self.num_lb_evaluations = 0
        self.num_obj_evaluations = 0

        self._check_rule_ids()

        self.__post_init__()

    def _print_rules_and_y(self):
        for r in self.rules:
            print(f"r{r.id:>3}: {bin(r.truthtable)[2:]:>20}")
        print(f"y   : {bin(self.y_mpz)[2:]:>20}")

    def __post_init__(self):
        """hook function to be called after __init__ is called"""

    def _check_rule_ids(self):
        """check the rule ids are consecutive integers starting from 0"""
        rule_ids = np.array([r.id for r in self.rules])
        np.testing.assert_allclose(
            rule_ids,
            np.arange(0, len(rule_ids)),
            err_msg="rule are not ordered by their ids starting from 0",
        )

    def reset(self):
        # logger.debug("initializing search tree and priority queue")
        self.status = SolverStatus()

    def _captured_by_rule(self, rule: Rule, parent_not_captured: mpz):
        """return the captured array for the rule in the context of parent"""
        return parent_not_captured & rule.truthtable

    def _calculate_lb(self, prefix: RuleSet) -> float:
        return calculate_lower_bound(
            self.rules, self.y_np, self.y_mpz, prefix, self.lmbd
        )

    def _calculate_obj(self, prefix: RuleSet) -> float:
        return calculate_obj(self.rules, self.y_np, self.y_mpz, prefix, self.lmbd)

    # @profile
    def generate(self, return_objective=False) -> Iterable:
        while not self.status.is_queue_empty():
            queue_item = self.status.pop_from_queue()
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

    def _not_captured_by_default_rule(self):
        """return the vector of not captured by default rule
        the dafault rule captures nothing
        """
        return mpz_all_ones(self.num_train_pts)

    def reset(self):
        self.status = SolverStatus()

        not_captured = self._not_captured_by_default_rule()
        lb = 0.0
        item = (tuple(), lb, not_captured)
        self.status.push_to_queue(lb, item)

    def _incremental_update_lb(self, v: mpz, y: np.ndarray) -> float:
        return incremental_update_lb(v, y, self.num_train_pts)

    def _incremental_update_obj(self, u: mpz, v: mpz) -> Tuple[float, mpz]:
        return incremental_update_obj(u, v, self.y_mpz, self.num_train_pts)

    # @profile
    def _loop(
        self,
        parent_prefix: Tuple[int],
        parent_lower_bound: float,
        parent_not_captured: mpz,
        return_objective=False,
    ) -> Iterable[Union[Tuple[Prefix, float], Prefix]]:
        """
        check one node in the search tree, update the queue, and yield feasible solution if exists

        parent_prefix: the prefix rule set
        parent_lower_bound: lower bound achieved by the parent prefix
        parent_not_captured: postives not captured by the current prefix
        return_objective: True if return the objective of the evaluated node
        """
        parent_prefix_length = len(parent_prefix)
        # if parent_prefix is [1]
        # start from 2
        # if parent_prefix is []
        # start from 0
        max_rule_idx = max(parent_prefix or [-1])
        length_ub = prefix_specific_length_upperbound(
            parent_lower_bound, parent_prefix_length, self.lmbd, self.ub
        )

        for rule in self.rules[(max_rule_idx + 1) :]:
            # prune by ruleset length
            if (parent_prefix_length + 1) > length_ub:
                continue

            prefix = RuleSet(parent_prefix + (rule.id,))
            captured = self._captured_by_rule(rule, parent_not_captured)
            prefix_lower_bound = (
                parent_lower_bound
                + self._incremental_update_lb(captured, self.y_mpz)
                + self.lmbd
            )
            self.num_lb_evaluations += 1
            if prefix_lower_bound <= self.ub:
                fn_fraction, not_captured = self._incremental_update_obj(
                    parent_not_captured, captured
                )
                self.num_obj_evaluations += 1
                prefix_obj = prefix_lower_bound + fn_fraction

                # apply look-ahead bound
                lookahead_lb = prefix_lower_bound + self.lmbd

                if lookahead_lb <= self.ub:
                    self.status.push_to_queue(
                        prefix_lower_bound,  # the choice of key shouldn't matter for complete enumeration
                        (prefix, prefix_lower_bound, not_captured),
                    )
                if prefix_obj <= self.ub:
                    self.status.add_to_solution_set(prefix)
                    if return_objective:
                        yield (prefix, float(prefix_obj))
                    else:
                        yield prefix


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
