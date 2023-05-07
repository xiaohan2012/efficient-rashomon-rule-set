# constrained branch-and-bound
from typing import Dict, List, Optional, Tuple, Union

import gmpy2 as gmp

import numpy as np
from logzero import logger
from gmpy2 import mpz, mpfr

from .bb import BranchAndBoundNaive, incremental_update_lb, incremental_update_obj
from .bounds_utils import *
from .bounds_v2 import equivalent_points_bounds, rule_set_size_bound_with_default
from .cache_tree import CacheTree, Node
from .queue import Queue
from .rule import Rule
from .utils import assert_binary_array, bin_ones, bin_zeros, mpz_all_ones


# @profile
def check_if_not_unsatisfied(
    j: int,
    A: np.ndarray,
    t: np.ndarray,
    u: mpz,
    s: mpz,
    z: mpz,
    *,
    rule2cst: Optional[Dict[int, List[int]]] = None,
    max_nz_idx_array: Optional[np.ndarray] = None,
) -> Tuple[mpz, mpz, mpz, bool]:
    """
    given:

    j: the index of the rule to be inserted (we assume rules are 1-indexed, i.e., the first rule's index is 1)
    A: the constraint matrix
    t: target parity vector
    u: the 'undetermined' vector for the constraints of a given prefix, 1 means "undecided" and 0 means "decided"
    s: the satisfication vector for the constraints of a given prefix, 1 means 'satisfied and 0 means 'unsatisfied'
    z: 'parity states vector for the constraints of a given preifx, 0 means 'even' and 1 means 'odd'
    rule2cst (optional): mapping from rule index to the indices of constraints that the rule is present
        provide it for better performance
    max_nz_idx_array (optional): the array of largest non-zero idx per constraint
        provide it for better performance

    (note that A and t determines the parity constraint system)

    return:

    - the updated undetermined vector after inserting the jth rule into the prefix
    - the updated satisfaction vector after inserting the jth rule into the prefix
    - the updated parity constraint
    - whether the constraint system is still not unsatisfied
    """
    # print(f"==== checking the parity system ===== ")
    # assert_binary_array(z)
    assert_binary_array(t)
    # assert s.shape == z.shape == t.shape

    up, sp, zp = mpz(u), mpz(s), mpz(z)
    # s.copy(), z.copy()
    num_constraints, num_variables = A.shape

    if rule2cst is None:
        # without caching
        iter_obj = [i for i in range(num_constraints) if A[i, j - 1]]
    else:
        # with caching
        iter_obj = rule2cst[j]

    for i in iter_obj:
        if gmp.bit_test(up, i) == 1:  # the ith constraint is undetermined
            zp = gmp.bit_flip(zp, i)  # flip the parity value

            # obtain the maximum non-zero index for the current constraint
            # either from cache or caculation from scratch
            if max_nz_idx_array is None:
                max_nz_idx = A[i].nonzero()[0].max()
            else:
                max_nz_idx = max_nz_idx_array[i]

            if j == (max_nz_idx + 1):  # we can evaluate this constraint
                up = gmp.bit_clear(up, i)  # the ith constraint is determined
                if gmp.bit_test(zp, i) == t[i]:
                    # this constraint evaluates to true
                    # we do not return because we might need to consider remaining constraints
                    # print(f"and it is satisfied")
                    sp = gmp.bit_set(sp, i)
                else:
                    # this constraint evaluates to false, thus the system evaluates to false
                    # print(f"and it is unsatisfied")
                    sp = gmp.bit_clear(sp, i)
                    return up, sp, zp, False
    return up, sp, zp, True


def check_if_satisfied(u: mpz, s: mpz, z: mpz, t: np.ndarray) -> bool:
    """check if yielding the current prefix as solution satisfies the parity constraint

    the calculation is based on incremental results from previous parity constraint checks
    """
    assert_binary_array(t)

    num_constraints = t.shape[0]
    for i in range(num_constraints):
        if (gmp.bit_test(u, i) == 0) and (
            gmp.bit_test(s, i) == 0
        ):  # constraint is determiend but failed
            return False
        elif (gmp.bit_test(u, i) == 1) and (
            gmp.bit_test(z, i) != t[i]
        ):  # constraint is undetermiend
            return False
        # elif (s[i] == -1) and (z[i] != t[i]):  # undecided
        #     return False
    return True


class ConstrainedBranchAndBoundNaive(BranchAndBoundNaive):
    def reset_queue(self, A: np.ndarray, t: np.ndarray):
        self.queue: Queue = Queue()
        not_captured = self._not_captured_by_default_rule()

        # assign the parity constraint system
        self.A = A
        self.t = t
        num_constraints = self.A.shape[0]

        # auxiliary data structures for caching and better performance
        self.max_nz_idx_array = np.array(
            [
                (nz.max() if len((nz := A[i].nonzero()[0])) > 0 else -1)
                for i in range(self.A.shape[0])
            ]
        )

        self.rule2cst = {
            r.id: list(
                map(
                    int,  # convert to int for mpz.bit_test campatibility
                    self.A[:, r.id - 1].nonzero()[0],
                )
            )
            for r in self.rules
        }

        assert (
            self.A.shape[0] == self.t.shape[0]
        ), f"dimension mismatch: {self.A.shape[0]} != {self.t.shape[0]}"

        num_constraints = self.A.shape[0]

        # the undetermined vector:
        # one entry per constraint, 1 means the constraint cannot be evaluated and 0 otherwise
        u = mpz_all_ones(num_constraints)
        # the satisfication status constraint
        # means unsatisified, 1 means satisfied
        s = mpz()
        # the parity status constraint
        # 0 mean an even number of rules are selected
        # 1 mean an odd number of rules are selected
        z = mpz()

        item = (self.tree.root, not_captured, u, s, z)
        self.queue.push(item, key=0)

    def reset(self, A: np.ndarray, t: np.ndarray):
        self.reset_tree()
        self.reset_queue(A, t)

    # @profile
    def _loop(
        self,
        parent_node: Node,
        parent_not_captured: mpz,
        u: mpz,
        s: mpz,
        z: mpz,
        return_objective=False,
    ):
        """
        check one node in the search tree, update the queue, and yield feasible solution if exists
        parent_node: the current node/prefix to check
        parent_not_captured: postives not captured by the current prefix
        u: the undetermined vector
        s: the satisfifaction state vector
        z: the parity state vector
        return_objective: True if return the objective of the evaluated node
        """
        parent_lb = parent_node.lower_bound
        for rule in self.rules:
            if rule.id > parent_node.rule_id:
                # logger.debug(f"considering rule {rule.id}")
                captured = self._captured_by_rule(rule, parent_not_captured)
                lb = (
                    parent_lb
                    + self._incremental_update_lb(captured, self.y)
                    + self.lmbd
                )
                if lb <= self.ub:
                    up, sp, zp, not_unsatisfied = check_if_not_unsatisfied(
                        rule.id,
                        self.A,
                        self.t,
                        u,
                        s,
                        z,
                        # provide the following cache data for better performance
                        rule2cst=self.rule2cst,
                        max_nz_idx_array=self.max_nz_idx_array,
                    )
                    if not_unsatisfied:
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
                            (child_node, not_captured, up, sp, zp),
                            key=child_node.lower_bound,  # TODO: consider other types of prioritization, does the prioritization method count for enumeration problem?
                        )

                        if obj <= self.ub and check_if_satisfied(up, sp, zp, self.t):
                            ruleset = child_node.get_ruleset_ids()
                            # logger.debug(
                            #     f"yield rule set {ruleset}: {child_node.objective:.4f} (obj) <= {self.ub:.4f} (ub)"
                            # )
                            if return_objective:
                                yield (ruleset, child_node.objective)
                            else:
                                yield ruleset


class ConstrainedBranchAndBoundV1(BranchAndBoundNaive):
    def reset_queue(self, A: np.ndarray, t: np.ndarray):
        self.queue: Queue = Queue()
        not_captured = bin_ones(self.y.shape)  # the dafault rule captures nothing

        # assign the parity constraint system
        self.A = A
        self.t = t

        self.max_nz_idx_array = np.array(
            [
                (nz.max() if len((nz := A[i].nonzero()[0])) > 0 else -1)
                for i in range(self.A.shape[0])
            ]
        )

        assert (
            self.A.shape[0] == self.t.shape[0]
        ), f"dimension mismatch: {self.A.shape[0]} != {self.t.shape[0]}"

        num_constraints = self.A.shape[0]
        # the satisfication status constraint
        # -1 means ?, 0 means unsatisified, 1 means satisfied
        s = np.ones(num_constraints, dtype=int) * -1
        # the parity status constraint
        # 0 mean an even number of rules are selected
        # 1 mean an odd number of rules are selected
        z = bin_zeros(num_constraints)

        item = (self.tree.root, not_captured, s, z)
        self.queue.push(item, key=0)

    def reset(self, A: np.ndarray, t: np.ndarray):
        self.reset_tree()
        self.reset_queue(A, t)

    # override method from the base class so we can compute the equivalence classes before starting the search
    # i guess we could instead compute self.equivalence_classes upon initialization?
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

    # @profile
    def _loop(
        self,
        parent_node: Node,
        parent_not_captured: np.ndarray,
        s: np.ndarray,
        z: np.ndarray,
        X_trn: np.array,
        equivalence_classes: dict,
        return_objective=False,
    ):
        """
        check one node in the search tree, update the queue, and yield feasible solution if exists

        parent_node: the current node/prefix to check
        parent_not_captured: postives not captured by the current prefix
        s: the satisfifaction state vector
        z: the parity state vector
        return_objective: True if return the objective of the evaluated node
        """
        parent_lb = parent_node.lower_bound
        for rule in self.rules:
            if rule.id > parent_node.rule_id:
                # logger.debug(f"considering rule {rule.id}")
                sp, zp, not_unsatisfied = check_if_not_unsatisfied(
                    rule.id, self.A, self.t, s, z, self.max_nz_idx_array
                )
                if not_unsatisfied:
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
                            (child_node, not_captured, sp, zp),
                            key=child_node.lower_bound,  # TODO: consider other types of prioritization
                        )

                        if obj <= self.ub and check_if_satisfied(sp, zp, self.t):
                            ruleset = child_node.get_ruleset_ids()
                            # logger.debug(
                            #     f"yield rule set {ruleset}: {child_node.objective:.4f} (obj) <= {self.ub:.4f} (ub)"
                            # )
                            if return_objective:
                                yield (ruleset, child_node.objective)
                            else:
                                yield ruleset
