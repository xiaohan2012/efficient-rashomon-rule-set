# constrained branch-and-bound
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from logzero import logger

from .utils import assert_binary_array, bin_ones, bin_zeros
from .cache_tree import CacheTree, Node
from .queue import Queue
from .rule import Rule
from .bb import BranchAndBoundNaive, incremental_update_lb, incremental_update_obj


def check_if_not_unsatisfied(
    j: int, A: np.ndarray, t: np.ndarray, s: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    given:

    j: the index of the rule to be inserted (we assume rules are 1-indexed, i.e., the first rule's index is 1)
    A: the constraint matrix
    t: target parity vector
    s: the satisfication vector of a given prefix, 0 means 'unsatisfied', 1 means 'satisfied', and -1 means "undetermined"
    z: parity states vector of a given preifx, 0 means 'even' and 1 means 'odd'

    (note that A and t determines the parity constraint system)

    return:

    - the updated satisfaction vector after inserting the jth rule into the prefix
    - the updated parity constraint
    - whether the constraint system is still not unsatisfied
    """
    # print(f"==== checking the parity system ===== ")
    assert_binary_array(z)
    assert_binary_array(t)
    assert s.shape == z.shape == t.shape

    sp, zp = s.copy(), z.copy()
    num_constraints, num_variables = A.shape
    for i in range(num_constraints):
        if s[i] == -1:  # s[i] == ?
            # print(f"constraint {i+1} is undetermined")
            if A[i, j - 1]:  # j-1 because we assume rule index is 1-indexed
                # print(f"rule {j} is active in this constraint")
                # print(f"parity value from {zp[i]} to {np.invert(zp[i])}")
                zp[i] = np.invert(zp[i])  # flip the sign
                max_nz_idx = A[i].nonzero()[0].max()  # TODO: cache this information
                if j == (max_nz_idx + 1):  # we can evaluate this constraint
                    # print(f"we can evaluate this constraint")
                    if zp[i] == t[i]:
                        # this constraint evaluates to tue, but we need to consider remaining constraints
                        # print(f"and it is satisfied")
                        sp[i] = 1
                    else:
                        # this constraint evaluates to false, thus the system evaluates to false
                        # print(f"and it is unsatisfied")
                        sp[i] = 0
                        return sp, zp, False
    return sp, zp, True


def check_if_satisfied(s: np.ndarray, z: np.ndarray, t: np.ndarray) -> bool:
    """check if yielding the current prefix as solution satisfies the parity constraint

    the calculation is based on incremental results from previous parity constraint checks
    """
    assert_binary_array(z)
    assert_binary_array(t)

    assert s.shape == z.shape == t.shape

    num_constraints = z.shape[0]
    for i in range(num_constraints):
        if s[i] == 0:  # failed this constraint
            return False
        elif (s[i] == -1) and (z[i] != t[i]):  # undecided
            return False
    return True


class ConstrainedBranchAndBoundNaive(BranchAndBoundNaive):
    def reset_queue(self, A: np.ndarray, t: np.ndarray):
        self.queue: Queue = Queue()
        not_captured = bin_ones(self.y.shape)  # the dafault rule captures nothing
        
        # assign the parity constraint system
        self.A = A
        self.t = t

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
        
    def _loop(
        self,
            parent_node: Node, parent_not_captured: np.ndarray,
            s: np.ndarray,
            z: np.ndarray,
            return_objective=False
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
                logger.debug(f"considering rule {rule.id}")
                sp, zp, not_unsatisfied = check_if_not_unsatisfied(
                    rule.id, self.A, self.t, s, z
                )
                if not_unsatisfied:
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
                            (child_node, not_captured, sp, zp),
                            key=child_node.lower_bound,  # TODO: consider other types of prioritization
                        )

                        if obj <= self.ub and check_if_satisfied(sp, zp, self.t):
                            logger.debug(f"yield rule {rule.id} as a feasible solution")
                            ruleset = child_node.get_ruleset_ids()
                            if return_objective:
                                yield (ruleset, child_node.objective)
                            else:
                                yield ruleset
