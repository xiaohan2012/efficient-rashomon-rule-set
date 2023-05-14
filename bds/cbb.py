# constrained branch-and-bound
import itertools
from typing import Dict, Iterable, List, Optional, Tuple, Union

import gmpy2 as gmp
import numpy as np
from gmpy2 import mpfr, mpz
from logzero import logger
from numba import jit

from .bb import BranchAndBoundNaive, incremental_update_lb, incremental_update_obj
from .bounds_utils import *
from .bounds_v2 import equivalent_points_bounds, rule_set_size_bound_with_default
from .cache_tree import CacheTree, Node
from .gf2 import GF, extended_rref
from .queue import Queue
from .rule import Rule
from .utils import (
    assert_binary_array,
    bin_array,
    bin_ones,
    bin_zeros,
    get_max_nz_idx_per_row,
    mpz_all_ones,
    get_indices_and_indptr,
)


@jit(nopython=True)
def check_if_not_unsatisfied(
    j: int,
    A: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    s: np.ndarray,
    z: np.ndarray,
    A_indices: np.ndarray,
    A_indptr: np.ndarray,
    max_nz_idx_array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    given:

    j: the index of the rule to be inserted (we assume rules are 1-indexed, i.e., the first rule's index is 1)
    A: the constraint matrix
    t: target parity vector
    u: the 'undetermined' vector for the constraints of a given prefix, 1 means "undecided" and 0 means "decided"
    s: the satisfication vector for the constraints of a given prefix, 1 means 'satisfied and 0 means 'unsatisfied'
    z: 'parity states vector for the constraints of a given preifx, 0 means 'even' and 1 means 'odd'
    A_indices and A_indptr: where the non-zero row indices for column/rule i are stored in A_indices[A_indptr[i-1]:A_indptr[i]]
        provide it for better performance
    max_nz_idx_array: the array of largest non-zero idx per constraint
        provide it for better performance

    (note that A and t determines the parity constraint system)

    return:

    - the updated undetermined vector after inserting the jth rule into the prefix
    - the updated satisfaction vector after inserting the jth rule into the prefix
    - the updated parity constraint
    - whether the constraint system is still not unsatisfied
    """
    cst_idxs = A_indices[
        A_indptr[j - 1] : A_indptr[j]
    ]  # get the constraint (row) indices corresponind to rule j

    # compute max_nz_idx_array from scratch if not given
    up: np.ndarray = u.copy()
    sp: np.ndarray = s.copy()
    zp: np.ndarray = z.copy()

    num_constraints, num_variables = A.shape

    # iter_obj = [i for i in range(num_constraints) if A[i, j - 1]]

    for i in cst_idxs:
        if up[i]:  # the ith constraint is undetermined
            zp[i] = not zp[i]  # flip the parity value
            # obtain the maximum non-zero index for the current constraint
            # either from cache or caculation from scratch
            max_nz_idx = max_nz_idx_array[i]

            if j == (max_nz_idx + 1):  # we can evaluate this constraint
                up[i] = 0  # the ith constraint is determined
                if zp[i] == t[i]:
                    # this constraint evaluates to true
                    # we do not return because we might need to consider remaining constraints
                    sp[i] = True
                else:
                    # this constraint evaluates to false, thus the system evaluates to false
                    sp[i] = False
                    return up, sp, zp, False
    return up, sp, zp, True


@jit(nopython=True)
def check_if_satisfied(
    u: np.ndarray, s: np.ndarray, z: np.ndarray, t: np.ndarray
) -> bool:
    """check if yielding the current prefix as solution satisfies the parity constraint

    the calculation is based on incremental results from previous parity constraint checks
    """

    num_constraints = t.shape[0]
    for i in range(num_constraints):
        if (not u[i]) and (not s[i]):  # constraint is determiend but failed
            return False
        elif u[i] and (z[i] != t[i]):  # constraint is undetermiend
            return False
    return True


class ConstrainedBranchAndBoundNaive(BranchAndBoundNaive):
    def _simplify_constraint_system(
        self, A: np.ndarray, t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """simplify the constraint system using reduced row echelon form"""
        logger.debug("simplifying A x = t using rref")
        A_rref, t_rref, p = extended_rref(
            GF(A.astype(int)), GF(t.astype(int)), verbose=False
        )
        n_total_entries = np.prod(A.shape)

        logger.debug(
            "density(A_rref) = {:.3%} (from {:.1%})".format(
                bin_array(A_rref).sum() / n_total_entries, A.sum() / n_total_entries
            )
        )
        return bin_array(A_rref), bin_array(t_rref)

    def reset_queue(self, A: np.ndarray, t: np.ndarray):
        assert_binary_array(t)

        self.queue: Queue = Queue()
        not_captured = self._not_captured_by_default_rule()

        # simplify theconstraint system
        self.A, self.t = self._simplify_constraint_system(A, t)

        num_constraints = self.A.shape[0]

        # auxiliary data structures for caching and better performance
        self.max_nz_idx_array = get_max_nz_idx_per_row(self.A)

        self.A_indices, self.A_indptr = get_indices_and_indptr(self.A)

        assert (
            self.A.shape[0] == self.t.shape[0]
        ), f"dimension mismatch: {self.A.shape[0]} != {self.t.shape[0]}"

        num_constraints = int(self.A.shape[0])

        # the undetermined vector:
        # one entry per constraint, 1 means the constraint cannot be evaluated and 0 otherwise
        u = bin_ones(num_constraints)

        # the satisfication status constraint
        # means unsatisified, 1 means satisfied
        s = bin_zeros(num_constraints)
        # the parity status constraint
        # 0 mean an even number of rules are selected
        # 1 mean an odd number of rules are selected
        z = bin_zeros(num_constraints)

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
        u: np.ndarray,
        s: np.ndarray,
        z: np.ndarray,
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

        # here we assume the rule ids are consecutive integers
        for rule in self.rules[parent_node.rule_id :]:
            captured = self._captured_by_rule(rule, parent_not_captured)
            lb = parent_lb + self._incremental_update_lb(captured, self.y) + self.lmbd
            if lb <= self.ub:
                up, sp, zp, not_unsatisfied = check_if_not_unsatisfied(
                    rule.id,
                    self.A,
                    self.t,
                    u,
                    s,
                    z,
                    # provide the following cache data for better performance
                    A_indices=self.A_indices,
                    A_indptr=self.A_indptr,
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
                        key=child_node.lower_bound,
                        # key=child_node.lower_bound / child_node.num_captured,  # using the curiosity function defined in CORELS
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
        data_points2rules, equivalence_classes = find_equivalence_classes(
            X_trn, self.y, self.rules
        )
        while not self.queue.is_empty:
            queue_item = self.queue.pop()
            yield from self._loop(
                *queue_item,
                X_trn,
                data_points2rules,
                equivalence_classes,
                return_objective=return_objective,
            )

    def _loop(
        self,
        parent_node: Node,
        parent_not_captured: np.ndarray,
        s: np.ndarray,
        z: np.ndarray,
        X_trn: np.array,
        data_points2rules: dict,
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

                    flag_rule_set_size = rule_set_size_bound_with_default(
                        parent_node, self.lmbd, self.ub
                    )

                    if not flag_rule_set_size:
                        lb = (
                            parent_lb
                            + incremental_update_lb(captured, self.y)
                            + self.lmbd
                        )

                        flag_equivalent_classes = equivalent_points_bounds(
                            lb,
                            self.lmbd,
                            self.ub,
                            parent_not_captured,
                            X_trn,
                            data_points2rules,
                            equivalence_classes,
                        )  # if true, we prune

                        if not flag_equivalent_classes:
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
