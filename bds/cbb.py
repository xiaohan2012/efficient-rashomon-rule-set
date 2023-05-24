# constrained branch-and-bound
import itertools
from typing import Dict, Iterable, List, Optional, Tuple, Union

import gmpy2 as gmp
import numpy as np
from gmpy2 import mpfr, mpz
from logzero import logger
from numba import jit

from .bb import BranchAndBoundNaive
from .bounds import (
    find_equivalence_points,
    get_equivalent_point_lb,
    incremental_update_lb,
    incremental_update_obj,
    prefix_specific_length_upperbound,
)
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
    count_iter,
    get_indices_and_indptr,
    get_max_nz_idx_per_row,
    mpz_all_ones,
)


@jit(nopython=True, cache=True)
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
    z: the parity states vector for the constraints of a given preifx, 0 means 'even' and 1 means 'odd'
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


@jit(nopython=True, cache=True)
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
    # @profile
    def _simplify_constraint_system(
        self, A: np.ndarray, t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """simplify the constraint system using reduced row echelon form"""
        logger.debug("simplifying A x = t using rref")
        A_rref, t_rref, rank = extended_rref(
            GF(A.astype(int)), GF(t.astype(int)), verbose=False
        )

        n_total_entries = np.prod(A.shape)

        logger.debug(
            "density(A_rref) = {:.3%} (from {:.1%})".format(
                bin_array(A_rref).sum() / n_total_entries, A.sum() / n_total_entries
            )
        )
        return bin_array(A_rref), bin_array(t_rref), rank

    # @profile
    def generate(self, return_objective=False) -> Iterable:
        if not self.is_linear_system_solvable:
            logger.debug("abort the search since the linear system is not solvable")
            yield from ()
        else:
            yield from super(ConstrainedBranchAndBoundNaive, self).generate(
                return_objective
            )

    # @profile
    def reset(self, A: np.ndarray, t: np.ndarray):
        self.setup_constraint_system(A, t)
        super(ConstrainedBranchAndBoundNaive, self).reset()

    def setup_constraint_system(self, A: np.ndarray, t: np.ndarray):
        """set the constraint system, e.g., simplify the system"""
        logger.debug("setting up the parity constraint system")
        assert_binary_array(t)

        # simplify the constraint system
        # TODO: if the constraint system tends to be denser, do not use the rref version
        self.A, self.t, rank = self._simplify_constraint_system(A, t)
        self.is_linear_system_solvable = (self.t[rank:] == 0).all()

        # auxiliary data structures for caching and better performance
        self.max_nz_idx_array = get_max_nz_idx_per_row(self.A)

        self.A_indices, self.A_indptr = get_indices_and_indptr(self.A)

        assert (
            self.A.shape[0] == self.t.shape[0]
        ), f"dimension mismatch: {self.A.shape[0]} != {self.t.shape[0]}"

        self.num_constraints = int(self.A.shape[0])

    def reset_queue(self):
        self.queue: Queue = Queue()
        not_captured = self._not_captured_by_default_rule()

        # the undetermined vector:
        # one entry per constraint, 1 means the constraint cannot be evaluated and 0 otherwise
        u = bin_ones(self.num_constraints)

        # the satisfication status constraint
        # means unsatisified, 1 means satisfied
        s = bin_zeros(self.num_constraints)
        # the parity status constraint
        # 0 mean an even number of rules are selected
        # 1 mean an odd number of rules are selected
        z = bin_zeros(self.num_constraints)

        item = (self.tree.root, not_captured, u, s, z)
        self.queue.push(item, key=0)

    def __post_init__(self):
        # disable equivalent_points for now
        # self._find_equivalent_points()
        pass

    def _find_equivalent_points(self):
        # call it only once after initialization
        _, self._pt2rules, self._equivalent_pts = find_equivalence_points(
            self.y_np, self.rules
        )
        print("number of points: {}".format(self.y_np.shape[0]))
        print("number of equivalent points: {}".format(len(self._equivalent_pts)))

    def _check_if_not_unsatisfied(
        self, rule: Rule, u: np.ndarray, s: np.ndarray, z: np.ndarray
    ):
        """wrapper of check_if_not_unsatisfied"""
        return check_if_not_unsatisfied(
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
        length_ub = prefix_specific_length_upperbound(
            parent_lb, parent_node.num_rules, self.lmbd, self.ub
        )

        # here we assume the rule ids are consecutive integers
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

                # the following variables might be assigned later
                # and if assigned
                # will be assigned only once in the check of the current rule
                child_node = None
                up = None

                # apply look-ahead bound
                lookahead_lb = (
                    lb
                    # equivalent point bound disabled for now, due to slower speed
                    # + get_equivalent_point_lb(  # belongs to equivalent point bound
                    #     captured, self._pt2rules, self._equivalent_pts
                    # )
                    + self.lmbd  # belongs to 'look-ahead' bound
                )
                if lookahead_lb <= self.ub:
                    up, sp, zp, not_unsatisfied = self._check_if_not_unsatisfied(
                        rule, u, s, z
                    )
                    if not_unsatisfied:
                        child_node = self._create_new_node_and_add_to_tree(
                            rule, lb, obj, captured, parent_node
                        )
                        self.queue.push(
                            (child_node, not_captured, up, sp, zp),
                            key=child_node.lower_bound,
                        )

                if obj <= self.ub:
                    if up is None:
                        # do the checking if not done
                        up, sp, zp, not_unsatisfied = self._check_if_not_unsatisfied(
                            rule, u, s, z
                        )

                    if check_if_satisfied(up, sp, zp, self.t):
                        if child_node is None:
                            # compute child_node if not done
                            child_node = self._create_new_node_and_add_to_tree(
                                rule, lb, obj, captured, parent_node
                            )
                        ruleset = child_node.get_ruleset_ids()

                        if return_objective:
                            yield (ruleset, child_node.objective)
                        else:
                            yield ruleset
