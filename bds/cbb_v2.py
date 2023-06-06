import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Iterable, Set
from logzero import logger

from gmpy2 import mpz, mpfr
from .utils import (
    bin_array,
    assert_binary_array,
    get_max_nz_idx_per_row,
    get_indices_and_indptr,
    bin_ones,
    bin_zeros,
)
from .bounds import prefix_specific_length_upperbound
from .queue import Queue
from .cache_tree import Node
from .bb import BranchAndBoundNaive
from .gf2 import GF, extended_rref


def update_pivot_variables(
    j: int,
    z: np.ndarray,
    t: np.ndarray,
    A_indices: np.ndarray,
    A_indptr: np.ndarray,
    max_nz_idx_array: np.ndarray,
    row2pivot_column: np.ndarray,
) -> Tuple[Set[int], np.ndarray]:
    """
    determine the set of selected pivot rules after adding the jth rule to the current prefix

    note that the rule index must correspond to a non-pivot column

    the parity states vector is provided for the current prefix

    the function also returns the updated parity states vector

    row2pivot_column is the mapping from row id to the corresponding pivot column

    for performance reasons, the following data structures are given:

    A_indices and A_indptr: where the non-zero row indices for column/rule i are stored in A_indices[A_indptr[i-1]:A_indptr[i]]
    max_nz_idx_array: the array of largest non-zero idx per constraint
    """
    if (j - 1) in set(row2pivot_column):
        raise ValueError(f"cannot set pivot variable of column {j - 1}")
    
    cst_idxs = A_indices[
        A_indptr[j - 1] : A_indptr[j]
    ]  # get the constraint (row) indices corresponind to rule j
    selected_rules = set()
    zp: np.ndarray = z.copy()
    for i in cst_idxs:
        zp[i] = not zp[i]  # flip the parity value
        max_nz_idx = max_nz_idx_array[i]

        # this constraint can be determined
        # and we should select the correponding pivot rule
        if (j == (max_nz_idx + 1)) and (zp[i] != t[i]):
            # flip again because due to the addition of pivot rule
            zp[i] = not zp[i]
            selected_rules.add(row2pivot_column[i] + 1)  # +1 because rule is 1-indexed
    return selected_rules, zp


def assign_pivot_variables():
    pass


class ConstrainedBranchAndBoundNaive(BranchAndBoundNaive):
    def _simplify_constraint_system(
        self, A: np.ndarray, t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """simplify the constraint system using reduced row echelon form"""
        logger.debug("simplifying A x = t using rref")
        A_rref, t_rref, rank = extended_rref(
            GF(A.astype(int)), GF(t.astype(int)), verbose=False
        )

        return bin_array(A_rref), bin_array(t_rref), rank

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

        # the parity status constraint
        # 0 mean an even number of rules are selected
        # 1 mean an odd number of rules are selected
        z = bin_zeros(self.num_constraints)

        item = (self.tree.root, not_captured, z)
        self.queue.push(item, key=0)

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
            if lb <= self.ub:  # parent + current rule
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
                # parent + current rule + any next rule
                lookahead_lb = lb + self.lmbd
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
                            # if the lower bound are the same for two nodes, resolve the order by the corresponding ruleset
                            key=(
                                child_node.lower_bound,
                                tuple(child_node.get_ruleset_ids()),
                            ),
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
                            # print("self.queue._items: {}".format(self.queue._items))
                            print(
                                f"yielding {ruleset} with lb {child_node.lower_bound}"
                            )
                            yield ruleset
