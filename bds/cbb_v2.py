from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import gmpy2 as gmp
import numpy as np
from gmpy2 import mpfr, mpz
from logzero import logger
from numba import jit

from .bounds import prefix_specific_length_upperbound
from .cache_tree import Node
from .cbb import ConstrainedBranchAndBoundNaive
from .gf2 import GF, extended_rref
from .queue import Queue
from .rule import Rule, lor_of_truthtable
from .utils import (
    assert_binary_array,
    bin_array,
    bin_ones,
    bin_zeros,
    get_indices_and_indptr,
    get_max_nz_idx_per_row,
)


@jit(nopython=True, cache=True)
def update_pivot_variables(
    j: int,
    z: np.ndarray,
    t: np.ndarray,
    A_indices: np.ndarray,
    A_indptr: np.ndarray,
    max_nz_idx_array: np.ndarray,
    row2pivot_column: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    return:

    - selected pivot rule indices after adding the jth rule to the current prefix
    - the function also returns the updated parity states vector

    note that the rule index must correspond to a non-pivot column

    the parity states vector is provided for the current prefix

    row2pivot_column is the mapping from row id to the corresponding pivot column

    for performance reasons, the following data structures are given:

    A_indices and A_indptr: where the non-zero row indices for column/rule i are stored in A_indices[A_indptr[i-1]:A_indptr[i]]
    max_nz_idx_array: the array of largest non-zero idx per constraint
    """
    # commented out the raise statement because numba does not allow it
    # if (j - 1) in row2pivot_column:
    #     raise ValueError(f"cannot set pivot variable of column {j - 1}")

    cst_idxs = A_indices[
        A_indptr[j - 1] : A_indptr[j]
    ]  # get the constraint (row) indices corresponind to rule j
    zp: np.ndarray = z.copy()

    selected_rules = np.empty(A_indices.shape, np.int_)
    num_rules_selected = 0
    for i in cst_idxs:
        zp[i] = not zp[i]  # flip the parity value
        max_nz_idx = max_nz_idx_array[i]

        # this constraint can be determined
        # and we should select the correponding pivot rule
        if (j == (max_nz_idx + 1)) and (zp[i] != t[i]):
            # flip again because due to the addition of pivot rule
            zp[i] = not zp[i]
            # +1 because rule is 1-indexed
            selected_rules[num_rules_selected] = row2pivot_column[i] + 1
            num_rules_selected += 1
    return selected_rules[:num_rules_selected], zp


@jit(nopython=True, cache=True)
def assign_pivot_variables(
    j: int,
    rank: int,
    z: np.ndarray,
    t: np.ndarray,
    A: np.ndarray,
    # A_indices: np.ndarray,
    # A_indptr: np.ndarray,
    max_nz_idx_array: np.ndarray,
    row2pivot_column: np.ndarray,
) -> np.ndarray:
    """
    return the pivot variables that are assigned to 1s if no rules with index larger than j are added further.
    """
    # if (j - 1) in set(row2pivot_column):
    #     raise ValueError(f"cannot set pivot variable of column {j - 1}")

    # if j > 0:
    #     cst_idxs = A_indices[
    #         A_indptr[j - 1] : A_indptr[j]
    #     ]  # get the constraint (row) indices corresponind to rule j
    # else:
    #     cst_idxs = np.array([], dtype=int)

    # print("cst_idxs: {}".format(cst_idxs))
    selected_rules = np.empty(A.shape[0], np.int_)
    num_rules_selected = 0

    for i in range(rank):  # loop up to rank
        max_nz_idx = max_nz_idx_array[i]
        # print("i: {}".format(i))
        # print("max_nz_idx: {}".format(max_nz_idx))
        # print("z[i]: {}".format(z[i]))
        # print("t[i]: {}".format(t[i]))

        if (
            # the ith constraint is not finalized by j
            (j < (max_nz_idx + 1))
            # j is not the default rule and jth rule is not relevant for the ith constraint
            or (j > 0 and A[i][j - 1] == 0)
        ) and (z[i] != t[i]):
            # +1 because rule is 1-indexed
            selected_rules[num_rules_selected] = row2pivot_column[i] + 1
            num_rules_selected += 1
    return selected_rules[:num_rules_selected]


class ConstrainedBranchAndBound(ConstrainedBranchAndBoundNaive):
    def _update_pivot_variables(self, rule: Rule, z: np.ndarray):
        """a wrapper around update_pivot_variables"""
        return update_pivot_variables(
            rule.id,
            z,
            self.t,
            self.A_indices,
            self.A_indptr,
            self.max_nz_idx_array,
            self.row2pivot_column,
        )

    def _assign_pivot_variables(self, rule: Rule, z: np.ndarray):
        """a wrapper around assign_pivot_variables"""
        return assign_pivot_variables(
            rule.id,
            self.rank,
            z,
            self.t,
            self.A,
            # self.A_indices,
            # self.A_indptr,
            self.max_nz_idx_array,
            self.row2pivot_column,
        )

    def setup_constraint_system(self, A: np.ndarray, t: np.ndarray):
        """set the constraint system, e.g., simplify the system"""
        # perform rref
        super(ConstrainedBranchAndBound, self).setup_constraint_system(A, t)

        self.num_vars = int(self.A.shape[1])

        self.pivot_rule_idxs = set(
            map(lambda v: v + 1, self.pivot_columns)
        )  # +1 because rules are 1-indexed
        self.free_rule_idxs = (
            set(map(lambda v: v + 1, range(self.num_vars))) - self.pivot_rule_idxs
        )
        # mapping from row index to the pivot column index
        self.row2pivot_column = np.array(self.pivot_columns, dtype=int)
        # print("self.row2pivot_column: {}".format(self.row2pivot_column))
        # print("self.A:\n {}".format(self.A.astype(int)))
        # print("self.t:\n {}".format(self.t.astype(int)))

    def reset_queue(self):
        self.queue: Queue = Queue()
        not_captured = self._not_captured_by_default_rule()

        # the parity status vector
        # 0 mean an even number of rules are selected
        # 1 mean an odd number of rules are selected
        z = bin_zeros(self.num_constraints)

        item = (self.tree.root, not_captured, z)
        self.queue.push(item, key=0)

    def _create_new_node_and_add_to_tree(
        self,
        rule: Rule,
        lb: mpfr,
        obj: mpfr,
        captured: mpz,
        parent_node: Node,
        pivot_rules_to_add: List[Rule] = [],
    ) -> Node:
        """create a node using information provided by rule, lb, obj, and captured
        and add it as a child of parent"""
        if rule.id not in parent_node.children:
            child_node = Node(
                rule_id=rule.id,
                lower_bound=lb,
                objective=obj,
                num_captured=gmp.popcount(captured),
                pivot_rule_ids=[r.id for r in pivot_rules_to_add],
            )
            self.tree.add_node(child_node, parent_node)
            return child_node
        else:
            return parent_node.children[rule.id]

    def _get_rules_by_idxs(self, idxs: List[int]) -> List[Rule]:
        """extract rules from a list of indices, the indices start from 1"""
        return [self.rules[idx - 1] for idx in idxs]

    def generate_solution_at_root(self, return_objective=False) -> Iterable:
        """check the solution at the root, e.g., all free variables assigned to zero"""
        # add more rules if needed to satistify Ax=b
        default_rule = Rule(0, "rule-0", 0, mpz())
        rule_idxs = self._assign_pivot_variables(
            default_rule, bin_zeros(self.num_constraints)
        )
        rules_to_add = self._get_rules_by_idxs(rule_idxs)

        # calculate the objective
        captured = lor_of_truthtable(rules_to_add)
        num_mistakes = gmp.popcount(captured ^ self.y_mpz)

        # print("bin(captured): {}".format(bin(captured)))
        obj = num_mistakes / self.num_train_pts + len(rule_idxs) * self.lmbd
        # print("num_mistakes: {}".format(num_mistakes))
        sol = {0} | set(rule_idxs)
        # logger.debug(f"solution at root: {sol} (obj={obj:.2f})")
        if len(sol) > 1 and obj <= self.ub:
            if return_objective:
                yield sol, obj
            else:
                yield sol

    def generate(self, return_objective=False) -> Iterable:
        if not self.is_linear_system_solvable:
            logger.debug("abort the search since the linear system is not solvable")
            yield from ()
        else:
            yield from self.generate_solution_at_root(return_objective)

            while not self.queue.is_empty:
                queue_item = self.queue.pop()
                yield from self._loop(*queue_item, return_objective=return_objective)

    # @profile
    def _loop(
        self,
        parent_node: Node,
        u: mpz,
        z: np.ndarray,
        return_objective=False,
    ):
        """
        check one node in the search tree, update the queue, and yield feasible solution if exists
        parent_node: the current node/prefix to check
        parent_not_captured: postives not captured by the current prefix
        z: the parity states vector
        return_objective: True if return the objective of the evaluated node
        """
        parent_lb = parent_node.lower_bound
        length_ub = prefix_specific_length_upperbound(
            parent_lb, parent_node.num_rules, self.lmbd, self.ub
        )

        # logger.debug(f"parent node: {parent_node.get_ruleset_ids()}")
        # here we assume the rule ids are consecutive integers
        for rule in self.rules[parent_node.rule_id :]:
            # consider adding only free rules
            # since the addition of pivot rules are determined "automatically" by Ax=b
            if rule.id in self.pivot_rule_idxs:
                continue

            # prune by ruleset length
            if (parent_node.num_rules + 1) > length_ub:
                continue

            e1_idxs, zp = self._update_pivot_variables(rule, z)
            e1_idxs = set(e1_idxs)
            e1 = [rule] + self._get_rules_by_idxs(e1_idxs)

            # logger.debug(f"adding free rule {rule.id}")
            # logger.debug(
            #     "[update_pivot_variables] e1 = {}".format(list(e1_idxs) + [rule.id])
            # )
            # consider the captured points not captured by the prefix
            v1 = lor_of_truthtable(e1) & u
            # for r in e1:
            #     logger.debug(f"r{r.id:2d}: {bin(r.truthtable)[2:]:>25}")
            # logger.debug(f"v1: {bin(v1)[2:]:>26}")
            # logger.debug(f"y: {bin(self.y_mpz)[2:]:>27}")
            lb = (
                parent_lb
                + self._incremental_update_lb(v1, self.y_mpz)
                + len(e1) * self.lmbd
            )
            # logger.debug(f"parent_lb: {parent_lb}")
            # logger.debug(
            #     f"addition to lb: {self._incremental_update_lb(v1, self.y_mpz)}"
            # )
            # logger.debug(f"lb: {lb}")
            # logger.debug(f"len(e1): {len(e1)}")
            if lb <= self.ub:  # parent + current rule
                e2_idxs = set(self._assign_pivot_variables(rule, zp))
                e2 = self._get_rules_by_idxs(e2_idxs)

                # logger.debug("[assign_pivot_variables] e2 = {}".format(e2_idxs))
                # captured by the prefix, current rule, and rules introduced by update_pivot_variables

                # a verbose way to do bitwise inverse on u
                # the ~ operator returns a negative number, e.g., -0b1000000000
                # and applying lor gives weird result
                # TODO: is there a way to do bitwise inverse in the "expected" way
                # def mpz_comp(v):
                #     not_v = v
                #     for i in range(len(self.rules)):
                #         not_v = gmp.bit_flip(not_v, i)
                #     return not_v

                not_u = ~u
                w = v1 | not_u

                # logger.debug(f"v1: {bin(v1)[2:]:>26}")
                # logger.debug(f"not_u: {bin(not_u)[2:]:>23}")
                # logger.debug(f"w: {bin(w)[2:]:>27}")

                not_w = ~w
                # logger.debug(f"not_w: {bin(not_w)[2:]:>23}")
                # logger.debug(f"e2: {bin(lor_of_truthtable(e2))[2:]:>26}")
                # captured by e2 but not by e1 + d'
                v2 = lor_of_truthtable(e2) & not_w

                # logger.debug(f"v2: {bin(v2)[2:]:>26}")
                # logger.debug(f"y: {bin(self.y_mpz)[2:]:>27}")
                # the FP mistakes incurred by e2
                fp_fraction = self._incremental_update_lb(v2, self.y_mpz)

                # the FN mistakes incurred by e2
                fn_fraction, _ = self._incremental_update_obj(not_w, v2)
                # logger.debug(f"fp_fraction: {fp_fraction}")
                # logger.debug(f"fn_fraction: {fn_fraction}")
                # logger.debug(f"len(e2): {len(e2)}")
                obj = lb + fn_fraction + fp_fraction + (self.lmbd * len(e2))

                # the child_node might be assigned later
                # and if assigned
                # will be assigned only once during the the check of the current rule
                child_node = None

                # apply look-ahead bound
                # parent + current rule + any next rule
                lookahead_lb = lb + self.lmbd
                if lookahead_lb <= self.ub:
                    child_node = self._create_new_node_and_add_to_tree(
                        rule,
                        lb,
                        obj,
                        w,
                        parent_node,
                        e1,
                    )
                    self.queue.push(
                        (child_node, not_w, zp),
                        key=child_node.lower_bound,
                    )

                if obj <= self.ub:
                    if child_node is None:
                        # compute child_node if not done
                        child_node = self._create_new_node_and_add_to_tree(
                            rule,
                            lb,
                            obj,
                            w,
                            parent_node,
                            e1,
                        )
                    ruleset = child_node.get_ruleset_ids() | e2_idxs

                    # logger.debug(f"yielding {ruleset} with obj {obj:.2f}")
                    if return_objective:
                        yield (ruleset, child_node.objective)
                    else:
                        # print("self.queue._items: {}".format(self.queue._items))
                        yield ruleset
