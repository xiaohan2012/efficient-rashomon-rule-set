from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import gmpy2 as gmp
from gmpy2 import mpfr, mpz
from logzero import logger

from .cbb import ConstrainedBranchAndBoundNaive
from .bounds import prefix_specific_length_upperbound
from .cache_tree import Node
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


def assign_pivot_variables(
    j: int,
    rank: int,
    z: np.ndarray,
    t: np.ndarray,
    A_indices: np.ndarray,
    A_indptr: np.ndarray,
    max_nz_idx_array: np.ndarray,
    row2pivot_column: np.ndarray,
) -> Set[int]:
    """
    return the set of pivot variables that are assigned to 1s if no rules with index larger than j are added further.
    """
    if (j - 1) in set(row2pivot_column):
        raise ValueError(f"cannot set pivot variable of column {j - 1}")

    if j > 0:
        cst_idxs = A_indices[
            A_indptr[j - 1] : A_indptr[j]
        ]  # get the constraint (row) indices corresponind to rule j
    else:
        cst_idxs = np.array([], dtype=int)

    # print("cst_idxs: {}".format(cst_idxs))
    selected_rules = set()
    for i in range(rank):
        max_nz_idx = max_nz_idx_array[i]
        print("i: {}".format(i))
        print("max_nz_idx: {}".format(max_nz_idx))
        print("z[i]: {}".format(z[i]))
        print("t[i]: {}".format(t[i]))
        # this constraint is not determined yet or the added rule is not relevant
        if ((j < (max_nz_idx + 1)) or (i not in cst_idxs)) and (z[i] != t[i]):
            selected_rules.add(row2pivot_column[i] + 1)  # +1 because rule is 1-indexed
    return selected_rules


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
            self.A_indices,
            self.A_indptr,
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
        print("self.row2pivot_column: {}".format(self.row2pivot_column))
        print("self.A:\n {}".format(self.A.astype(int)))
        print("self.t:\n {}".format(self.t.astype(int)))

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
        sol = {0} | rule_idxs
        logger.debug(f"solution at root: {sol} (obj={obj:.2f})")
        if len(sol) > 1 and obj <= self.ub:
            if return_objective:
                yield sol, obj
            else:
                yield sol

    def generate(self, return_objective=False) -> Iterable:
        yield from self.generate_solution_at_root(return_objective)

        while not self.queue.is_empty:
            queue_item = self.queue.pop()
            yield from self._loop(*queue_item, return_objective=return_objective)

    def _loop(
        self,
        parent_node: Node,
        parent_not_captured: mpz,
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

        logger.debug(f"parent node: {parent_node.get_ruleset_ids()}")
        # here we assume the rule ids are consecutive integers
        for rule in self.rules[parent_node.rule_id :]:
            # consider adding only free rules
            # since the addition of pivot rules are determined "automatically" by Ax=b
            if rule.id in self.pivot_rule_idxs:
                continue

            # prune by ruleset length
            if (parent_node.num_rules + 1) > length_ub:
                continue

            pivot_rule_idxs_1, zp = self._update_pivot_variables(rule, z)
            pivot_rules_to_add_1 = self._get_rules_by_idxs(pivot_rule_idxs_1)
            rules_to_add = [rule] + pivot_rules_to_add_1

            logger.debug(f"adding rule {rule.id}")
            logger.debug(
                "[update_pivot_variables]: indices of pivot rules to add: {}".format(
                    pivot_rule_idxs_1
                )
            )
            captured_1 = lor_of_truthtable(rules_to_add)

            lb = (
                parent_lb
                + self._incremental_update_lb(
                    # consider the captured points not captured by the prefix
                    captured_1 & parent_not_captured,
                    self.y_mpz,
                )
                + len(rules_to_add) * self.lmbd
            )
            logger.debug(f"parent_lb: {parent_lb}")
            logger.debug(
                f"addition to lb: {self._incremental_update_lb(captured_1 & parent_not_captured, self.y_mpz)}"
            )
            logger.debug(f"lb: {lb}")
            logger.debug(f"len(rules_to_add): {len(rules_to_add)}")
            if lb <= self.ub:  # parent + current rule
                pivot_rule_idxs_2 = self._assign_pivot_variables(rule, zp)
                pivot_rules_to_add_2 = self._get_rules_by_idxs(pivot_rule_idxs_2)

                logger.debug(
                    "[assign_pivot_variables] indices of pivot rules to add: {}".format(
                        pivot_rule_idxs_2
                    )
                )
                captured_previously = captured_1 | (
                    ~parent_not_captured
                )  # captured by the prefix, current rule, and rules introduced by update_pivot_variables
                not_captured_previously = ~captured_previously

                # captured by the rules introduced by assign_pivot_variables
                # but not by the previously added rules
                captured_2 = (
                    lor_of_truthtable(pivot_rules_to_add_2) & not_captured_previously
                )

                logger.debug(f"bin(~captured_previously): {bin(~captured_previously)}")
                logger.debug(f"bin(captured_2): {bin(captured_2)}")
                # captured = (captured_1 | captured_2) & parent_not_captured
                # the FP mistakes incurred by adding pivot_rules_to_add_2
                fp_fraction = self._incremental_update_lb(
                    captured_2,
                    self.y_mpz,
                )

                # the FN mistakes incurred by adding pivot_rules_to_add_2
                fn_fraction, _ = self._incremental_update_obj(
                    not_captured_previously, captured_2
                )
                logger.debug(f"fp_fraction: {fp_fraction}")
                logger.debug(f"fn_fraction: {fn_fraction}")
                logger.debug(f"len(pivot_rule_idxs_2): {len(pivot_rule_idxs_2)}")
                obj = (
                    lb
                    + fn_fraction
                    + fp_fraction
                    + (self.lmbd * len(pivot_rule_idxs_2))
                )

                # the child_node might be assigned later
                # and if assigned
                # will be assigned only once during the the check of the current rule
                child_node = None

                # apply look-ahead bound
                # parent + current rule + any next rule
                lookahead_lb = lb + self.lmbd
                if lookahead_lb <= self.ub:
                    # TODO: what is captured, by whom?
                    # by current rule and updated pivot rules
                    child_node = self._create_new_node_and_add_to_tree(
                        rule,
                        lb,
                        obj,
                        captured_previously,
                        parent_node,
                        pivot_rules_to_add_1,
                    )
                    # TODO: what is not_captured, by whom? by current rule and udpated pivot rules
                    self.queue.push(
                        (child_node, not_captured_previously, zp),
                        # if the lower bound are the same for two nodes, resolve the order by the corresponding ruleset
                        key=child_node.lower_bound,
                    )

                if obj <= self.ub:
                    if child_node is None:
                        # compute child_node if not done
                        child_node = self._create_new_node_and_add_to_tree(
                            rule,
                            lb,
                            obj,
                            captured_previously,
                            parent_node,
                            pivot_rules_to_add_1,
                        )
                    ruleset = child_node.get_ruleset_ids() | pivot_rule_idxs_2

                    logger.debug(f"yielding {ruleset} with obj {obj:.2f}")
                    if return_objective:
                        yield (ruleset, child_node.objective)
                    else:
                        # print("self.queue._items: {}".format(self.queue._items))
                        yield ruleset
