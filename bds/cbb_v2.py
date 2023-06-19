import functools
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
def ensure_no_violation(
    j: int,
    z: np.ndarray,
    s: np.ndarray,
    t: np.ndarray,
    A: np.ndarray,
    max_nz_idx_array: np.ndarray,
    row2pivot_column: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    upon adding rule j to the current prefix, add a set of pivot rules to ensure that Ax=b is not violated

    return:

    - selected pivot rule indices after adding the jth rule to the current prefix
    - the satisfiability vector
    - the updated parity states vector

    note that the rule index must correspond to a non-pivot column

    the parity states vector and satisfiability vector for the current prefix are provided for incremental computation

    row2pivot_column is the mapping from row id to the corresponding pivot column

    for performance reasons, the following data structures are given:

    - max_nz_idx_array: the array of largest non-zero idx per constraint
    """
    zp: np.ndarray = z.copy()
    sp: np.ndarray = s.copy()

    selected_rules = np.empty(A.shape[1], np.int_)
    num_rules_selected = 0

    for i in range(A.shape[0]):
        max_nz_idx = max_nz_idx_array[i]
        if not sp[i]:
            # constraint i is not satisfied yet
            if j >= (max_nz_idx + 1):
                # j is exterior
                # the corresponding pivot rule maybe added
                sp[i] = True
                if A[i][j - 1]:
                    #  j is relevant
                    if t[i] == zp[i]:
                        # add the pivot rule if t[i] equals zp[i]
                        # an example to explain the rationale:
                        # say t[i] = zp[i] = False
                        # note that adding j flips zp[i] to True
                        # to make the constraint satisfied,
                        # we need to add the pivot, which flips zp[i] back to False
                        selected_rules[num_rules_selected] = row2pivot_column[i] + 1
                        num_rules_selected += 1
                    else:
                        # otherwise, we add j only
                        # and flip zp[i] to make zp[i] = t[i]
                        zp[i] = not zp[i]
                elif (A[i][j - 1] == 0) and (t[i] != zp[i]):
                    selected_rules[num_rules_selected] = row2pivot_column[i] + 1
                    num_rules_selected += 1
                    zp[i] = not zp[i]
            elif A[i][j - 1]:
                # j is interior and relevant
                zp[i] = not zp[i]
    return selected_rules[:num_rules_selected], zp, sp


@jit(nopython=True, cache=True)
def count_added_pivots(j: int, A: np.ndarray, t: np.ndarray, z: np.ndarray) -> int:
    """count the number of added pivot rules

    rationale:

    we basically count the number of which passes one of the conditions below:

    - A[i][j-1] is False and (z[i] == t[i]) is False
    - A[i][j-1] is True and (z[i] == t[i]) is True

    equivalently, we count the number entries in A[:, j-1] == (z == t) that are true
    """
    return (A[:, j - 1] == (z == t)).sum()


@jit(nopython=True, cache=True)
def ensure_satisfiability(
    j: int,
    rank: int,
    z: np.ndarray,
    t: np.ndarray,
    A: np.ndarray,
    max_nz_idx_array: np.ndarray,
    row2pivot_column: np.ndarray,
) -> np.ndarray:
    """
    return the pivot variables that are assigned to 1s if no rules with index larger than j are added further.

    j must corresponds to a free variable, meaning:

    1. it is not the default one (j=0)
    2. and it does not correspond to any pivot variable
    """
    selected_rules = np.empty(A.shape[1], np.int_)
    num_rules_selected = 0

    for i in range(rank):  # loop up to rank
        if (A[i][j - 1] == 0) and (z[i] != t[i]):
            # the rule is irrelevant
            selected_rules[num_rules_selected] = row2pivot_column[i] + 1
            num_rules_selected += 1
        elif (A[i][j - 1] == 1) and (z[i] == t[i]):
            # the rule is relevant
            selected_rules[num_rules_selected] = row2pivot_column[i] + 1
            num_rules_selected += 1
    return selected_rules[:num_rules_selected]


def ensure_satisfiability_at_root(
    rank: int,
    z: np.ndarray,
    t: np.ndarray,
    num_rules: int,
    row2pivot_column: np.ndarray,
) -> np.ndarray:
    """
    return the pivot variables that are assigned to 1s if no free rules are added
    """
    selected_rules = np.empty(num_rules, np.int_)
    num_rules_selected = 0

    for i in range(rank):  # loop up to rank
        if z[i] != t[i]:
            # j is the default rule
            selected_rules[num_rules_selected] = row2pivot_column[i] + 1
            num_rules_selected += 1
    return selected_rules[:num_rules_selected]


def lor(vs: List[mpz]) -> mpz:
    """logical OR over a list of bit arrays"""
    return functools.reduce(lambda x, y: x | y, vs, mpz())


@jit(nopython=True, cache=True, nogil=True)
def negate_at_idxs(v: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    """negate the values of v at idxs"""
    vp = v.copy()
    for i in idxs:
        vp[i] = ~vp[i]
    return vp


@jit(nopython=True, cache=True)
def check_look_ahead_bound(lb: float, lmbd: float, ub: float) -> bool:
    return (lb + lmbd) <= ub


@jit(nopython=True, cache=True)
def check_pivot_length_bound(
    prefix_length: int, pvt_count: int, length_ub: int
) -> bool:
    return (prefix_length + 1 + pvt_count) > length_ub


class ConstrainedBranchAndBound(ConstrainedBranchAndBoundNaive):
    def __post_init__(self):
        # pad None in front so that self.truthtable_list is 1-indexed
        self.truthtable_list = [None] + [r.truthtable for r in self.rules]

    def _lor(self, idxs: np.ndarray) -> mpz:
        """given a set of rule idxs, return the logical OR over the truthtables of the corresponding rules"""
        r = mpz()
        for i in idxs:
            r |= self.truthtable_list[i]
        return r

    def _ensure_no_violation(
        self, rule: Rule, z: np.ndarray, s: np.ndarray, u: mpz
    ) -> Tuple[np.ndarray, np.ndarray, mpz, int]:
        """a wrapper around ensure_no_violation"""
        e1_idxs, zp, sp = ensure_no_violation(
            rule.id,
            z,
            s,
            self.t,
            self.A,
            self.max_nz_idx_array,
            self.row2pivot_column,
        )
        return (
            e1_idxs,
            zp,
            sp,
            (self._lor(e1_idxs) | rule.truthtable) & u,
            len(e1_idxs) + 1,
        )

    def _ensure_satisfiability(self, rule: Rule, z: np.ndarray):
        """a wrapper around assign_pivot_variables"""
        return ensure_satisfiability(
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

        # a rule is a border rule if it appears as the last rule in at least one constraint
        self.border_rule_idxs = set(self.max_nz_idx_array + 1) - {0}

    def reset_queue(self):
        self.queue: Queue = Queue()
        not_captured = self._not_captured_by_default_rule()

        # the parity status vector
        # 0 mean an even number of rules are selected
        # 1 mean an odd number of rules are selected
        z = bin_zeros(self.num_constraints)

        # the satisfiability vector
        # 0 means not satisfied yet
        # 1 means satisfied
        s = bin_zeros(self.num_constraints)

        item = (self.tree.root, not_captured, z, s)
        self.queue.push(item, key=0)

    def _create_new_node_and_add_to_tree(
        self,
        rule: Rule,
        lb: mpfr,
        obj: mpfr,
        captured: mpz,
        parent_node: Node,
        pivot_rule_idxs_to_add: List[Rule] = [],
    ) -> Node:
        """create a node using information provided by rule, lb, obj, and captured
        and add it as a child of parent"""
        if rule.id not in parent_node.children:
            child_node = Node(
                rule_id=rule.id,
                lower_bound=lb,
                objective=obj,
                num_captured=gmp.popcount(captured),
                pivot_rule_ids=pivot_rule_idxs_to_add,
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
        # default_rule = Rule(0, "rule-0", 0, mpz())
        rule_idxs = ensure_satisfiability_at_root(
            self.rank,
            bin_zeros(self.num_constraints),
            self.t,
            self.num_rules,
            self.row2pivot_column,
        )
        captured = self._lor(rule_idxs)
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
        s: np.ndarray,
        return_objective=False,
    ):
        """
        check one node in the search tree, update the queue, and yield feasible solution if exists
        parente_node: the current node/prefix to check
        u: postives not captured by the current prefix
        z: the parity states vector
        s: the satisfiability state vector
        return_objective: True if return the objective of the evaluated node
        """
        parent_lb = parent_node.lower_bound
        prefix_length = parent_node.num_rules
        length_ub = prefix_specific_length_upperbound(
            parent_lb, prefix_length, self.lmbd, self.ub
        )

        # logger.debug(f"parent node: {parent_node.get_ruleset_ids()}")
        # here we assume the rule ids are consecutive integers
        # padding = ' ' * (2 * parent_node.depth)
        # print("{}prefix: {}".format(padding, tuple(sorted(parent_node.get_ruleset_ids()))))
        for rule in self.rules[parent_node.rule_id :]:
            # consider adding only free rules
            # since the addition of pivot rules are determined "automatically" by Ax=b
            if rule.id in self.pivot_rule_idxs:
                continue

            # print("{}  rule.id: {}".format(padding, rule.id))
            self.num_prefix_evaluations += 1

            # prune by ruleset length
            if (prefix_length + 1) > length_ub:
                continue

            lb = (
                parent_lb
                + self._incremental_update_lb(rule.truthtable & u, self.y_mpz)
                + self.lmbd
            )

            # check hierarchical lower bound
            if lb <= self.ub:
                # child_node might be assigned later
                # and if assigned
                # will be assigned only once during the the check of the current rule
                child_node = None
                # apply look-ahead bound
                # parent + current rule + any next rule

                if check_look_ahead_bound(lb, self.lmbd, self.ub):
                    # ensure that Ax=b is not unsatisfied
                    # v1 and not_w are updated here
                    e1_idxs, zp, sp, v1, extension_size = self._ensure_no_violation(
                        rule, z, s, u
                    )

                    w = v1 | ~u  # captured by the current rule set, i.e., d, e1, and r
                    not_w = ~w  # not captured by the above

                    # update the hierarchical lower bound only if at least one pivot rules are added
                    if extension_size > 1:
                        _ = 1 + 1  # dummy statement for profiling purposes
                        if (prefix_length + extension_size) > length_ub:
                            continue

                        # check new hierarchical lower bound
                        lb = (
                            parent_lb
                            + self._incremental_update_lb(v1, self.y_mpz)
                            + extension_size * self.lmbd
                        )

                        if lb > self.ub:
                            continue

                    # all checks are passed and we add the new node
                    child_node = self._create_new_node_and_add_to_tree(
                        rule,
                        lb,
                        -1,  # set obj to -1 temporarily, will be updated later (if the node is feasible)
                        w,
                        parent_node,
                        e1_idxs,
                    )

                    self.queue.push(
                        (child_node, not_w, zp, sp),
                        key=child_node.lower_bound,
                    )
                # next we consider the feasibility d + r + the extension rules needed to satisfy Ax=b
                # note that ext_idxs exclude the current rule

                # we first check the solution length exceeds the length bound
                # the number of added pivots are calcualted in a fast way
                pvt_count = count_added_pivots(rule.id, self.A, self.t, z)

                # add 1 for the current rule, because ext_idx does not include the current rule

                # (prefix_length + 1 + pvt_count) > length_ub:
                if check_pivot_length_bound(prefix_length, pvt_count, length_ub):
                    continue

                # then we get the actual of added pivots to calculate the objective
                ext_idxs = self._ensure_satisfiability(rule, z)

                # prepare to calculate the obj of the final solution
                # v_ext: points captured by extention + current rule in the context of the prefix
                v_ext = (rule.truthtable | self._lor(ext_idxs)) & u
                # the FP mistakes incurred by extension
                fp_fraction = self._incremental_update_lb(v_ext, self.y_mpz)

                # check if adding fp mistakes exceeds the ub
                obj_with_fp = (
                    parent_lb + fp_fraction + (self.lmbd * (1 + len(ext_idxs)))
                )
                if obj_with_fp > self.ub:
                    continue

                # calculate the true obj
                # by adding the FN mistakes incurred by extention rules
                fn_fraction, _ = self._incremental_update_obj(u, v_ext)
                obj = obj_with_fp + fn_fraction

                if obj <= self.ub:
                    if child_node is None:
                        # compute child_node if not done
                        e1_idxs, _, _, _, _ = self._ensure_no_violation(rule, z, s, u)
                        child_node = self._create_new_node_and_add_to_tree(
                            rule,
                            lb,
                            obj,
                            v_ext | (~u),
                            parent_node,
                            e1_idxs,
                        )
                    else:
                        child_node.objective = obj  # update the obj
                    ruleset = child_node.get_ruleset_ids() | set(ext_idxs)

                    # logger.debug(f"yielding {ruleset} with obj {obj:.2f}")
                    # print(f"{padding}    yield: {tuple(ruleset)}")
                    if return_objective:
                        yield (ruleset, child_node.objective)
                    else:
                        # print("self.queue._items: {}".format(self.queue._items))
                        yield ruleset