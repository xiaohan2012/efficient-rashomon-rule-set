import functools
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import gmpy2 as gmp
import numpy as np
from gmpy2 import mpfr, mpz
from logzero import logger
from numba import jit

from .bb import BranchAndBoundNaive
from .bounds import prefix_specific_length_upperbound
from .gf2 import GF, extended_rref
from .queue import Queue
from .rule import Rule, lor_of_truthtable
from .utils import (
    assert_binary_array,
    bin_array,
    bin_ones,
    bin_zeros,
    get_indices_and_indptr,
    calculate_lower_bound,
    calculate_obj,
)


@jit(nopython=True, cache=True)
def ensure_minimal_no_violation(
    j: int,
    z: np.ndarray,
    s: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    B: np.ndarray,
    row2pivot_column: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    upon adding rule j to the current prefix (represented  by `z` and `s`),
    add a set of pivot rules to ensure that the new prefix is minimally non-violating

    return:

    - selected pivot rule indices after adding the jth rule to the current prefix
    - the satisfiability vector
    - the updated parity states vector

    note that the rule index must correspond to a non-pivot column

    the parity states vector `s` and satisfiability vector `z` for the current prefix are provided for incremental computation

    row2pivot_column is the mapping from row id to the corresponding pivot column

    for performance reasons, the following data structures are given:

    - max_nz_idx_array: the array of largest non-zero idx per constraint
    """
    zp: np.ndarray = z.copy()
    sp: np.ndarray = s.copy()

    selected_rules = np.empty(A.shape[1], np.int_)
    num_rules_selected = 0
    for i in range(A.shape[0]):
        if j == -1:
            # the initial case, where no rules are added
            if j == B[i]:
                sp[i] = 1
                if b[i] == 1:
                    selected_rules[num_rules_selected] = row2pivot_column[i]
                    num_rules_selected += 1
                    zp[i] = not zp[i]
            continue
        if sp[i] == 0:
            # constraint i is not satisfied yet
            if j >= B[i]:
                # j is exterior
                # the corresponding pivot rule maybe added
                sp[i] = 1
                if A[i][j]:
                    #  j is relevant
                    if b[i] == zp[i]:
                        selected_rules[num_rules_selected] = row2pivot_column[i]
                        num_rules_selected += 1
                    else:
                        zp[i] = not zp[i]
                elif b[i] != zp[i]:
                    # j is irrelevant
                    selected_rules[num_rules_selected] = row2pivot_column[i]
                    num_rules_selected += 1
                    zp[i] = not zp[i]
            elif A[i][j]:
                # j is interior and relevant
                zp[i] = not zp[i]
    return selected_rules[:num_rules_selected], zp, sp


@jit(nopython=True, cache=True)
def count_added_pivots(j: int, A: np.ndarray, b: np.ndarray, z: np.ndarray) -> int:
    """count the number of pivot rules to add in order to satisfy Ax=b

    which is equivalent to counting the number of constraints that satisfies either of the conditions below:

    - A[i][j] is False and (z[i] == b[i]) is False
    - A[i][j] is True and (z[i] == b[i]) is True

    which is equivalent to counting the number entries in A[:, j] == (z == b) that are True
    """
    assert j >= 0
    return (A[:, j] == (z == b)).sum()


@jit(nopython=True, cache=True)
def ensure_satisfiability(
    j: int,
    rank: int,
    z: np.ndarray,
    s: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
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
        if j == -1:
            if b[i] == 1:
                selected_rules[num_rules_selected] = row2pivot_column[i]
                num_rules_selected += 1
        elif s[i] == 0:
            if ((A[i][j] == 0) and (z[i] != b[i])) or (
                (A[i][j] == 1) and (z[i] == b[i])
            ):
                # case 1: the rule is irrelevant
                # case 2: the rule is relevant
                selected_rules[num_rules_selected] = row2pivot_column[i]
                num_rules_selected += 1
    return selected_rules[:num_rules_selected]


# def ensure_satisfiability_at_root(
#     rank: int,
#     z: np.ndarray,
#     t: np.ndarray,
#     num_rules: int,
#     row2pivot_column: np.ndarray,
# ) -> np.ndarray:
#     """
#     return the pivot variables that are assigned to 1s if no free rules are added
#     """
#     selected_rules = np.empty(num_rules, np.int_)
#     num_rules_selected = 0

#     for i in range(rank):  # loop up to rank
#         if z[i] != t[i]:
#             # j is the default rule
#             selected_rules[num_rules_selected] = row2pivot_column[i] + 1
#             num_rules_selected += 1
#     return selected_rules[:num_rules_selected]


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


def build_boundary_table(
    A: np.ndarray, rank: int, pivot_columns: np.ndarray
) -> np.ndarray:
    """for a 2D matrix A, compute the maximum non-zero non-pivot index per row, if it does not exist, use -1"""
    assert A.ndim == 2
    Ap = A.copy()
    result = []
    for i in range(rank):
        Ap[i, pivot_columns[i]] = 0
        if Ap[i, :].sum() == 0:
            result.append(-1)
        else:
            result.append((Ap[i, :] > 0).nonzero()[0].max())
    return np.array(result, dtype=int)


class ConstrainedBranchAndBound(BranchAndBoundNaive):
    def _simplify_constraint_system(
        self, A: np.ndarray, b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """simplify the constraint system using reduced row echelon form"""
        # logger.debug("simplifying A x = t using rref")
        A_rref, b_rref, rank, pivot_columns = extended_rref(
            GF(A.astype(int)), GF(b.astype(int)), verbose=False
        )
        return bin_array(A_rref), bin_array(b_rref), rank, pivot_columns

    def setup_constraint_system(self, A: np.ndarray, t: np.ndarray):
        """set the constraint system, e.g., simplify the system"""
        logger.debug("setting up the parity constraint system")
        assert_binary_array(t)

        assert (
            A.shape[0] == t.shape[0]
        ), f"dimension mismatch: {A.shape[0]} != {t.shape[0]}"

        # simplify the constraint system
        (
            self.A,
            self.b,
            self.rank,
            self.pivot_columns,
        ) = self._simplify_constraint_system(A, t)
        # print("A\n: {}".format(self.A.astype(int)))
        # print("b\n: {}".format(self.b.astype(int)))
        self.is_linear_system_solvable = (self.b[self.rank :] == 0).all()

        self.num_constraints = int(self.A.shape[0])
        self.num_vars = self.num_rules = int(self.A.shape[1])

        # build the boundary table
        self.B = build_boundary_table(self.A, self.rank, self.pivot_columns)

        self.pivot_rule_idxs = set(map(lambda v: v, self.pivot_columns))
        self.free_rule_idxs = (
            set(map(lambda v: v, range(self.num_vars))) - self.pivot_rule_idxs
        )
        # mapping from row index to the pivot column index
        self.row2pivot_column = np.array(self.pivot_columns, dtype=int)

    # @profile
    def generate(self, return_objective=False) -> Iterable:
        if not self.is_linear_system_solvable:
            logger.debug("abort the search since the linear system is not solvable")
            yield from ()
        else:
            yield from super(ConstrainedBranchAndBound, self).generate(return_objective)

    def reset(self, A: np.ndarray, b: np.ndarray):
        self.setup_constraint_system(A, b)
        super(ConstrainedBranchAndBound, self).reset()

    def __post_init__(self):
        self.truthtable_list = [r.truthtable for r in self.rules]

    def _lor(self, idxs: np.ndarray) -> mpz:
        """given a set of rule idxs, return the logical OR over the truthtables of the corresponding rules"""
        r = mpz()
        for i in idxs:
            r |= self.truthtable_list[i]
        return r

    def _ensure_minimal_non_violation(
        self, rule_id: int, u: mpz, z: np.ndarray, s: np.ndarray
    ) -> Tuple[np.ndarray, mpz, np.ndarray, np.ndarray, int]:
        """a wrapper around ensure_no_violation
        returns (ext_idxs, v, z, s, ext_size)

        where v is the captured vector in the context of the current prefix
        """
        assert rule_id >= -1
        e1_idxs, zp, sp = ensure_minimal_no_violation(
            rule_id,
            z,
            s,
            self.A,
            self.b,
            self.B,
            self.row2pivot_column,
        )
        # print("e1_idxs: {}".format(e1_idxs))
        if rule_id == -1:
            v = self._lor(e1_idxs) & u
            ext_size = len(e1_idxs)
        else:
            ext_size = len(e1_idxs) + 1
            v = (self._lor(e1_idxs) | self.truthtable_list[rule_id]) & u

        return (e1_idxs, v, zp, sp, ext_size)

    def _ensure_satisfiability(self, rule_id: int, z: np.ndarray, s: np.ndarray):
        """a wrapper around assign_pivot_variables"""
        return ensure_satisfiability(
            rule_id,
            self.rank,
            z,
            s,
            self.A,
            self.b,
            self.row2pivot_column,
        )

    def reset_queue(self):
        self.queue: Queue = Queue()
        u = self._not_captured_by_default_rule()

        # the parity status vector
        # 0 mean an even number of rules are selected
        # 1 mean an odd number of rules are selected
        z = bin_zeros(self.num_constraints)

        # the satisfiability vector
        # 0 means not satisfied yet
        # 1 means satisfied
        s = bin_zeros(self.num_constraints)
        prefix_idxs, v, zp, sp, ext_size = self._ensure_minimal_non_violation(
            -1, u, z, s
        )
        up = ~(v | (~u))  # not captured by the prefix
        lb = calculate_lower_bound(
            self.rules, self.y_np, self.y_mpz, prefix_idxs, self.lmbd
        )
        prefix = tuple(sorted(prefix_idxs))
        item = (prefix, lb, up, zp, sp)
        self.queue.push(item, key=lb)

    # def _get_rules_by_idxs(self, idxs: List[int]) -> List[Rule]:
    #     """extract rules from a list of indices, the indices start from 1"""
    #     return [self.rules[idx] for idx in idxs]

    def generate_solution_at_root(self, return_objective=False) -> Iterable:
        """check the solution at the root, e.g., all free variables assigned to zero"""
        # print(f"generate at root")
        # add pivot rules if necessary to satistify Ax=b
        rule_idxs = self._ensure_satisfiability(
            -1, bin_zeros(self.num_constraints), bin_zeros(self.num_constraints)
        )
        captured = self._lor(rule_idxs)
        num_mistakes = gmp.popcount(captured ^ self.y_mpz)

        # print("bin(captured): {}".format(bin(captured)))
        obj = num_mistakes / self.num_train_pts + len(rule_idxs) * self.lmbd
        # print("num_mistakes: {}".format(num_mistakes))
        sol = set(rule_idxs)
        # logger.debug(f"solution at root: {sol} (obj={obj:.2f})")
        if len(sol) >= 1 and obj <= self.ub:
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
        parent_prefix: Tuple[int],
        parent_lb: float,
        parent_u: mpz,
        parent_z: np.ndarray,
        parent_s: np.ndarray,
        return_objective=False,
    ):
        """
        check one node in the search tree, update the queue, and yield feasible solution if exists

        parent_prefix: the parent prefix to extend from
        parent_u: points not captured by the current prefix
        parent_z: the parity states vector
        parent_s: the satisfiability state vector
        return_objective: True if return the objective of the evaluated node
        """
        prefix_length = len(parent_prefix)
        length_ub = prefix_specific_length_upperbound(
            parent_lb, prefix_length, self.lmbd, self.ub
        )
        free_rules_in_prefix = set(parent_prefix) - self.pivot_rule_idxs
        max_rule_idx = max(free_rules_in_prefix or [-1])
        # print("parent_prefix: {}".format(parent_prefix))
        for rule in self.rules[(max_rule_idx + 1) :]:
            # print("rule.id: {}".format(rule.id))
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
                + self._incremental_update_lb(rule.truthtable & parent_u, self.y_mpz)
                + self.lmbd
            )

            # check hierarchical lower bound
            if lb <= self.ub:
                # apply look-ahead bound
                # parent + current rule + any next rule

                if check_look_ahead_bound(lb, self.lmbd, self.ub):
                    # ensure that Ax=b is not violated
                    # v1, zp, and sp are updated here
                    (
                        e1_idxs,
                        v1,
                        zp,
                        sp,
                        extension_size,
                    ) = self._ensure_minimal_non_violation(
                        rule.id, parent_u, parent_z, parent_s
                    )

                    # update the hierarchical lower bound only if at least one pivot rules are added
                    # prune if needed
                    if extension_size > 1:
                        _ = 1 + 1  # dummy statement for profiling purposes
                        if (prefix_length + extension_size) > length_ub:
                            continue

                        # overwrite the hierarchical objective lower bound
                        lb = (
                            parent_lb
                            + self._incremental_update_lb(v1, self.y_mpz)
                            + extension_size * self.lmbd
                        )

                        if lb > self.ub:
                            continue

                    new_prefix = tuple(
                        sorted(parent_prefix + tuple(e1_idxs) + (rule.id,))
                    )
                    w = v1 | ~parent_u  # captured by the new prefix
                    up = ~w  # not captured by the new prefix
                    self.queue.push(
                        (new_prefix, lb, up, zp, sp),
                        key=lb,
                    )
                # next we consider the feasibility d + r + the extension rules needed to satisfy Ax=b
                # note that ext_idxs exclude the current rule

                # we first check the solution length exceeds the length bound
                # the number of added pivots are calculated in a fast way
                pvt_count = count_added_pivots(rule.id, self.A, self.b, parent_z)

                # add 1 for the current rule, because ext_idx does not include the current rule

                # (prefix_length + 1 + pvt_count) > length_ub:
                if check_pivot_length_bound(prefix_length, pvt_count, length_ub):
                    continue

                # then we get the actual of added pivots to calculate the objective
                ext_idxs = self._ensure_satisfiability(rule.id, parent_z, parent_s)

                # prepare to calculate the obj of the final solution
                # v_ext: points captured by extention + current rule in the context of the prefix
                v_ext = (rule.truthtable | self._lor(ext_idxs)) & parent_u
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
                fn_fraction, _ = self._incremental_update_obj(parent_u, v_ext)
                obj = obj_with_fp + fn_fraction

                if obj <= self.ub:
                    solution_prefix = tuple(
                        sorted(parent_prefix + tuple(ext_idxs) + (rule.id,))
                    )

                    # logger.debug(f"yielding {ruleset} with obj {obj:.2f}")
                    # print(f"{padding}    yield: {tuple(ruleset)}")
                    if return_objective:
                        yield (solution_prefix, obj)
                    else:
                        # print("self.queue._items: {}".format(self.queue._items))
                        yield solution_prefix
