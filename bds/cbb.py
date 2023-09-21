import functools
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import gmpy2 as gmp
import numpy as np
from gmpy2 import mpfr, mpz
from logzero import logger
from numba import jit
from copy import deepcopy

from .bb import BranchAndBoundNaive
from .bounds import (
    prefix_specific_length_upperbound,
    check_look_ahead_bound,
    check_pivot_length_bound,
)
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
from .parity_constraints import (
    ensure_minimal_no_violation,
    ensure_satisfiability,
    count_added_pivots,
    build_boundary_table,
)
from .types import RuleSet
from .solver_status import SolverStatus


# def lor(vs: List[mpz]) -> mpz:
#     """logical OR over a list of bit arrays"""
#     return functools.reduce(lambda x, y: x | y, vs, mpz())


class ConstrainedBranchAndBound(BranchAndBoundNaive):
    def __init__(self, *args, reorder_columns=True, **kwargs):
        super(ConstrainedBranchAndBound, self).__init__(*args, **kwargs)
        self.reorder_column = reorder_columns
        # copy the rules for later use
        self.rules_before_ordering = deepcopy(self.rules)

    def _simplify_constraint_system(
        self, A: np.ndarray, b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """simplify the constraint system using reduced row echelon form"""
        # logger.debug("simplifying A x = t using rref")
        A_rref, b_rref, rank, pivot_columns = extended_rref(
            GF(A.astype(int)), GF(b.astype(int)), verbose=False
        )
        return bin_array(A_rref), bin_array(b_rref), rank, pivot_columns

    def _do_reorder_columns(self):
        """re-order the columns of A and reflect the new ordering in the rules"""
        free_cols = np.array(
            list(set(np.arange(self.A.shape[1])) - set(self.pivot_columns)), dtype=int
        )

        if self.A.shape[0] >= 1:
            row_idx = self.A.sum(axis=1).argmin()
            ordered_free_idxs = np.argsort(
                np.array(self.A[row_idx, free_cols], dtype=int)
            )[::-1]
        else:
            # when A is empty, do not re-order
            ordered_free_idxs = np.arange(len(free_cols))

        ordered_idxs = np.concatenate(
            [self.pivot_columns, free_cols[ordered_free_idxs]]
        )
        # mapping from new column idx to old idx
        self.idx_map_new2old = ordered_idxs.copy()

        self.pivot_columns = np.arange(self.rank)

        # re-order the columns, rule, and truthtable_list
        self.A = self.A[:, ordered_idxs]
        self.rules = [self.rules[i] for i in ordered_idxs]
        self.truthtable_list = [r.truthtable for r in self.rules]

        # re-assign the rule ids
        for i, rule in enumerate(self.rules):
            rule.id = i

    def setup_constraint_system(self, A: np.ndarray, b: np.ndarray):
        """set the constraint system, e.g., simplify the system"""
        logger.debug("setting up the parity constraint system")
        assert_binary_array(b)

        assert (
            A.shape[0] == b.shape[0]
        ), f"dimension mismatch: {A.shape[0]} != {b.shape[0]}"

        # simplify the constraint system
        (
            self.A,
            self.b,
            self.rank,
            self.pivot_columns,
        ) = self._simplify_constraint_system(A, b)

        if self.reorder_column:
            self._do_reorder_columns()

        self.is_linear_system_solvable = (self.b[self.rank :] == 0).all()

        self.num_constraints = int(self.A.shape[0])
        self.num_vars = self.num_rules = int(self.A.shape[1])

        # build the boundary table
        self.B = build_boundary_table(self.A, self.rank, self.pivot_columns)

        self.pivot_rule_idxs = set(self.pivot_columns)
        self.free_rule_idxs = set(range(self.num_vars)) - self.pivot_rule_idxs
        # mapping from row index to the pivot column index
        self.row2pivot_column = np.array(self.pivot_columns, dtype=int)

    def reset(
        self, A: np.ndarray, b: np.ndarray, solver_status: Optional[SolverStatus] = None
    ):
        """
        if queue and d_last is given, the search continues from that queue | {d_last}
        solutions and reserve solutions are added to S and S, respectively
        """
        # important: restore the original ordering first
        # otherwise, previous calls may mess up the ordering
        self.rules = deepcopy(self.rules_before_ordering)

        self.setup_constraint_system(A, b)

        if solver_status is not None:
            # continuation search
            self.status = solver_status.copy()
            self.status.push_d_last_to_queue(self._calculate_lb(self.status.d_last))
        else:
            self.reset_status()
            self.reset_queue()

    def __post_init__(self):
        self.truthtable_list = [r.truthtable for r in self.rules]

    def _lor(self, idxs: np.ndarray) -> mpz:
        """given a set of rule idxs, return the logical OR over the truthtables of the corresponding rules"""
        r = mpz()
        for i in idxs:
            r |= self.truthtable_list[i]
        return r

    def _restore_rule_ids(self, prefix: RuleSet) -> RuleSet:
        """return the rule ids if reorder_column is enabled"""
        if self.reorder_column:
            return RuleSet([self.idx_map_new2old[i] for i in prefix])
        return RuleSet(prefix)

    def _pack_solution(
        self, prefix: RuleSet, obj=None
    ) -> Union[RuleSet, Tuple[RuleSet, float]]:
        """return the solution (and objective if it is given)"""
        if obj is not None:
            return prefix, obj
        else:
            return prefix

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

    def reset_status(self):
        self.status = SolverStatus()

    def reset_queue(self):
        assert hasattr(self, "status") and isinstance(
            self.status, SolverStatus
        ), "self.status is not set"

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
        lb = self._calculate_lb(prefix_idxs)
        prefix = RuleSet(prefix_idxs)
        item = (prefix, lb, up, zp, sp)
        self.status.push_to_queue(lb, item)

    def _generate_solution_at_root(self, return_objective=False) -> Iterable:
        """check the solution at the root, e.g., all free variables assigned to zero"""
        # add pivot rules if necessary to satistify Ax=b
        prefix = RuleSet(
            self._ensure_satisfiability(
                -1, bin_zeros(self.num_constraints), bin_zeros(self.num_constraints)
            )
        )

        obj = self._calculate_obj(prefix)

        # restore the rule ids in the prefix
        prefix_restored = self._restore_rule_ids(prefix)

        # record R
        if self._calculate_lb(prefix) <= self.ub:
            self.status.add_to_reserve_set(prefix_restored)

        if len(prefix_restored) >= 1 and obj <= self.ub:
            # record and yield the prefix if it is not yet in the solution set
            # this check is needed for incremental/continuation search
            if prefix_restored not in self.status.solution_set:
                self.status.add_to_solution_set(prefix_restored)

                yield self._pack_solution(
                    prefix_restored, (obj if return_objective else None)
                )

    def generate(self, return_objective=False) -> Iterable:
        if not self.is_linear_system_solvable:
            logger.debug("abort the search since the linear system is not solvable")
            yield from ()
        else:
            yield from self._generate_solution_at_root(return_objective)

            while not self.status.is_queue_empty():
                queue_item = self.status.pop_from_queue()
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

        # udpate d_last in search status
        self.status.update_d_last(self._restore_rule_ids(free_rules_in_prefix))

        max_rule_idx = max(free_rules_in_prefix or [-1])

        for rule in self.rules[(max_rule_idx + 1) :]:
            # consider adding only free rules
            # since the addition of pivot rules are determined "automatically" by Ax=b
            if rule.id in self.pivot_rule_idxs:
                continue

            self.num_prefix_evaluations += 1

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

                    if (extension_size == 1) or (
                        (extension_size > 1)
                        and ((prefix_length + extension_size) <= length_ub)
                    ):
                        # update lb if pivots are added
                        if extension_size > 1:
                            lb = (
                                parent_lb
                                + self._incremental_update_lb(v1, self.y_mpz)
                                + extension_size * self.lmbd
                            )

                        if (
                            lb + self.lmbd
                        ) <= self.ub:  # + 1 because we apply look-ahead bound
                            new_prefix = RuleSet(
                                parent_prefix + tuple(e1_idxs) + (rule.id,)
                            )
                            w = v1 | ~parent_u  # captured by the new prefix
                            up = ~w  # not captured by the new prefix

                            self.status.push_to_queue(lb, (new_prefix, lb, up, zp, sp))

                # next we consider the feasibility d + r + the extension rules needed to satisfy Ax=b
                # note that ext_idxs exclude the current rule

                # we first check the solution length exceeds the length bound
                # the number of added pivots are calculated in a fast way
                pvt_count = count_added_pivots(rule.id, self.A, self.b, parent_z)

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

                prefix_restored = self._restore_rule_ids(
                    RuleSet(parent_prefix + tuple(ext_idxs) + (rule.id,))
                )

                self.status.add_to_reserve_set(prefix_restored)

                if obj <= self.ub:
                    self.status.add_to_solution_set(prefix_restored)
                    yield self._pack_solution(
                        prefix_restored, (obj if return_objective else None)
                    )
