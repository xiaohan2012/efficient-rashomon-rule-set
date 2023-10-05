import numpy as np
from contexttimer import Timer

from typing import List, Optional, Iterable
from logzero import logger
from copy import copy
from .solver_status import SolverStatus

from .cbb import (
    ConstrainedBranchAndBound,
    inc_ensure_satisfiability,
    ensure_minimal_non_violation,
)
from .queue import NonRedundantQueue
from .types import SolutionSet


class IncrementalConstrainedBranchAndBound(ConstrainedBranchAndBound):
    # @profile
    def reset(
        self,
        A: np.ndarray,
        b: np.ndarray,
        solver_status: Optional[SolverStatus] = None,
    ):
        # print("calling reset in ICBB")
        self.setup_constraint_system(A, b)

        if solver_status is not None:
            # continuation search
            self.status = solver_status.copy()
            self._push_last_checked_prefix_to_queue()
        else:
            self.reset_status()
            self.reset_queue()

        # print("self.rules: {}".format(self.rules))
        # print("self.truthtable_list: {}".format(self.truthtable_list))

    def _examine_R_and_S(self, return_objective=False):
        """check the solution set and reserve set from previous runsand yield them if feasible"""
        # print("examing R and S")
        candidate_prefixes = copy(self.status.solution_set | self.status.reserve_set)

        # clear the solution and reserve set
        self.status.reset_reserve_set()
        self.status.reset_solution_set()

        # only check the solutions if the new Ax=b is solvable
        # print("self.is_linear_system_solvable: {}".format(self.is_linear_system_solvable))
        if self.is_linear_system_solvable:
            for prefix in candidate_prefixes:
                # transform the prefix according to the new ordering
                prefix_permutated = self._permutate_prefix(prefix)
                prefix_with_free_rules = prefix_permutated - self.pivot_rule_idxs
                extension = self._ensure_satisfiability(prefix_with_free_rules)

                extended_prefix = (
                    prefix_with_free_rules + extension
                )  # using the new ordering if present
                # using the original ordering
                extended_prefix_restored = self._restore_prefix(extended_prefix)
                # print("prefix: {} -> {}".format(prefix, prefix - self.pivot_rule_idxs))
                # print("extension: {}".format(extension))
                # add to reserve set if needed
                if (
                    extended_prefix_restored not in self.status.reserve_set
                    and self._calculate_lb(extended_prefix) <= self.ub
                    and len(extended_prefix) >= 1
                ):
                    self.status.add_to_reserve_set(extended_prefix_restored)

                # yield and add to solution set if needed
                obj = self._calculate_obj(extended_prefix)
                if (
                    extended_prefix_restored not in self.status.solution_set
                    and obj <= self.ub
                    and len(extended_prefix) >= 1
                ):
                    # print(
                    #     f"-> inheriting {extended_prefix_restored} (obj={obj:.2}) as solution from {prefix}"
                    # )
                    self.status.add_to_solution_set(extended_prefix_restored)
                    yield self._pack_solution(
                        extended_prefix_restored, (obj if return_objective else None)
                    )
        else:
            logger.debug("Ax=b is unsolvable, skip solution checking")
        # print("examine_R_and_S: done ")

    def _update_queue(self):
        """
        check each item in the queue from previous run and push them to the current queue if bound checking is passed
        """
        new_queue = NonRedundantQueue()
        # print("updating queue")
        if self.is_linear_system_solvable:
            # only check the queue items if the new linear system is solvable
            for queue_item in self.status.queue:
                prefix = queue_item[0]
                # we need to transform the rule ids in prefix if column reordering is applied
                prefix_permutated = self._permutate_prefix(prefix)

                prefix_with_free_rules_only = (
                    prefix_permutated - self.pivot_rule_idxs
                )  # remove the pivots
                extension, u, z, s = self._ensure_minimal_non_violation(
                    prefix_with_free_rules_only
                )

                extended_prefix = prefix_with_free_rules_only + extension
                # print(
                #     f"prefix: {prefix} -> {prefix_with_free_rules_only} -> {extended_prefix}"
                # )
                lb = self._calculate_lb(extended_prefix)

                # print("extended_prefix: {}".format(extended_prefix))
                # print("lb: {}".format(lb))
                if (lb + self.lmbd) <= self.ub:
                    # print(f"-> queue")
                    # print(f"pushing {extended_prefix} to queue")
                    new_queue.push(
                        (extended_prefix, lb, ~u, z, s), key=(lb, extended_prefix)
                    )
        else:
            logger.debug("Ax=b is unsolvable, skip queue items checking")
        # use the new queue in status
        self.status.set_queue(new_queue)
        # print("update_queue: done ")

    # @profile
    def generate(self, return_objective=False) -> Iterable:
        # print(f"m={self.A.shape[0]}")
        # self.print_Axb()
        # print("self.rules: {}".format(self.rules))
        # print("self.truthtable_list: {}".format(self.truthtable_list))
        yield from self._examine_R_and_S(return_objective)
        logger.debug("inheritted {} solutions".format(len(self.status.solution_set)))
        with Timer() as timer:
            logger.debug(f'update_queue processes {self.status.queue_size()} items')
            self._update_queue()
            logger.debug(f'  which takes {timer.elapsed:.2f}s')

        yield from self._generate_solution_at_root(return_objective)

        while not self.status.is_queue_empty():
            queue_item = self.status.pop_from_queue()
            yield from self._loop(*queue_item, return_objective=return_objective)
