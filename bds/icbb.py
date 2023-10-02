import numpy as np
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
    def reset(
        self,
        A: np.ndarray,
        b: np.ndarray,
        solver_status: Optional[SolverStatus] = None,
    ):
        self.setup_constraint_system(A, b)

        if solver_status is not None:
            # continuation search
            self.status = solver_status.copy()
            self._push_last_checked_prefix_to_queue()
        else:
            self.reset_status()
            self.reset_queue()

    def _examine_R_and_S(self, return_objective=False):
        """check the solution set and reserve set from previous runsand yield them if feasible"""
        candidate_prefixes = copy(self.status.solution_set | self.status.reserve_set)

        # clear the solution and reserve set
        self.status.reset_reserve_set()
        self.status.reset_solution_set()

        # only check the solutions if the new Ax=b is solvable
        # print("self.is_linear_system_solvable: {}".format(self.is_linear_system_solvable))
        if self.is_linear_system_solvable:
            for prefix in candidate_prefixes:
                preifx_with_free_rules = prefix - self.pivot_rule_idxs
                extension = self._ensure_satisfiability(preifx_with_free_rules)
                prefix_new = preifx_with_free_rules + extension
                print("prefix: {} -> {}".format(prefix, prefix - self.pivot_rule_idxs))
                print("extension: {}".format(extension))
                # add to reserve set if needed
                if (
                    prefix_new not in self.status.reserve_set
                    and self._calculate_lb(prefix_new) <= self.ub
                    and len(prefix_new) >= 1
                ):
                    self.status.add_to_reserve_set(prefix_new)

                # yield and add to solution set if needed
                if (
                    prefix_new not in self.status.solution_set
                    and self._calculate_obj(prefix_new) <= self.ub
                    and len(prefix_new) >= 1
                ):
                    print(f"-> inheritting {prefix_new}")
                    self.status.add_to_solution_set(prefix_new)
                    yield prefix_new
        else:
            print("Ax=b is unsolvable, skip solution checking")

    def _update_queue(self):
        """
        check each item in the queue from previous run and push them to the current queue if bound checking is passed
        """
        new_queue = NonRedundantQueue()
        print("inherited queue: {}".format(list(self.status.queue)))
        if self.is_linear_system_solvable:
            # only check the queue items if the new linear system is solvable
            for queue_item in self.status.queue:
                prefix = queue_item[0]
                prefix_with_free_rules_only = (
                    prefix - self.pivot_rule_idxs
                )  # remove the pivots
                extension, u, z, s = self._ensure_minimal_non_violation(
                    prefix_with_free_rules_only
                )

                prefix_new = prefix_with_free_rules_only + extension
                # print(
                #     f"prefix: {prefix} -> {prefix_with_free_rules_only} -> {prefix_new}"
                # )
                lb = self._calculate_lb(prefix_new)

                # print("prefix_new: {}".format(prefix_new))
                # print("lb: {}".format(lb))
                if (lb + self.lmbd) <= self.ub:
                    # print(f"-> queue")
                    # print(f"pushing {prefix_new} to queue")
                    new_queue.push((prefix_new, lb, ~u, z, s), key=(lb, prefix_new))
        else:
            print("Ax=b is unsolvable, skip queue items checking")
        # use the new queue in status
        self.status.set_queue(new_queue)

    def generate(self, return_objective=False) -> Iterable:
        yield from self._examine_R_and_S()

        self._update_queue()

        yield from self._generate_solution_at_root(return_objective)

        while not self.status.is_queue_empty():
            queue_item = self.status.pop_from_queue()
            yield from self._loop(*queue_item, return_objective=return_objective)
