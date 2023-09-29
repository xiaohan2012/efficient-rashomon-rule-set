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
        candidate_prefixes = copy(self.status.solution_set | self.status.reserve_set)

        # clear the solution and reserve set
        self.status.reset_reserve_set()
        self.status.reset_solution_set()
        for prefix in candidate_prefixes:
            extention = self._ensure_satisfiability(prefix - self.pivot_rule_idxs)
            prefix_new = prefix + extention
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
                self.status.add_to_solution_set(prefix_new)
                yield prefix_new

    def _update_queue(self):
        """
        check each item in the queue and
        """
        new_queue = NonRedundantQueue()
        print("inherited queue: {}".format(list(self.status.queue)))
        for queue_item in self.status.queue:
            prefix = queue_item[0]
            extension, u, z, s = self._ensure_minimal_non_violation(
                prefix - self.pivot_rule_idxs  # remove the pivots
            )

            prefix_new = prefix + extension
            lb = self._calculate_lb(prefix_new)
            
            print("prefix_new: {}".format(prefix_new))            
            print("lb: {}".format(lb))
            if (lb + self.lmbd) <= self.ub:
                print(f"pushing {prefix_new} to queue")
                new_queue.push((prefix_new, lb, ~u, z, s), key=(lb, prefix_new))
        # use the new queue in status
        self.status.set_queue(new_queue)

    def generate(self, return_objective=False) -> Iterable:
        if not self.is_linear_system_solvable:
            logger.debug("abort the search since the linear system is not solvable")
            yield from ()
        else:
            yield from self._examine_R_and_S()

            self._update_queue()

            yield from self._generate_solution_at_root(return_objective)

            while not self.status.is_queue_empty():
                queue_item = self.status.pop_from_queue()
                yield from self._loop(*queue_item, return_objective=return_objective)
