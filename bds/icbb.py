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


def ensure_minimal_non_violation_plus(d):
    pass


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
        print("self.status.solution_set: {}".format(self.status.solution_set))
        print("self.status.reserve_set: {}".format(self.status.reserve_set))        
        candidate_prefixes = copy(self.status.solution_set | self.status.reserve_set)
        print("candidate_prefixes: {}".format(candidate_prefixes))

        # clear the solution and reserve set
        self.status.reset_reserve_set()
        self.status.reset_solution_set()
        print("self.status.solution_set: {}".format(self.status.solution_set))
        print("self.status.reserve_set: {}".format(self.status.reserve_set))

        print("self.pivot_rule_idxs: {}".format(self.pivot_rule_idxs))
        for prefix in candidate_prefixes:
            extention = self._ensure_satisfiability(prefix - self.pivot_rule_idxs)
            prefix_new = prefix + extention
            print("prefix_new: {}".format(prefix_new))
            # add to reserve set if needed
            if (
                prefix_new not in self.status.reserve_set
                and self._calculate_lb(prefix_new) <= self.ub
                and len(prefix_new) >= 1
            ):
                self.status.add_to_reserve_set(prefix_new)

            print("obj: {}".format(self._calculate_obj(prefix_new)))
            # yield and add to solution set if needed
            if (
                prefix_new not in self.status.solution_set
                and self._calculate_obj(prefix_new) <= self.ub
                and len(prefix_new) >= 1
            ):
                self.status.add_to_solution_set(prefix_new)
                print(f"adding {prefix_new}")
                yield prefix_new

    def _update_queue(self):
        """
        check each item in the queue and
        """
        new_queue = NonRedundantQueue()
        for prefix in self.status.queue:
            prefix_new, u, z, s = self._ensure_minimal_non_violation(prefix)
            lb = self._calculate_lb(prefix_new)
            if (lb + self.lmbd) <= self.ub:
                new_queue.push((prefix_new, lb, u, z, s), key=(lb, prefix_new))
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
