from typing import List, Optional

import numpy as np

from .cbb import (
    ConstrainedBranchAndBound,
    ensure_satisfiability,
    ensure_minimal_non_violation,
)
from .queue import Queue
from .types import SolutionSet

def ensure_minimal_non_violation_plus(d):
    pass


class IncrementalConstrainedBranchAndBound:
    def __post_init__(self):
        # TODO: add R, S, and dlast
        pass

    def reset(
        self,
        A: np.ndarray,
        t: np.ndarray,
        previous_cbb: Optional["IncrementalConstrainedBranchAndBound"] = None,
    ):
        if previous_cbb is None:
            previous_cbb = IncrementalConstrainedBranchAndBound(
                self.rules, self.XX
            )  # TODO: fill this
        self.previous_cbb = previous_cbb

        # setup the queue, R, and S
        self.S = set()
        self.R = set()
        self.queue = Queue()

    def _examine_R_and_S(self, return_objective=False):
        for d in self.previous_cbb.S | self.previous_cbb.R:
            dp = ensure_satisfiability(d)
            if dp not in self.R and self._lower_bound(dp) <= self.ub:
                self.R.add(dp)
            if dp not in self.S and self._objective(dp) <= self.ub:
                self.S.add(dp)
                yield dp

    def _update_queue(self):
        for d in self.previous_cbb.queue | {self.previous_cbb.dlast}:
            dp, u, z, s = ensure_minimal_non_violation_plus(d)
            lb = self._lower_bound(dp)
            if lb <= self.ub and dp not in self.queue:
                self.queue.push(lb, (dp, u, z, s))

    def _search_from_queue(
        self,
        return_objective: bool = False,
    ):
        """
        continue branch-and-bound search from an existing queue,

        in the meanwhile, update the set of feasible solutions and "reserve" soltuions.
        """
        cbb = ConstrainedBranchAndBound(self.rules, self.ub, self.y, self.lmbd)
        cbb.set_status(self.queue, self.S, self.R)
        cbb.setup_constraint_system(self.A, self.b)
        yield from cbb.bounded_sols(return_objective=return_objective)

    def generate(self, return_objective=False) -> Iterable:
        yield from self._examine_R_and_S()
        self._update_queue()
        yield from self._search_from_queue(return_objective)
