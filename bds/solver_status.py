from numbers import Number
from typing import Any, Optional, Set, Tuple, Dict
from copy import deepcopy
from .queue import Queue
from .types import RuleSet


class SolverStatus:
    """a utility class to record the solver status"""

    def __init__(self, queue_class=Queue):
        self._reserve_set: Set[RuleSet] = set()
        self._solution_set: Set[RuleSet] = set()
        self._queue: Queue = queue_class()
        self._last_checked_prefix: Optional[RuleSet] = None
        self._other_data_for_last_checked_prefix: Optional[Dict[Any]] = None

    def set_status(self, new_status: "SolverStatus"):
        pass

    def add_to_reserve_set(self, prefix: RuleSet):
        """add a prefix to the reserve set"""
        self._reserve_set.add(prefix)

    def add_to_solution_set(self, prefix: RuleSet):
        """add a prefix to the solution set"""
        self._solution_set.add(prefix)

    def push_to_queue(self, key: Number, item: Any):
        self._queue.push(item, key)

    def queue_front(self) -> Any:
        return self._queue.front()

    def pop_from_queue(self) -> Any:
        return self._queue.pop()

    def queue_size(self):
        return self._queue.size

    def is_queue_empty(self):
        return self._queue.is_empty

    @property
    def queue(self):
        return self._queue

    @property
    def reserve_set(self):
        """the set of prefixes whose lower bound is below the objective upper bound"""
        return self._reserve_set

    @property
    def solution_set(self):
        """the set of feasible prefixes, i.e., whose objective value is below the objective upper bound"""
        return self._solution_set

    @property
    def last_checked_prefix(self):
        """return the prefix that was last checked by some branch-and-bound procedure

        if it is not set, raise an error

        return associated data if self._other_data_for_last_checked_prefix is set
        """
        if self._last_checked_prefix is None:
            return None

        if self._other_data_for_last_checked_prefix is not None:
            return self._last_checked_prefix, self._other_data_for_last_checked_prefix
        else:
            return self._last_checked_prefix

    def update_last_checked_prefix(
        self, prefix: RuleSet, other_data: Optional[Dict[str, Any]] = None
    ):
        self._last_checked_prefix = prefix
        self._other_data_for_last_checked_prefix = other_data

    def copy(self):
        return deepcopy(self)

    def __eq__(self, other: "SolverStatus") -> bool:
        assert isinstance(other, SolverStatus)
        return (
            (self.queue == other.queue)
            and (self.last_checked_prefix == other.last_checked_prefix)
            and (self.solution_set == other.solution_set)
            and (self.reserve_set == other.reserve_set)
        )
