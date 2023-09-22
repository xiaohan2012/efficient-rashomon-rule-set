from numbers import Number
from typing import Any, Optional, Set, Tuple
from copy import deepcopy
from .queue import Queue
from .types import RuleSet


class SolverStatus:
    """a utility class to record the solver status"""

    def __init__(
        self,
    ):
        self._reserve_set: Set[RuleSet] = set()
        self._solution_set: Set[RuleSet] = set()
        self._queue: Queue = Queue()
        self._d_last: Optional[RuleSet] = None

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
    def d_last(self):
        """the prefix that was last checked by some branch-and-bound procedure"""
        return self._d_last

    def push_d_last_to_queue(self, key: float, other_data: Optional[Tuple[Any]] = None):
        """if d_last is set, push d_last to the queue using key"""
        if self._d_last is not None:
            data_to_push = (
                ((self._d_last,) + other_data)
                if other_data is not None
                else self.d_last
            )
            self.push_to_queue(key, data_to_push)

    def update_d_last(self, prefix: RuleSet):
        self._d_last = prefix

    def copy(self):
        return deepcopy(self)

    def __eq__(self, other: "SolverStatus") -> bool:
        assert isinstance(other, SolverStatus)
        return (
            (self.queue == other.queue)
            and (self.d_last == other.d_last)
            and (self.solution_set == other.solution_set)
            and (self.reserve_set == other.reserve_set)
        )
