import heapq
from typing import Any, Union
from copy import deepcopy


class Queue:
    """a priority queue (as a min heap) for branch-and-bound search"""

    def __init__(self):
        """a simple queue implementation"""
        self._items = []

        # a counter to record how many times items are popped
        self.popped_count = 0

        # a counter to record how many times items are pushed into
        self.pushed_count = 0

    def push(self, item: Any, key: Union[float, int]):
        item_with_key = (key, item)
        heapq.heappush(self._items, item_with_key)
        self.pushed_count += 1

    def front(self) -> Any:
        key, item = self._items[0]
        return item

    def pop(self) -> Any:
        key, item = heapq.heappop(self._items)
        self.popped_count += 1
        return item

    @property
    def size(self) -> int:
        return len(self._items)

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    def copy(self) -> "Queue":
        return deepcopy(self)
        # new_queue = self.__class__()
        # new_queue._items = deepcopy(self._items)
        # new_queue.popped_count = self.popped_count
        # new_queue.pushed_count = self.pushed_count
        
        # return new_queue

    def __eq__(self, other: "Queue") -> bool:
        assert isinstance(other, Queue)
        return (
            (self._items == other._items)
            and (self.popped_count == other.popped_count)
            and (self.pushed_count == other.pushed_count)
        )
