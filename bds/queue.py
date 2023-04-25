import heapq
from typing import Any, Union


class Queue:
    """a priority queue (as a min heap) for branch-and-bound search"""

    def __init__(self):
        """a simple queue implementation"""
        self._items = []

    def push(self, item: Any, key: Union[float, int]):
        item_with_key = (key, item)
        heapq.heappush(self._items, item_with_key)

    def front(self) -> Any:
        key, item = self._items[0]
        return item

    def pop(self) -> Any:
        key, item = heapq.heappop(self._items)
        return item

    @property
    def size(self) -> int:
        return len(self._items)

    @property
    def is_empty(self) -> bool:
        return self.size == 0
        
