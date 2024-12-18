import heapq
from copy import deepcopy
from operator import itemgetter
from typing import Any, Union

import numpy as np


class Queue:
    """a priority queue (as a min heap) for branch-and-bound search"""

    def __init__(self):
        """a simple queue implementation"""
        self._items = []

        # a counter to record how many times items are popped
        self.popped_count = 0

        # a counter to record how many times items are pushed into
        self.pushed_count = 0

        # store the maximum queue size
        self._max_size = 0

    def push(self, item: Any, key: Union[float, int]):
        item_with_key = (key, item)
        heapq.heappush(self._items, item_with_key)
        self.pushed_count += 1

        self._max_size = max(self._max_size, self.size)

    def front(self) -> Any:
        key, item = self._items[0]
        return item

    def pop(self) -> Any:
        key, item = heapq.heappop(self._items)
        self.popped_count += 1
        return item

    @property
    def max_size(self):
        return self._max_size

    @property
    def size(self) -> int:
        return len(self._items)

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    def copy(self) -> "Queue":
        return deepcopy(self)

    def __items_eq__(self, other_queue: "Queue") -> bool:
        """check if the items are equal"""
        items = other_queue._items
        # print("self._items: {}".format(self._items))
        if len(self._items) != len(items):
            print(
                "len(self._items) != len(items): {} != {}".format(
                    len(self._items), len(items)
                )
            )
            return False

        for i1, i2 in zip(self._items, items):
            if len(i1) != len(i2):
                print(f"len(i1) != len(i2): {len(i1) != len(i2)}")
                return False
            k1, k2 = i1[0], i2[0]
            if k1 != k2:
                # keys are different
                print(f"k1 != k2: {k1 != k2}")
                return False
            for e1, e2 in zip(i1[1], i2[1]):
                # check the item content
                print(f"e1: {e1}")
                print(f"e2: {e2}")
                if type(e1) != type(e2):
                    print(f"type(e1) != type(e2): {type(e1) != type(e2)}")
                    return False
                if isinstance(e1, np.ndarray):
                    if not np.allclose(e1, e2):
                        print(
                            "not np.allclose(e1, e2): {}".format(
                                not np.allclose(e1, e2)
                            )
                        )
                        return False
                elif e1 != e2:
                    print(f"{e1} != {e2}")
                    return False
        return True

    def __eq__(self, other: "Queue") -> bool:
        assert isinstance(other, Queue)
        # print("self.__items_eq__(other._items): {}".format(self.__items_eq__(other)))
        return (
            (self.__items_eq__(other))
            and (self.popped_count == other.popped_count)
            and (self.pushed_count == other.pushed_count)
        )

    def __iter__(self):
        """return an iterator of the items (without the keys)"""
        return map(itemgetter(1), self._items)


class NonRedundantQueue(Queue):
    """a subclass of Queue which avoids pushing keys that already exist in the queue"""

    def __init__(self):
        super().__init__()
        self._existing_keys = set()

    def push(self, item: Any, key: Union[float, int]):
        if key not in self._existing_keys:
            super().push(item, key)
            self._existing_keys.add(key)

    def pop(self):
        key, item = heapq.heappop(self._items)
        self.popped_count += 1
        self._existing_keys.remove(key)
        return item
