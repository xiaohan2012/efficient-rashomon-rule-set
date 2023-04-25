import numpy as np
import pytest
from bds.queue import Queue
from bds.utils import randints


class TestQueue:
    def get_simple_queue(self, seed=None):
        np.random.seed(seed)
        self.items_with_key = [("first", -100), ("second", -10), ("third", -1)]
        self.items_with_key_permutated = self.items_with_key.copy()
        np.random.shuffle(self.items_with_key_permutated)

        q = Queue()

        print("self.items_with_key_permutated: {}".format(self.items_with_key_permutated))
        for item, key in self.items_with_key_permutated:
            q.push(item, key=key)

        return q

    @pytest.mark.parametrize("seed", randints(1))
    def test_all(self, seed):
        q = self.get_simple_queue(seed)

        total_queue_size = len(self.items_with_key)
        assert q.size == total_queue_size
        assert not q.is_empty

        for i, expected_item_and_key in zip(
            range(q.size), self.items_with_key
        ):
            actual_item = q.front()
            expected_item, _ = expected_item_and_key
            assert expected_item == actual_item
            actual_item = q.pop()
            assert expected_item == actual_item
            assert q.size == total_queue_size - i - 1
