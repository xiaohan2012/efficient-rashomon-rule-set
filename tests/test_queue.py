import numpy as np
import pytest
from gmpy2 import mpz
from bds.queue import Queue, NonRedundantQueue
from bds.utils import randints, bin_array


class TestQueue:
    def get_simple_queue(self, seed=None):
        np.random.seed(seed)
        self.items_with_key = [("first", -100), ("second", -10), ("third", -1)]
        self.items_with_key_permutated = self.items_with_key.copy()
        np.random.shuffle(self.items_with_key_permutated)

        q = Queue()

        print(
            "self.items_with_key_permutated: {}".format(self.items_with_key_permutated)
        )
        for item, key in self.items_with_key_permutated:
            q.push(item, key=key)

        return q

    @pytest.mark.parametrize("seed", randints(1))
    def test_all(self, seed):
        q = self.get_simple_queue(seed)

        total_queue_size = len(self.items_with_key)
        assert q.size == total_queue_size
        assert not q.is_empty

        for i, expected_item_and_key in zip(range(q.size), self.items_with_key):
            actual_item = q.front()
            expected_item, _ = expected_item_and_key
            assert expected_item == actual_item
            actual_item = q.pop()
            assert expected_item == actual_item
            assert q.size == total_queue_size - i - 1

    def test_copy(self):
        q = self.get_simple_queue(1234)
        q_cp = q.copy()

        assert q._items == q_cp._items
        assert q.pushed_count == q_cp.pushed_count
        assert q.popped_count == q_cp.popped_count

        # modifying the original queue shouldn't affect the new queue
        q.pop()
        assert q._items != q_cp._items
        assert q.size == (q_cp.size - 1)
        assert q.popped_count == (q_cp.popped_count + 1)

    def test___eq__(self):
        q = self.get_simple_queue(1234)
        q_cp = q.copy()
        assert q == q_cp

        q.pop()
        assert q != q_cp

    def test__items_eq__case_1(self):
        """different lengths"""
        q1 = Queue()
        item0 = (mpz("0b001"), bin_array([0, 0, 1]))
        item1 = (mpz("0b010"), bin_array([0, 1, 0]))

        q1.push(item0, key=0)

        q2 = Queue()
        q2.push(item0, key=0)

        assert q1.__items_eq__(q2._items)

        q1.push(item1, key=1)
        assert not q1.__items_eq__(q2._items)

    def test__items_eq__case_2(self):
        """different content"""
        q1 = Queue()
        item0 = (mpz("0b001"), bin_array([0, 0, 1]))
        item1 = (mpz("0b010"), bin_array([0, 1, 0]))

        q1.push(item0, key=0)

        q2 = Queue()
        q2.push(item1, key=0)

        assert not q1.__items_eq__(q2._items)

    def test__items_eq__case_3(self):
        """different types"""
        q1 = Queue()
        item0 = (mpz("0b001"), bin_array([0, 0, 1]))
        item1 = (bin_array([0, 1, 0]), mpz("0b010"))

        q1.push(item0, key=0)

        q2 = Queue()
        q2.push(item1, key=0)

        assert not q1.__items_eq__(q2._items)

    def test_iterators(self):
        q = Queue()
        q.push("zero", 0)
        q.push("one", 1)
        q.push("two", 2)
        q.push("two", 2)

        assert len(list(q)) == 4
        assert set(q) == {"zero", "one", "two"}


class TestNonRedundantQueue:
    def test_basic(self):
        non_redundant_queue = NonRedundantQueue()

        item0 = (mpz("0b001"), bin_array([0, 0, 1]))
        item1 = (mpz("0b101"), bin_array([0, 1, 0]))

        non_redundant_queue.push(item0, key=0)
        assert non_redundant_queue.size == 1

        non_redundant_queue.push(item1, key=0)
        non_redundant_queue._existing_keys == {0}
        assert non_redundant_queue.size == 1

        non_redundant_queue.pop()
        non_redundant_queue._existing_keys == set()
        assert non_redundant_queue.size == 0
        non_redundant_queue.push(item1, key=1)
        non_redundant_queue._existing_keys == {1}
        assert non_redundant_queue.size == 1
