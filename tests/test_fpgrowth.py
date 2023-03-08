import pytest
from bds.fpgrowth import (
    FPTree,
    add_frequency_to_transaction_list,
    extract_frequent_items,
    filter_and_reorder_transaction_list,
    squash_path_by_leaf_frequency,
)


@pytest.fixture
def a_path():
    root = FPTree(0, count=3)
    child = root.add_child(FPTree(1, count=2))
    child.add_child(FPTree(2, count=1))
    return root


class TestFPTree:
    def test_add_child(self):
        root = FPTree(0, count=3)
        child = root.add_child(FPTree(1, count=2))
        assert root.children[0].name == child.name
        assert root.children[0].count == child.count

        assert child.parent.name == root.name
        assert child.parent.count == root.count

    def test_is_path(self):
        # a single node is a path
        root = FPTree(0)
        assert root.is_path

        # multiple nodes in a line is a path
        child = root.add_child(FPTree(1))
        grandchild = child.add_child(FPTree(2))
        assert root.is_path

        # turn it into a non-path
        root.add_child(FPTree(3))
        assert not root.is_path

    def test_is_leaf(self):
        root = FPTree(0)
        child = root.add_child(FPTree(1))
        assert not root.is_leaf
        assert child.is_leaf

    def test_all_names(self):
        root = FPTree(0)
        assert root.all_names == {0}
        child = root.add_child(FPTree(1))
        assert root.all_names == {0, 1}
        child.add_child(FPTree(2))
        assert root.all_names == {0, 1, 2}
        root.add_child(FPTree(3))
        assert root.all_names == {0, 1, 2, 3}

    def test_item_count_on_path_to_root(self, a_path):
        a_leaf = a_path.children[0].children[0]
        assert a_leaf.item_count_on_path_to_root == ((0, 3), (1, 2), (2, 1))


class Test_add_frequency_to_transaction_list:
    def test_simple(self):
        input_data = [{"a", "c", "d"}, {"a", "b", "c"}]
        expected = [{"a": 1, "c": 1, "d": 1}, {"a": 1, "b": 1, "c": 1}]
        actual = add_frequency_to_transaction_list(input_data)
        assert expected == actual


class Test_extract_frequent_items:
    def test_simple(self):
        input_data = [{"a": 2, "c": 1, "d": 1}, {"a": 2, "b": 2, "c": 2}]
        actual = extract_frequent_items(input_data, 3)
        expected = ["a", "c"]
        assert actual == expected

        actual = extract_frequent_items(input_data, 2)
        expected = ["a", "c", "b"]
        assert actual == expected

        actual = extract_frequent_items(input_data, 10)
        expected = []
        assert actual == expected


class Test_filter_and_reorder_transaction_list:
    def test_simple(self):
        input_data = [{"a": 2, "c": 1, "d": 1}, {"a": 2, "b": 2, "c": 2}]
        frequent_items = ["a", "c"]
        actual = filter_and_reorder_transaction_list(input_data, frequent_items)
        expected = [[("a", 2), ("c", 1)], [("a", 2), ("c", 2)]]
        assert actual == expected

        frequent_items = ["a", "c", "b"]
        actual = filter_and_reorder_transaction_list(input_data, frequent_items)
        expected = [[("a", 2), ("c", 1)], [("a", 2), ("c", 2), ("b", 2)]]
        assert actual == expected

        frequent_items = []
        actual = filter_and_reorder_transaction_list(input_data, frequent_items)
        expected = [[], []]
        assert actual == expected


class Test_squash_path_by_leaf_frequency:
    def test_simple(self):
        input_data = (("f", 4), ("c", 3), ("a", 3), ("m", 2), ("p", 2))
        expected = (("f", 2), ("c", 2), ("a", 2), ("m", 2), ("p", 2))
        actual = squash_path_by_leaf_frequency(input_data)
        assert actual == expected


class Test_insert_tree:
    pass


class TestFPGrowth:
    pass
