import pytest
import itertools
from bds.utils import powerset
from bds.fpgrowth import (
    FPTree,
    _add_frequency_to_transaction_list,
    _extract_frequent_items_and_order,
    _filter_and_reorder_transaction_list,
    squash_path_by_leaf_frequency,
    insert_node,
    build_fptree,
    fpgrowth_on_tree,
    preprocess_transaction_list,
)


@pytest.fixture
def a_path():
    root = FPTree(0, count=3)
    child = root.add_child(FPTree(1, count=2))
    child.add_child(FPTree(2, count=1))
    return root


class TestFPTree:
    def test_as_nested_tuple(self, a_path):
        actual = a_path.as_nested_tuple()
        expected = ((0, 3), [((1, 2), [((2, 1), [])])])  # root  # children
        assert actual == expected

        a_path.add_child(FPTree(3, 1))
        expected = (
            (0, 3),  # root
            [((1, 2), [((2, 1), [])]), ((3, 1), [])],  # children
        )
        actual = a_path.as_nested_tuple()

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
        assert root.all_names(exclude_root=False) == {0}
        child = root.add_child(FPTree(1))
        assert root.all_names(exclude_root=False) == {0, 1}
        child.add_child(FPTree(2))
        assert root.all_names(exclude_root=False) == {0, 1, 2}
        root.add_child(FPTree(3))
        assert root.all_names(exclude_root=False) == {0, 1, 2, 3}

        # now we exclude the root
        assert root.all_names(exclude_root=True) == {1, 2, 3}

    def test_item_count_on_path_to_root(self, a_path):
        a_leaf = a_path.children[0].children[0]
        # (0, 3),  should be excluded
        assert a_leaf.item_count_on_path_to_root == [(1, 2), (2, 1)]


class Test_add_frequency_to_transaction_list:
    def test_simple(self):
        input_data = [{"a", "c", "d"}, {"a", "b", "c"}]
        expected = [{"a": 1, "c": 1, "d": 1}, {"a": 1, "b": 1, "c": 1}]
        actual = _add_frequency_to_transaction_list(input_data)
        assert expected == actual


class Test_extract_frequent_items:
    def test_simple(self):
        input_data = [{"a": 2, "c": 1, "d": 1}, {"a": 2, "b": 2, "c": 2}]
        actual = _extract_frequent_items_and_order(input_data, 3)
        expected = ["a", "c"]
        assert actual == expected

        actual = _extract_frequent_items_and_order(input_data, 2)
        expected = ["a", "c", "b"]
        assert actual == expected

        actual = _extract_frequent_items_and_order(input_data, 10)
        expected = []
        assert actual == expected


class Test_filter_and_reorder_transaction_list:
    def test_simple(self):
        input_data = [{"a": 2, "c": 1, "d": 1}, {"a": 2, "b": 2, "c": 2}]
        frequent_items = ["a", "c"]
        actual = _filter_and_reorder_transaction_list(input_data, frequent_items)
        expected = [[("a", 2), ("c", 1)], [("a", 2), ("c", 2)]]
        assert actual == expected

        frequent_items = ["a", "c", "b"]
        actual = _filter_and_reorder_transaction_list(input_data, frequent_items)
        expected = [[("a", 2), ("c", 1)], [("a", 2), ("c", 2), ("b", 2)]]
        assert actual == expected

        frequent_items = []
        actual = _filter_and_reorder_transaction_list(input_data, frequent_items)
        expected = [[], []]
        assert actual == expected


class Test_squash_path_by_leaf_frequency:
    def test_simple(self):
        input_data = (("f", 4), ("c", 3), ("a", 3), ("m", 2), ("p", 2))
        expected = (("f", 2), ("c", 2), ("a", 2), ("m", 2), ("p", 2))
        actual = squash_path_by_leaf_frequency(input_data)
        assert actual == expected


class Test_insert_tree:
    def test_no_new_node(self, a_path):
        # a_path: (0, 2), (1, 2), (2, 1)
        itemset = [(1, 1), (2, 1)]
        head, tail = itemset[0], itemset[1:]

        # no new node is created, we simply increment the count
        insert_node(head, tail, a_path)

        actual = a_path.as_nested_tuple()
        expected = ((0, 3), [((1, 3), [((2, 2), [])])])
        assert actual == expected

    def test_with_new_node(self, a_path):
        itemset = [(2, 1), (1, 1)]
        head, tail = itemset[0], itemset[1:]

        # two new nodes are created
        insert_node(head, tail, a_path)

        actual = a_path.as_nested_tuple()
        expected = (
            (0, 3),
            [
                ((1, 2), [((2, 1), [])]),
                ((2, 1), [((1, 1), [])]),  # the new branch (and nodes)
            ],
        )
        assert actual == expected


class Test_build_fptree:
    def test_simple(self):
        input_data = [[(0, 1), (1, 1), (2, 1)], [(0, 1), (2, 1), (1, 1)]]
        tree = build_fptree(input_data)
        actual = tree.as_nested_tuple()
        expected = (
            ("ROOT", 0),
            [((0, 2), [((1, 1), [((2, 1), [])]), ((2, 1), [((1, 1), [])])])],
        )
        assert actual == expected


class TestFPGrowth:
    def test_single_node(self):
        single_node_tree = FPTree("ROOT", 1)
        single_node_tree.add_child(FPTree(0, 1))

        actual = set(fpgrowth_on_tree(single_node_tree, set(), 1))
        expected = set([(0,)])
        assert actual == expected

    def test_a_path(self):
        path = FPTree("ROOT", 1)
        child = path.add_child(FPTree(0, 1))
        child.add_child(FPTree(1, 1))

        actual = set(fpgrowth_on_tree(path, set(), 1))
        expected = set([(0,), (1,), (0, 1)])
        assert actual == expected

    @pytest.mark.parametrize('min_support, expected',
                             [
                                 (2, powerset({0, 1}, min_size=1)),
                                 (1, itertools.chain(
                                     powerset({0, 1, 2}, min_size=1),
                                     powerset({0, 1, 3}, min_size=1),
                                 ))
                             ])
    def test_allgether(self, min_support, expected):
        input_transactions = [{0, 1, 2}, {0, 1, 3}]

        ordered_input_data = preprocess_transaction_list(
            input_transactions, min_support
        )
        
        tree = build_fptree(ordered_input_data)
        actual = set(fpgrowth_on_tree(tree, set(), min_support))
        expected = set(
            map(
                lambda seq: tuple(sorted(seq)),
                expected
            )
        )
        assert actual == expected
