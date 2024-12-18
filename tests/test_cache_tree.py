import pytest

from bds.cache_tree import CacheTree, Node


def create_dummy_node(
    rule_id, parent=None, lower_bound=0, objective=0, num_captured=0, pivot_rule_ids=[]
):
    return Node(
        rule_id=rule_id,
        lower_bound=lower_bound,
        objective=objective,
        num_captured=num_captured,
        parent=parent,
        pivot_rule_ids=pivot_rule_ids,
    )


class TestNode:
    def test__init__(self):
        parent = create_dummy_node(0)
        child = create_dummy_node(1, parent)
        grand_child = create_dummy_node(2, child)

        assert parent.children[child.rule_id] == child
        assert child.depth == 1
        assert grand_child.depth == 2

    def test___eq__(self):
        n1 = create_dummy_node(0)

        n2 = create_dummy_node(0)

        n3 = create_dummy_node(1)

        assert n1 == n2
        assert n1 == n1
        assert n2 == n2

        assert n1 != n3  # node ids are different
        assert n2 != n3

        # parent of n2 is changed to n3
        n2.parent = n3
        n3.children[n2.rule_id] = n2
        assert n1 != n2

        # parent of n1 is changed to n3 too
        # and n1 and n2 become equal
        n1.parent = n3
        n3.children[n1.rule_id] = n1
        assert n1 == n2

        # n2.lower_bound is changed
        n2.lower_bound = 1
        assert n1 != n2

    def test_simple(self):
        parent = create_dummy_node(0)

        child1 = create_dummy_node(1)

        child2 = create_dummy_node(2)
        grand_child = create_dummy_node(3)

        parent.add_child(child1)
        assert child1.parent == parent
        assert parent.num_children == 1
        assert child1.depth == 1

        # no duplicate is added
        with pytest.raises(KeyError, match=".*is already a child.*"):
            parent.add_child(child1)

        with pytest.raises(ValueError, match='cannot add "self" as a child of itself'):
            parent.add_child(parent)

        parent.add_child(child2)
        assert child2.parent == parent
        assert child2.depth == 1
        assert parent.num_children == 2

        assert parent.total_num_nodes == 3
        assert child1.total_num_nodes == 1
        assert child1.total_num_nodes == 1

        child1.add_child(grand_child)
        assert grand_child.depth == 2

    def test_make_root(self):
        num_train_pts = 10
        fnr = 0.2
        root = Node.make_root(fnr=fnr, num_train_pts=num_train_pts)

        assert root.num_captured == num_train_pts
        assert root.rule_id == 0
        assert root.depth == 0
        assert root.lower_bound == 0
        assert root.objective == fnr

    def test_get_ruleset_ids(self):
        leaf = create_dummy_node(
            2, parent=create_dummy_node(1, parent=create_dummy_node(0))
        )
        parent = leaf.parent
        grand_parent = leaf.parent.parent

        assert leaf.get_ruleset_ids() == {0, 1, 2}
        assert parent.get_ruleset_ids() == {0, 1}
        assert grand_parent.get_ruleset_ids() == {0}

    def test_get_ruleset_ids_with_pivot_rule_ids(self):
        leaf = create_dummy_node(
            2, parent=create_dummy_node(1, parent=create_dummy_node(0))
        )
        parent = leaf.parent
        grand_parent = leaf.parent.parent

        # add some pivots
        leaf.pivot_rule_ids = [3, 4]
        parent.pivot_rule_ids = [5]
        grand_parent.pivot_rule_ids = [6]
        assert leaf.get_ruleset_ids() == {0, 1, 2, 3, 4, 5, 6}

        assert parent.get_ruleset_ids() == {0, 1, 5, 6}

        assert grand_parent.get_ruleset_ids() == {0, 6}

    def test_num_rules(self):
        root = create_dummy_node(0, pivot_rule_ids=[2, 3])
        child = create_dummy_node(1, parent=root, pivot_rule_ids=[4])
        assert child.num_rules == (2 + 2)
        # root._num_rules should be updated if any of its descendants' _num_rules are updated
        assert root._num_rules == 2

        # this call uses the cache
        assert root.num_rules == 2


class TestCacheTree:
    def test_simple(self):
        root = Node.make_root(0, 10)
        tree = CacheTree()
        with pytest.raises(ValueError, match="root is not set yet"):
            tree.root

        tree.add_node(root)
        assert tree.root == root

        assert tree.num_nodes == 1
        child = create_dummy_node(1)  # do not set parent here
        grand_child = create_dummy_node(2)  # do not set parent here

        tree.add_node(child, parent=root)
        assert tree.num_nodes == 2

        tree.add_node(grand_child, parent=child)
        assert tree.num_nodes == 3

        with pytest.raises(ValueError, match="Root has already been set!"):
            tree.add_node(child)
