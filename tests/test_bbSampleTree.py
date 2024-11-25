import gmpy2 as gmp
import numpy as np
import pytest
from gmpy2 import mpfr, mpz

from bds.bbSampleTreeApproxCounting import BranchAndBoundNaive
from bds.utils import mpz_all_ones

from .fixtures import rules, y  # noqa


@pytest.mark.skip("TODO: fix them.")
class TestSampleTreeApproxCounting:  # Sampling uses the same procedures
    def create_ST_object(self, rules, y, lmbd=0.1, l=1, k=150):
        return BranchAndBoundNaive(rules, ub=10, y=y, lmbd=0.1, l=1, k=150)

    def test__create_new_node_and_add_to_tree(self, rules, y):
        bb = self.create_ST_object(rules, y, lmbd=0.1, l=1, k=150)
        bb.reset_tree()
        assert bb.tree.num_nodes == 1

        rule1 = rules[0]
        rule2 = rules[1]
        child = bb._create_new_node_and_add_to_tree(
            rule1, lb=mpfr(), obj=mpfr(), captured=mpz(), parent_node=bb.tree.root
        )
        assert bb.tree.num_nodes == 2  # tree is updated

        grandchild = bb._create_new_node_and_add_to_tree(
            rule2, lb=mpfr(), obj=mpfr(), captured=mpz(), parent_node=child
        )
        assert bb.tree.num_nodes == 3  # tree is updated

        # depth should be correct
        # parent should be correct
        assert child.depth == 1
        assert child.parent == bb.tree.root

        assert grandchild.depth == 2
        assert grandchild.parent == child

        # add an already-added node just return the added node
        grandchild_same = bb._create_new_node_and_add_to_tree(
            rule2, lb=mpfr(), obj=mpfr(), captured=mpz(), parent_node=child
        )
        assert grandchild_same == grandchild

    def test_bb_init(self, rules, y):
        bb = self.create_ST_object(rules, y, lmbd=0.1, l=1, k=150)

        assert bb.num_train_pts == mpz(5)
        assert isinstance(bb.y_mpz, mpz)
        assert isinstance(bb.y_np, np.ndarray)

        assert bb.default_rule_fnr == 0.4
        assert bb.rules == rules

    def test_prepare(self, rules, y):
        bb = self.create_ST_object(rules, y, lmbd=0.1, l=1, k=150)
        bb.reset()

        # front and tree root are both accessible
        # and refer to the same node
        node, not_captured = bb.queue.front()
        assert node == bb.tree.root
        assert not_captured == mpz_all_ones(y.shape[0])
        assert gmp.popcount(not_captured) == bb.num_train_pts

    def test_generate_single_level(self, rules, y):
        bb = BranchAndBoundNaive(rules, ub=1, y=y, lmbd=0.1, l=1, k=150)
        bb.reset_tree()
        # #
        not_captured_root = bb._not_captured_by_default_rule()
        pseudosolutions = [
            ({0}, (bb.tree._root, not_captured_root), bb.tree._root.objective)
        ]  # empty -
        # #
        # L = math.ceil(bb.n/bb.l)
        bb.current_length = 0

        new_pseudosolutions, this_ratio = bb.generate_single_level(pseudosolutions)

        all_rule_sets = []
        all_objectives = []
        for sol in new_pseudosolutions:
            all_rule_sets.append(sol[0])
            all_objectives.append(float(sol[-1]))

        assert all_rule_sets == [{0}, {0, 1}]
        np.testing.assert_allclose(all_objectives, [0.4, 0.9])

    def test_generate_single_level_v2(self, rules, y):
        bb = BranchAndBoundNaive(rules, ub=0.49, y=y, lmbd=0.1, l=1, k=150)
        bb.reset_tree()
        # #
        not_captured_root = bb._not_captured_by_default_rule()
        pseudosolutions = [
            ({0}, (bb.tree._root, not_captured_root), bb.tree._root.objective)
        ]  # empty -
        # #
        # L = math.ceil(bb.n/bb.l)
        bb.current_length = 0

        new_pseudosolutions, this_ratio = bb.generate_single_level(pseudosolutions)

        all_rule_sets = []
        all_objectives = []
        for sol in new_pseudosolutions:
            all_rule_sets.append(sol[0])
            all_objectives.append(float(sol[-1]))

        assert all_rule_sets == [{0}]
        np.testing.assert_allclose(all_objectives, [0.4])

    def test_counter_complete(self, rules, y):
        # no bounds triggered
        bb = BranchAndBoundNaive(rules, ub=10, y=y, lmbd=0.1, l=1, k=1000000)
        Z = bb.runST()
        # #
        assert Z == 8  ## all rule sets are found because of the value of k and l

    def test_queue_and_tree(self, rules, y):
        bb = BranchAndBoundNaive(rules, ub=1, y=y, lmbd=0.1, l=1, k=150)
        bb.reset_tree()
        # #
        not_captured_root = bb._not_captured_by_default_rule()
        pseudosolutions = [
            ({0}, (bb.tree._root, not_captured_root), bb.tree._root.objective)
        ]  # empty -
        # #
        # L = math.ceil(bb.n/bb.l)
        bb.current_length = 0

        new_pseudosolutions, this_ratio = bb.generate_single_level(pseudosolutions)

        this_sample = new_pseudosolutions[0]

        bb.reset_queue_arbitrary_node(this_sample[1])
        bb.reset_tree_arbitrary_node(this_sample[1][0])

        assert bb.tree.num_nodes == 1
        assert bb.queue.front() == this_sample[1]
