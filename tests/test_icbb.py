import pytest
import numpy as np


from bds.cache_tree import CacheTree, Node
from bds.cbb import ConstrainedBranchAndBoundNaive
from bds.icbb import IncrementalConstrainedBranchAndBound
from bds.queue import Queue
from bds.random_hash import generate_h_and_alpha
from bds.utils import bin_array, randints

from .fixtures import rules, y
from .utils import generate_random_rules_and_y
from .test_cache_tree import create_dummy_node


class TestEquivalenceToNonincrementalVersion:
    """test the case without incremental computation,
    that IncrementalConstrainedBranchAndBound returns the same result as ConstrainedBranchAndBound,
    for the same input
    """

    @pytest.mark.parametrize("thresh", [None, 1, 2, 3])
    def test_equivalence_toy_data(self, rules, y, thresh):
        ub = float("inf")
        lmbd = 0.1
        A = bin_array([[0, 1, 0], [1, 0, 1]])
        t = bin_array([1, 0])

        icbb = IncrementalConstrainedBranchAndBound(rules, ub, y, lmbd)
        cbb = ConstrainedBranchAndBoundNaive(rules, ub, y, lmbd)

        expected = cbb.bounded_sols(thresh, A=A, t=t)
        actual = icbb.bounded_sols(thresh, A=A, t=t)

        assert actual == expected

    @pytest.mark.parametrize("thresh", [10, 20, 30])
    @pytest.mark.parametrize("seed", randints(3))
    def test_equivalence_random_data(self, thresh, seed):
        ub = 0.8
        lmbd = 0.1
        num_rules = 15
        A, t = generate_h_and_alpha(
            num_rules, int(num_rules / 2), seed=seed, as_numpy=True
        )
        rand_rules, rand_y = generate_random_rules_and_y(10, num_rules, rand_seed=seed)
        icbb = IncrementalConstrainedBranchAndBound(rand_rules, ub, rand_y, lmbd)
        cbb = ConstrainedBranchAndBoundNaive(rand_rules, ub, rand_y, lmbd)

        expected = cbb.bounded_sols(thresh, A=A, t=t)
        actual = icbb.bounded_sols(thresh, A=A, t=t)

        assert actual == expected


class Test2:
    def test__recalculate_satisfiability_vectors(self):
        # the tree is a path 0 -> 1 -> 2 -> 3
        root = create_dummy_node(0)
        node1 = create_dummy_node(1, root)
        node2 = create_dummy_node(2, node1)
        node3 = create_dummy_node(3, node2)

        # the constraint system encodes that:
        # 1, 0, 1| 0  -> x1 and x3 either both present or none
        # 0, 1, 1| 1  -> x2 and x3: only one is present
        A = bin_array([[1, 0, 1], [0, 1, 1]])
        t = bin_array([0, 1])

        rand_rules, rand_y = generate_random_rules_and_y(10, 3)
        icbb = IncrementalConstrainedBranchAndBound(
            rand_rules, float("inf"), rand_y, 0.1
        )
        icbb.setup_constraint_system(A, t)

        u, s, z, not_unsatisfied = icbb._recalculate_satisfiability_vectors(node1)
        np.testing.assert_allclose(u, [1, 1])
        np.testing.assert_allclose(s, [0, 0])
        np.testing.assert_allclose(z, [1, 0])
        assert not_unsatisfied is True

        u, s, z, not_unsatisfied = icbb._recalculate_satisfiability_vectors(node2)
        np.testing.assert_allclose(u, [1, 1])
        np.testing.assert_allclose(s, [0, 0])
        np.testing.assert_allclose(z, [1, 1])
        assert not_unsatisfied is True

        u, s, z, not_unsatisfied = icbb._recalculate_satisfiability_vectors(node3)
        np.testing.assert_allclose(u, [0, 0])
        np.testing.assert_allclose(s, [1, 0])
        np.testing.assert_allclose(z, [0, 0])
        assert not_unsatisfied is False


class TestIncrementalConstrainedBranchAndBound:
    def create_solver(self, rules, y):
        return IncrementalConstrainedBranchAndBound(rules, float("inf"), y, 0.1)

    def reset_solver_without_solver_status(self, solver):
        A = bin_array([[0, 1, 0], [1, 0, 1]])
        t = bin_array([1, 0])
        solver.reset(A, t)
        return solver

    def reset_solver_with_solver_status(self, solver):
        A = bin_array([[0, 1, 0], [1, 0, 1]])
        t = bin_array([1, 0])
        solver.reset(A, t, solver.solver_status)  # use its own status, which is weird
        return solver

    def test__init_solver_status(self, rules, y):
        solver = self.create_solver(rules, y)
        self.reset_solver_with_solver_status(solver)

        attr_names_to_check = [
            "_last_node",
            "_last_not_captured",
            "_last_rule",
            "_last_u",
            "_last_s",
            "_last_z",
        ]

        for name in attr_names_to_check:
            assert getattr(solver, name) is None

        assert getattr(solver, "_feasible_nodes") == []

    def test_solver_status(self, rules, y):
        solver = self.create_solver(rules, y)
        solver = self.reset_solver_without_solver_status(solver)

        status = solver.solver_status
        assert isinstance(status, dict)

        keys = [
            "last_node",
            "last_not_captured",
            "last_rule",
            "last_u",
            "last_s",
            "last_z",
            "feasible_nodes",
            "queue",
            "tree",
        ]
        for key in keys:
            assert key in status
        assert isinstance(status["feasible_nodes"], list)
        assert isinstance(status["queue"], Queue)
        assert isinstance(status["tree"], CacheTree)

    def test__update_solver_status(self):
        pass

    def test__record_feasible_solution(self):
        pass

    def test_is_incremental(self):
        pass

    def test_generate(self):
        pass

    def test__recalculate_satisfiability_vectors(self):
        pass

    def test__copy_solver_status(self):
        pass

    def test_reset(self):
        pass

    def test__loop(self):
        pass

    def test_basic(self):
        pass
