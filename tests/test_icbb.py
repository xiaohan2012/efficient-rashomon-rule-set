import itertools

import numpy as np
import pytest
from gmpy2 import mpz, mpfr

from bds.cache_tree import CacheTree, Node
from bds.cbb import ConstrainedBranchAndBoundNaive
from bds.icbb import IncrementalConstrainedBranchAndBound
from bds.queue import Queue
from bds.random_hash import generate_h_and_alpha
from bds.rule import Rule
from bds.utils import bin_array, bin_zeros, randints

from typing import Optional, List, Dict, Tuple, Union
from .fixtures import rules, y
from .test_cache_tree import create_dummy_node
from .utils import generate_random_rules_and_y


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

        assert icbb.is_incremental is False
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

        assert icbb.is_incremental is False
        assert actual == expected


class TestRecalculateSatisfiabilityVectors:
    def test_simple(self):
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


class Utility:
    """a utility class that provides data loading, solver creation support, etc"""

    @property
    def num_rules(self):
        return 5

    @property
    def ub(self):
        return float("inf")

    @property
    def lmbd(self):
        return 0.1

    @property
    def num_constraints(self):
        return self.num_rules - 1

    @property
    def vacuum_constraints(self):
        """a constraint system in vacuum (= no constraint at all)"""
        A = bin_zeros((1, self.num_rules))
        t = bin_array([0])
        return A, t

    def get_A_and_t_that_exclude_rules(
        self, excluded_rules: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        # construct a new constraint system
        # excluding the ith rule, for i in excluded_rules
        num_constraints = len(excluded_rules)
        A = bin_zeros((num_constraints, self.num_rules))
        t = bin_zeros(num_constraints)

        for i in range(num_constraints):
            A[i, i] = 1  # exclude the (i+1)th rule

        return A, t

    def create_icbb(self):
        """create incremental solver"""
        rand_rules, rand_y = generate_random_rules_and_y(
            10, self.num_rules, rand_seed=12345
        )
        return IncrementalConstrainedBranchAndBound(
            rand_rules, self.ub, rand_y, self.lmbd
        )

    def create_cbb(self):
        """create non-incremental solver"""
        rand_rules, rand_y = generate_random_rules_and_y(
            10, self.num_rules, rand_seed=12345
        )
        return ConstrainedBranchAndBoundNaive(rand_rules, self.ub, rand_y, self.lmbd)

    @property
    def A_and_t(self):
        return generate_h_and_alpha(
            self.num_rules, self.num_rules - 1, seed=12345, as_numpy=True
        )


class TestGenerate(Utility):
    """cannot find a better test name"""

    @pytest.mark.parametrize(
        "starting_rule, expected",
        [
            (None, 10),
            (Rule(2, "rule-2", 0, mpz()), 10),
            (Rule(11, "rule-11", 0, mpz()), 11),
        ],
    )
    def test__get_continuation_idx(self, starting_rule, expected):
        icbb = self.create_icbb()
        parent_node = create_dummy_node(10)
        assert icbb._get_continuation_idx(parent_node, starting_rule) == expected

    def test__copy_solver_status(self):
        thresh = 1
        # computation from scratch
        icbb1 = self.create_icbb()

        A, t = self.A_and_t
        icbb1.bounded_count(thresh, A=A, t=t)

        # incremental computation
        icbb2 = self.create_icbb()
        icbb2._copy_solver_status(icbb1.solver_status)
        assert icbb2.is_incremental is True

        assert isinstance(icbb1.tree, CacheTree)
        assert isinstance(icbb1.queue, Queue)

        attrs_to_check = [
            "_last_node",
            "_last_not_captured",
            "_last_rule",
            "_feasible_nodes",
        ]
        for attr in attrs_to_check:
            assert getattr(icbb1, attr) == getattr(icbb2, attr)

        # modify the queue and the original queue shouldn't be affected
        icbb2.queue.push("random stuff", (11, tuple()))
        assert icbb1.solver_status["queue"].size < icbb2.queue.size

        # modify the feasible node list of icbb2 should not affect icbb1
        icbb2._feasible_nodes.append("blah")
        assert len(icbb1.solver_status["feasible_nodes"]) < len(icbb2._feasible_nodes)

        # modify the tree of icbb2 should not affect icbb1
        icbb2.tree.root.children[1].lower_bound += 1
        assert (
            icbb1.solver_status["tree"].root.children[1].lower_bound
            != icbb2.tree.root.children[1].lower_bound
        )

    def test__generate_from_last_checked_node_feasible_case(self):
        """test generating from the last checked node
        in this case, the last checked node is feasible"""
        thresh = 1
        # computation from scratch
        icbb1 = self.create_icbb()

        # a constraint system in vacuum (= no constraint at all)
        A, t = self.vacuum_constraints
        sol1 = icbb1.bounded_sols(thresh, A=A, t=t)[0]
        assert sol1 == {0, 1}  # the first solution should be {0, 1} by construction

        # incremental computation from icbb1
        # using the same set of constraints
        icbb2 = self.create_icbb()
        icbb2.reset(A=A, t=t, solver_status=icbb1.solver_status)
        sol2 = list(itertools.islice(icbb2._generate_from_last_checked_node(), thresh))[
            0
        ]

        assert icbb2.is_incremental is True
        assert sol2 == {0, 2}  # and sol2 should be {0, 2} by construction

        # incremental computation from icbb2
        # using the same set of constraints
        icbb3 = self.create_icbb()
        icbb3.reset(A=A, t=t, solver_status=icbb2.solver_status)
        # now we yield 2 solutions
        sols3 = list(itertools.islice(icbb3._generate_from_last_checked_node(), 2))

        assert icbb3.is_incremental is True
        assert sols3 == [{0, 3}, {0, 4}]

    def test__generate_from_last_checked_node_infeasible_case(self):
        """test generating from the last checked node
        in this case, the last checked node becomes infeasible due to the new unsatisfiable constraint system
        """
        thresh = 6
        # generate the {0, 1}, ..., {0, 5}, and some other ruleset
        # computation from scratch
        icbb1 = self.create_icbb()

        # a constraint system in vacuum (= no constraint at all)
        A, t = self.vacuum_constraints
        sols = icbb1.bounded_sols(thresh, A=A, t=t)
        assert len(sols) == 6

        # construct a new constraint sysmte that cannot be satisfied
        # thus making any further search in vain
        A1 = bin_zeros((2, self.num_rules))
        t1 = bin_zeros(2)

        # the first rule are involved in both constraints
        A1[0, 0] = 1
        A1[1, 0] = 1
        # however, it is both present and absent
        t1[0], t1[1] = 0, 1

        icbb2 = self.create_icbb()
        icbb2.reset(A=A1, t=t1, solver_status=icbb1.solver_status)

        # since the last checked node becomes infeasible
        # we do not continue from that node any more
        # thus no solution yielded
        sols = list(itertools.islice(icbb2._generate_from_last_checked_node(), thresh))
        assert len(sols) == 0

    def test_solver_status_case_1(self):
        """we yield just one solution and check if the solver status is as expected

        the solving procedure is something like this:

        1. initial state:

        queue = [root]
        tree:
        - root

        2. after yielding {0, 1}

        tree:
        - root
          - 1

        queue = [(0, 1)]  # other children of root has not been checked yet
        last_node = root
        last_rule = rule_1
        """
        thresh = 1
        # computation from scratch
        icbb1 = self.create_icbb()

        A, t = self.vacuum_constraints
        icbb1.bounded_sols(thresh, A=A, t=t)[0]

        ss = icbb1.solver_status
        assert ss["last_node"] == icbb1.tree.root
        assert ss["last_rule"] == icbb1.rules[0]
        assert ss["queue"].size == 1
        assert ss["queue"].front()[0].rule_id == 1
        assert ss["tree"].num_nodes == 2  # root + {0, 1}

    def test_solver_status_case_2(self):
        """we yield more solutions and check if the solver status is as expected

        the solving procedure is something like this:

        1. initial state:

        queue = [root]
        tree:
        - root

        2. after yielding {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}

        tree:
        - root
          - 1
          - 2
          - 3
          - 4
          - 5

        queue = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
        last_node = root
        last_rule = rule_4
        """
        thresh = 5
        # computation from scratch
        icbb1 = self.create_icbb()

        A, t = self.vacuum_constraints
        icbb1.bounded_sols(thresh, A=A, t=t)[0]

        ss = icbb1.solver_status
        assert ss["last_node"] == icbb1.tree.root
        assert ss["last_rule"] == icbb1.rules[4]
        assert ss["queue"].size == 5

        # check the nodes in the queue are correct
        # item[1] is the stored item
        # item[1][0] is the stored node
        assert set(
            [tuple(item[1][0].get_ruleset_ids()) for item in ss["queue"]._items]
        ) == {
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
        }
        assert (
            ss["tree"].num_nodes == 6
        )  # root, {0, 1}, {0, 2} , {0, 3} , {0, 4}, {0, 5}

    def test_solver_status_case_3(self):
        """we go down to the next level fo the search tree

        the solving procedure is something like this:

        1. initial state:

        queue = [root]
        tree:
        - root

        2. after yielding {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}

        tree:
        - root
          - 1
          - 2
          - 3
          - 4
          - 5

        queue = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
        last_node = root
        last_rule = rule_5

        3. pick one from queue and search down one level, could be any child of root

        for instance, if we pop (0, 1), we have

        tree:
        - root
          - 1
            - 2
          - 2
          - 3
          - 4
          - 5

        queue = [(0, 2), (0, 3), (0, 4), (0, 5), (0, 1, 2)]
        last_node = (0, 1)
        last_rule = rule_2
        """
        thresh = 6  # {0, 1}, ..., {0, 5}, another arbitrary ruleset
        # computation from scratch
        icbb1 = self.create_icbb()

        A, t = self.vacuum_constraints
        sols = icbb1.bounded_sols(thresh, A=A, t=t)

        ss = icbb1.solver_status
        for sol in sols[:-1]:
            assert len(sol) == 2
        assert (
            len(sols[-1]) == 3
        )  # the last ruleset should contain 3 rules (one of them is the default one)
        # it is hard to check which node is last_node due to the randomness in data generation
        # assert ss["last_node"] == icbb1.tree.root.children[1]
        # assert ss["last_rule"] == icbb1.rules[1]

        # if (0, 5) is popped, another node needs to be popped, leaving 3 in the queue
        # then after searching down, the new yielded node is pushed in the queue
        # given 4 nodes in the queue
        # for the other cases, only one node is popped, leaving 5 nodes in the queue finally
        assert 4 <= ss["queue"].size <= 5
        assert ss["tree"].num_nodes == 7

    @pytest.mark.parametrize(
        "excluded_rules, expected_sols",
        [
            ([1, 2, 3, 4], set()),
            ([1, 2, 3], {(0, 4, 5)}),
            ([1, 2], {(0, 4, 5), (0, 3, 4, 5), (0, 3, 5), (0, 3, 4)}),
        ],
    )
    def test__generate_from_queue_element(self, excluded_rules, expected_sols):
        """check that:
        - nodes inserted in the queue by previous run are indeed filtered out
        - nodes inserted in the queue by current run are not filtered out

        note that:
        we explicitly inject a constraint system with a different number of constraints from the previous run (1 constraint)
        so that the solver can tell we are solving a different problem
        """
        thresh = 5  # {0, 1}, ..., {0, 5}
        # computation from scratch
        icbb1 = self.create_icbb()

        A, t = self.vacuum_constraints
        icbb1.bounded_sols(thresh, A=A, t=t)

        A1, t1 = self.get_A_and_t_that_exclude_rules(excluded_rules)

        icbb2 = self.create_icbb()
        icbb2.reset(A=A1, t=t1, solver_status=icbb1.solver_status)
        sols = list(icbb2._generate_from_queue())

        # the solutions should not contain any ruleset that includes any one of 1, 2, or 3
        sols = set(map(tuple, map(sorted, sols)))
        assert sols == expected_sols

    @pytest.mark.parametrize(
        "excluded_rules, expected_sols",
        [
            ([1, 2, 3, 4], {(0, 5)}),
            ([1, 2, 3], {(0, 4), (0, 5)}),
            ([1, 2], {(0, 3), (0, 4), (0, 5)}),
        ],
    )
    def test__generate_from_feasible_solutions(self, excluded_rules, expected_sols):
        thresh = 5  # yield {0, 1}, ..., {0, 5}
        # computation from scratch
        icbb1 = self.create_icbb()

        A, t = self.vacuum_constraints
        icbb1.bounded_sols(thresh, A=A, t=t)

        A1, t1 = self.get_A_and_t_that_exclude_rules(excluded_rules)

        icbb2 = self.create_icbb()
        icbb2.reset(A=A1, t=t1, solver_status=icbb1.solver_status)
        sols = list(icbb2._generate_from_feasible_solutions())

        sols = set(map(tuple, map(sorted, sols)))
        # none of the previously-found solutions are feasible
        assert sols == expected_sols

        # and the feasible node list is updated accordingly
        assert (
            set(map(lambda n: tuple(n.get_ruleset_ids()), icbb2._feasible_nodes))
            == sols
        )

    def test__generate_from_feasible_solutions_when_not_incremental(self):
        """calling this method in non-incremental mode is not allowed"""
        thresh = 1
        # computation from scratch
        icbb = self.create_icbb()

        A, t = self.vacuum_constraints
        icbb.bounded_sols(thresh, A=A, t=t)

        with pytest.raises(RuntimeError, match="forbidden.*non-incremental mode"):
            list(icbb._generate_from_feasible_solutions())

    @pytest.mark.parametrize(
        "excluded_rules, expected_sols",
        [
            ([1, 2, 3, 4], {(0, 5)}),
            ([1, 2, 3], {(0, 4), (0, 5), (0, 4, 5)}),
            (
                [1, 2],
                {(0, 3), (0, 4), (0, 5), (0, 3, 4), (0, 3, 5), (0, 4, 5), (0, 3, 4, 5)},
            ),
        ],
    )
    def test_generate(self, excluded_rules, expected_sols):
        thresh = 5  # yield {0, 1}, ..., {0, 5}
        # computation from scratch
        icbb1 = self.create_icbb()

        A, t = self.vacuum_constraints
        icbb1.bounded_sols(thresh, A=A, t=t)

        A1, t1 = self.get_A_and_t_that_exclude_rules(excluded_rules)

        icbb2 = self.create_icbb()
        icbb2.reset(A=A1, t=t1, solver_status=icbb1.solver_status)
        sols = list(icbb2.generate())

        sols = set(map(tuple, map(sorted, sols)))
        # none of the previously-found solutions are feasible
        assert sols == expected_sols

        # collected solutions should be consistent with yielded solutions
        collected_feasible_sols = set(
            map(lambda n: tuple(n.get_ruleset_ids()), icbb2._feasible_nodes)
        )
        assert collected_feasible_sols == expected_sols


class TestEnd2End(Utility):
    """end2end functional test"""

    # @pytest.mark.parametrize("target_thresh", randints(3))
    # @pytest.mark.parametrize("seed", randints(3))
    @pytest.mark.parametrize("target_thresh", [10])
    @pytest.mark.parametrize("seed", [123])
    def test_return_objective(self, target_thresh, seed):
        thresh1 = 1  # yield just 1 solution
        # computation from scratch
        icbb1 = self.create_icbb()

        A, t = self.vacuum_constraints
        icbb1.bounded_sols(thresh1, A=A, t=t)

        # create random constraints
        A1, t1 = generate_h_and_alpha(self.num_rules, 2, seed=seed, as_numpy=True)

        print(
            "tree size before icbb2 solving: {}".format(
                icbb1.solver_status["tree"].num_nodes
            )
        )
        # 1. return objective
        icbb2 = self.create_icbb()
        sols_with_obj = icbb2.bounded_sols(
            target_thresh,
            A=A1,
            t=t1,
            solver_status=icbb1.solver_status,
            return_objective=True,
        )
        assert 0 < len(sols_with_obj) <= target_thresh
        for sol, obj in sols_with_obj:
            assert isinstance(sol, set)
            assert isinstance(obj, mpfr)

        # but do not return objective
        icbb3 = self.create_icbb()
        sols = icbb3.bounded_sols(
            target_thresh,
            A=A1,
            t=t1,
            solver_status=icbb1.solver_status,
            return_objective=False,
        )
        assert 0 < len(sols) <= target_thresh
        for sol in sols:
            assert isinstance(sol, set)

    @pytest.mark.parametrize("target_thresh", randints(3))
    @pytest.mark.parametrize("seed", randints(3))
    def test_bounded_count(self, target_thresh, seed):
        thresh1 = 1  # yield just 1 solution
        # computation from scratch
        icbb1 = self.create_icbb()

        A, t = self.vacuum_constraints
        icbb1.bounded_sols(thresh1, A=A, t=t)

        # create random constraints
        A1, t1 = generate_h_and_alpha(self.num_rules, 2, seed=seed, as_numpy=True)

        icbb2 = self.create_icbb()
        count2 = icbb2.bounded_count(
            target_thresh, A=A1, t=t1, solver_status=icbb1.solver_status
        )
        assert 0 < count2 <= target_thresh

    @pytest.mark.parametrize("thresh1", randints(3, vmin=1, vmax=int(2**5 - 1)))
    @pytest.mark.parametrize("thresh2", randints(3, vmin=1, vmax=int(2**5 - 1)))
    @pytest.mark.parametrize(
        "num_constraints", [1, 2, 3, 4]
    )  # at most 4, since there are only 5 rules
    @pytest.mark.parametrize("seed", randints(3))
    def test_equivalence_to_nonincremental_solver(
        self, thresh1, thresh2, num_constraints, seed
    ):
        # the constraint system to be shared
        # create random constraints
        A, t = generate_h_and_alpha(
            self.num_rules, num_constraints, seed=seed, as_numpy=True
        )

        # 1. solve incrementally
        icbb1 = self.create_icbb()

        # take a sub system of Ax=b
        A1, t1 = A[:1, :], t[:1]
        icbb1.bounded_sols(thresh1, A=A1, t=t1)
        assert not icbb1.is_incremental

        # icbb2 solves incrementally based on icbb1
        icbb2 = self.create_icbb()
        sols_icbb = icbb2.bounded_sols(
            thresh2, A=A, t=t, solver_status=icbb1.solver_status
        )
        assert icbb2.is_incremental

        # 2. solve non-incrementally
        cbb = self.create_cbb()
        sols_cbb = cbb.bounded_sols(thresh2, A=A, t=t)

        # and the output should be exactly the same
        assert set(map(tuple, sols_icbb)) == set(map(tuple, sols_cbb))

    @pytest.mark.parametrize("thresh", [1, 2, 5, 10])
    @pytest.mark.parametrize(
        "num_constraints", [1, 2, 3, 4]
    )  # at most 4, since there are only 5 rules
    @pytest.mark.parametrize("seed", randints(3))
    def test_solving_by_multiple_steps(self, thresh, num_constraints, seed):
        """we call incremental solver multiple times with the same thresh
        this simulates how incremental CBB is used e.g., by log_search
        """
        # the random constraint system to be shared
        A, t = generate_h_and_alpha(
            self.num_rules, num_constraints, seed=seed, as_numpy=True
        )

        # initially, solver status is None
        # we solve from scratch
        solver_status = None

        for i in range(1, num_constraints + 1):
            # solve using the first i constraint(s)
            icbb_i = self.create_icbb()
            # take a sub system of Ax=b
            Ai, ti = A[:i, :], t[:i]
            print("Ai: \n{}".format(Ai.astype(int)))
            print("ti: \n{}".format(ti.astype(int)))
            sols_icbb = icbb_i.bounded_sols(
                thresh, A=Ai, t=ti, solver_status=solver_status
            )
            print("sols_icbb: {}".format(sols_icbb))
            # update solver status
            solver_status = icbb_i.solver_status

        # 2. solve non-incrementally
        cbb = self.create_cbb()
        sols_cbb = cbb.bounded_sols(thresh, A=A, t=t)

        # and the output should be exactly the same
        assert set(map(tuple, sols_icbb)) == set(map(tuple, sols_cbb))
