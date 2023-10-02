import numpy as np
import pytest
from gmpy2 import mpfr, mpz

from bds.cbb import ConstrainedBranchAndBound
from bds.icbb import IncrementalConstrainedBranchAndBound
from bds.random_hash import generate_h_and_alpha
from bds.utils import bin_array, bin_zeros, randints
from bds.gf2 import extended_rref

from typing import List, Tuple
from .utils import generate_random_rules_and_y


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

    def generate_input(self, rand_seed=12345):
        return generate_random_rules_and_y(10, self.num_rules, rand_seed=rand_seed)

    def create_icbb(self, ub=float("inf"), reorder_columns=False, rand_seed=42):
        """create incremental solver"""
        rand_rules, rand_y = self.generate_input(rand_seed)
        return IncrementalConstrainedBranchAndBound(
            rand_rules, ub, rand_y, self.lmbd, reorder_columns=reorder_columns
        )

    def create_cbb(self, ub=float("inf"), reorder_columns=False, rand_seed=42):
        """create non-incremental solver"""
        rand_rules, rand_y = self.generate_input(rand_seed)
        return ConstrainedBranchAndBound(
            rand_rules, ub, rand_y, self.lmbd, reorder_columns=reorder_columns
        )

    def create_A_and_b(self, rand_seed=42, do_rref=True):
        A, b = generate_h_and_alpha(
            self.num_rules, self.num_rules - 1, seed=rand_seed, as_numpy=True
        )
        # remark: it is important to perform rref before calling ICBB so that
        # the the constraints of Aix=bi is always a subset of Ajx=bj, where i <= j
        A_rref, b_rref = extended_rref(A, b, verbose=False)[:2]
        return bin_array(A_rref), bin_array(b_rref)

    def unpack_sols_and_objs(self, sols_with_obj):
        if len(sols_with_obj) == 0:
            return [], np.array([], dtype=float)
        else:
            sols, objs = zip(*sols_with_obj)
            return list(sols), np.array(objs, dtype=float)


class TestNonIncremental(Utility):
    @pytest.mark.parametrize("threshold", [5, 10, None])
    @pytest.mark.parametrize("num_constraints", [2, 4, 6])
    @pytest.mark.parametrize("ub", [0.21, 0.51, float("inf")])
    @pytest.mark.parametrize("rand_seed", randints(3))
    def test(self, threshold, num_constraints, ub, rand_seed):
        """calling cbb and icbb from scratch should yield the same set of solutions and objectives"""
        cbb = self.create_cbb(ub=ub, rand_seed=rand_seed)
        A_full, b_full = self.create_A_and_b(rand_seed)
        A, b = A_full[:num_constraints], b_full[:num_constraints]
        expected_sols_with_obj = cbb.bounded_sols(
            threshold, A=A, b=b, return_objective=True
        )
        expected_sols, expected_objs = self.unpack_sols_and_objs(expected_sols_with_obj)
        expected_S = cbb.status.solution_set
        expected_R = cbb.status.reserve_set

        icbb = self.create_icbb(ub=ub, rand_seed=rand_seed)
        icbb.reset(A=A, b=b)

        actual_sols_with_obj = icbb.bounded_sols(
            threshold, A=A, b=b, return_objective=True
        )
        actual_sols, actual_objs = self.unpack_sols_and_objs(actual_sols_with_obj)

        actual_S = icbb.status.solution_set
        actual_R = icbb.status.reserve_set

        assert len(expected_sols) == len(actual_sols)
        assert set(expected_sols) == set(actual_sols)
        np.testing.assert_allclose(np.sort(expected_objs), np.sort(actual_objs))
        assert expected_S == actual_S
        assert expected_R == actual_R


class TestExamineRAndS(Utility):
    @pytest.mark.parametrize("num_constraints", [2, 4, 6])
    @pytest.mark.parametrize("ub", [0.21, 0.51, float("inf")])
    @pytest.mark.parametrize("rand_seed", randints(3))
    # @pytest.mark.parametrize("num_constraints, ub, rand_seed", [(2, 0.21001, 1539280901)])
    def test_consistency_under_the_same_constraint_system(
        self, num_constraints, ub, rand_seed
    ):
        """
        icbb._examine_R_and_S should yield the same set of solutions as cbb, if they are subject to the same Ax=b
        """
        cbb = self.create_cbb(ub=ub, rand_seed=rand_seed)
        A_full, b_full = self.create_A_and_b(rand_seed)
        A, b = A_full[:num_constraints], b_full[:num_constraints]
        expected_sols_with_obj = cbb.bounded_sols(10, A=A, b=b, return_objective=True)
        expected_sols, expected_objs = self.unpack_sols_and_objs(expected_sols_with_obj)
        expected_S = cbb.status.solution_set
        expected_R = cbb.status.reserve_set

        icbb = self.create_icbb(ub=ub, rand_seed=rand_seed)
        icbb.reset(A=A, b=b, solver_status=cbb.status)

        actual_sols_with_obj = list(icbb._examine_R_and_S(return_objective=True))
        actual_sols, actual_objs = self.unpack_sols_and_objs(actual_sols_with_obj)

        actual_S = icbb.status.solution_set
        actual_R = icbb.status.reserve_set

        assert len(expected_sols) == len(actual_sols)
        assert set(expected_sols) == set(actual_sols)
        np.testing.assert_allclose(np.sort(expected_objs), np.sort(actual_objs))
        assert expected_S == actual_S
        assert expected_R == actual_R

        # check that the prefix is indeed feasible
        for prefix in actual_sols:
            assert icbb.is_prefix_feasible(prefix)

    @pytest.mark.parametrize("num_constraints", [2, 4, 6])
    @pytest.mark.parametrize("diff_in_num_constraints", randints(1, 1, 4))
    @pytest.mark.parametrize("threshold", [5, 10, 15])
    @pytest.mark.parametrize("ub", [0.21, 0.51, float("inf")])
    @pytest.mark.parametrize("rand_seed", randints(3))
    def test_under_the_different_constraint_systems(
        self,
        num_constraints,
        diff_in_num_constraints,
        threshold,
        ub,
        rand_seed
        # num_constraints = 2, diff_in_num_constraints = 3, threshold = 5, ub = 0.21, rand_seed = 360269222
    ):
        """
        now the constraint system changes, the non-incremental version uses 1 fewer constraint than the incremental version
        we should expect that the generated solutions by ICBB is a subset of that by CBB

        we do not check the objective calculation here because it is tested previously and it is not very easy to test in this case
        """
        cbb = self.create_cbb(ub=ub, rand_seed=rand_seed)
        A_full_rref, b_full_rref = self.create_A_and_b(rand_seed)

        i = num_constraints
        j = num_constraints + diff_in_num_constraints
        Ai, bi = A_full_rref[:i], b_full_rref[:i]
        Aj, bj = A_full_rref[:j], b_full_rref[:j]

        expected_sols_with_obj = cbb.bounded_sols(
            threshold, A=Ai, b=bi, return_objective=True
        )
        expected_sols, _ = self.unpack_sols_and_objs(expected_sols_with_obj)
        expected_S = cbb.status.solution_set
        expected_R = cbb.status.reserve_set

        cbb.print_Axb()

        icbb = self.create_icbb(ub=ub, rand_seed=rand_seed)
        icbb.reset(A=Aj, b=bj, solver_status=cbb.status)

        icbb.print_Axb()

        actual_sols_with_obj = list(icbb._examine_R_and_S(return_objective=True))
        actual_sols, _ = self.unpack_sols_and_objs(actual_sols_with_obj)

        actual_S = icbb.status.solution_set
        actual_R = icbb.status.reserve_set

        assert len(expected_sols) >= len(actual_sols)
        assert len(set(expected_sols)) >= len(set(actual_sols))
        # assert set(expected_sols).issuperset(set(actual_sols))  # this line may not hold because the new constraint system may add new pivots
        assert len(expected_S) >= len(actual_S)
        assert len(expected_R) >= len(actual_R)

        # check that the prefix is indeed feasible
        for prefix in actual_sols:
            assert icbb.is_prefix_feasible(prefix)


class TestUpdateQueue(Utility):
    @pytest.mark.parametrize("num_constraints", [2, 4, 6])
    @pytest.mark.parametrize("ub", [0.21, 0.51, float("inf")])
    @pytest.mark.parametrize("rand_seed", randints(3))
    @pytest.mark.parametrize("threshold", [5, 10, 20])
    # @pytest.mark.parametrize("num_constraints, ub, rand_seed", [(2, 0.51001, 1534479381)])
    def test_consistency_under_the_same_constraint_system(
        self, num_constraints, ub, rand_seed, threshold
    ):
        """
        icbb._update_queue should yield the same queue as cbb, if they are subject to the same Ax=b
        """
        cbb = self.create_cbb(ub=ub, rand_seed=rand_seed)
        A_full, b_full = self.create_A_and_b(rand_seed)
        A, b = A_full[:num_constraints], b_full[:num_constraints]
        cbb.bounded_sols(threshold, A=A, b=b)

        icbb = self.create_icbb(ub=ub, rand_seed=rand_seed)
        icbb.reset(A=A, b=b, solver_status=cbb.status)

        icbb._update_queue()

        cbb._push_last_checked_prefix_to_queue()  # to be consistent with icbb

        # they are equal in the queue items
        assert cbb.status.queue == cbb.status.queue
        assert (
            cbb.status.queue is not icbb.status.queue
        )  # but they do not point to the same object

    @pytest.mark.parametrize("num_constraints", [2, 4, 6])
    @pytest.mark.parametrize("diff_in_num_constraints", randints(3, 1, 4))
    @pytest.mark.parametrize("ub", [0.21, 0.51, float("inf")])
    @pytest.mark.parametrize("rand_seed", randints(3))
    @pytest.mark.parametrize("threshold", [5, 10, 20])
    def test_under_the_different_constraint_systems(
        self,
        # num_constraints = 2, diff_in_num_constraints = 2, ub = 0.21, rand_seed = 39462156, threshold = 5
        num_constraints,
        diff_in_num_constraints,
        ub,
        rand_seed,
        threshold,
    ):
        """
        now the constraint system changes, the non-incremental version uses 1 fewer constraint than the incremental version
        we should expect that the number of queue items in ICBB  is smaller than CBB

        note that the queue items in ICBB is not necessarily a subset of CBB, because the new constraint system may add new pivot rules
        """
        cbb = self.create_cbb(ub=ub, rand_seed=rand_seed)
        A_full_rref, b_full_rref = self.create_A_and_b(rand_seed)

        i = num_constraints
        j = num_constraints + diff_in_num_constraints
        Ai, bi = A_full_rref[:i], b_full_rref[:i]
        Aj, bj = A_full_rref[:j], b_full_rref[:j]

        cbb.bounded_sols(threshold, A=Ai, b=bi)
        cbb._push_last_checked_prefix_to_queue()  # to be consistent with icbb

        icbb = self.create_icbb(ub=ub, rand_seed=rand_seed)
        icbb.reset(A=Aj, b=bj, solver_status=cbb.status)

        icbb._update_queue()

        cbb.print_Axb()
        icbb.print_Axb()

        prefixes_in_queue_cbb = {el[0] for el in cbb.status.queue}
        prefixes_in_queue_icbb = {el[0] for el in icbb.status.queue}

        assert len(prefixes_in_queue_icbb) <= len(prefixes_in_queue_cbb)
        # the following assertion is wrong
        # assert prefixes_in_queue_cbb.issuperset(prefixes_in_queue_icbb), prefixes_in_queue_cbb - prefixes_in_queue_icbb
        for prefix in prefixes_in_queue_icbb:
            assert icbb.is_preflix_qualified_for_queue(prefix), prefix


class TestEquivalenceToNonIncremental(Utility):
    @pytest.mark.parametrize("i", [2, 4, 6])
    @pytest.mark.parametrize("delta_i_j", randints(3, 1, 4))
    @pytest.mark.parametrize("ub", [0.21, 0.51, float("inf")])
    @pytest.mark.parametrize("rand_seed", randints(3))
    @pytest.mark.parametrize("threshold", [5, 10, 20])
    def test_two_step_solving(
        self,
        i,
        delta_i_j,
        ub,
        rand_seed,
        threshold
        # i=2,
        # delta_i_j=1,
        # ub=float("inf"),
        # rand_seed=1761324163,
        # threshold=5,
    ):
        A, b = self.create_A_and_b(rand_seed)
        j = i + delta_i_j

        Ai, bi = A[:i], b[:i]
        Aj, bj = A[:j], b[:j]

        # solving from scratch with i constraints
        cbb_i = self.create_icbb(ub, rand_seed=rand_seed)
        cbb_i.bounded_sols(threshold=threshold, A=Ai, b=bi, solver_status=None)

        # solve the problem with j constraints incrementally based on cbb_i
        cbb_j = self.create_icbb(ub, rand_seed=rand_seed)
        actual_sols_with_obj = cbb_j.bounded_sols(
            threshold, A=Aj, b=bj, solver_status=cbb_i.status, return_objective=True
        )
        actual_sols, actual_objs = self.unpack_sols_and_objs(actual_sols_with_obj)
        cbb_i.print_Axb()
        cbb_j.print_Axb()

        # expected results are calculated from the non-incremental CBB
        cbb_ref = self.create_cbb(ub, rand_seed=rand_seed)
        expected_sols_with_obj = cbb_ref.bounded_sols(
            threshold, A=Aj, b=bj, solver_status=None, return_objective=True
        )
        expected_sols, expected_objs = self.unpack_sols_and_objs(expected_sols_with_obj)
        assert set(actual_sols) == set(expected_sols)
        assert len(actual_sols) == len(expected_sols)
        assert set(actual_sols) == cbb_j.status.solution_set
        np.testing.assert_allclose(np.sort(expected_objs), np.sort(actual_objs))
        assert cbb_ref.status.reserve_set == cbb_j.status.reserve_set

    @pytest.mark.parametrize("init_i", randints(3, 1, 5))
    @pytest.mark.parametrize("delta_i_j", randints(3, 2, 6))
    @pytest.mark.parametrize("ub", [0.21, 0.51, float("inf")])
    @pytest.mark.parametrize("rand_seed", randints(3))
    @pytest.mark.parametrize("threshold", [5, 10, 20])
    def test_k_step_solving(
        self,
        init_i,
        delta_i_j,
        ub,
        rand_seed,
        threshold
        # i = 2, delta_i_j = 3, ub = 0.51, rand_seed = 1734416845, threshold = 5
    ):
        """the general case where icbb is invoked k times

        we consider the setting of starting with i constraints, and ending up with j constraints,

        each invocation of ICBB increments i until j is reached

        we expect that ICBB under Ajx = bj gives the same set of solutions as CBB under Ajx = bj
        """
        A, b = self.create_A_and_b(rand_seed)
        j = init_i + delta_i_j

        solver_status = None
        for i in range(init_i, j + 1):
            Ai, bi = A[:i], b[:i]
            icbb = self.create_icbb(ub, rand_seed=rand_seed)
            actual_sols_with_obj = icbb.bounded_sols(
                threshold=threshold,
                A=Ai,
                b=bi,
                solver_status=solver_status,
                return_objective=True,
            )
            actual_sols, actual_objs = self.unpack_sols_and_objs(actual_sols_with_obj)
            solver_status = icbb.status

        # cbb_i.print_Axb()
        # cbb_j.print_Axb()

        # expected results are calculated from the non-incremental CBB
        Aj, bj = A[:j], b[:j]
        cbb = self.create_cbb(ub, rand_seed=rand_seed)
        expected_sols_with_obj = cbb.bounded_sols(
            threshold, A=Aj, b=bj, solver_status=None, return_objective=True
        )
        expected_sols, expected_objs = self.unpack_sols_and_objs(expected_sols_with_obj)

        assert set(actual_sols) == set(expected_sols)
        assert len(actual_sols) == len(expected_sols)
        assert set(expected_sols) == icbb.status.solution_set
        assert cbb.status.reserve_set == icbb.status.reserve_set
        np.testing.assert_allclose(np.sort(expected_objs), np.sort(actual_objs))


# class TestEnd2End(Utility):
#     """end2end functional test"""

#     # @pytest.mark.parametrize("target_thresh", randints(3))
#     # @pytest.mark.parametrize("seed", randints(3))
#     @pytest.mark.parametrize("target_thresh", [10])
#     @pytest.mark.parametrize("seed", [123])
#     def test_return_objective(self, target_thresh, seed):
#         thresh1 = 1  # yield just 1 solution
#         # computation from scratch
#         icbb1 = self.create_icbb()

#         A, t = self.vacuum_constraints
#         icbb1.bounded_sols(thresh1, A=A, t=t)

#         # create random constraints
#         A1, t1 = generate_h_and_alpha(self.num_rules, 2, seed=seed, as_numpy=True)

#         print(
#             "tree size before icbb2 solving: {}".format(
#                 icbb1.solver_status["tree"].num_nodes
#             )
#         )
#         # 1. return objective
#         icbb2 = self.create_icbb()
#         sols_with_obj = icbb2.bounded_sols(
#             target_thresh,
#             A=A1,
#             t=t1,
#             solver_status=icbb1.solver_status,
#             return_objective=True,
#         )
#         assert 0 < len(sols_with_obj) <= target_thresh
#         for sol, obj in sols_with_obj:
#             assert isinstance(sol, set)
#             assert isinstance(obj, mpfr)

#         # but do not return objective
#         icbb3 = self.create_icbb()
#         sols = icbb3.bounded_sols(
#             target_thresh,
#             A=A1,
#             t=t1,
#             solver_status=icbb1.solver_status,
#             return_objective=False,
#         )
#         assert 0 < len(sols) <= target_thresh
#         for sol in sols:
#             assert isinstance(sol, set)

#     @pytest.mark.parametrize("target_thresh", randints(3, vmin=1))
#     @pytest.mark.parametrize("seed", randints(3))
#     def test_bounded_count(self, target_thresh, seed):
#         thresh1 = 1  # yield just 1 solution
#         # computation from scratch
#         icbb1 = self.create_icbb()

#         A, t = self.vacuum_constraints
#         icbb1.bounded_sols(thresh1, A=A, t=t)

#         # create random constraints
#         A1, t1 = generate_h_and_alpha(self.num_rules, 2, seed=seed, as_numpy=True)

#         icbb2 = self.create_icbb()
#         count2 = icbb2.bounded_count(
#             target_thresh, A=A1, t=t1, solver_status=icbb1.solver_status
#         )
#         assert 0 < count2 <= target_thresh

#     @pytest.mark.parametrize("thresh1", randints(3, vmin=1, vmax=int(2**5 - 1)))
#     @pytest.mark.parametrize("thresh2", randints(3, vmin=1, vmax=int(2**5 - 1)))
#     @pytest.mark.parametrize(
#         "num_constraints", [1, 2, 3, 4]
#     )  # at most 4, since there are only 5 rules
#     @pytest.mark.parametrize("seed", randints(3))
#     def test_equivalence_to_nonincremental_solver(
#         self, thresh1, thresh2, num_constraints, seed
#     ):
#         # the constraint system to be shared
#         # create random constraints
#         A, t = generate_h_and_alpha(
#             self.num_rules, num_constraints, seed=seed, as_numpy=True
#         )

#         # 1. solve incrementally
#         icbb1 = self.create_icbb()

#         # take a sub system of Ax=b
#         A1, t1 = A[:1, :], t[:1]
#         icbb1.bounded_sols(thresh1, A=A1, t=t1)
#         assert not icbb1.is_incremental

#         # icbb2 solves incrementally based on icbb1
#         icbb2 = self.create_icbb()
#         sols_icbb = icbb2.bounded_sols(
#             thresh2, A=A, t=t, solver_status=icbb1.solver_status
#         )
#         assert icbb2.is_incremental

#         # 2. solve non-incrementally
#         cbb = self.create_cbb()
#         sols_cbb = cbb.bounded_sols(thresh2, A=A, t=t)

#         # and the output should be exactly the same
#         assert set(map(tuple, sols_icbb)) == set(map(tuple, sols_cbb))

#     @pytest.mark.parametrize("thresh", [1, 2, 5, 10])
#     @pytest.mark.parametrize(
#         "num_constraints", [1, 2, 3, 4]
#     )  # at most 4, since there are only 5 rules
# @pytest.mark.parametrize("seed", randints(3))
# def test_solving_by_multiple_steps(self, thresh, num_constraints, seed):
#     """we call incremental solver multiple times with the same thresh
#     this simulates how incremental CBB is used e.g., by log_search
#     """
#     # the random constraint system to be shared
#     A, t = generate_h_and_alpha(
#         self.num_rules, num_constraints, seed=seed, as_numpy=True
#     )

#     # initially, solver status is None
#     # we solve from scratch
#     solver_status = None

#     for i in range(1, num_constraints + 1):
#         # solve using the first i constraint(s)
#         icbb_i = self.create_icbb()
#         # take a sub system of Ax=b
#         Ai, ti = A[:i, :], t[:i]
#         print("Ai: \n{}".format(Ai.astype(int)))
#         print("ti: \n{}".format(ti.astype(int)))
#         sols_icbb = icbb_i.bounded_sols(
#             thresh, A=Ai, t=ti, solver_status=solver_status
#         )
#         print("sols_icbb: {}".format(sols_icbb))
#         # update solver status
#         solver_status = icbb_i.solver_status

#     # 2. solve non-incrementally
#     cbb = self.create_cbb()
#     sols_cbb = cbb.bounded_sols(thresh, A=A, t=t)

#     # and the output should be exactly the same
#     assert set(map(tuple, sols_icbb)) == set(map(tuple, sols_cbb))
