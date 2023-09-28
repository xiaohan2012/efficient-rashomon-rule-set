import numpy as np
import pytest
from gmpy2 import mpfr

from bds.cbb import ConstrainedBranchAndBound
from bds.icbb import IncrementalConstrainedBranchAndBound
from bds.random_hash import generate_h_and_alpha
from bds.utils import bin_array, bin_zeros, randints

from typing import List, Tuple
from .utils import generate_random_rules_and_y


class Utility:
    """a utility class that provides data loading, solver creation support, etc"""

    @property
    def num_rules(self):
        return 10

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

    def create_A_and_b(self, rand_seed=42):
        return generate_h_and_alpha(
            self.num_rules, self.num_rules - 1, seed=rand_seed, as_numpy=True
        )


class TestExamineRAndS(Utility):
    @pytest.mark.parametrize(
        'num_constraints', [2, 4, 6]
    )
    @pytest.mark.parametrize(
        'ub', [0.21, 0.51, float("inf")]
    )
    @pytest.mark.parametrize(
        'rand_seed', randints(3)
    )
    # @pytest.mark.parametrize("num_constraints, ub, rand_seed", [(2, 0.21001, 1539280901)])
    def test_consistency_under_the_same_constraint_system(
        self, num_constraints, ub, rand_seed
    ):
        """
        icbb._examine_R_andS should yield the same set of solutions as cbb, if they are subject to the same Ax=b
        """
        cbb = self.create_cbb(ub=ub, rand_seed=rand_seed)
        A_full, b_full = self.create_A_and_b(rand_seed)
        A, b = A_full[:num_constraints], b_full[:num_constraints]
        expected_sols = cbb.bounded_sols(10, A=A, b=b)
        expected_S = cbb.status.solution_set
        expected_R = cbb.status.reserve_set

        icbb = self.create_icbb(ub=ub, rand_seed=rand_seed)
        icbb.reset(A=A, b=b, solver_status=cbb.status)

        actual_sols = list(icbb._examine_R_and_S())
        actual_S = icbb.status.solution_set
        actual_R = icbb.status.reserve_set

        assert len(expected_sols) == len(actual_sols)
        assert set(expected_sols) == set(actual_sols)
        assert expected_S == actual_S
        assert expected_R == actual_R


class TestSimple:
    def test(self):
        num_pts, num_rules = 20, 10
        rand_rules, rand_y = generate_random_rules_and_y(
            num_pts, num_rules, rand_seed=42
        )
        A, b = generate_h_and_alpha(num_rules, num_rules - 1, seed=32, as_numpy=True)

        i = 2
        j = 5

        Ai, bi = A[:i], b[:i]
        Aj, bj = A[:j], b[:j]

        # solving
        cbb_i = ConstrainedBranchAndBound(
            rand_rules, float("inf"), rand_y, lmbd=0.1, reorder_columns=False
        )
        cbb_i.setup(A=Ai, b=bi, solver_status=None)
        cbb_i.bounded_sols(threshold=10)

        cbb_j = ConstrainedBranchAndBound(
            rand_rules, float("inf"), rand_y, lmbd=0.1, reorder_columns=False
        )
        cbb_j.setup(A=Aj, b=bj, solver_status=cbb_i.solver_status)
        actual_sols = cbb_j.bounded_sols(threshold=10)

        # expected results are calculated from the non-incremental CBB
        cbb_ref = ConstrainedBranchAndBound(
            rand_rules, float("inf"), rand_y, lmbd=0.1, reorder_columns=False
        )
        cbb_ref.setup(A=Aj, b=bj, solver_status=None)
        expected_sols = cbb_ref.bounded_sols(threshold=10)

        assert actual_sols == expected_sols


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

    @pytest.mark.parametrize("target_thresh", randints(3, vmin=1))
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
