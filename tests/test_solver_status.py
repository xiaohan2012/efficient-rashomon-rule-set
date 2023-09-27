import numpy as np
import pytest

from copy import copy
from bds.solver_status import SolverStatus
from gmpy2 import mpz
from bds.types import RuleSet


class TestSolverStatus:
    def test_reserve_set(self):
        s = SolverStatus()
        s.add_to_reserve_set((0, 1))
        assert s.reserve_set == {(0, 1)}

        s.add_to_reserve_set((0, 1))
        assert s.reserve_set == {(0, 1)}

        s.add_to_reserve_set((0, 2))
        assert s.reserve_set == {(0, 1), (0, 2)}

    def test_solution_set(self):
        s = SolverStatus()
        s.add_to_solution_set((0, 1))
        assert s.solution_set == {(0, 1)}

        s.add_to_solution_set((0, 1))
        assert s.solution_set == {(0, 1)}

        s.add_to_solution_set((0, 2))
        assert s.solution_set == {(0, 1), (0, 2)}

    def test_queue(self):
        s = SolverStatus()
        key = 0
        item = "zero"
        s.push_to_queue(key, item)
        assert s.queue_size() == 1

        key = 1
        item = "one"
        s.push_to_queue(key, item)
        assert s.queue_size() == 2
        assert not s.is_queue_empty()

        assert s.pop_from_queue() == "zero"
        assert s.queue_size() == 1
        assert not s.is_queue_empty()

        assert s.pop_from_queue() == "one"
        assert s.queue_size() == 0
        assert s.is_queue_empty()

    def test_update_last_checked_prefix(self):
        s = SolverStatus()
        assert s.last_checked_prefix is None
        other_data = {"u": mpz(), "s": mpz("0b11")}
        s.update_last_checked_prefix(RuleSet([0, 1]), other_data=other_data)

        assert s.last_checked_prefix == (0, 1)
        assert s.other_data_for_last_checked_prefix == other_data

    def test___eq__(self):
        s = SolverStatus()
        key = 0
        item = "zero"
        s.push_to_queue(key, item)

        s_cp = s.copy()

        assert s is not s_cp
        assert s == s_cp

        s.pop_from_queue()
        assert s is not s_cp
        assert s != s_cp

    def test_reset_reserve_set(self):
        s = SolverStatus()
        s.add_to_reserve_set(RuleSet([0, 1]))
        s.add_to_reserve_set(RuleSet([1, 2]))
        reserve_set = copy(s.reserve_set)

        s.reset_reserve_set()
        assert s.reserve_set == set()
        assert len(reserve_set) == 2  # we can restore the reserve set before

    def test_reset_solution_set(self):
        s = SolverStatus()
        s.add_to_solution_set(RuleSet([0, 1]))
        s.add_to_solution_set(RuleSet([1, 2]))
        solution_set = copy(s.solution_set)

        s.reset_solution_set()
        assert s.solution_set == set()
        assert len(solution_set) == 2        
