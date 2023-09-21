from bds.solver_status import SolverStatus
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

    def test_d_last(self):
        s = SolverStatus()
        assert s.d_last is None
        s.update_d_last(RuleSet([0, 1]))
        assert s.d_last == (0, 1)

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
