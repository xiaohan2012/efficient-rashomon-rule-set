from bds.types import RuleSet


class TestRule:
    def test_1(self):
        rs = RuleSet([0, 1, 2])
        assert rs == (0, 1, 2)

    def test_2(self):
        # it is sorted
        rs = RuleSet([2, 1, 0])
        assert rs == (0, 1, 2)
