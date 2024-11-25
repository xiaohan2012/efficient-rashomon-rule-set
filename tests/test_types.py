import pytest

from bds.types import RuleSet


class TestRule:
    def test_1(self):
        rs = RuleSet([0, 1, 2])
        assert rs == (0, 1, 2)

    def test_2(self):
        # it is sorted
        rs = RuleSet([2, 1, 0])
        assert rs == (0, 1, 2)

    @pytest.mark.parametrize(
        "left, right, expected",
        [
            ((2, 1, 0), (0, 1, 2), tuple()),
            ((2, 1, 0), (1, 2, 3, 4, 5), (0,)),
            ((2, 1, 0), tuple(), (0, 1, 2)),
            ((2, 1, 0), (1,), (0, 2)),
        ],
    )
    def test___sub__(self, left, right, expected):
        assert (RuleSet(left) - RuleSet(right)) == RuleSet(expected)

    @pytest.mark.parametrize(
        "left, right, expected",
        [
            ((2, 1, 0), (0, 1, 2), (0, 1, 2)),
            ((2, 1, 0), (1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)),
            ((2, 1, 0), tuple(), (0, 1, 2)),
            ((2, 1, 0), (1,), (0, 1, 2)),
        ],
    )
    def test___add__(self, left, right, expected):
        assert (RuleSet(left) + RuleSet(right)) == RuleSet(expected)
