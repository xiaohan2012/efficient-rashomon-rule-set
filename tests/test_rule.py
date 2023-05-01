import pytest
import numpy as np

from bds.rule import Rule, RuleEntry, RuleSet
from bds.utils import randints, bin_zeros


class TestRule:
    def test_init(self):
        n_samples = 200

        id = 1
        name = "a rule"
        card = 100
        supp = 5
        ids = np.sort(np.random.permutation(n_samples)[:supp])
        truthtable = bin_zeros(n_samples)
        truthtable[ids] = 1
        r = Rule(id, name, card, truthtable)

        assert r.id == id
        assert r.cardinality == card
        assert r.name == name
        assert r.support == supp
        np.testing.assert_allclose(r.ids, ids)  # r.ids calculated in __post_init__
        np.testing.assert_allclose(truthtable, truthtable)

    @pytest.mark.parametrize("random_seed", randints(5))
    def test_random(self, random_seed):
        num_pts = 100
        r = Rule.random(1, num_pts, random_seed=random_seed)
        r_copy = Rule.random(1, num_pts, random_seed=random_seed)
        assert r == r_copy


class TestRuleEntry:
    def test_init(self):
        n_samples = 200
        rule_id = 0
        n_captured = 10

        ids = np.random.permutation(n_samples)[:n_captured]
        captured = np.zeros(n_samples)
        captured[ids] = 1

        entry = RuleEntry(rule_id, n_captured, captured)

        assert entry.rule_id == rule_id
        assert entry.n_captured == n_captured
        np.testing.assert_allclose(entry.captured, captured)


class TestRuleSet:
    def random_rule_entry(self, rule_id=0):
        n_samples = 200
        n_captured = 10

        ids = np.random.permutation(n_samples)[:n_captured]
        captured = np.zeros(n_samples)
        captured[ids] = 1

        return RuleEntry(rule_id, n_captured, captured)

    def test_init(self):
        n_rules = 5
        rs = RuleSet([self.random_rule_entry(i) for i in range(n_rules)])
        assert rs.n_rules == n_rules

        for entry in rs:
            assert isinstance(entry, RuleEntry)
