import pytest
import numpy as np

from bds.rule import Rule, RuleEntry, RuleSet
from bds.utils import randints


class TestRule:
    def test_init(self):
        n_samples = 200

        id = 1
        name = "a rule"
        card = 10
        supp = 100
        ids = np.random.permutation(n_samples)[:supp]
        truthtable = np.zeros(n_samples)
        truthtable[ids] = 1
        r = Rule(id, name, card, ids, truthtable)

        assert r.id == id
        assert r.cardinality == card
        assert r.name == name
        assert r.support == supp
        np.testing.assert_allclose(r.ids, ids)
        np.testing.assert_allclose(truthtable, truthtable)


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
