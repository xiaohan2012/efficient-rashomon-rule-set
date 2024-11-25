import numpy as np
import pytest
from gmpy2 import mpz

from bds.rule import Rule, lor_of_truthtable
from bds.utils import mpz_set_bits, randints


class TestRule:
    def test_init(self):
        n_samples = 200

        id = 1
        name = "a rule"
        card = 100
        supp = 5
        ids = np.sort(np.random.permutation(n_samples)[:supp])

        truthtable = mpz_set_bits(mpz(), ids)

        r = Rule(id, name, card, truthtable)

        assert r.id == id
        assert r.cardinality == card
        assert r.name == name
        assert r.support == supp
        assert r.truthtable == truthtable

    @pytest.mark.parametrize("random_seed", randints(5))
    def test_random(self, random_seed):
        num_pts = 100
        r = Rule.random(1, num_pts, random_seed=random_seed)
        r_copy = Rule.random(1, num_pts, random_seed=random_seed)
        assert r == r_copy


def make_rule(idx, truthtable: mpz):
    return Rule(idx, "", 1, truthtable)


class TestLOROfTruthtable:
    @pytest.mark.parametrize(
        "rules, expected",
        [
            ([], mpz()),
            ([make_rule(1, mpz("0b001010"))], mpz("0b001010")),
            (
                [make_rule(1, mpz("0b001010")), make_rule(2, mpz("0b001101"))],
                mpz("0b001111"),
            ),
        ],
    )
    def test(self, rules, expected):
        assert lor_of_truthtable(rules) == expected


# class TestRuleEntry:
#     def test_init(self):
#         n_samples = 200
#         rule_id = 0
#         n_captured = 10

#         ids = np.random.permutation(n_samples)[:n_captured]
#         captured = np.zeros(n_samples)
#         captured[ids] = 1

#         entry = RuleEntry(rule_id, n_captured, captured)

#         assert entry.rule_id == rule_id
#         assert entry.n_captured == n_captured
#         np.testing.assert_allclose(entry.captured, captured)


# class TestRuleSet:
#     def random_rule_entry(self, rule_id=0):
#         n_samples = 200
#         n_captured = 10

#         ids = np.random.permutation(n_samples)[:n_captured]
#         captured = np.zeros(n_samples)
#         captured[ids] = 1

#         return RuleEntry(rule_id, n_captured, captured)

#     def test_init(self):
#         n_rules = 5
#         rs = RuleSet([self.random_rule_entry(i) for i in range(n_rules)])
#         assert rs.n_rules == n_rules

#         for entry in rs:
#             assert isinstance(entry, RuleEntry)
