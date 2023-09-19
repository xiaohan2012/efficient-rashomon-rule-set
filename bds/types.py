from typing import Set, Iterable


class RuleSet(tuple):
    def __new__(self, rule_ids: Iterable[int]):
        rule_ids = list(sorted(rule_ids))
        return tuple.__new__(RuleSet, rule_ids)


SolutionSet = Set[RuleSet]
