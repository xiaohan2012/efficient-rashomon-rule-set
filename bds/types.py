from typing import Iterable, Set, Union


class RuleSet(tuple):
    def __new__(self, rule_ids: Iterable[int]):
        rule_ids = list(sorted(rule_ids))
        return tuple.__new__(RuleSet, rule_ids)

    def __sub__(self, other: Union[Set[int], "RuleSet"]) -> "RuleSet":
        """remove ids in other from self"""
        return RuleSet(tuple(set(self) - set(other)))

    def __add__(self, other: Union[Set[int], "RuleSet"]) -> "RuleSet":
        """add ids in other to self"""
        return RuleSet(tuple(set(self) | set(other)))


SolutionSet = Set[RuleSet]


class ParityConstraintViolation(Exception):
    """an exception class meaning a partiy constraint system Ax=b must be violated given a partial assignment of x"""
