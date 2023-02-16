import pandas as pd
import numpy as np
from typing import Tuple, Union, List

from sklearn.base import BaseEstimator
from tqdm import tqdm

from aix360.algorithms.rule_induction.trxf.core.dnf_ruleset import (
    DnfRuleSet,
    Conjunction,
)
from aix360.algorithms.rule_induction.trxf.core.predicate import Predicate, Feature
from aix360.algorithms.rule_induction.rbm.utils import OPERATOR_MAPS
from aix360.algorithms.rule_induction.trxf.core.utils import batch_evaluate

from .common import PatternSet


def construct_dnf_rule(
    patterns: List[PatternSet], column_names: Union[pd.MultiIndex, pd.Index]
) -> DnfRuleSet:
    """given one or more patterns, each representing a conjunction of predicates, construct a DNF ruleset"""
    conj = []
    for pattern in patterns:
        pred_list = []
        for bf_id in pattern:
            feat, op, val = column_names[bf_id]
            pred_list.append(Predicate(Feature(feat), OPERATOR_MAPS[op], val))
        conj.append(Conjunction(predicate_list=pred_list))
    return DnfRuleSet(conjunctions=conj, then_part=1)


class DNFRuleSetClassifier(BaseEstimator):
    """a DNF ruleset-based classifier

    example of constructing a classifier and doing prediction::

        from sklearn.metrics import accuracy_score

        # X_df_fb is a binarized feature matrix
        # cb is the callback of a CP solver
        patterns = cb.solutions_found[0]
        ruleset = construct_dnf_rule(patterns, X_df_fb.columns)
        print(str(ruleset))

        clf = DNFRuleSetClassifier(dnf_ruleset=ruleset)
        pred = clf.predict(X_df)

        # evaluate
        accuracy_score(y, pred)
    """

    def __init__(self, dnf_ruleset: DnfRuleSet):
        self.dnf_ruleset = dnf_ruleset

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X: pd.DataFrame):
        """X: the original feature matrix (not discretized yet)"""
        return batch_evaluate(self.dnf_ruleset, X)


class EnsembleDNFClassifier(BaseEstimator):
    """an ensemble of multiple DNFRuleSetClassifier"""

    def __init__(self, base_estimators: List[DNFRuleSetClassifier]):
        assert len(base_estimators) > 0, 'at least one base estimator should be given'
        self.base_estimators = base_estimators

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X: pd.DataFrame, show_progress: bool = True) -> np.ndarray:
        outputs = []
        iter_obj = self.base_estimators
        if show_progress:
            iter_obj = tqdm(iter_obj)
        for est in iter_obj:
            outputs.append(est.predict(X).to_numpy()[:, None])
        return np.hstack(outputs).mean(axis=1)
