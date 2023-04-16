import pandas as pd
import numpy as np

from typing import Set, List, Dict, Tuple
from itertools import combinations

from tqdm import tqdm
from mlxtend.frequent_patterns import fpgrowth
from logzero import logger

from ..common import Pattern, SupportSet, PatternSet


def support_set(itemset: Set[int], column_names: pd.MultiIndex, dataset: pd.DataFrame):
    """
    computes the support set of an itemset (a set of integers) in a dataset
    """
    columns = column_names[[i for i in itemset]]
    match_flag = dataset[columns].sum(axis=1) == len(itemset)
    return set(match_flag.to_numpy().nonzero()[0])


def support(itemset: Set[int], column_names: pd.MultiIndex, dataset: pd.DataFrame):
    """computes the support of an itemset (a set of integers) in a dataset

    TODO: check if this procedure can be done in batch using matrix operations
    """
    return len(support_set(itemset, column_names, dataset))


def extract_feature_group_info(column_names: pd.MultiIndex) -> List[Dict]:
    """
    group the discretized features by the original feature name and comparison operator

    output the feature group information, where each group contains the indices of the discretized features and the operator name (e.g., >= or <)

    an example:

    column_names = MultiIndex([
            ('x1', '<=',   -0.9386006984208207),
            ('x1', '<=',   -0.5569271816330525),
            ('x1', '<=',  -0.23005535230376395),
            ...
            ('x1',  '>',   -0.9386006984208207),
            ...
            ('x2', '<=',   -0.5438620708814287),
            ('x2', '<=',    -0.281728855074761),
            ...
            ('x2',  '>',   -0.5438620708814287),
            ('x2',  '>',    -0.281728855074761),
            ...
            ],
           names=['feature', 'operation', 'value'])

    extract_feature_group_info(column_names) # which gives

    [{'feature_indices': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'operator': '<='},
     {'feature_indices': (9, 10, 11, 12, 13, 14, 15, 16, 17), 'operator': '>'},
     {'feature_indices': (18, 19, 20, 21, 22, 23, 24, 25, 26), 'operator': '<='},
     {'feature_indices': (27, 28, 29, 30, 31, 32, 33, 34, 35), 'operator': '>'}]
    """
    cols_df = pd.DataFrame(list(column_names), columns=["col", "cmp", "val"])

    def _mapper(subdf):
        return {
            "feature_indices": tuple(subdf.index),
            "operator": subdf["cmp"].unique()[0],
        }

    return cols_df.groupby(["col", "cmp"]).apply(_mapper).to_list()


def reduce_pattern_by_numerical_containment(
    pattern: Tuple, feature_group_info: Dict
) -> Tuple:
    """
    given a pattern and the numerical containment information of each feature group,

    reduce the pattern so that each feature group has at most one element inside the reduced pattern
    """
    # we assume inside each feature group, the order of the feature indices aligns with the order of the corresponding feature values
    reduced_pattern = set()

    for group in feature_group_info:
        operator = group["operator"]
        shared_indices = set(pattern).intersection(set(group["feature_indices"]))
        if len(shared_indices) == 0:
            continue
        else:
            if operator == "<=" or operator == "<":
                reduced_pattern.add(min(shared_indices))
            elif operator == ">=" or operator == ">":
                reduced_pattern.add(max(shared_indices))
            else:
                # the group is intact
                reduced_pattern |= shared_indices
    assert set(reduced_pattern).issubset(
        set(pattern)
    ), f"{reduced_pattern} is not a subset of {pattern}"
    return tuple(reduced_pattern)


def dichotomize_dataset(X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """split the feature dataframe X into two sub dataframes using the label information provided in y"""
    pos_idx, neg_idx = (y == 1).nonzero()[0], (y == 0).nonzero()[0]

    return (X.iloc[pos_idx], X.iloc[neg_idx])


def get_contrastive_patterns(pos_df, neg_df, min_pos_frequency, max_neg_frequency):
    """get the list of contrastive patterns, such that each pattern P satisfies the following:

    1. freq(P, pos_df) >= min_pos_frequency
    2. supp(P, neg_df) <= supp(P, pos_df) x max_neg_frequency
    """
    # 1. obtain the frequent itemsets w.r.t. the positive data points
    logger.info(
        "1 - mining frequent itemsets on the positive data points using FPgrowth"
    )
    itemsets_pos_only = fpgrowth(pos_df, min_support=min_pos_frequency, verbose=0)

    num_pos_itemsets = itemsets_pos_only.shape[0]
    columns = pos_df.columns

    feature_group_info = extract_feature_group_info(columns)

    num_pos = pos_df.shape[0]

    logger.info(
        f"obtained {num_pos_itemsets} frequent itemsets on the positive data points"
    )

    # 2. simplify the patterns by numerical containment
    itemsets_pos_only["reduced_itemsets"] = itemsets_pos_only["itemsets"].apply(
        lambda itemset: reduce_pattern_by_numerical_containment(
            itemset, feature_group_info
        )
    )

    # remove duplicate itemsets and obtain the support (renamed to "frequency" in the new df)
    reduced_itemsets_pos_only = (
        itemsets_pos_only.groupby("reduced_itemsets")
        .apply(lambda subdf: subdf["support"].mean())
        .to_frame(name="frequency")
        .reset_index()
    )

    num_reduced_pos_itemsets = reduced_itemsets_pos_only.shape[0]
    logger.info(
        f"2 - reducing each itemset by numerical containment,  obtained {num_reduced_pos_itemsets} unique itemsets"
    )

    # 3. filter out those frequent itemsets w.r.t. the negative data points
    logger.info("3 - filtering by thresholding on negative support")
    contrast_itemsets = []
    for _, r in tqdm(
        reduced_itemsets_pos_only.iterrows(), total=num_reduced_pos_itemsets
    ):
        itemset = r["reduced_itemsets"]
        pos_supp = r["frequency"] * num_pos
        neg_supp = support(itemset, columns, neg_df)
        if neg_supp <= pos_supp * max_neg_frequency:
            contrast_itemsets.append(itemset)

    logger.info(
        f"obtained {len(contrast_itemsets)} contrast itemsets from {num_reduced_pos_itemsets} candidates"
    )

    return contrast_itemsets


def enumerate_feasible_combinations(
    contrastive_patterns: List[Pattern],
    min_pos_ratio: float,
    max_neg_ratio: float,
    num_rules: int,
    pos_supp_set_by_pattern: Dict[Pattern, SupportSet],
    neg_supp_set_by_pattern: Dict[Pattern, SupportSet],
    n_pos: int,
    n_neg: int,
) -> List[PatternSet]:
    """enumerate all feasible combinations"""
    feasible_solutions = []
    for rule_set in tqdm(combinations(contrastive_patterns, num_rules)):
        pos_cov = set.union(*[pos_supp_set_by_pattern[r] for r in rule_set])
        neg_cov = set.union(*[neg_supp_set_by_pattern[r] for r in rule_set])

        pos_freq = len(pos_cov) / n_pos
        neg_freq = len(neg_cov) / n_neg

        accuracy = (len(pos_cov) + n_neg - len(neg_cov)) / (n_pos + n_neg)
        if (pos_freq >= min_pos_ratio) and (neg_freq <= max_neg_ratio):
            feasible_solutions.append(
                {
                    "rule_set": rule_set,
                    "pos_freq": pos_freq,
                    "neg_freq": neg_freq,
                    "accuracy": accuracy,
                }
            )
    logger.info(
        f"obtained {len(feasible_solutions)} feasible decision sets (each of size {num_rules})"
    )
    return feasible_solutions
