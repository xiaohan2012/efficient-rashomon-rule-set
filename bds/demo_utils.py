import pandas as pd
import numpy as np
from typing import List, Tuple
from logzero import logger

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from aix360.algorithms.rbm import FeatureBinarizer
from aix360.algorithms.rule_induction.rbm.boolean_rule_cg import BooleanRuleCG as BRCG


from contrastive_patterns import construct_program, BoundedWeightSATCallback
from weight_gen import WeightGen
from models import construct_dnf_rule, DNFRuleSetClassifier, EnsembleDNFClassifier

from common import PatternSet


def binarize_data(X_train_df, X_test_df, return_fb=False):
    fb = FeatureBinarizer(negations=True)
    X_train_df_fb = fb.fit_transform(X_train_df)
    X_test_df_fb = fb.transform(X_test_df)
    ret = (X_train_df_fb, X_test_df_fb)
    if return_fb:
        ret += (fb,)
    return ret


# extract data for our CP
def construct_program_and_variables(
    X_train_df_fb: pd.DataFrame,
    y_train: np.ndarray,
    min_pos_ratio: float = 0.6,
    max_neg_ratio: float = 0.3,
    num_rules: int = 2,
):
    """
    given the input binarized data frame and label vector: X_train_df_fb and y_train,
    construct a constrained program
    """
    X_fb = X_train_df_fb.to_numpy()
    pos_idx, neg_idx = (y_train == 1).nonzero()[0], (y_train == 0).nonzero()[0]
    Xp, Xn = X_fb[pos_idx], X_fb[neg_idx]

    min_pos_freq = int(Xp.shape[0] * min_pos_ratio)
    max_neg_freq = int(Xn.shape[0] * max_neg_ratio)

    # print(f"num. pos: {Xp.shape[0]}")
    # print(f"num. neg: {Xn.shape[0]}")
    # print(f"min_pos_freq: {min_pos_freq}")
    # print(f"max_neg_freq: {max_neg_freq}")

    # extract feature groups
    cols_df = pd.DataFrame(list(X_train_df_fb.columns), columns=["col", "cmp", "val"])

    # single feature groups, in which at most one feature is selected
    single_feature_groups = (
        cols_df.groupby(["col", "cmp"])
        .apply(lambda subdf: (tuple(subdf.index), 1))
        .to_list()
    )

    # pair feature groups, in which at most two feature is selected
    pair_feature_groups = (
        cols_df.groupby(["col"]).apply(lambda subdf: (tuple(subdf.index), 2)).to_list()
    )

    feature_groups_with_max_cardinality = single_feature_groups + pair_feature_groups

    program, I, Tp, Tn = construct_program(
        Xp,
        Xn,
        num_patterns=num_rules,
        min_pos_freq=min_pos_freq,
        max_neg_freq=max_neg_freq,
        feature_groups_with_max_cardinality=feature_groups_with_max_cardinality,
    )

    return program, I, Tp, Tn


def make_accuracy(num_pos, num_neg):
    """the weight is between 0 and 1, and is essentially accuracy"""

    def weight_func(pattern, TP_examples, FP_examples):
        TP, FP = len(TP_examples), len(FP_examples)
        TN = num_neg - FP
        return (TP + TN) / (num_pos + num_neg)

    return weight_func


def calculate_r(num_pos, num_neg, min_pos_ratio, max_neg_ratio):
    max_TP = num_pos
    max_TN = num_neg
    max_weight = (
        max_TP + max_TN
    )  # max. weight achieved when everything is predicted correctly

    min_TP = int(num_pos * min_pos_ratio)
    min_TN = num_neg - int(num_neg * max_neg_ratio)
    min_weight = min_TP + min_TN  # min. weight achieved when ...
    logger.info(f"min_TP: {min_TP}")
    logger.info(f"min_TN: {min_TN}")

    logger.info(f"max_weight: {max_weight}")
    logger.info(f"min_weight: {min_weight}")
    r = max_weight / min_weight
    return r


def sample_decision_sets(
    program, I, weight_func, make_callback, r, num_samples=100, eps=16.0
):
    wg = WeightGen(
        weight_func=weight_func,
        r=r,
        eps=eps,
        verbose=True,
        parallel=True,
    )

    wg.prepare(program, I, make_callback)

    wg.verbose = False
    samples_with_weights = wg.sample(
        num_samples, return_weight=True, show_progress=True
    )
    print(f"successfully obtained {len(samples_with_weights)} samples")
    return samples_with_weights


def gen_two_clusters_gaussian(n_pts_per_cluster, seed=124):
    pos_mean_1 = [-1, -1]
    pos_mean_2 = [1, 1]
    neg_mean_1 = [1, -1]
    neg_mean_2 = [-1, 1]

    std = 0.3

    np.random.seed(seed)
    X1p = np.random.multivariate_normal(
        pos_mean_1, cov=np.eye(2) * std, size=n_pts_per_cluster
    )
    X2p = np.random.multivariate_normal(
        pos_mean_2, cov=np.eye(2) * std, size=n_pts_per_cluster
    )

    X1n = np.random.multivariate_normal(
        neg_mean_1, cov=np.eye(2) * std, size=n_pts_per_cluster
    )
    X2n = np.random.multivariate_normal(
        neg_mean_2, cov=np.eye(2) * std, size=n_pts_per_cluster
    )

    X = np.concatenate([X1p, X2p, X1n, X2n])
    y = np.concatenate(
        [np.ones(n_pts_per_cluster * 2), np.zeros(n_pts_per_cluster * 2)]
    )
    return X, y


def make_linearly_separable(n_samples):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=1,
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)
    return linearly_separable


def construct_bayesian_decision_sets(
    decision_sets: List[PatternSet], feature_names: pd.MultiIndex
):
    """
    given a list of pattern sets, construct an ensemble rule-based classifier

    feature_names is the names of the discretized features
    """
    clf_list = []
    for i, ds in enumerate(decision_sets):
        dnf_rules = construct_dnf_rule(ds, feature_names)

        clf = DNFRuleSetClassifier(dnf_ruleset=dnf_rules)
        clf_list.append(clf)

    return EnsembleDNFClassifier(clf_list)


def train_BDS(
    X_train_df_fb: pd.DataFrame,
    y_train: np.ndarray,
    num_rules=2,
    min_pos_ratio=0.6,
    max_neg_ratio=0.3,
    num_decision_sets=100,
    eps=8.0,
):
    program, I, Tp, Tn = construct_program_and_variables(
        X_train_df_fb,
        y_train,
        min_pos_ratio=min_pos_ratio,
        max_neg_ratio=max_neg_ratio,
        num_rules=num_rules,
    )

    # use accuracy as the weight function
    num_pos, num_neg = (y_train == 1).sum(), (y_train == 0).sum()

    # make the weight function we will use
    accuracy = make_accuracy(num_pos, num_neg)

    # estimating $r$, the tilt
    r = calculate_r(num_pos, num_neg, min_pos_ratio, max_neg_ratio)
    logger.info(f"r = {r:.2f}")

    # define the callback for the BoundedWeightSAT subroutine
    def make_callback(weight_func, pivot, w_max, r):
        return BoundedWeightSATCallback(
            I,
            Tp,
            Tn,
            weight_func=weight_func,
            pivot=pivot,
            w_max=w_max,
            r=r,
            save_stat=True,
            verbose=False,
        )

    # obtain samples
    logger.info("Sampling decision sets: ")
    samples_with_weights = sample_decision_sets(
        program, I, accuracy, make_callback, r=r, num_samples=num_decision_sets, eps=eps
    )

    patterns = [p for p, _ in samples_with_weights]

    logger.info("Constructing rule sets ensemble: ")
    return construct_bayesian_decision_sets(patterns, X_train_df_fb.columns)


def train_BRCG(X_train_df_fb: pd.DataFrame, y_train: np.ndarray):
    brcg = BRCG(silent=True)
    brcg.fit(X_train_df_fb, y_train)
    return brcg


def split_and_binarize_data(X, y, test_ratio=0.5, random_state=1234):
    """split a dataset into train and test parts, and further binarize the features"""
    # split the data into train and test sets
    test_ratio = 0.5
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state
    )

    # turn the data to dataframes, so that feature binarizer can accept them
    X_train_df = pd.DataFrame(X_train, columns=["x1", "x2"])
    X_test_df = pd.DataFrame(X_test, columns=["x1", "x2"])

    # discretizing features
    X_train_df_fb, X_test_df_fb, fb = binarize_data(
        X_train_df, X_test_df, return_fb=True
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_df,
        X_test_df,
        X_train_df_fb,
        X_test_df_fb,
        fb,
    )


def prettify_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
