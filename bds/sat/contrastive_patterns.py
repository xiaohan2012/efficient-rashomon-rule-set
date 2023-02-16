import numpy as np
import re
from ortools.sat.python import cp_model

from typing import Tuple, List, Callable
from itertools import chain
from logzero import logger

from ..common import Program, CPVarList2D, CPVarList, PATTERN_LOG_LEVEL
from .printers import PatternSolutionPrinter


def add_coverage_constraints(
    model: Program,
    D: np.ndarray,
    I: CPVarList,
    T: CPVarList,
    C: CPVarList2D,
    P: CPVarList,
) -> Program:
    """
    add coverage constraints for a given dataset D

    model: the CP program
    D the data matrix
    I: the variables for y
    T: coverage variables for transactions
    C: auxiliary variables, C_{kt} = 0 <=> sum_i I_{ki} (1 - D_{ti}) == 0
    P: auxiliary variables, P_t = prod_k C_{kt}
    """

    # may C and P can be created inside this function
    num_pts, num_feats = D.shape
    num_patterns = int(len(I) / num_feats)
    assert num_patterns == (len(I) / num_feats)

    logger.debug("adding coverage constraints...")
    logger.debug(f"num. of y: {num_patterns}")
    logger.debug(f"num. of data points: {num_pts}")

    for t in range(num_pts):
        absent_features = (D[t] == 0).nonzero()[0]
        for k in range(num_patterns):
            num_matched_absent_features = sum(
                I[k * num_feats + i] for i in absent_features
            )

            # define: C_{kt} = 0 <=> \sum_i I_{ki} (1 - D_{ti}) == 0
            model.Add(num_matched_absent_features > 0).OnlyEnforceIf(C[k][t])
            model.Add(num_matched_absent_features == 0).OnlyEnforceIf(C[k][t].Not())

        # next, add the following constraints P_t = \prod_k C_{kt}
        # if you do not understand the code below:
        # read this: https://github.com/google/or-tools/blob/stable/ortools/sat/docs/boolean_logic.md#product-of-two-boolean-variables
        model.AddBoolOr(P[t], *[C[k][t].Not() for k in range(num_patterns)])
        for k in range(num_patterns):
            model.AddImplication(P[t], C[k][t])

        # finally, T_t = 1 <=> (P_t = 0)
        model.Add(P[t] == 0).OnlyEnforceIf(T[t])
        model.Add(P[t] > 0).OnlyEnforceIf(T[t].Not())
    return model


def construct_program(
    Dp: np.ndarray,
    Dn: np.ndarray,
    num_patterns: int,
    min_pos_freq: int,
    max_neg_freq: int,
    feature_groups_with_max_cardinality: List[Tuple[Tuple[int], int]] = [],
) -> Tuple[Program, CPVarList2D, CPVarList, CPVarList]:
    """
    num_patterns: number of conjunctions
    min_pos_freq: minimum number of covered examples among the positive records (minimum True Positive)
    max_neg_freq: maximum number of covered examples among the negative records (maximum False Positive)

    feature_groups: groups of features. in each group, at most one feature can be selected
    """
    logger.info("constructing a constrainted program for finding contrastive y")
    num_pos_pts, pos_num_feats = Dp.shape
    num_neg_pts, neg_num_feats = Dn.shape

    assert (
        pos_num_feats == neg_num_feats
    ), f"feature dimension of Dp and Dn mismatch: {pos_num_feats} != {neg_num_feats}"

    num_feats = pos_num_feats

    model = cp_model.CpModel()

    # ------- variable declaration ---------
    # our two primary binary variables.
    Tp = [
        model.NewBoolVar(f"Tp[{t}]") for t in range(num_pos_pts)
    ]  # Tp for positive examples
    Tn = [
        model.NewBoolVar(f"Tn[{t}]") for t in range(num_neg_pts)
    ]  # Tn for negative examples
    # remark: Tp and Tn do NOT stand for true positive and true negative

    # num_patterns x num_features
    # it is a 1D array, where the (k * num_feats + i)th entry corresponds the jth feature variable for the ith rule
    I = [  # I for the y
        model.NewBoolVar(f"I[{k},{i}]")
        for k in range(num_patterns)
        for i in range(num_feats)
    ]
    # the coverage (auxiliary) variable on the pattern level
    # num_patterns x num_pts
    Cp = [
        [model.NewBoolVar(f"Cp[{k},{i}]") for i in range(num_pos_pts)]
        for k in range(num_patterns)
    ]
    Cn = [
        [model.NewBoolVar(f"Cn[{k},{i}]") for i in range(num_neg_pts)]
        for k in range(num_patterns)
    ]
    # the product (auxiliary) variable, P_t = \prod_k C_{kt}
    Pp = [model.NewBoolVar(f"Pp[{i}]") for i in range(num_pos_pts)]
    Pn = [model.NewBoolVar(f"Pn[{i}]") for i in range(num_neg_pts)]

    # ------- constraints ---------
    # 1. coverage constraints for Dp and Dn
    add_coverage_constraints(model, Dp, I, Tp, Cp, Pp)
    add_coverage_constraints(model, Dn, I, Tn, Cn, Pn)

    # 2. mininum positive threshold
    model.Add(sum(Tp) >= min_pos_freq)

    # 3. maximum negative threshold
    model.Add(sum(Tn) <= max_neg_freq)

    # 4. the pattern is not empty
    for k in range(num_patterns):
        # model.Add(sum(I[k][i] for i in range(pos_num_feats)) > 0)
        model.Add(sum(I[k * num_feats + i] for i in range(pos_num_feats)) > 0)

    # 5. impose ordering of the y to reduce the solution space
    # here we use a hierarchical ordering based on a sequence of attributes
    for k in range(num_patterns - 1):
        # 5.1: the numbers of TPs are in non-decreasing order
        num_TP_k = sum(Cp[k][t] for t in range(num_pos_pts))
        num_TP_kp1 = sum(Cp[k + 1][t] for t in range(num_pos_pts))
        model.Add(num_TP_k >= num_TP_kp1)

        # an auxiliary variable indicating whether the TP is equal for the kth pair
        TP_EQ_k = model.NewBoolVar(f"TP_EQ[{k}]")
        model.Add(num_TP_k == num_TP_kp1).OnlyEnforceIf(TP_EQ_k)
        model.Add(num_TP_k > num_TP_kp1).OnlyEnforceIf(TP_EQ_k.Not())

        # 5.2: if the pair have the same TPs, sort by pattern length
        # an auxiliary variable indicating whether the lengths are equal for the kth pair
        L_EQ_k = model.NewBoolVar(f"L_EQ[{k}]")

        len_k = sum(I[k + num_feats + i] for i in range(num_feats))
        len_kp1 = sum(I[(k + 1) * num_feats + i] for i in range(num_feats))
        model.Add(len_k <= len_kp1).OnlyEnforceIf(TP_EQ_k)

        # L_EQ[k] = 1 <=> len of k == len of (k+1)
        model.Add(len_k == len_kp1).OnlyEnforceIf(L_EQ_k)
        model.Add(len_k > len_kp1).OnlyEnforceIf(L_EQ_k.Not())

        # 5.3: if the pair have both equal TP numbers and equal pattern lengths
        # sort by the numbers of FPs
        # an auxiliary variable indicating whether the lengths are equal and the TP numbers for the kth pair
        TP_L_EQ_k = model.NewBoolVar(f"TP_L_EQ[{k}]")
        # TP_L_EQ[k] = 1 <=> L_EQ[k] and TP_EQ[k]
        model.AddBoolAnd(L_EQ_k, TP_EQ_k).OnlyEnforceIf(TP_L_EQ_k)
        model.AddBoolOr(L_EQ_k.Not(), TP_EQ_k.Not()).OnlyEnforceIf(TP_L_EQ_k.Not())

        num_FP_k = sum(Cn[k][t] for t in range(num_neg_pts))
        num_FP_kp1 = sum(Cn[k + 1][t] for t in range(num_neg_pts))
        model.Add(num_FP_k <= num_FP_kp1).OnlyEnforceIf(TP_L_EQ_k)

    # 6: the y should be different
    # implication: strictly `num_patterns` different y should be found
    for k in range(num_patterns - 1):
        for j in range(k + 1, num_patterns):
            # compare the kth and jth pattern
            # feature-wise inequality indicator
            I_NEQ = [
                model.NewBoolVar(f"I_NEQ[{k}, {j}, {i}]") for i in range(num_feats)
            ]
            for i in range(num_feats):
                model.Add(I[k * num_feats + i] == I[j * num_feats + i]).OnlyEnforceIf(
                    I_NEQ[i].Not()
                )
                model.Add(I[k * num_feats + i] != I[j * num_feats + i]).OnlyEnforceIf(
                    I_NEQ[i]
                )
            # at least one P_NEQ should be true
            model.AddBoolOr(*I_NEQ)

    model.AddDecisionStrategy(I, cp_model.CHOOSE_FIRST, cp_model.SELECT_MAX_VALUE)

    # 7: constraints on feature groups: at most one feature in each group can be selected
    for grp, max_card in feature_groups_with_max_cardinality:
        for k in range(num_patterns):
            model.Add(sum(I[k * num_feats + i] for i in grp) <= max_card)
    logger.info("construction done")
    return model, I, Tp, Tn


class ContrastPatternSolutionPrinter(PatternSolutionPrinter):
    """print contrast pattern solutions for a binary classification dataset

    an example:


        # given some data points
        Xp, Xn = ...

        # and the binarized data
        # whose column names are used to infer feature group information
        X_train_df_fb = ...

        # define min positive frequency and max negative frequency
        min_pos_freq = ...
        max_neg_freq = ...

        solver = construct_solver()

        # extract feature groups
        cols_df = pd.DataFrame(list(X_train_df_fb.columns), columns=['col', 'cmp', 'val'])

        # single feature groups, in which at most one feature is selected
        single_feature_groups = cols_df.groupby(['col', 'cmp']).apply(lambda subdf: (tuple(subdf.index), 1)).to_list()

        # pair feature groups, in which at most two feature is selected
        pair_feature_groups = cols_df.groupby(['col']).apply(lambda subdf: (tuple(subdf.index), 2)).to_list()

        feature_groups_with_max_cardinality = single_feature_groups + pair_feature_groups

        program, I, Tp, Tn = construct_program(
            Xp, Xn, num_patterns=2, min_pos_freq=min_pos_freq, max_neg_freq=max_neg_freq,
            feature_groups_with_max_cardinality=feature_groups_with_max_cardinality
        )

        # count the number of solutions
        import time
        start_time = time.time()

        cb = ContrastPatternSolutionPrinter(I, Tp, Tn, num_feats=Xp.shape[1])
        solver.Solve(program, cb)

        print('number of solutions:', cb.solution_count)

        end_time = time.time() - start_time
        print(f'it took {end_time:.2f} seconds')
    """

    def __init__(
        self,
        pattern_variables: CPVarList,
        pos_coverage_variables: CPVarList,
        neg_coverage_variables: CPVarList,
        verbose=0,
    ):
        cp_model.CpSolverSolutionCallback.__init__(self)

        self.__pattern_variables = pattern_variables
        self.__pos_coverage_variables = pos_coverage_variables
        self.__neg_coverage_variables = neg_coverage_variables

        self.__solution_count = 0
        self.__solutions = []
        self.__solution_stat = []

        # infer the number of patterns from the variable name, e.g., 'I[0, 1]', where the first index is the pattern index
        self.num_patterns = len(
            set(
                [re.findall("I\[(\d+),\d+\]", str(i))[0] for i in pattern_variables]
            )
        )
        self.num_feats = int(len(pattern_variables) / self.num_patterns)
        assert self.num_feats == (len(pattern_variables) / self.num_patterns)

        self.verbose = verbose

    def on_solution_callback(self):
        self.__solution_count += 1
        y = [
            tuple(
                [
                    i
                    for i, v in enumerate(
                        self.__pattern_variables[
                            k * self.num_feats : (k + 1) * self.num_feats
                        ]
                    )
                    if self.Value(v) == 1
                ]
            )
            for k in range(self.num_patterns)
        ]
        y = tuple(sorted(y))
        self.__solutions.append(y)

        pos_covered_examples = tuple(
            t for t, v in enumerate(self.__pos_coverage_variables) if self.Value(v) == 1
        )
        neg_covered_examples = tuple(
            t for t, v in enumerate(self.__neg_coverage_variables) if self.Value(v) == 1
        )
        self.__solution_stat.append(
            {
                "solution": y,
                "stat": {
                    "pos_freq": len(pos_covered_examples),
                    "neg_freq": len(neg_covered_examples),
                    "cov_pos": pos_covered_examples,
                    "cov_neg": neg_covered_examples,
                },
            }
        )

        if self.verbose > 0:
            print(
                f"y: {y}, \npos. freq.: {len(pos_covered_examples)}, neg. freq.: {len(neg_covered_examples)}, covered pos.: {pos_covered_examples}, covered neg.: {neg_covered_examples}"
            )
            print("-" * 5)

    @property
    def solution_count(self):
        # Q: why cannot I inherit from the parent class?
        return self.__solution_count

    @property
    def solutions_found(self):
        return self.__solutions

    @property
    def solution_stat(self):
        return self.__solution_stat


class BoundedWeightSATCallback(cp_model.CpSolverSolutionCallback):
    """
    the BoundedWeightSAT algorithm (for contrastive pattern mining), which collects a number of y (the solutions in a SAT program) with total weight at most `pivot`
    """

    def __init__(
        self,
        pattern_variables: CPVarList,
        pos_coverage_variables: CPVarList,
        neg_coverage_variables: CPVarList,
        weight_func: Callable,  # thr weight function
        pivot: float,  # the total weight upper bound (normalized)
        w_max: float,  # estimation of maximum of pattern weight
        r: float,  # upperbound of the tilt parameter
        save_stat: bool = True,
        verbose: bool = False,
    ):
        """pattern_variables: one for each feature, whether the feature is selected or not
        coverage_variables: one for each transaction, whether the transaction is covered or not

        if limit < 0, all solutions are printed
        """
        assert (
            pivot > 0
        ), "pivot should be positive (how can cell sizes be non-positive?)"

        cp_model.CpSolverSolutionCallback.__init__(self)

        self.verbose = verbose

        self.__pattern_variables = pattern_variables
        self.__pos_coverage_variables = pos_coverage_variables
        self.__neg_coverage_variables = neg_coverage_variables

        self.weight_func = weight_func
        self.save_stat = save_stat

        self.pivot = pivot
        self.w_max = w_max
        self.r = r

        # print("self.pivot (init of callback): ", self.pivot)

        # infer the number of patterns from the variable name, e.g., 'I[0, 1]', where the first index is the pattern index
        self.num_patterns = len(
            set(
                [re.findall("I\[(\d+),\d+\]", str(i))[0] for i in pattern_variables]
            )
        )
        self.num_feats = int(len(pattern_variables) / self.num_patterns)
        assert self.num_feats == (len(pattern_variables) / self.num_patterns)

        self.reset()

    def reset(self):
        self.Y = []
        self.weights = []

        self.__solution_stat = {}

        self.w_total = 0
        self.w_min = self.w_max / self.r

        # a flag which indicates if the search procedure exits because the w_total overflows the limit
        # being False also means we cannot find any more solutions
        self.overflows_w_total = False
        if self.verbose:
            logger.debug(f"reset: w_min={self.w_min}")

    def log_pattern(
        self, pattern, pos_covered_examples, neg_covered_examples, pattern_weight
    ):
        msg1 = f"y: {pattern}, w={pattern_weight}, |supp+|: {len(pos_covered_examples)}, |supp-|: {len(neg_covered_examples)}"
        msg2 = f"supp+: {pos_covered_examples}, covered neg.: {neg_covered_examples}"
        logger.log(
            level=PATTERN_LOG_LEVEL,
            msg=msg1,
        )
        logger.log(
            level=PATTERN_LOG_LEVEL - 1,
            msg=msg2,
        )

    def on_solution_callback(self):
        # extract the patterns from the variable values
        y = [
            tuple(
                [
                    i
                    for i, v in enumerate(
                        self.__pattern_variables[
                            k * self.num_feats : (k + 1) * self.num_feats
                        ]
                    )
                    if self.Value(v) == 1
                ]
            )
            for k in range(self.num_patterns)
        ]
        y = tuple(sorted(y))

        pos_covered_examples = tuple(
            t for t, v in enumerate(self.__pos_coverage_variables) if self.Value(v) == 1
        )
        neg_covered_examples = tuple(
            t for t, v in enumerate(self.__neg_coverage_variables) if self.Value(v) == 1
        )
        pattern_weight = self.weight_func(y, pos_covered_examples, neg_covered_examples)

        if self.save_stat:
            self.__solution_stat[y] = {
                "w": pattern_weight,
                "stat": {
                    "pos_freq": len(pos_covered_examples),
                    "neg_freq": len(neg_covered_examples),
                    "cov_pos": pos_covered_examples,
                    "cov_neg": neg_covered_examples,
                },
            }

        if self.verbose > 0:
            self.log_pattern(
                y, pos_covered_examples, neg_covered_examples, pattern_weight
            )

        assert pattern_weight >= 0, "non-negative weights are assumed"

        self.Y.append(y)
        self.weights.append(pattern_weight)

        self.w_total += pattern_weight
        self.w_min = min(self.w_min, pattern_weight)

        # self.log_pattern(y, covered_examples, pattern_weight)

        w_total_normalized = self.w_total / self.w_min / self.r
        # print("self.pivot (before stop search): ", self.pivot)
        if w_total_normalized > self.pivot:
            if self.verbose:
                logger.debug(
                    f"BoundedWeightSATCallback: search stops after overflowing pivot: {w_total_normalized} > {self.pivot}, where the former = {self.w_total} (w_total) / {self.w_min} (w_min) / {self.r} (r)"
                )
            self.overflows_w_total = True

            self.StopSearch()

    @property
    def new_w_max(self):
        return self.w_min * self.r

    @property
    def solutions_found(self):
        return self.Y

    @property
    def solution_stat(self):
        return self.__solution_stat
