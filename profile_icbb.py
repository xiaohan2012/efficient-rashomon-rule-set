import numpy as np
from logzero import logger
from bds.rule import Rule
from bds.utils import bin_random, randints
from bds.icbb import IncrementalConstrainedBranchAndBound
from bds.random_hash import generate_h_and_alpha
from contexttimer import Timer

num_pts = 1000
num_rules = 150
seed = 12345
np.random.seed(seed)
rand_seeds = randints(num_rules)

rules = [
    Rule.random(i + 1, num_pts, random_seed=rand_seeds[i]) for i in range(num_rules)
]
y = bin_random(num_pts)

ub = 0.8
lmbd = 0.1

rand_seed = 12345

num_vars = len(rules)
num_constraints = num_vars - 1
A, t = generate_h_and_alpha(num_vars, num_constraints, seed=rand_seed, as_numpy=True)

thresh = 50
m1 = 8

icbb1 = IncrementalConstrainedBranchAndBound(rules, ub, y, lmbd)
Y_size_1 = icbb1.bounded_count(thresh, A=A[:m1], t=t[:m1])
logger.debug(f"|Y_1|: {Y_size_1}")

m2 = 12

with Timer() as timer:
    icbb2 = IncrementalConstrainedBranchAndBound(rules, ub, y, lmbd)
    Y_size_2 = icbb2.bounded_count(
        thresh, A=A[:m2], t=t[:m2], solver_status=icbb1.solver_status
    )
    logger.debug(f"solving takes {timer.elapsed:.2f} secs")
    logger.debug(f"tree size: {icbb2.tree.num_nodes}")
    logger.debug(f"|Y_2|: {Y_size_2}")
