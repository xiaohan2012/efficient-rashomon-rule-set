from contexttimer import Timer
import numpy as np

from bds.meel import approx_mc2_core
from bds.rule import Rule
from bds.utils import bin_random, randints

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

eps = 5
delta = 0.99

with Timer() as timer:
    num_cells, num_sols = approx_mc2_core(
        rules, y, lmbd=lmbd, ub=ub, thresh=50, prev_num_cells=2, rand_seed=12345
    )
    print(f"elapsed time: {timer.elapsed:.2f}s")
