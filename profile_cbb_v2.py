import numpy as np
from bds.cbb_v2 import ConstrainedBranchAndBound
from bds.rule import Rule
from bds.utils import bin_random, randints
from contexttimer import Timer
from bds.random_hash import generate_h_and_alpha


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
B = 72
num_constraints = 8

A, t = generate_h_and_alpha(num_rules, num_constraints, seed=12345, as_numpy=True)
with Timer() as timer:
    cbb = ConstrainedBranchAndBound(rules, ub, y, lmbd)
    cnt = cbb.bounded_count(threshold=B, A=A, t=t)
    print("{} / {}".format(cnt, B))
    print(f"elapsed time: {timer.elapsed}")

print("cbb.num_prefix_evaluations: {}".format(cbb.num_prefix_evaluations))    
# print("cbb.tree.root.total_num_nodes: {}".format(cbb.tree.root.total_num_nodes))
# print("cbb.tree.num_nodes: {}".format(cbb.tree.num_nodes))
