import numpy as np
from bds.bb import BranchAndBoundNaive
from bds.rule import Rule
from bds.utils import bin_random, randints
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

with Timer() as timer:
    bb = BranchAndBoundNaive(rules, ub, y, lmbd)
    cnt = bb.bounded_count()
    print(f"elapsed time: {timer.elapsed}")


# print("bb.tree.root.total_num_nodes: {}".format(bb.tree.root.total_num_nodes))
# print("bb.tree.num_nodes: {}".format(bb.tree.num_nodes))
