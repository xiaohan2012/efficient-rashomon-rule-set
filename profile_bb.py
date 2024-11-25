import pickle as pkl

import numpy as np
from contexttimer import Timer

from bds.bb import BranchAndBoundNaive

rules = pkl.load(open("rules/rules_compas_06_104", "rb"))

f = open("data/compas_test.label")
labels = []
for row in f.readlines():
    labels.append(list(map(int, row.split(" ")[1:])))

y = np.array(labels[1], dtype=bool)
print(f"number of rules: {len(rules)}")
print(f"number of points: {y.shape[0]}")

ub = 0.3
lmbd = 0.05

with Timer() as timer:
    bb = BranchAndBoundNaive(rules, ub, y, lmbd)
    cnt = bb.bounded_count()
    print(f"elapsed time: {timer.elapsed}")
    print(f"|Rashomon set| = {cnt}")


# print("bb.tree.root.total_num_nodes: {}".format(bb.tree.root.total_num_nodes))
# print("bb.tree.num_nodes: {}".format(bb.tree.num_nodes))
