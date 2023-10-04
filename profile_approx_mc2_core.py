from contexttimer import Timer
import numpy as np
import pickle as pkl
from bds.meel import approx_mc2_core
from bds.utils import randints

# seed = 42

rules = pkl.load(open("rules/rules_compas_06_104", "rb"))

f = open("data/compas_test.label", "r")
labels = []
for row in f.readlines():
    labels.append(list(map(int, row.split(" ")[1:])))

y = np.array(labels[1], dtype=bool)
print(f"number of rules: {len(rules)}")
print(f"number of points: {y.shape[0]}")

ub = 0.3
lmbd = 0.05

thresh = 72
prev_m = 1

reps = 5
with Timer() as timer:
    # for _ in range(reps):
    num_cells, num_sols = approx_mc2_core(
        rules,
        y,
        lmbd=lmbd,
        ub=ub,
        thresh=thresh,
        prev_num_cells=int(2**prev_m),
        rand_seed=None,
    )
print(f"elapsed time: {timer.elapsed:.2f}s")
print(f"estimated count = {num_cells * num_sols}")
