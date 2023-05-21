from bds.cbb import ConstrainedBranchAndBoundNaive
from bds.rule import Rule
from bds.utils import bin_random
from contexttimer import Timer
from bds.random_hash import generate_h_and_alpha


num_pts = 1000
num_rules = 150

rules = [Rule.random(i + 1, num_pts) for i in range(num_rules)]
y = bin_random(num_pts)

ub = 0.8
lmbd = 0.1
B = 50
num_constraints = 13

A, t = generate_h_and_alpha(num_rules, num_constraints, seed=12345, as_numpy=True)
with Timer() as timer:
    cbb = ConstrainedBranchAndBoundNaive(rules, ub, y, lmbd)
    cnt = cbb.bounded_count(threshold=B, A=A, t=t)
    print(f"elapsed time: {timer.elapsed}")

print("cbb.tree.root.total_num_nodes: {}".format(cbb.tree.root.total_num_nodes))
print("cbb.tree.num_nodes: {}".format(cbb.tree.num_nodes))
