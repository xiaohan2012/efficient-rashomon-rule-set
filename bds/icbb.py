from typing import Any, Dict, Iterable, List, Optional, Tuple, Set
import copy

import numpy as np
from gmpy2 import mpz
from logzero import logger

from .bounds import prefix_specific_length_upperbound
from .cache_tree import Node
from .cbb import ConstrainedBranchAndBoundNaive, check_if_satisfied
from .rule import Rule
from .utils import bin_ones, bin_zeros


class IncrementalConstrainedBranchAndBound(ConstrainedBranchAndBoundNaive):
    """constrained branch-and-bound with incremental computation"""

    def __post_init__(self):
        self._init_solver_status()
        self._is_incremental = False

    def _init_solver_status(self):
        """intialize solver status

        the status of a solver is characterized by:

        - the last node being checked
        - the not_captured vector for the above node
        - the last rule being checked
        - the queue
        - the search tree
        """
        self._last_node = None  # the last node that is being expanded/checked
        self._last_not_captured = None  # the not_captured vector for the last node
        self._last_rule = None  # the last rule that was checked

        # the following variables record satisfiability information of the parity constraint system
        # self._last_u = None  # the undecided vector
        # self._last_s = None  # the satisfiability states vector
        # self._last_z = None  # the parity states vector

        # a list of nodes that correspond to discovered feasible solutions
        self._feasible_nodes = []

    @property
    def feasible_rulesets(
        self,
    ) -> List[Tuple]:
        """return the collected set of rulesets"""
        return list(
            map(lambda n: tuple(sorted(n.get_ruleset_ids())), self._feasible_nodes)
        )

    @property
    def solver_status(self):
        """return the status of the solver.

        the status can be used for subsequent solving of problems related to the current one.

        subsequent problems are expected to be more constrained than the current one
        """
        return dict(
            last_node=self._last_node,
            last_not_captured=self._last_not_captured,
            last_rule=self._last_rule,
            # last_u=self._last_u,
            # last_s=self._last_s,
            # last_z=self._last_z,
            feasible_nodes=self._feasible_nodes,
            queue=self.queue,
            tree=self.tree,
        )

    def _update_solver_status(
        self,
        parent_node: Node,
        parent_not_captured: mpz,
        rule: Rule,
        # u: np.ndarray,
        # s: np.ndarray,
        # z: np.ndarray,
    ):
        """update the current solver status

        the method is called whenever the solver checks the addition of a new rule to a certain prefix

        refer to ConstrainedBranchAndBoundNaive._loop to see when this method is called
        """
        self._last_node = parent_node
        self._last_not_captured = parent_not_captured
        self._last_rule = rule
        # no need to save u, s, and z since they need to be updated
        # self._last_u = u
        # self._last_s = s
        # self._last_z = z

    def _record_feasible_solution(self, node: Node):
        """record a feasible solution encoded by node

        this method is called whenever a new feasible solution is yielded

        refer to ConstrainedBranchAndBoundNaive._loop to see when this method is called
        """
        self._feasible_nodes.append(node)

    @property
    def is_incremental(self):
        """return True if incremental computation is enabled"""
        return self._is_incremental

    def _generate_from_last_checked_node(self, return_objective=False):
        """continue checking the last checked node in previous run (if the node does not fail the parity constraints)"""

        u, s, z, not_unsatisfied = self._recalculate_satisfiability_vectors(
            self._last_node
        )
        if not_unsatisfied:
            logger.debug("continue checking the last checked node in previous run")
            args = (
                # _last_node and _last_not_captured will be overriden once _loop is called
                self._last_node,
                self._last_not_captured,
                u,
                s,
                z,
            )
            yield from self._loop(
                *args, return_objective=return_objective, starting_rule=self._last_rule
            )
        else:
            logger.debug("unsatisfying Ax=b, thus stop checking last node ")
            yield from ()

    def _generate_from_queue(self, return_objective=False):
        """continue the normal search (e.g., check the nodes one by one in the queue)

        some nodes in the queue may become infeasible in the new constraint system
        """

        logger.debug("continue the normal checking on the nodes in the queue")

        while not self.queue.is_empty:
            parent, not_captured, u, s, z = self.queue.pop()

            kwargs_to_loop = dict(
                parent_node=parent,
                parent_not_captured=not_captured,
                u=u,
                s=s,
                z=z,
                starting_rule=None,
                return_objective=return_objective,
            )
            # if in incremental mode
            # the node maybe inserted by previous run or current run
            # if the former, re-check the satisfication of the constraints and conditionally continue the search
            # otherwise, search directly
            if self.is_incremental:
                # the node is inserted by previous run
                # here we assume that it is in the same run if and only if the length of u = the number of constraints
                # this assumption could fail in general of course
                assert (
                    self.num_constraints >= u.shape[0]
                ), "the previous problem must be less constrained than the current one"
                if u.shape[0] != self.num_constraints:
                    # logger.debug("recalculate u, s, and z")
                    (
                        up,
                        sp,
                        zp,
                        not_unsatisfied,
                    ) = self._recalculate_satisfiability_vectors(parent)
                    if not_unsatisfied:
                        # the node does not dissatisfy Ax=b,
                        # continue the search
                        # but with the updated u, s, and z
                        kwargs_to_loop["u"] = up
                        kwargs_to_loop["s"] = sp
                        kwargs_to_loop["z"] = zp
                        yield from self._loop(**kwargs_to_loop)
                    else:
                        pass
                        # logger.debug(
                        #     "the node failed the current Ax=b, thus do not check it"
                        # )
                else:
                    # the node is inserted by current run
                    yield from self._loop(**kwargs_to_loop)
            else:
                # in non-incremental mode
                # search directly
                yield from self._loop(**kwargs_to_loop)

    def generate(self, return_objective=False) -> Iterable:
        """return a generator which yields the feasible solutions
        the feasible solution can be yielded from two "sources":

        1. the unfinished checked from previous run (if exists)
        2. and the normal search (by checking nodes in the queue)
        """
        if not self.is_linear_system_solvable:
            logger.debug("abort the search since the linear system is not solvable")
            yield from ()
        else:
            # apply the following logic:
            # if in incremental mode
            # generate from feasible solutions found by previous run
            # search from the last checked node from previous run
            # and in any case, continue check the nodes in the queue
            if self.is_incremental:
                # generate from previously-found solutions
                yield from self._generate_from_feasible_solutions(return_objective)

                # and continue checking from the last node checked by previous run
                yield from self._generate_from_last_checked_node(return_objective)

            yield from self._generate_from_queue(return_objective)

    def _generate_from_feasible_solutions(
        self, return_objective: bool = False
    ) -> Iterable:
        """generate from previously collected feasible solutions"""
        if not self.is_incremental:
            raise RuntimeError(
                "it is forbidden to call this method in non-incremental mode"
            )

        new_feasible_nodes = []
        for node in self._feasible_nodes:
            # sol = node.get_ruleset_ids()
            u, s, z, not_unsatisfied = self._recalculate_satisfiability_vectors(node)
            # logger.debug(f"checking {sol}: not_unsatisfied={not_unsatisfied}, satisfied={check_if_satisfied(u, s, z, self.t)}")
            # logger.debug(f"u: {u.astype(int)}")
            # logger.debug(f"s: {s.astype(int)}")
            # logger.debug(f"z: {z.astype(int)}")
            # logger.debug(f"self.A:\n{self.A.astype(int)}")
            # logger.debug(f"self.t:\n{self.t.astype(int)}")
            if not_unsatisfied and check_if_satisfied(u, s, z, self.t):
                ruleset = node.get_ruleset_ids()
                if return_objective:
                    yield (ruleset, node.objective)
                else:
                    yield ruleset
                new_feasible_nodes.append(node)

        logger.debug(
            "inheritted {} solutions out of {}".format(
                len(new_feasible_nodes), len(self._feasible_nodes)
            )
        )
        # update _feasible_nodes
        self._feasible_nodes = new_feasible_nodes

    def _recalculate_satisfiability_vectors(
        self, node: Node
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """recalcuate the satisfiability vectors for a given node

        the vectors include:

        - u: the undecided vector
        - s: the satisfiability states vector
        - z: the parity states vector
        """

        # print("node: {}".format(node))
        rule_ids = node.get_ruleset_ids() - {0}

        assert len(rule_ids) == node.depth, f"{len(rule_ids)} != {node.depth}"

        u = bin_ones(self.num_constraints)
        s = bin_zeros(self.num_constraints)
        z = bin_zeros(self.num_constraints)
        not_unsatisfied = True
        # print("checking satisfiability")
        # print("rule_ids: {}".format(rule_ids))
        # TODO: can we do it in one run e.g., using vectorized operation?

        # we sort the rule ids because small ids are scanned first
        for idx in sorted(rule_ids):
            u, s, z, not_unsatisfied = self._check_if_not_unsatisfied(
                self.rules[idx - 1],  # minus 1 because rule id starts from 1
                u,
                s,
                z,
            )

        return u, s, z, not_unsatisfied

    def _copy_solver_status(self, solver_status: Optional[Dict[str, Any]] = None):
        """copy the solver status into self"""
        self._is_incremental = True  # mark the run as incremental

        self.queue = solver_status["queue"].copy()
        self.tree = copy.copy(solver_status["tree"])
        self._feasible_nodes = copy.copy(solver_status["feasible_nodes"])

        # assume the following 3 attributes are immutable
        self._last_node = copy.copy(solver_status["last_node"])
        self._last_not_captured = copy.copy(solver_status["last_not_captured"])
        self._last_rule = copy.copy(solver_status["last_rule"])

    def reset(
        self,
        A: np.ndarray,
        t: np.ndarray,
        solver_status: Optional[Dict[str, Any]] = None,
    ):
        """
        reset the solving status based on the status from previous solving

        if solving_status is not given, solve the problem from scratch
        """
        self.setup_constraint_system(A, t)

        if solver_status is None:
            # create the tree and queue
            logger.debug("solve from scratch")
            self.reset_tree()
            self.reset_queue()
        else:
            logger.debug(
                "solve incrementally "
            )
            self._copy_solver_status(solver_status)

    def run(self, return_objective=False, **kwargs) -> Iterable:
        """incrementally solve the problem if solver_status is given"""
        self.reset(**kwargs)

        yield from self.generate(return_objective=return_objective)

    def _get_continuation_idx(
        self, parent_node: Node, starting_rule: Optional[Rule]
    ) -> int:
        """
        get the index where the search continues
        the continuation index is determined by whether starting_rule is given

        - if yes, then we start from max(starting_rule.rule_id, parent_node.rule_id)
           (we do not subtract the index by 1 because rule indices start from 1)
        - otherwise, start from parent_node.rule_id
        """
        return (
            max(starting_rule.id, parent_node.rule_id)
            if starting_rule is not None
            else parent_node.rule_id
        )

    def _loop(
        self,
        parent_node: Node,
        parent_not_captured: mpz,
        u: np.ndarray,
        s: np.ndarray,
        z: np.ndarray,
        starting_rule: Optional[Rule] = None,
        return_objective=False,
    ):
        """
        check one node in the search tree, update the queue, and yield feasible solution if exists
        parent_node: the current node/prefix to check
        parent_not_captured: postives not captured by the current prefix
        u: the undetermined vector
        s: the satisfifaction state vector
        z: the parity state vector
        starting_rule: the rule from which the iterating starts
        return_objective: True if return the objective of the evaluated node
        """
        parent_lb = parent_node.lower_bound
        length_ub = prefix_specific_length_upperbound(
            parent_lb, parent_node.num_rules, self.lmbd, self.ub
        )

        continuation_idx = self._get_continuation_idx(parent_node, starting_rule)

        # here we assume the rule ids are consecutive integers
        for rule in self.rules[continuation_idx:]:
            self._update_solver_status(
                parent_node,
                parent_not_captured,
                rule,
            )

            # prune by ruleset length
            assert parent_node.num_rules == (len(parent_node.get_ruleset_ids()) - 1)
            if (parent_node.num_rules + 1) > length_ub:
                continue

            captured = self._captured_by_rule(rule, parent_not_captured)
            lb = (
                parent_lb
                + self._incremental_update_lb(captured, self.y_mpz)
                + self.lmbd
            )
            if lb <= self.ub:
                fn_fraction, not_captured = self._incremental_update_obj(
                    parent_not_captured, captured
                )
                obj = lb + fn_fraction

                # the following variables might be assigned later
                # and if assigned
                # will be assigned only once in the check of the current rule
                child_node = None
                up = None

                # apply look-ahead bound
                lookahead_lb = lb + self.lmbd
                if lookahead_lb <= self.ub:
                    up, sp, zp, not_unsatisfied = self._check_if_not_unsatisfied(
                        rule, u, s, z
                    )
                    if not_unsatisfied:
                        child_node = self._create_new_node_and_add_to_tree(
                            rule, lb, obj, captured, parent_node
                        )
                        self.queue.push(
                            (child_node, not_captured, up, sp, zp),
                            key=(
                                child_node.lower_bound,
                                tuple(child_node.get_ruleset_ids()),
                            ),  # if the lower bound are the same for two nodes, resolve the order by the corresponding ruleset
                        )

                if obj <= self.ub:
                    if up is None:
                        # do the checking if not done
                        up, sp, zp, not_unsatisfied = self._check_if_not_unsatisfied(
                            rule, u, s, z
                        )

                    if check_if_satisfied(up, sp, zp, self.t):
                        if child_node is None:
                            # compute child_node if not done
                            child_node = self._create_new_node_and_add_to_tree(
                                rule, lb, obj, captured, parent_node
                            )
                        ruleset = child_node.get_ruleset_ids()
                        self._record_feasible_solution(child_node)

                        if return_objective:
                            yield (ruleset, child_node.objective)
                        else:
                            yield ruleset
