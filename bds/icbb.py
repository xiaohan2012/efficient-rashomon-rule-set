from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from gmpy2 import mpz
from logzero import logger

from .bounds import prefix_specific_length_upperbound
from .utils import bin_ones, bin_zeros
from .cache_tree import Node
from .cbb import ConstrainedBranchAndBoundNaive, check_if_satisfied
from .rule import Rule

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
        - satisfiability (to the parity constraint system) information of the last node
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
        """continue checking the last checked node in previous run"""
        logger.debug("continue checking the last checked node in previous run")

        # TODO: recalculate u, s and z
        u, s, z = self._recalculate_satisfiability_vectors(self._last_node)
        args = (
            self._last_node,  # these attributes will be overriden once _loop is called
            self._last_not_captured,
            u,
            s,
            z,
        )
        yield from self._loop(
            *args, return_objective=return_objective, starting_rule=self._last_rule
        )

    def _generate_from_queue(self, return_objective=False):
        """continue the normal search (e.g., check the nodes one by one in the queue)"""

        logger.debug("continue the normal search (e.g., check the nodes in the queue)")

        while not self.queue.is_empty:
            parent, parent_not_captured, u, s, z = self.queue.pop()

            if self.is_incremental and (
                u.shape[0] != self.num_constraints
            ):  # the node is inserted by previous run
                logger.debug("recalculate u, s, and z")
                # the stored u, s, and z in queue_item[3:] can be outdated
                # TOOD: re-calculate them for the new constraint system
                u, s, z = self._recalculate_satisfiability_vectors(parent)

                # otherwise, we use the queue item as it is
            yield from self._loop(
                parent,
                parent_not_captured,
                u,  # u, s, and z maybe updated
                s,
                z,
                return_objective=return_objective,
            )

    def generate(self, return_objective=False) -> Iterable:
        logger.debug("calling generate in icbb")
        if not self.is_linear_system_solvable:
            logger.debug("abort the search since the linear system is not solvable")
            yield from ()
        else:
            # apply the following logic:
            # if in incremental mode
            # search from the last checked node from previous one
            # and then check the nodes in the queue
            if self.is_incremental:
                yield from self._generate_from_last_checked_node(return_objective)

            yield from self._generate_from_queue(return_objective)

    def _recalculate_satisfiability_vectors(
        self, node: Node
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """recalcuate the satisfiability vectors for a given node

        the vectors include:

        - u: the undecided vector
        - s: the satisfiability states vector
        - z: the parity states vector
        """
        rule_ids = node.get_ruleset_ids() - {0}

        assert len(rule_ids) == node.depth, f"{len(rule_ids)} != {node.depth}"

        u = bin_ones(self.num_constraints)
        s = bin_zeros(self.num_constraints)
        z = bin_zeros(self.num_constraints)

        # TODO: can we do it in one run e.g., using vectorized operation?
        for idx in rule_ids:
            u, s, z, not_unsatisfied = self._check_if_not_unsatisfied(
                self.rules[idx - 1],  # -1 because rule id starts from 1
                u,
                s,
                z,
            )
        return u, s, z, not_unsatisfied

    def _copy_solver_status(self, solver_status: Optional[Dict[str, Any]] = None):
        """copy the solver status into self"""
        self._is_incremental = True  # mark the run as incremental

        self.queue = solver_status["queue"]
        self.tree = solver_status["tree"]

        self._last_node = solver_status["last_node"]
        self._last_not_captured = solver_status["last_not_captured"]
        self._last_rule = solver_status["last_rule"]

        self._last_u = solver_status["last_u"]
        self._last_s = solver_status["last_s"]
        self._last_z = solver_status["last_z"]

        self._feasible_nodes = solver_status["feasible_nodes"]

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
            logger.debug("solve from scratch, since solver_stats is None")
            self.reset_tree()
            self.reset_queue()
        else:
            logger.debug("solve based on previous run, since solver_stats is not None")
            self._copy_solver_status(solver_status)

    def run(self, return_objective=False, **kwargs) -> Iterable:
        """incrementally solve the problem if solver_status is given"""
        self.reset(**kwargs)

        yield from self.generate(return_objective=return_objective)

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

        # the starting point is determined by whether starting_rule is given
        # if yes, then we start from starting_rule.rule_id (not minus one because rule ids are one-indexed)
        # otherwise, start from the rule id of the parent node
        starting_rule_idx = (
            starting_rule.id if starting_rule is not None else parent_node.rule_id
        )

        # here we assume the rule ids are consecutive integers
        for rule in self.rules[starting_rule_idx:]:
            self._update_solver_status(
                parent_node,
                parent_not_captured,
                rule,
                # u,
                # s,
                # z,
            )

            # prune by ruleset length
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
                            key=child_node.lower_bound,
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
