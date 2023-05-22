import numpy as np
import itertools


from typing import Optional, List, Iterable, Dict, Any, Tuple
from logzero import logger
from gmpy2 import mpz

from .cache_tree import Node
from .cbb import ConstrainedBranchAndBoundNaive
from .rule import Rule
from .utils import count_iter


class IncrementalConstrainedBranchAndBound(ConstrainedBranchAndBoundNaive):
    """constrained branch-and-bound with incremental computation"""

    def __post_init__(self):
        self._init_solver_status()

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
        self._last_u = None  # the undecided vector
        self._last_s = None  # the satisfiability states vector
        self._last_z = None  # the parity states vector

        # a list of nodes that correspond to discovered feasible solutions
        self._feasible_nodes = []

    @property
    def solver_status(self):
        """return the status of the solver.

        the status can be used for subsequent solving of problems related to the current one.

        subsequent problems are expected to be more constrained than the current one
        """
        return dict(
            last_node=self._parent_node,
            last_not_captured=self._last_not_captured,
            last_rule=self._rule,
            last_u=self._u,
            last_s=self._s,
            last_z=self._z,
            feasible_nodes=self._feasible_nodes,
            queue=self.queue,
            tree=self.tree,
        )

    def _update_solver_status(
        self,
        parent_node: Node,
        parent_not_captured: mpz,
        rule: Rule,
        u: np.ndarray,
        s: np.ndarray,
        z: np.ndarray,
    ):
        """update the current solver status

        the method is called whenever the solver checks the addition of a new rule to a certain prefix

        refer to ConstrainedBranchAndBoundNaive._loop to see when this method is called
        """
        self._last_node = parent_node
        self._last_not_captured = parent_not_captured
        self._last_rule = rule
        self._last_u = u
        self._last_s = s
        self._last_z = z

    def _record_feasible_solution(self, node: Node):
        """record a feasible solution encoded by node

        this method is called whenever a new feasible solution is yielded

        refer to ConstrainedBranchAndBoundNaive._loop to see when this method is called
        """
        self._feasible_nodes.append(node)

    @property
    def is_incremental(self):
        """return True if incremental computation is enabled"""
        # we assume that if self._last_node is given, incremental computation is enabled
        return self._last_node is not None

    def generate(self, return_objective=False) -> Iterable:
        if not self.is_linear_system_solvable:
            logger.debug("abort the search since the linear system is not solvable")
            yield from ()
        else:
            # apply the following logic:
            # if _last_node is given, meaning we re-use previous run
            # then search from that node first before the popped node
            if self._last_node:
                args = (
                    self._last_node,
                    self._last_not_captured,
                    self._last_u,
                    self._last_s,
                    self._last_z,
                )
                yield from self._loop(*args, return_objective=return_objective)

            # continue the "normal" search
            while not self.queue.is_empty:
                queue_item = self.queue.pop()

                if self.is_incremental:
                    parent, parent_not_captured = queue_item[:2]
                    # the stored u, s, and z in queue_item[3:] can be outdated
                    # TOOD: re-calculate them for the new constraint system
                    u, s, z = self._recalculate_satisfiability_vectors(parent)
                    queue_item = (parent, parent_not_captured, u, s, z)

                    # TODO: distinguish the cases that the node is inserted by previous run or by current run
                    # the decision is made by whether the length of u (s or z) matches that of t
                    # if yes, then the node is inserted in this run (and no need to re-calculate),
                    # otherwise, by the previous one (and should be re-calculated)

                # otherwise, we use the queue item as it is
                yield from self._loop(*queue_item, return_objective=return_objective)

    def _recalculate_satisfiability_vectors(
        self, node: Node
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """recalcuate the satisfiability vectors for a given node

        the vectors include:

        - u: the undecided vector
        - s: the satisfiability states vector
        - z: the parity states vector
        """
        raise NotImplementedError()

    def _copy_solver_status(self, solver_status: Optional[Dict[str, Any]] = None):
        """copy the solver status into self"""
        self.queue = solver_status["queue"]
        self.tree = solver_status["tree"]

        self._last_node = solver_status["last_node"]
        self._last_not_captured = solver_status["not_captured"]
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
            super(ConstrainedBranchAndBoundNaive, self).reset()
        else:
            self._copy_solver_status(solver_status)

    def run(self, return_objective=False, **kwargs) -> Iterable:
        """incrementally solve the problem if solver_status is given"""
        self.reset(**kwargs)

        yield from self.generate(return_objective=return_objective)
