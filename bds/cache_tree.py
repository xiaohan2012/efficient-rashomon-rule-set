import numpy as np

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from .rule import RuleSet


@dataclass
class Node:
    node_id: int
    lower_bound: float
    objective: float
    depth: int
    num_captured: int
    equivalent_minority: float = 0
    children: Dict[int, "Node"] = field(default_factory=list)
    parent: Optional["Node"] = None

    @property
    def num_children(self):
        return len(self.children)

    def get_ruleset(self):
        """get the rule set or prefix associated with this node/rule"""
        return RuleSet

    @classmethod
    def get_root(cls, captured: np.ndarray):
        return Node(
            node_id=0,
            lower_bound=0.0,
            objective=0.0,
            depth=0,
            num_captured=captured.sum(),
        )


class CacheTree:
    """a prefix tree"""

    def __init__(self):
        self._num_nodes = 0

        self._root = None

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def root(self):
        if self._root is None:
            raise ValueError("root is not set yet")
        self._root

    def _set_root(self, node: Node):
        assert (
            self._root is None
        ), "Root has already been set! Do not do it again on the same tree."
        self._root = node

    def add_node(self, node: Node, parent: Optional[Node]):
        """add node as a child to parent"""
        node.parent = parent
        parent.children[node.node_id] = node

        if parent is None:
            self._set_root(node)

        self._num_nodes += 1
