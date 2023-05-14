from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class Node:
    rule_id: int
    lower_bound: float
    objective: float
    num_captured: int
    # TODO: should I store not captured?
    equivalent_minority: float = 0

    children: Dict[int, "Node"] = field(default_factory=dict)
    depth: int = 0
    parent: Optional["Node"] = None
    equivalent_lower_bound: Optional[
        float
    ] = None  # it should default to none if not given

    def __post_init__(self):
        """if parent is not None, 'bind' self and parent by upting children and depth accordingly"""
        if self.parent is not None:
            assert isinstance(self.parent, Node)
            self.parent.add_child(self)
            self.depth = self.parent.depth + 1

    @property
    def num_rules(self):
        return self.depth

    @property
    def num_children(self):
        return len(self.children)

    def add_child(self, node: "Node"):
        if node == self:
            raise ValueError('cannot add "self" as a child of itself')

        if node.rule_id in self.children:
            raise KeyError(f"{node} is already a child!")
        self.children[node.rule_id] = node
        node.parent = self

    def get_ruleset_ids(self):
        """get the rule ids of the rule set associated with this node/rule"""
        ret = {self.rule_id}
        if self.parent is not None:
            ret |= self.parent.get_ruleset_ids()
        return ret

    @classmethod
    def make_root(cls, fnr: float, num_train_pts: int) -> "Node":
        """create the root of a cache tree

        fnr: the false positive rate of the default rule which captures all training pts and predict them to be negative
        num_train_pts: number of training points

        the root corresponds to the "default rule", which captures all points and predicts the default label (negative)
        """
        return Node(
            rule_id=0,
            lower_bound=0.0,  # the false positive rate, which is zero
            objective=fnr,  # the false negative rate, the complexity is zero since the default rule does not add into the complexity term
            depth=0,
            num_captured=num_train_pts,
            parent=None,
        )

    def __eq__(self, other):
        """two nodes are equal if:

        - they have exactly the same attribute values, including parent
        - the above equiality condition carries to the parents (and recursively)
        """
        if type(other) != type(self):
            return False

        attributes_to_compare_directly = [
            "rule_id",
            "lower_bound",
            "objective",
            "depth",
            "num_captured",
            "equivalent_minority",
        ]
        for attr_name in attributes_to_compare_directly:
            if getattr(self, attr_name) != getattr(other, attr_name):
                return False

        if (self.parent is None and other.parent is not None) or (
            self.parent is not None and other.parent is None
        ):
            return False

        if self.parent is not None and other.parent is not None:
            return self.parent == other.parent

        return True

    def __gt__(self, other):
        if type(other) != type(self):
            raise TypeError(f"{other} is not of type {type(self)}")
        return self.rule_id > other.rule_id

    def __repr__(self):
        return f"Node(rule_id={self.rule_id}, lower_bound={self.lower_bound}, objective={self.objective})"


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
        return self._root

    def _set_root(self, node: Node):
        if self._root is not None:
            raise ValueError("Root has already been set!")
        self._root = node

    def add_node(self, node: Node, parent: Optional[Node] = None):
        """add node as a child (to parent if it is given)"""
        if parent is not None:
            parent.add_child(node)
        else:
            self._set_root(node)

        self._num_nodes += 1
