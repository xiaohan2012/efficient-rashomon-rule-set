from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union



@dataclass
class NodeST:
    rule_id: str
    lower_bound: float
    objective: float
    num_captured: int
    # TODO: should I store not captured?
    equivalent_minority: float = 0

    children: Dict[str, "NodeST"] = field(default_factory=dict)
    depth: int = 0
    parent: Optional["NodeST"] = None

    equivalent_lower_bound: Optional[
        float
    ] = None  # it should default to none if not given

    pivot_rule_ids: List[int] = field(
        default_factory=list
    )  # a list of ids of the pivod rules that are added due to the addition of rule_id

    def __post_init__(self):
        """if parent is not None, 'bind' self and parent by upting children and depth accordingly"""
        if self.parent is not None:
            assert isinstance(self.parent, NodeST)
            self.parent.add_child(self)
        self._num_rules = None

    def _get_num_rules(self):
        """return and update the num_rules of the node"""
        if self._num_rules is not None:
            # computed already
            return self._num_rules

        # otherwise, compute from scratch
        if self.parent is None:
            num_rules = len(self.pivot_rule_ids)
        else:
            num_rules = self.parent._get_num_rules() + len(self.pivot_rule_ids) + 1

        # cache the result
        self._num_rules = num_rules
        return self._num_rules

    @property
    def num_rules(self):
        """lazily calculate the number of rules"""
        if self._num_rules is None:
            self._num_rules = self._get_num_rules()

        return self._num_rules

    @property
    def num_children(self):
        return len(self.children)

    @property
    def total_num_nodes(self):
        """return the total number of nodes in the tree rooted at self"""
        if self.num_children == 0:
            # leaf
            return 1
        else:
            return 1 + sum([c.total_num_nodes for c in self.children.values()])

    def add_child(self, child: "NodeST"):
        if child == self:
            raise ValueError('cannot add "self" as a child of itself')

        if child.rule_id in self.children:
            raise KeyError(f"{child} is already a child!")
        # two-way binding
        self.children[child.rule_id] = child
        child.parent = self
        # update depth of the child
        child.depth = self.depth + 1

    def get_ruleset_ids(self):
        """get the rule ids of the rule set associated with this node/rule"""
        ret = {self.rule_id} | set(self.pivot_rule_ids)
        #print(ret) 
        if self.parent is not None:
            ret |= self.parent.get_ruleset_ids()
        return ret

    @classmethod
    def make_root(cls, fnr: float, num_train_pts: int) -> "NodeST":
        
        
        """create the root of a cache tree

        fnr: the false positive rate of the default rule which captures all training pts and predict them to be negative
        num_train_pts: number of training points

        the root corresponds to the "default rule", which captures all points and predicts the default label (negative)
        """
        
        return NodeST(
            rule_id="0",
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



class CacheTreeST:
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

    def _set_root(self, node: NodeST):
        if self._root is not None:
            raise ValueError("Root has already been set!")
        self._root = node

    def add_node(self, node: NodeST, parent: Optional[NodeST] = None):
        """add node as a child (to parent if it is given)"""
        if parent is not None:
            parent.add_child(node)
        else:
            self._set_root(node)

        self._num_nodes += 1

