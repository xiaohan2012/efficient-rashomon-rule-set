from dataclasses import dataclass
from collections import OrderedDict, defaultdict, Counter
from typing import List, Set, Optional, Dict

Item = int
Transaction = Set[Item]
OrderedTransaction = OrderedDict
Database = List[Transaction]

ROOT = -1



HeaderTable = Dict[Item, List['FPTree']]

@dataclass
class FPTree:
    name: str  # the item index
    count: int = 0
    parent: Optional["FPTree"] = None
    _children: Optional[List["FPTree"]] = None

    @property
    def is_path(self):
        if self.is_leaf:
            return True
        elif len(self.children) == 1:
            return self.children[0].is_path
        else:
            return False

    @property
    def all_names(self):
        if self.is_leaf:
            return {self.name}
        else:
            ret = {self.name}
            for c in self.children:
                ret |= c.all_names
            return ret

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def path_to_root(self):
        if self.parent is not None:
            return self.parent.path_to_root + (self,)
        else:
            return tuple()

    @property
    def item_count_on_path_to_root(self):
        if self.parent is not None:
            return self.parent.item_count_on_path_to_root + ((self.name, self.count),)
        else:
            return ((self.name, self.count), )

    @property
    def children(self):
        if self._children is None:
            return []
        else:
            return self._children

    def add_child(self, child: "FPTree"):
        if self._children is None:
            self._children = []

        self._children.append(child)
        child.parent = self
        return child

    def __str__(self, level=0):
        ret = "  " * level + repr(self) + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return "{}:{}".format(self.name, self.count)


def insert_tree(head, remain, tree):
    """insert an itemset (head + remain) into the tree"""
    # if the node has a child with the same name
    head_name, head_count = head
    for child in tree.children:
        if child.name == head_name:
            child.count += head_count
            break

    else:
        # otherwise, create a new child
        child = FPTree(name=head_name, count=head_count, parent=tree)
        tree.add_child(child)

    if len(remain) > 0:
        insert_tree(remain[0], remain[1:], child)


# build the FPtree
def build_fptree(preprocessed_trans_list):
    root = FPTree(name="ROOT")

    for trans in preprocessed_trans_list:
        insert_tree(trans[0], trans[1:], root)

    return root


def build_header_table(tree):
    """
    a header table is a dict of item name to the list of nodes with the same item name
    """
    header_table = defaultdict(list)

    def aux(node):
        for child in node.children:
            header_table[child.name].append(child)
            aux(child)

    aux(tree)
    return header_table


def add_frequency_to_transaction_list(trans_list):
    """
    add frequency information to each item in each transaction

    by converting each transaction (a list) to a dict
    """
    trans_list_with_freq = []
    for trans in trans_list:
        new_trans = dict([(item, 1) for item in trans])
        trans_list_with_freq.append(new_trans)
    return trans_list_with_freq


def extract_frequent_items(trans_list_with_freq, min_support) -> List:
    """extract items whose frequency to at least min_support

    return the frequent items ordered by their frequency in descending order
    """
    # extract frequent items
    item_counts = Counter()
    for trans in trans_list_with_freq:
        for item, count in trans.items():
            item_counts[item] += count

    frequent_items = filter(
        lambda item: item_counts[item] >= min_support, item_counts.keys()
    )
    ordered_frequent_items = sorted(
        frequent_items, key=lambda item: item_counts[item], reverse=True
    )
    return ordered_frequent_items


def filter_and_reorder_transaction_list(trans_list_with_freq, ordered_frequent_items: List):
    """
    convert the transactions to ordered and filtered transactions
    """
    preprocessed_trans_list = []
    for trans_with_freq in trans_list_with_freq:
        # order and filter the items in each transaction
        preprocessed_trans = [
            (item, trans_with_freq[item])
            for item in ordered_frequent_items
            if item in trans_with_freq
        ]
        preprocessed_trans_list.append(preprocessed_trans)
    return preprocessed_trans_list

def squash_path_by_leaf_frequency(path):
    """
    given a list of item, count pairs, change the count of each entry  to that of the last entry in the list
    
    (('f', 4), ('c', 3), ('a', 3), ('m', 2), ('p', 2))

    ->

    (('f', 2), ('c', 2), ('a', 2), ('m', 2), ('p', 2))
    """
    tail_freq = path[-1][1]
    return tuple((item[0], tail_freq) for item in path)
