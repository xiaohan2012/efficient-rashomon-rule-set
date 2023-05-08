"""
an implementation of the FPGrowth algorithm

example usage:

```python
    input_transactions = [{0, 1, 2}, {0, 1, 3}]  # the input transactions, represented as a list of sets
    min_support = 1
    ordered_input_data = preprocess_transaction_list(
        input_transactions, min_support
    )

    tree = build_fptree(ordered_input_data)
    frequent_itemsets = set(fpgrowth_on_tree(tree, set(), min_support))
```
"""
from dataclasses import dataclass
from collections import OrderedDict, defaultdict, Counter

from typing import List, Set, Optional, Dict, Tuple

from .utils import powerset

Item = int
Freq = int
ItemFreq = Tuple[Item, Freq]
Transaction = Set[Item]
TransactionDict = Dict[Item, Freq]
TransactionDictList = List[TransactionDict]
OrderedTransaction = List[Tuple[Item, Freq]]
Database = List[Transaction]

TransactionList = List[Transaction]
ROOT = -1


HeaderTable = Dict[Item, List["FPTree"]]


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
    def is_root(self):
        return self.parent is None

    def all_names(self, exclude_root=False):
        if self.is_leaf:
            return {self.name}
        else:
            ret = set() if self.is_root and exclude_root else {self.name}
            for c in self.children:
                ret |= c.all_names(exclude_root=exclude_root)
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
    def item_count_on_path_to_root(self) -> List[Tuple[Item, int]]:
        """
        traversing the path from the current node to the root (excluding the root)
        and collect (item, count) information

        note that root is excluded
        """
        if self.is_root:
            return []
        elif self.parent is not None:
            return self.parent.item_count_on_path_to_root + [(self.name, self.count)]
        else:
            return [(self.name, self.count)]

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

    def as_nested_tuple(self):
        """
        return the tree as a nested tuple

        ((name, count), (child0, child1, ...))
        """
        ret = ((self.name, self.count), list())
        for child in self.children:
            ret[1].append(child.as_nested_tuple())
        return ret


def insert_node(head: Tuple[Item, int], remain, tree):
    """insert an itemset (head + remain) into the tree

    head is the first item in the itemset, together with its frequency
    """
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
        # recurse on
        insert_node(remain[0], remain[1:], child)


# build the FPtree
def build_fptree(preprocessed_trans_list: List[OrderedTransaction]) -> FPTree:
    """
    build a FPtree for a list of preprocessed transactions

    the input transaction list should already filter out infrequent items
    and items in each transaction should be ordered by their frequency in descending order
    """
    root = FPTree(name="ROOT")

    for trans in preprocessed_trans_list:
        # print("trans: ", trans)
        if len(trans) > 0:
            insert_node(trans[0], trans[1:], root)

    return root


def build_header_table(tree):
    """
    build the header table for a tree,

    a header table is a dict of item name to the list of nodes with the same item name
    """
    header_table = defaultdict(list)

    def aux(node):
        for child in node.children:
            header_table[child.name].append(child)
            aux(child)

    aux(tree)
    return header_table


def _add_frequency_to_transaction_list(
    trans_list: TransactionList,
) -> TransactionDictList:
    """
    add frequency information to each item in each transaction

    by converting each transaction (a list) to a dict
    """
    trans_list_with_freq = []
    for trans in trans_list:
        new_trans = dict([(item, 1) for item in trans])
        trans_list_with_freq.append(new_trans)
    return trans_list_with_freq


def _extract_frequent_items_and_order(
    trans_list_with_freq: List[TransactionDict], min_support: int
) -> List[Item]:
    """for each transaction, extract items whose frequency to at least min_support

    and order them by their frequency in descending order
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


def _filter_and_reorder_transaction_list(
    trans_list_with_freq: List[TransactionDict], ordered_frequent_items: List[Item]
) -> List[OrderedTransaction]:
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


def preprocess_transaction_list(
    trans_list: List[Transaction], min_support: int
) -> List[OrderedTransaction]:
    """given a list of "raw" transactions and minimum support,

    convert each transaction into an ordered form, where only the frequent items are included
    and they are furthered ordered by their frequency
    """
    transactions_with_freq = _add_frequency_to_transaction_list(trans_list)

    frequent_items = _extract_frequent_items_and_order(
        transactions_with_freq, min_support
    )

    ordered_input_data = _filter_and_reorder_transaction_list(
        transactions_with_freq, frequent_items
    )

    return ordered_input_data


def squash_path_by_leaf_frequency(path: List[ItemFreq]) -> List[ItemFreq]:
    """
    given a list of item, count pairs, change the count of each entry  to that of the last entry in the list

    (('f', 4), ('c', 3), ('a', 3), ('m', 2), ('p', 2))

    ->

    (('f', 2), ('c', 2), ('a', 2), ('m', 2), ('p', 2))
    """
    tail_freq = path[-1][1]
    return tuple((item[0], tail_freq) for item in path)


def fpgrowth_on_tree(tree: FPTree, prefix: set, min_support: int):
    """run fpgrowth on a FPTree and return the frequent itemsets"""
    if tree.is_path:
        # enumerate all combinations
        all_names = set(prefix) | tree.all_names(exclude_root=True)
        # print("prefix: ", prefix)
        # print("tree: \n", tree)
        # return list(
        yield from map(lambda seq: tuple(sorted(seq)), powerset(all_names, min_size=1))
        # )
        # for frequent_itemset in powerset(all_names, min_size=1):
        #     yield frequent_itemset
    else:
        header_table = build_header_table(tree)
        # print("header_table: ", header_table)
        # construct conditional tree
        for item in header_table.keys():
            paths = [node.item_count_on_path_to_root for node in header_table[item]]
            cond_trans_list = list(map(squash_path_by_leaf_frequency, paths))

            cond_trans_list = list(map(dict, cond_trans_list))  # turn to dict

            cond_frequent_items = _extract_frequent_items_and_order(
                cond_trans_list, min_support=min_support
            )

            cond_preprocessed_trans_list = _filter_and_reorder_transaction_list(
                cond_trans_list, cond_frequent_items
            )

            # print("prefix: ", prefix)
            # print("cond_preprocessed_trans_list: ", cond_preprocessed_trans_list)
            cond_tree = build_fptree(cond_preprocessed_trans_list)

            # cond_header_table = build_header_table(tree)
            yield from fpgrowth_on_tree(cond_tree, prefix | {item}, min_support)
