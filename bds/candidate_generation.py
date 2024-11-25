from typing import List

import numpy as np

from bds.fpgrowth import build_fptree, fpgrowth_on_tree, preprocess_transaction_list
from bds.rule import Rule
from bds.utils import compute_truthtable


def extract_rules_with_min_support(
    X: np.ndarray, attribute_names: List[str], min_support: int
) -> List[Rule]:
    """given a binarized feature matrix X, extract all association rules whoe support is at least min_support

    the list of returned rules are sorted by their support in ascending order

    an example:

    ```
    dataset = "compas"
    data = pd.read_csv('data/compas_train-binary.csv')  # the features are binary
    X = data.to_numpy()[:,:-2]
    attribute_names = list(data.columns[:-2])
    sorted_rules = extract_rules_with_min_support(X, attribute_names, min_support=70)
    ```
    """
    X_bag = [{j for j, x in enumerate(row) if x} for row in X]

    ordered_input_data = preprocess_transaction_list(X_bag, min_support)
    tree = build_fptree(ordered_input_data)
    frequent_itemsets = set(fpgrowth_on_tree(tree, set(), min_support))

    # Now create rules
    all_rules = []
    for i, itemset in enumerate(frequent_itemsets):
        rule = Rule(
            id=i,
            name=f"rule_{i}",
            cardinality=len(itemset),
            truthtable=compute_truthtable(X, itemset),
            predicates=[attribute_names[idx] for idx in itemset],
        )
        all_rules.append(rule)

    sorted_rules = list(sorted(all_rules, key=lambda r: r.support, reverse=True))
    for i, rule in enumerate(sorted_rules):
        rule.name = f"rule_{i}"  # rename the rules
        rule.id = i
    return sorted_rules
