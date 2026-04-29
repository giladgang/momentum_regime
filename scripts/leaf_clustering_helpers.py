"""Helpers for leaf-membership rule-path analysis.

Functions:
  route_stock_through_tree(x, tree_dict, feature_names) -> leaf_node_id
"""
import numpy as np


def route_stock_through_tree(x, tree_dict, feature_names):
    """Walk a stock's feature vector through a single XGBoost tree-dict.

    Parameters
    ----------
    x : array-like shape (n_features,)
        Standardised feature vector for one stock-month.
    tree_dict : dict
        node_id -> dict. Internal nodes have keys
        {'leaf': False, 'feature', 'threshold', 'yes', 'no'}.
        Leaf nodes have keys {'leaf': True, 'value'}.
    feature_names : list of str
        Maps feature names (used in tree_dict) to indices in x.

    Returns
    -------
    int  Leaf node id where the stock landed.

    Notes
    -----
    The xgboost tree-dict format from `pi_verify_trees_seeds50.pkl` wraps
    each tree as {'has_pi': bool, 'tree': <node_dict>}. Pass the inner
    `tree` dict here (caller unwraps).
    """
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    nid = 0
    while not tree_dict[nid].get('leaf', False):
        node = tree_dict[nid]
        col = name_to_idx[node['feature']]
        nid = node['yes'] if x[col] < node['threshold'] else node['no']
    return nid
