"""Helpers for leaf-membership rule-path analysis.

Functions:
  route_stock_through_tree(x, tree_dict, feature_names) -> leaf_node_id
  hamming_distance_matrix(L) -> pairwise normalised Hamming distances
  one_hot_encode_leaves(L, n_leaves_per_tree) -> sparse one-hot matrix
"""
import numpy as np
from scipy import sparse


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


def one_hot_encode_leaves(L, n_leaves_per_tree):
    """Sparse one-hot encoding of leaf indices.

    Parameters
    ----------
    L : array (n, k)
        Integer leaf indices: ``L[i, t] in [0, n_leaves_per_tree[t])``.
    n_leaves_per_tree : array-like of length k
        Number of leaves in each tree.

    Returns
    -------
    scipy.sparse.csr_matrix shape (n, sum(n_leaves_per_tree))
        Each row has exactly k ones (one per tree).
    """
    L = np.asarray(L)
    n, k = L.shape
    offsets = np.cumsum([0] + list(n_leaves_per_tree[:-1]))
    rows = np.repeat(np.arange(n), k)
    cols = (L + offsets).ravel()
    data = np.ones(n * k, dtype=np.int8)
    n_features = int(sum(n_leaves_per_tree))
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n_features))


def hamming_distance_matrix(L):
    """Pairwise normalised Hamming distance between rows of L.

    Parameters
    ----------
    L : array shape (n, k)
        Integer leaf-index matrix (n stock-months, k trees).

    Returns
    -------
    array (n, n)
        D[i, j] = mean of (L[i] != L[j]) over k trees.
        Symmetric, with zeros on the diagonal, values in [0, 1].
    """
    L = np.asarray(L)
    n, _ = L.shape
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        D[i] = (L != L[i]).mean(axis=1)
    return D
