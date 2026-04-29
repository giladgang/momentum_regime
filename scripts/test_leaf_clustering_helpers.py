"""Tests for scripts/leaf_clustering_helpers.py.

Run with:
    pytest scripts/test_leaf_clustering_helpers.py -v
"""

import numpy as np
import pytest

from leaf_clustering_helpers import (
    route_stock_through_tree,
    route_all_stocks_through_tree,
    hamming_distance_matrix,
    one_hot_encode_leaves,
    count_leaves,
    leaf_id_remap,
    silhouette_sparse_subsample,
)


# ----------------------------------------------------------------------
# route_stock_through_tree
# ----------------------------------------------------------------------
def test_route_stock_through_tree_simple_split():
    # Tiny tree: root splits on feature 0 at threshold 0; left leaf = -1, right leaf = +1.
    tree = {
        0: {'leaf': False, 'feature': 'f0', 'threshold': 0.0, 'yes': 1, 'no': 2},
        1: {'leaf': True, 'value': -1.0},
        2: {'leaf': True, 'value': +1.0},
    }
    feature_names = ['f0', 'f1']
    # x[0] = -0.5 -> goes 'yes' branch -> leaf node 1
    leaf = route_stock_through_tree(np.array([-0.5, 0.0]), tree, feature_names)
    assert leaf == 1
    # x[0] = +0.5 -> goes 'no' branch -> leaf node 2
    leaf = route_stock_through_tree(np.array([+0.5, 0.0]), tree, feature_names)
    assert leaf == 2


def test_route_stock_through_tree_deeper():
    tree = {
        0: {'leaf': False, 'feature': 'a', 'threshold': 0.0, 'yes': 1, 'no': 2},
        1: {'leaf': False, 'feature': 'b', 'threshold': 5.0, 'yes': 3, 'no': 4},
        2: {'leaf': True, 'value': 1.0},
        3: {'leaf': True, 'value': 2.0},
        4: {'leaf': True, 'value': 3.0},
    }
    fn = ['a', 'b']
    assert route_stock_through_tree(np.array([-1.0, 4.0]), tree, fn) == 3
    assert route_stock_through_tree(np.array([-1.0, 6.0]), tree, fn) == 4
    assert route_stock_through_tree(np.array([+1.0, 0.0]), tree, fn) == 2


def test_route_stock_handles_root_only_leaf():
    tree = {0: {'leaf': True, 'value': 0.5}}
    assert route_stock_through_tree(np.array([0.0]), tree, ['f0']) == 0


def test_route_stock_threshold_boundary_exact_value():
    # x[feature] == threshold should go to 'no' branch (since condition is x < threshold).
    tree = {
        0: {'leaf': False, 'feature': 'f0', 'threshold': 1.0, 'yes': 1, 'no': 2},
        1: {'leaf': True, 'value': -1.0},
        2: {'leaf': True, 'value': +1.0},
    }
    # exact match: x[0] == 1.0 is NOT < 1.0, so goes 'no' (node 2)
    assert route_stock_through_tree(np.array([1.0]), tree, ['f0']) == 2


# ----------------------------------------------------------------------
# hamming_distance_matrix
# ----------------------------------------------------------------------
def test_hamming_distance_matrix_shape():
    L = np.array([[1, 2, 3], [1, 2, 4], [5, 5, 5]])  # 3 rows x 3 trees
    D = hamming_distance_matrix(L)
    assert D.shape == (3, 3)


def test_hamming_distance_matrix_values():
    L = np.array([[1, 2, 3], [1, 2, 4], [5, 5, 5]])
    D = hamming_distance_matrix(L)
    # row 0 vs row 1: 1 mismatch out of 3 -> 1/3
    # row 0 vs row 2: 3 mismatches out of 3 -> 1
    # row 1 vs row 2: 3 mismatches out of 3 -> 1
    assert D[0, 0] == pytest.approx(0.0)
    assert D[0, 1] == pytest.approx(1 / 3)
    assert D[0, 2] == pytest.approx(1.0)
    assert D[1, 2] == pytest.approx(1.0)
    assert D[1, 0] == D[0, 1]  # symmetric


def test_hamming_distance_zero_for_identical_rows():
    L = np.tile([1, 2, 3, 4], (5, 1))
    D = hamming_distance_matrix(L)
    assert (D == 0).all()


# ----------------------------------------------------------------------
# one_hot_encode_leaves
# ----------------------------------------------------------------------
def test_one_hot_encode_leaves_shape():
    L = np.array([[0, 0], [1, 0], [2, 1]])
    n_leaves_per_tree = [3, 2]
    X = one_hot_encode_leaves(L, n_leaves_per_tree)
    assert X.shape == (3, sum(n_leaves_per_tree))


def test_one_hot_encode_leaves_correctness():
    L = np.array([[0, 0], [1, 0], [2, 1]])
    n_leaves_per_tree = [3, 2]
    X = one_hot_encode_leaves(L, n_leaves_per_tree).toarray()
    # Row 0: leaf 0 in tree 0 -> col 0; leaf 0 in tree 1 -> col 3
    assert (X[0] == np.array([1, 0, 0, 1, 0])).all()
    # Row 1: leaf 1 in tree 0 -> col 1; leaf 0 in tree 1 -> col 3
    assert (X[1] == np.array([0, 1, 0, 1, 0])).all()
    # Row 2: leaf 2 in tree 0 -> col 2; leaf 1 in tree 1 -> col 4
    assert (X[2] == np.array([0, 0, 1, 0, 1])).all()


def test_one_hot_encode_leaves_row_sum_equals_n_trees():
    # Each row should have exactly k 1's (one leaf per tree).
    L = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    n_leaves_per_tree = [3, 4, 5]
    X = one_hot_encode_leaves(L, n_leaves_per_tree)
    row_sums = np.asarray(X.sum(axis=1)).flatten()
    assert (row_sums == 3).all()  # 3 trees -> 3 ones per row


# ----------------------------------------------------------------------
# count_leaves and leaf_id_remap
# ----------------------------------------------------------------------
def test_count_leaves_simple():
    tree = {
        0: {'leaf': False, 'feature': 'a', 'threshold': 0, 'yes': 1, 'no': 2},
        1: {'leaf': True, 'value': 1},
        2: {'leaf': True, 'value': 2},
    }
    assert count_leaves(tree) == 2


def test_count_leaves_root_only_leaf():
    assert count_leaves({0: {'leaf': True, 'value': 0}}) == 1


def test_count_leaves_deeper_tree():
    tree = {
        0: {'leaf': False, 'feature': 'a', 'threshold': 0, 'yes': 1, 'no': 2},
        1: {'leaf': False, 'feature': 'b', 'threshold': 5, 'yes': 3, 'no': 4},
        2: {'leaf': True, 'value': 1},
        3: {'leaf': True, 'value': 2},
        4: {'leaf': True, 'value': 3},
    }
    assert count_leaves(tree) == 3


def test_leaf_id_remap_assigns_dense_indices():
    # Tree where leaf node ids are non-contiguous: 1, 4, 6
    tree = {
        0: {'leaf': False, 'feature': 'a', 'threshold': 0, 'yes': 1, 'no': 2},
        1: {'leaf': True, 'value': 1},
        2: {'leaf': False, 'feature': 'a', 'threshold': 1, 'yes': 4, 'no': 6},
        4: {'leaf': True, 'value': 2},
        6: {'leaf': True, 'value': 3},
    }
    remap = leaf_id_remap(tree)
    assert sorted(remap.values()) == [0, 1, 2]
    assert set(remap.keys()) == {1, 4, 6}


def test_leaf_id_remap_root_only_leaf():
    remap = leaf_id_remap({0: {'leaf': True, 'value': 0}})
    assert remap == {0: 0}


# ----------------------------------------------------------------------
# route_all_stocks_through_tree
# ----------------------------------------------------------------------
def test_route_all_stocks_matches_per_stock():
    # Vectorised batch routing must match per-stock routing on every row.
    tree = {
        0: {'leaf': False, 'feature': 'a', 'threshold': 0.0, 'yes': 1, 'no': 2},
        1: {'leaf': False, 'feature': 'b', 'threshold': 5.0, 'yes': 3, 'no': 4},
        2: {'leaf': True, 'value': 1.0},
        3: {'leaf': True, 'value': 2.0},
        4: {'leaf': True, 'value': 3.0},
    }
    fn = ['a', 'b']
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 2)) * 5  # spread across all branches
    expected = np.array([route_stock_through_tree(X[i], tree, fn) for i in range(100)])
    actual = route_all_stocks_through_tree(X, tree, fn)
    assert (actual == expected).all()


def test_route_all_stocks_root_only_leaf():
    # Tree that's just a root leaf — every stock should land at node 0.
    X = np.random.default_rng(0).standard_normal((20, 3))
    leaves = route_all_stocks_through_tree(X, {0: {'leaf': True, 'value': 0.5}}, ['a', 'b', 'c'])
    assert (leaves == 0).all()


def test_route_all_stocks_threshold_boundary():
    # x < threshold -> yes, x >= threshold -> no. Exact-match goes 'no'.
    tree = {
        0: {'leaf': False, 'feature': 'a', 'threshold': 1.0, 'yes': 1, 'no': 2},
        1: {'leaf': True, 'value': -1.0},
        2: {'leaf': True, 'value': +1.0},
    }
    X = np.array([[0.5], [1.0], [1.5]])
    leaves = route_all_stocks_through_tree(X, tree, ['a'])
    assert leaves[0] == 1   # 0.5 < 1.0  -> yes
    assert leaves[1] == 2   # 1.0 == 1.0 -> no (not strictly less)
    assert leaves[2] == 2   # 1.5 > 1.0  -> no


# ----------------------------------------------------------------------
# silhouette_sparse_subsample
# ----------------------------------------------------------------------
def test_silhouette_sparse_subsample_returns_float_in_range():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
    sil = silhouette_sparse_subsample(X, km.labels_, n_samples=50, seed=42)
    assert isinstance(sil, float)
    assert -1.0 <= sil <= 1.0


def test_silhouette_sparse_subsample_well_separated_blobs():
    # Two clearly separated 5-d blobs — silhouette should be > 0.5.
    rng = np.random.default_rng(0)
    A = rng.standard_normal((50, 5)) + np.array([10, 0, 0, 0, 0])
    B = rng.standard_normal((50, 5)) + np.array([-10, 0, 0, 0, 0])
    X = np.vstack([A, B])
    labels = np.array([0] * 50 + [1] * 50)
    sil = silhouette_sparse_subsample(X, labels, n_samples=80, seed=42)
    assert sil > 0.5


def test_silhouette_sparse_subsample_is_deterministic_for_same_seed():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X)
    s1 = silhouette_sparse_subsample(X, km.labels_, n_samples=50, seed=42)
    s2 = silhouette_sparse_subsample(X, km.labels_, n_samples=50, seed=42)
    assert s1 == s2


def test_silhouette_sparse_subsample_handles_n_samples_geq_n():
    # If n_samples >= n, should compute silhouette on all rows (no subsample).
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4))
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
    sil_full = silhouette_sparse_subsample(X, km.labels_, n_samples=1000, seed=42)
    from sklearn.metrics import silhouette_score
    sil_ref = float(silhouette_score(X, km.labels_, metric='euclidean'))
    assert sil_full == pytest.approx(sil_ref)
