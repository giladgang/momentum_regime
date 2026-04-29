"""Tests for scripts/leaf_clustering_helpers.py.

Run with:
    pytest scripts/test_leaf_clustering_helpers.py -v
"""

import numpy as np
import pytest

from leaf_clustering_helpers import route_stock_through_tree


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
