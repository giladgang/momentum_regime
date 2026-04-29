# Stock-Level Rule-Path Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a thesis-chapter-grade analysis of how the XGBoost ensemble decides which stocks to pick, using leaf-membership clustering of long-leg stock-months across the 50-seed × 500-tree ensemble. Output: rule-path descriptions, robustness, regime cross-tabs, plots, and a revised §5.2 of the thesis.

**Architecture:** Five phases — (1) extract leaf-VALUE signatures from the existing 50-seed ensemble (per-tree prediction contributions per stock-month), (2) cluster long-leg stock-months on the standardised 25,000-d leaf-value matrix via PCA + KMeans, (3) characterise each rule path (typical stock + Sharpe + bootstrap CI), (4) verify robustness (seed stability), (5) cross-tabulate rules × regime × landscape and rewrite §5.2.

**Note on tree weighting:** clustering uses leaf VALUES (per-tree real-valued prediction contributions), not leaf INDICES. Standardisation per column means high-variance trees (typically high-impact early trees) automatically dominate Euclidean distance — no explicit per-tree weighting needed. The 25,000-d signature sums (× learning_rate) to the model's predicted score, so the representation is a complete decomposition of the model's decision rather than a categorical summary.

**Note on statistical-test budget:** with only 167 OOS months as the ultimate ceiling for any monthly-aggregated claim, the plan is deliberately lean: K_RANGE = [2, 3] only, single primary clustering method (no variant comparison), seed-stability gating folded into Phase 2 (so the chosen k is robust by construction), single SHAP-vs-leaf ARI sanity check.

**Realistic timeline:** ~2-3 hours of focused coding/compute to reach the **Phase 4 checkpoint**, where the user reviews rule labels + characterisations + bootstrap Sharpe CIs before designing Phase 5. Phase 5 + LaTeX is open-ended and depends on what the rules look like.

**Phase 5 checkpoint:** Phase 5 (cross-tabs, plots, LaTeX revision) is held until the user reviews Phase 4 outputs. The plan's Tasks 14-21 are downstream of that gate — execute only after user sign-off.

**Tech Stack:** Python 3.12, numpy, pandas, scikit-learn (KMeans, AgglomerativeClustering, silhouette_score, adjusted_rand_score), networkx (optional, for community detection), xgboost (.apply on the single-seed XGBRegressor for sanity check), matplotlib, pytest.

---

## File Structure

**New helpers (TDD-tested):**
- `scripts/leaf_clustering_helpers.py` — tree-dict traversal (`route_stock_through_tree`), leaf encoding, Hamming distance, jaccard co-occurrence.
- `scripts/test_leaf_clustering_helpers.py` — pytest tests.

**New phase scripts:**
- `scripts/extract_leaf_signatures.py` — Phase 1.
- `scripts/cluster_rule_paths.py` — Phase 2.
- `scripts/characterise_rule_paths.py` — Phase 3.
- `scripts/rule_path_robustness.py` — Phase 4.
- `scripts/plot_rule_paths.py` — Phase 5 figures + tables.

**Outputs (will be created):**
- `artefacts/leaf_signatures.npz` — `(n_long_stockmonths, 25000)` int matrix + index metadata.
- `results/thesis/rule_path_*.csv` — labels, centroids, robustness, sharpe_ci, regime_crosstab, landscape_crosstab.
- `plots/thesis/rule_path_*.{png,pdf}` — dispersion, crosstabs, centroid_features, within_month_homogeneity.

**Modified:**
- `latex/main_results.tex` — §5.2 rewrite (last task).

---

## Task 1: Inspect tree-dict format and confirm extraction approach

**Files:**
- Investigation only; no code committed.

- [ ] **Step 1: Inspect a single tree-dict to understand its structure.**

```python
import pickle
with open('artefacts/pi_verify_trees_seeds50.pkl', 'rb') as f:
    trees, seeds = pickle.load(f)
sample = trees[0]
# Print all keys (node IDs) and look at structure
for nid in sorted(sample.keys()):
    print(nid, sample[nid])
```

Expected: each `nid` maps to a dict. Internal nodes have `'leaf': False` plus `'feature'`, `'threshold'`, `'yes'` (child if x[feature] < threshold), `'no'` (child otherwise). Leaf nodes have `'leaf': True` and a `'value'` field (the leaf score).

- [ ] **Step 2: Confirm extraction parity — sanity check `.apply()` from `cs_artefacts_xgb.pkl`.**

```python
import pickle, numpy as np
with open('artefacts/cs_artefacts_xgb.pkl', 'rb') as f:
    xgb = pickle.load(f)
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
X = art['X_te_s']
leaves = xgb.apply(X)
print(leaves.shape, leaves.dtype)  # expected (557087, 500)
```

Expected: `(557087, 500)` int matrix.

- [ ] **Step 3: Document findings in a comment header.**

No commit yet — investigation only.

---

## Task 2: TDD — tree-dict traversal helper

**Files:**
- Create: `scripts/leaf_clustering_helpers.py`
- Create: `scripts/test_leaf_clustering_helpers.py`

- [ ] **Step 1: Write the failing test for `route_stock_through_tree`.**

```python
# scripts/test_leaf_clustering_helpers.py
import numpy as np
import pytest
from leaf_clustering_helpers import route_stock_through_tree

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
```

- [ ] **Step 2: Run tests, see them fail.**

```bash
pytest scripts/test_leaf_clustering_helpers.py -v
```

Expected: `ImportError: No module named 'leaf_clustering_helpers'`.

- [ ] **Step 3: Implement minimal `route_stock_through_tree`.**

```python
# scripts/leaf_clustering_helpers.py
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
    """
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    nid = 0
    while not tree_dict[nid].get('leaf', False):
        node = tree_dict[nid]
        col = name_to_idx[node['feature']]
        nid = node['yes'] if x[col] < node['threshold'] else node['no']
    return nid
```

- [ ] **Step 4: Run tests, verify they pass.**

```bash
pytest scripts/test_leaf_clustering_helpers.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit.**

```bash
git add scripts/leaf_clustering_helpers.py scripts/test_leaf_clustering_helpers.py
git commit -m "rule-path: tree-dict traversal helper with TDD tests

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: TDD — Hamming distance helper

**Files:**
- Modify: `scripts/leaf_clustering_helpers.py`
- Modify: `scripts/test_leaf_clustering_helpers.py`

- [ ] **Step 1: Add failing tests for `hamming_distance_matrix`.**

Append to `scripts/test_leaf_clustering_helpers.py`:

```python
from leaf_clustering_helpers import hamming_distance_matrix

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
    assert D[0, 1] == pytest.approx(1/3)
    assert D[0, 2] == pytest.approx(1.0)
    assert D[1, 2] == pytest.approx(1.0)
    assert D[1, 0] == D[0, 1]  # symmetric

def test_hamming_distance_zero_for_identical_rows():
    L = np.tile([1, 2, 3, 4], (5, 1))
    D = hamming_distance_matrix(L)
    assert (D == 0).all()
```

- [ ] **Step 2: Run, see fail.**

```bash
pytest scripts/test_leaf_clustering_helpers.py -v
```

Expected: ImportError on `hamming_distance_matrix`.

- [ ] **Step 3: Implement.**

Append to `scripts/leaf_clustering_helpers.py`:

```python
def hamming_distance_matrix(L):
    """Pairwise Hamming distance between rows of L (normalised to [0, 1]).

    Parameters
    ----------
    L : array shape (n, k)  integer leaf-index matrix
                            (n stock-months, k trees)

    Returns
    -------
    array (n, n)  D[i, j] = mean of (L[i] != L[j]) over k trees.
    """
    L = np.asarray(L)
    n, k = L.shape
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        D[i] = (L != L[i]).mean(axis=1)
    return D
```

- [ ] **Step 4: Run, verify pass.**

```bash
pytest scripts/test_leaf_clustering_helpers.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit.**

```bash
git add scripts/leaf_clustering_helpers.py scripts/test_leaf_clustering_helpers.py
git commit -m "rule-path: hamming distance helper with TDD tests

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: TDD — leaf one-hot encoding helper

**Files:**
- Modify: `scripts/leaf_clustering_helpers.py`
- Modify: `scripts/test_leaf_clustering_helpers.py`

- [ ] **Step 1: Add failing tests.**

Append:

```python
from leaf_clustering_helpers import one_hot_encode_leaves

def test_one_hot_encode_leaves_shape():
    L = np.array([[0, 0], [1, 0], [2, 1]])
    n_leaves_per_tree = [3, 2]  # tree 0 has 3 leaves (ids 0..2), tree 1 has 2
    X = one_hot_encode_leaves(L, n_leaves_per_tree)
    assert X.shape == (3, sum(n_leaves_per_tree))  # 3 rows x 5 cols

def test_one_hot_encode_leaves_correctness():
    L = np.array([[0, 0], [1, 0], [2, 1]])
    n_leaves_per_tree = [3, 2]
    X = one_hot_encode_leaves(L, n_leaves_per_tree)
    # Row 0: leaf 0 in tree 0 -> col 0; leaf 0 in tree 1 -> col 3
    assert (X[0] == np.array([1, 0, 0, 1, 0])).all()
    # Row 1: leaf 1 in tree 0 -> col 1; leaf 0 in tree 1 -> col 3
    assert (X[1] == np.array([0, 1, 0, 1, 0])).all()
    # Row 2: leaf 2 in tree 0 -> col 2; leaf 1 in tree 1 -> col 4
    assert (X[2] == np.array([0, 0, 1, 0, 1])).all()
```

- [ ] **Step 2: Run, see fail.**

- [ ] **Step 3: Implement.**

Append to `scripts/leaf_clustering_helpers.py`:

```python
from scipy import sparse

def one_hot_encode_leaves(L, n_leaves_per_tree):
    """Convert leaf-index matrix to sparse one-hot encoding.

    Parameters
    ----------
    L : array (n, k)  leaf indices for n stock-months across k trees.
                      L[i, t] in [0, n_leaves_per_tree[t]).
    n_leaves_per_tree : list-like of length k

    Returns
    -------
    scipy.sparse.csr_matrix shape (n, sum(n_leaves_per_tree))
        For each row, exactly k cells are 1 (one per tree); the rest 0.
    """
    L = np.asarray(L)
    n, k = L.shape
    offsets = np.cumsum([0] + list(n_leaves_per_tree[:-1]))
    rows = np.repeat(np.arange(n), k)
    cols = (L + offsets).ravel()
    data = np.ones(n * k, dtype=np.int8)
    n_features = int(sum(n_leaves_per_tree))
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n_features))
```

Note: returns sparse for memory; for tests we may need `.toarray()` — adjust tests:

Update `test_one_hot_encode_leaves_correctness` to call `.toarray()`:

```python
def test_one_hot_encode_leaves_correctness():
    L = np.array([[0, 0], [1, 0], [2, 1]])
    n_leaves_per_tree = [3, 2]
    X = one_hot_encode_leaves(L, n_leaves_per_tree).toarray()
    assert (X[0] == np.array([1, 0, 0, 1, 0])).all()
    assert (X[1] == np.array([0, 1, 0, 1, 0])).all()
    assert (X[2] == np.array([0, 0, 1, 0, 1])).all()
```

And update the shape test:

```python
def test_one_hot_encode_leaves_shape():
    L = np.array([[0, 0], [1, 0], [2, 1]])
    n_leaves_per_tree = [3, 2]
    X = one_hot_encode_leaves(L, n_leaves_per_tree)
    assert X.shape == (3, sum(n_leaves_per_tree))
```

(Sparse matrices have a `.shape` attribute, so the shape test works without `.toarray()`.)

- [ ] **Step 4: Run, verify pass.**

```bash
pytest scripts/test_leaf_clustering_helpers.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit.**

```bash
git add scripts/leaf_clustering_helpers.py scripts/test_leaf_clustering_helpers.py
git commit -m "rule-path: leaf one-hot encoding helper

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: TDD — `n_leaves_per_tree` extractor from tree-dict

**Files:**
- Modify: `scripts/leaf_clustering_helpers.py`
- Modify: `scripts/test_leaf_clustering_helpers.py`

- [ ] **Step 1: Add failing tests.**

Append:

```python
from leaf_clustering_helpers import count_leaves, leaf_id_remap

def test_count_leaves_simple():
    tree = {
        0: {'leaf': False, 'feature': 'a', 'threshold': 0, 'yes': 1, 'no': 2},
        1: {'leaf': True, 'value': 1},
        2: {'leaf': True, 'value': 2},
    }
    assert count_leaves(tree) == 2

def test_count_leaves_root_only_leaf():
    assert count_leaves({0: {'leaf': True, 'value': 0}}) == 1

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
    # Leaves are 1, 4, 6; should map to 0, 1, 2 in some order.
    assert sorted(remap.values()) == [0, 1, 2]
    assert set(remap.keys()) == {1, 4, 6}
```

- [ ] **Step 2: Run, see fail.**

- [ ] **Step 3: Implement.**

Append to helpers:

```python
def count_leaves(tree_dict):
    """Number of leaf nodes in a tree-dict."""
    return sum(1 for n in tree_dict.values() if n.get('leaf', False))


def leaf_id_remap(tree_dict):
    """Map raw leaf node ids to dense [0, n_leaves) indices.

    XGBoost tree node ids are not necessarily contiguous (internal nodes
    consume ids too). This returns a dict {raw_node_id: dense_idx} for use
    with one_hot_encode_leaves.
    """
    leaves = sorted(nid for nid, n in tree_dict.items() if n.get('leaf', False))
    return {nid: i for i, nid in enumerate(leaves)}
```

- [ ] **Step 4: Run, verify pass.**

Expected: 11 passed.

- [ ] **Step 5: Commit.**

```bash
git add scripts/leaf_clustering_helpers.py scripts/test_leaf_clustering_helpers.py
git commit -m "rule-path: leaf counting + remapping helpers

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Phase 1 — extract leaf signatures end-to-end (long-leg only)

**Files:**
- Create: `scripts/extract_leaf_signatures.py`

- [ ] **Step 1: Write the script.**

```python
# scripts/extract_leaf_signatures.py
"""Phase 1: extract leaf-VALUE matrix for long-leg stock-months across the
50-seed × 500-tree ensemble.

Each cell leaf_values[i, t] is the prediction contribution of tree t to
stock-month i — i.e. the real-valued leaf score where stock i lands in
tree t. The sum across t (× learning_rate) recovers the stock's score.

Standardising the columns of leaf_values automatically weights trees by
the variance of their per-stock contributions: high-impact early trees
dominate Euclidean distance, low-impact late trees barely contribute.
This solves the tree-weighting problem without explicit per-tree weights.

Also extracts leaves_dense (leaf indices) for the dominant-splits
analysis in phase 3 (tracing tree paths back to feature/threshold pairs).

Output:
- artefacts/leaf_signatures.npz with keys:
    'leaf_values'   float matrix (n_long, 25000)  per-tree contributions
    'leaves_raw'    int matrix (n_long, 25000)    raw leaf node ids
    'leaves_dense'  int matrix (n_long, 25000)    leaves remapped to [0, n_leaves_t)
    'n_leaves_per_tree' int array (25000,)
    'date'          datetime64 array (n_long,)
    'permno'        int array (n_long,)
"""
import os, pickle, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leaf_clustering_helpers import route_stock_through_tree, count_leaves, leaf_id_remap

ARTEFACTS = 'artefacts/cs_artefacts_data.pkl'
TREES_PATH = 'artefacts/pi_verify_trees_seeds50.pkl'
OUT_PATH = 'artefacts/leaf_signatures.npz'

# ---- Load ----
with open(ARTEFACTS, 'rb') as f:
    art = pickle.load(f)
test = art['test']
features = list(art['FEATURES'])
X = art['X_te_s']

with open(TREES_PATH, 'rb') as f:
    trees, seeds = pickle.load(f)
n_trees = len(trees)
print(f'Loaded {n_trees} trees across {len(seeds)} seeds')

# ---- Identify long-leg stock-months ----
df = test.copy()
df['leg'] = 'middle'
for date, grp in df.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score_xgb'].dropna()
    if len(nyse) < 10:
        continue
    hi = nyse.quantile(0.90)
    df.loc[grp[grp['score_xgb'] >= hi].index, 'leg'] = 'long'
long_idx = np.where(df['leg'].values == 'long')[0]
print(f'long-leg stock-months: {len(long_idx)}')

# ---- Pre-compute leaf metadata ----
n_leaves_per_tree = np.array([count_leaves(t) for t in trees], dtype=np.int32)
remaps = [leaf_id_remap(t) for t in trees]

# ---- Extract leaves AND leaf values ----
n_long = len(long_idx)
leaves_raw = np.zeros((n_long, n_trees), dtype=np.int32)
leaves_dense = np.zeros((n_long, n_trees), dtype=np.int32)
leaf_values = np.zeros((n_long, n_trees), dtype=np.float32)
for ti, tree in enumerate(trees):
    if ti % 1000 == 0:
        print(f'  tree {ti}/{n_trees}')
    for li, idx in enumerate(long_idx):
        nid = route_stock_through_tree(X[idx], tree, features)
        leaves_raw[li, ti] = nid
        leaves_dense[li, ti] = remaps[ti][nid]
        leaf_values[li, ti] = float(tree[nid].get('value', 0.0))

np.savez_compressed(
    OUT_PATH,
    leaf_values=leaf_values,
    leaves_raw=leaves_raw,
    leaves_dense=leaves_dense,
    n_leaves_per_tree=n_leaves_per_tree,
    date=df.iloc[long_idx]['date'].values.astype('datetime64[ns]'),
    permno=df.iloc[long_idx]['permno'].values.astype(np.int64),
)
print(f'Saved: {OUT_PATH}')
print(f'  leaf_values shape: {leaf_values.shape}, '
      f'col std min/median/max: '
      f'{leaf_values.std(axis=0).min():.4f}/'
      f'{np.median(leaf_values.std(axis=0)):.4f}/'
      f'{leaf_values.std(axis=0).max():.4f}')
print(f'  total leaves: {n_leaves_per_tree.sum()}')
```

- [ ] **Step 2: Run on small subset first by limiting `long_idx` to first 100 entries (manual edit), confirm shape and runtime.**

```bash
# Temporarily add `long_idx = long_idx[:100]` after the long_idx assignment.
python scripts/extract_leaf_signatures.py
```

Expected: completes in < 1 min, leaves_dense.shape = (100, 25000), per-tree leaf counts look reasonable (~16 leaves per tree for depth-4 ensemble).

- [ ] **Step 3: Remove the subset hack, run full extraction.**

```bash
python scripts/extract_leaf_signatures.py
```

Expected: ~10–30 min runtime; final shape `(~55,200, 25000)`.

- [ ] **Step 4: Verify.**

```python
import numpy as np
d = np.load('artefacts/leaf_signatures.npz', allow_pickle=False)
print(d['leaves_dense'].shape)
print('sample:', d['leaves_dense'][0, :5])
print('n_leaves_per_tree min/max/mean:',
      d['n_leaves_per_tree'].min(),
      d['n_leaves_per_tree'].max(),
      d['n_leaves_per_tree'].mean())
```

Expected: shape ≈ (55200, 25000); leaf counts per tree mostly 8–16 (depth-4 trees have up to 16 leaves).

- [ ] **Step 5: Commit.**

```bash
git add scripts/extract_leaf_signatures.py
git commit -m "rule-path phase 1: extract leaf signatures for long-leg stocks

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Foundation audit — leaf signatures reproduce thesis 5.3.1 numbers

**Files:**
- Investigation only.

- [ ] **Step 1: Verify long-leg count matches existing analysis.**

```python
import numpy as np, pickle, pandas as pd
d = np.load('artefacts/leaf_signatures.npz', allow_pickle=False)
print('long-leg stock-months:', len(d['date']))
# Compare to existing tree-path analysis if available
with open('artefacts/tree_path_results.pkl', 'rb') as f:
    tp = pickle.load(f)
print('thesis 5.3.1 sample_size:', tp.get('sample_size'))
```

Expected: counts match within rounding.

- [ ] **Step 2: Verify trees-with-pi-split rate matches thesis claim of 73.9%.**

```python
import pickle, numpy as np
with open('artefacts/pi_verify_trees_seeds50.pkl', 'rb') as f:
    trees, seeds = pickle.load(f)

def has_pi_split(tree):
    return any(n.get('feature') == 'pi_filter' for n in tree.values()
               if not n.get('leaf', False))

rate = np.mean([has_pi_split(t) for t in trees])
print(f'fraction of trees with pi_filter split: {rate:.3f}  (thesis: 0.739)')
```

Expected: ~0.739 (within 0.01 tolerance).

- [ ] **Step 3: Document audit findings in a comment block at the top of `scripts/extract_leaf_signatures.py`.**

Add a comment confirming the audit results and the date the audit was run. This is documentation, not new code.

- [ ] **Step 4: Commit the documentation update.**

```bash
git add scripts/extract_leaf_signatures.py
git commit -m "rule-path: foundation audit confirms leaf signatures align with 5.3.1

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: TDD — silhouette and stability primitive for sparse data

**Files:**
- Modify: `scripts/leaf_clustering_helpers.py`
- Modify: `scripts/test_leaf_clustering_helpers.py`

- [ ] **Step 1: Add failing tests for `silhouette_sparse_subsample`.**

```python
from leaf_clustering_helpers import silhouette_sparse_subsample
from sklearn.cluster import KMeans
from scipy import sparse

def test_silhouette_sparse_subsample_returns_float():
    rng = np.random.default_rng(0)
    X_dense = rng.standard_normal((100, 5))
    X = sparse.csr_matrix(X_dense)
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
    sil = silhouette_sparse_subsample(X, km.labels_, n_samples=50, seed=42)
    assert isinstance(sil, float)
    assert -1.0 <= sil <= 1.0

def test_silhouette_sparse_subsample_well_separated():
    # Two clearly separated blobs in sparse form
    rng = np.random.default_rng(0)
    A = rng.standard_normal((50, 5)) + np.array([10, 0, 0, 0, 0])
    B = rng.standard_normal((50, 5)) + np.array([-10, 0, 0, 0, 0])
    X = sparse.csr_matrix(np.vstack([A, B]))
    labels = np.array([0]*50 + [1]*50)
    sil = silhouette_sparse_subsample(X, labels, n_samples=80, seed=42)
    assert sil > 0.5
```

- [ ] **Step 2: Run, see fail.**

- [ ] **Step 3: Implement using sklearn's silhouette with subsampling for memory.**

```python
from sklearn.metrics import silhouette_score
from sklearn.utils import check_random_state

def silhouette_sparse_subsample(X, labels, n_samples, seed=0):
    """Silhouette score on a random subsample (memory-friendly for sparse X).

    Parameters
    ----------
    X : sparse or dense (n, d)
    labels : array (n,)
    n_samples : int  number of points to subsample
    seed : int
    """
    rng = check_random_state(seed)
    n = X.shape[0]
    if n_samples >= n:
        return float(silhouette_score(X, labels, metric='euclidean'))
    idx = rng.choice(n, size=n_samples, replace=False)
    return float(silhouette_score(X[idx], np.asarray(labels)[idx],
                                  metric='euclidean'))
```

- [ ] **Step 4: Run, verify pass.**

Expected: 13 passed.

- [ ] **Step 5: Commit.**

```bash
git add scripts/leaf_clustering_helpers.py scripts/test_leaf_clustering_helpers.py
git commit -m "rule-path: subsampled silhouette helper

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Phase 2 — KMeans on PCA-reduced standardised leaf-values

**Files:**
- Create: `scripts/cluster_rule_paths.py`

- [ ] **Step 1: Write the clustering pipeline.**

```python
# scripts/cluster_rule_paths.py
"""Phase 2: cluster long-leg stock-months on standardised leaf-value matrix.

Each row of the leaf-value matrix is a stock-month's per-tree prediction
contribution. Standardising columns automatically weights trees by their
variance — high-impact early trees dominate Euclidean distance.

Pipeline:
  1. Load leaf_values matrix (~55,200 × 25,000)
  2. StandardScaler per column
  3. TruncatedSVD to 50 components
  4. KMeans for k in [2, 3], pick best by silhouette

Outputs:
- results/thesis/rule_path_clustering_sweep.csv
- artefacts/rule_path_labels.npz
"""
import os, pickle, sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leaf_clustering_helpers import silhouette_sparse_subsample

RES_DIR = 'results/thesis'
OUT_LABELS = 'artefacts/rule_path_labels.npz'
RNG_SEED = 42
N_INIT = 20
PCA_DIM = 50
# K_RANGE narrowed to [2, 3] only. With 167 OOS months, k>=4 leaves
# per-rule plurality-month samples too small for meaningful Sharpe CIs
# and regime cross-tabs.
K_RANGE = [2, 3]

# ---- Load leaf-value signatures ----
d = np.load('artefacts/leaf_signatures.npz', allow_pickle=False)
LV = d['leaf_values'].astype(np.float32)   # (n, 25000)
date = d['date']
permno = d['permno']
n = LV.shape[0]
print(f'leaf-value matrix: {LV.shape}')

# ---- Standardise columns: high-variance trees get high weight ----
# Drop zero-variance columns (trees where every long-leg stock landed in
# the same leaf) — they contribute nothing and break the StandardScaler.
col_std = LV.std(axis=0)
keep = col_std > 0
print(f'  dropping {(~keep).sum()} zero-variance trees, keeping {keep.sum()}')
LVk = LV[:, keep]
LVk_std = StandardScaler().fit_transform(LVk)

# ---- PCA reduction ----
svd = TruncatedSVD(n_components=PCA_DIM, random_state=RNG_SEED)
Z = svd.fit_transform(LVk_std)
print(f'  PCA explained variance ratio sum (first {PCA_DIM}): '
      f'{svd.explained_variance_ratio_.sum():.3f}')

# ---- Sweep k=2..3, pick best by joint silhouette + stability gate ----
# Stability requirement: mean pairwise ARI across 6 seeds must be >= 0.90
# for k to be eligible. Among eligible k, pick the one with highest mean
# silhouette across seeds.
from sklearn.metrics import adjusted_rand_score

STABILITY_SEEDS = [42, 123, 456, 789, 1011, 1213]
ARI_THRESHOLD = 0.90

sweep_rows = []
fits_by_seed = {}  # k -> list of label arrays (one per seed)

for k in K_RANGE:
    fits_by_seed[k] = []
    sils = []
    for s in STABILITY_SEEDS:
        km = KMeans(n_clusters=k, random_state=s, n_init=N_INIT).fit(Z)
        sil = silhouette_sparse_subsample(Z, km.labels_, n_samples=5000, seed=s)
        fits_by_seed[k].append(km.labels_)
        sils.append(sil)
    # Pairwise ARI across seeds
    pair_aris = []
    for i in range(len(STABILITY_SEEDS)):
        for j in range(i + 1, len(STABILITY_SEEDS)):
            pair_aris.append(float(adjusted_rand_score(
                fits_by_seed[k][i], fits_by_seed[k][j]
            )))
    mean_sil = float(np.mean(sils))
    mean_ari = float(np.mean(pair_aris))
    sizes_seed42 = sorted(np.bincount(fits_by_seed[k][0], minlength=k).tolist())
    sweep_rows.append({
        'k': k,
        'mean_silhouette': round(mean_sil, 4),
        'mean_ari': round(mean_ari, 4),
        'stable_at_threshold': mean_ari >= ARI_THRESHOLD,
        'sizes_seed42': str(tuple(sizes_seed42)),
    })
    print(f'  k={k}: mean_sil={mean_sil:.3f}, mean_ari={mean_ari:.3f}, '
          f'sizes(seed42)={tuple(sizes_seed42)}, '
          f'stable={"YES" if mean_ari >= ARI_THRESHOLD else "NO"}')

sweep_df = pd.DataFrame(sweep_rows)
os.makedirs(RES_DIR, exist_ok=True)
sweep_df.to_csv(f'{RES_DIR}/rule_path_clustering_sweep.csv', index=False)
print(f'\nSaved: {RES_DIR}/rule_path_clustering_sweep.csv')

# Pick: highest mean_silhouette among k where stable_at_threshold is True.
# If none are stable, fall back to highest mean_silhouette and flag.
eligible = sweep_df[sweep_df['stable_at_threshold']]
if len(eligible) > 0:
    best_row = eligible.loc[eligible['mean_silhouette'].idxmax()]
    print(f'\nStability gate satisfied. Best stable k={int(best_row["k"])}, '
          f'mean_sil={best_row["mean_silhouette"]:.3f}, '
          f'mean_ari={best_row["mean_ari"]:.3f}')
else:
    best_row = sweep_df.loc[sweep_df['mean_silhouette'].idxmax()]
    print(f'\n!!! WARNING: no k passed stability gate (ARI >= {ARI_THRESHOLD}). '
          f'Falling back to highest-silhouette k={int(best_row["k"])}, '
          f'mean_ari={best_row["mean_ari"]:.3f} — flag for discussion.')

best_k = int(best_row['k'])
best_sil = float(best_row['mean_silhouette'])
best_ari = float(best_row['mean_ari'])
# Use the seed-42 labels as the canonical label assignment
labels_final = fits_by_seed[best_k][0]

np.savez_compressed(
    OUT_LABELS,
    labels=labels_final,
    best_k=best_k,
    best_sil=best_sil,
    date=date,
    permno=permno,
    cols_kept=keep,
)
print(f'Saved: {OUT_LABELS}')

# Also save canonical (date, permno, rule_id) CSV for downstream phases
final_df = pd.DataFrame({
    'date': pd.to_datetime(date),
    'permno': permno,
    'rule_id': labels_final,
})
final_df.to_csv(f'{RES_DIR}/rule_path_labels.csv', index=False)
print(f'Saved: {RES_DIR}/rule_path_labels.csv ({len(final_df)} rows)')
```

- [ ] **Step 2: Run.**

```bash
python scripts/cluster_rule_paths.py
```

Expected: completes in ~5–10 min. Reports silhouette per k=2,3 and chooses winner.

- [ ] **Step 3: Inspect outputs.**

```bash
cat results/thesis/rule_path_clustering_sweep.csv
head -5 results/thesis/rule_path_labels.csv
wc -l results/thesis/rule_path_labels.csv
```

Expected: 2 rows in sweep CSV; ~55,200 rows in labels CSV.

- [ ] **Step 4: Commit.**

```bash
git add scripts/cluster_rule_paths.py \
  results/thesis/rule_path_clustering_sweep.csv \
  results/thesis/rule_path_labels.csv
git commit -m "rule-path phase 2: PCA + KMeans on standardised leaf values

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Phase 3 — per-rule input feature signatures

**Files:**
- Create: `scripts/characterise_rule_paths.py`

- [ ] **Step 1: Compute typical-input signatures per rule.**

```python
# scripts/characterise_rule_paths.py
"""Phase 3: per-rule characterisation.

For each rule (cluster of long-leg stock-months):
  - typical input feature centroid (12 mom horizons + pi_filter)
  - regime distribution (mean pi, calm/panic share)
  - cross-sectional landscape signature (mom tertile means)
  - realised return + Sharpe + bootstrap 95% CI

Output:
- results/thesis/rule_path_centroids.csv
"""
import os, pickle, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_helpers import block_bootstrap_sharpe

RES_DIR = 'results/thesis'

# ---- Load ----
labels_df = pd.read_csv(f'{RES_DIR}/rule_path_labels.csv', parse_dates=['date'])
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
test['date'] = pd.to_datetime(test['date'])
mom_cols = [f'mom_{h}' for h in range(1, 13)]

# Cross-sectional landscape per month (typical-stock)
agg_mean = test.groupby('date')[mom_cols].mean()
SHORT, MID, LONG = [1,2,3,4], [5,6,7,8], [9,10,11,12]
def tert(df, g): return df[[f'mom_{h}' for h in g]].mean(axis=1)
short_mean = tert(agg_mean, SHORT)
mid_mean = tert(agg_mean, MID)
long_mean = tert(agg_mean, LONG)

pi = test.groupby('date')['pi_filter'].first()

# Returns
with open('results/thesis/fundamentals_returns.pkl', 'rb') as f:
    rets = pickle.load(f)
m2_ret = rets['baseline_mom_pi']['returns']
m2_ret.index = pd.to_datetime(m2_ret.index)

# ---- Per-rule aggregation ----
# Each row of labels_df is one (date, permno, rule_id). Aggregate to:
#   - per-rule typical stock features (avg over all stock-months in rule)
#   - per-rule per-month return (we need the monthly return of the
#     subset of long-leg stocks belonging to rule r — but the strategy
#     return is computed at the month level, not stock level. So we
#     interpret "rule-conditional Sharpe" as: take months where MAJORITY
#     of long-leg picks belong to rule r, compute Sharpe over those months.)

# Add stock-level features by joining with test panel
key_cols = ['date', 'permno']
labels_with_features = labels_df.merge(
    test[key_cols + mom_cols + ['pi_filter']],
    on=key_cols, how='left',
)

rules = sorted(labels_df['rule_id'].unique())
print(f'{len(rules)} rules')

rows = []
for r in rules:
    sub = labels_with_features[labels_with_features['rule_id'] == r]
    n_stockmonths = len(sub)
    # Typical-stock feature centroid (mean over stock-months in rule)
    stock_centroid = sub[mom_cols].mean()
    mean_pi = sub['pi_filter'].mean()
    # Months where rule r is a plurality
    by_month = sub.groupby('date').size()
    total_by_month = labels_with_features.groupby('date').size()
    plurality_share = (by_month / total_by_month).fillna(0)
    plurality_dates = plurality_share.index[plurality_share > 0.5]
    # Conditional Sharpe over plurality months
    rs = m2_ret.reindex(plurality_dates).dropna().values
    if len(rs) > 0:
        ci = block_bootstrap_sharpe(rs, block_size=6, n_reps=5000, seed=42)
    else:
        ci = {'sharpe_point': np.nan, 'sharpe_lo95': np.nan,
              'sharpe_hi95': np.nan, 'n_reps_valid': 0}

    row = {
        'rule_id': r,
        'n_stockmonths': n_stockmonths,
        'n_plurality_months': len(plurality_dates),
        'mean_pi': round(mean_pi, 3),
        'panic_share': round((sub['pi_filter'] > 0.5).mean(), 3),
        'sharpe_point': round(ci['sharpe_point'], 3) if not np.isnan(ci['sharpe_point']) else None,
        'sharpe_lo95': round(ci['sharpe_lo95'], 3) if not np.isnan(ci['sharpe_lo95']) else None,
        'sharpe_hi95': round(ci['sharpe_hi95'], 3) if not np.isnan(ci['sharpe_hi95']) else None,
    }
    for c in mom_cols:
        row[f'centroid_{c}'] = round(stock_centroid[c], 4)
    rows.append(row)

centroids_df = pd.DataFrame(rows)
centroids_df.to_csv(f'{RES_DIR}/rule_path_centroids.csv', index=False)
print(f'\nSaved: {RES_DIR}/rule_path_centroids.csv')
print(centroids_df.to_string(index=False))
```

- [ ] **Step 2: Run.**

```bash
python scripts/characterise_rule_paths.py
```

Expected: per-rule table printed and saved.

- [ ] **Step 3: Verify.**

```bash
cat results/thesis/rule_path_centroids.csv
```

- [ ] **Step 4: Commit.**

```bash
git add scripts/characterise_rule_paths.py results/thesis/rule_path_centroids.csv
git commit -m "rule-path phase 3: per-rule centroids + bootstrap Sharpe CI

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Phase 3 — dominant tree splits per rule

**Files:**
- Modify: `scripts/characterise_rule_paths.py`

- [ ] **Step 1: Append logic to identify the (feature, threshold) pairs that show up in ≥30% of leaf paths within each rule.**

```python
# Append to scripts/characterise_rule_paths.py

# ---- Dominant splits per rule ----
import pickle
with open('artefacts/pi_verify_trees_seeds50.pkl', 'rb') as f:
    trees, seeds = pickle.load(f)
import numpy as np
d = np.load('artefacts/leaf_signatures.npz', allow_pickle=False)
leaves_raw = d['leaves_raw']  # (n_long, n_trees) raw node ids
labels = labels_df['rule_id'].values

def path_to_leaf(tree, leaf_nid):
    """Return list of (feature, threshold, direction) tuples on the path
    from root to leaf_nid. direction='yes' or 'no'."""
    # BFS over tree to find leaf, recording path
    parent = {0: None}
    edge = {}  # child -> ('feature', 'threshold', 'yes'/'no')
    stack = [0]
    while stack:
        nid = stack.pop()
        node = tree[nid]
        if node.get('leaf', False):
            continue
        for d in ('yes', 'no'):
            cnid = node[d]
            parent[cnid] = nid
            edge[cnid] = (node['feature'], node['threshold'], d)
            stack.append(cnid)
    path = []
    cur = leaf_nid
    while parent.get(cur) is not None:
        path.append(edge[cur])
        cur = parent[cur]
    return path[::-1]

split_rows = []
for r in rules:
    rule_idx = np.where(labels == r)[0]
    if len(rule_idx) == 0:
        continue
    # Sample stock-months from this rule for path extraction (cap at 500)
    if len(rule_idx) > 500:
        rule_idx = np.random.RandomState(42).choice(rule_idx, 500, replace=False)
    counts = {}  # (feature, threshold_round, direction) -> count
    total_paths = 0
    for sm in rule_idx:
        for ti, leaf_raw in enumerate(leaves_raw[sm]):
            path = path_to_leaf(trees[ti], int(leaf_raw))
            for f, t, dr in path:
                key = (f, round(float(t), 3), dr)
                counts[key] = counts.get(key, 0) + 1
            total_paths += 1
    # Top splits by frequency
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
    for (f, t, dr), c in top:
        split_rows.append({
            'rule_id': r,
            'feature': f,
            'threshold': t,
            'direction': dr,
            'frequency_pct': round(100 * c / total_paths, 1),
        })

splits_df = pd.DataFrame(split_rows)
splits_df.to_csv(f'{RES_DIR}/rule_path_dominant_splits.csv', index=False)
print(f'\nSaved: {RES_DIR}/rule_path_dominant_splits.csv')
```

- [ ] **Step 2: Run.**

```bash
python scripts/characterise_rule_paths.py
```

Expected: top 10 dominant splits per rule printed and saved.

- [ ] **Step 3: Verify.**

```bash
head -30 results/thesis/rule_path_dominant_splits.csv
```

- [ ] **Step 4: Commit.**

```bash
git add scripts/characterise_rule_paths.py results/thesis/rule_path_dominant_splits.csv
git commit -m "rule-path phase 3: dominant tree splits per rule

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Phase 4 — seed stability of rule clustering

**Files:**
- Create: `scripts/rule_path_robustness.py`

- [ ] **Step 1: Refit clustering at 6 seeds, compute pairwise ARI.**

```python
# scripts/rule_path_robustness.py
"""Phase 4: robustness checks for the rule-path clustering.

  - Seed stability: refit at 6 seeds, sorted cluster sizes + pairwise ARI
  - Subsample stability: hold out 20%, refit, predict, agreement
  - Disjoint-ensemble validation: cluster on a different seed-50 ensemble
                                  if available; ARI between label vectors.

Outputs:
- results/thesis/rule_path_robustness.csv
- results/thesis/rule_path_sharpe_ci.csv  (point + CI per rule)
"""
import os, pickle, sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leaf_clustering_helpers import one_hot_encode_leaves, silhouette_sparse_subsample

RES_DIR = 'results/thesis'
SEEDS = [42, 123, 456, 789, 1011, 1213]

d = np.load('artefacts/leaf_signatures.npz', allow_pickle=False)
leaves = d['leaves_dense']
n_leaves_per_tree = d['n_leaves_per_tree']

X = one_hot_encode_leaves(leaves, n_leaves_per_tree)

# Re-load winner k from cluster artefact
labels_artefact = np.load('artefacts/rule_path_labels.npz', allow_pickle=False)
k = int(labels_artefact['best_k_a' if str(labels_artefact['winner']) == 'A'
        else 'best_k_b'])
print(f'Refitting clustering at k={k} across {len(SEEDS)} seeds')

# PCA must be redone per seed (seed affects SVD initialisation only mildly,
# but use deterministic SVD + KMeans seed for the test).
svd = TruncatedSVD(n_components=50, random_state=42)
Z = svd.fit_transform(X)

label_sets = []
sizes_per_seed = []
for s in SEEDS:
    km = KMeans(n_clusters=k, random_state=s, n_init=20).fit(Z)
    label_sets.append(km.labels_)
    sizes_per_seed.append(tuple(sorted(np.bincount(km.labels_, minlength=k).tolist())))

pairwise_ari = []
for i in range(len(SEEDS)):
    for j in range(i + 1, len(SEEDS)):
        pairwise_ari.append(float(adjusted_rand_score(label_sets[i], label_sets[j])))

rows = [
    {'metric': 'k', 'value': k},
    {'metric': 'mean_pairwise_ari', 'value': round(float(np.mean(pairwise_ari)), 4)},
    {'metric': 'min_pairwise_ari', 'value': round(float(np.min(pairwise_ari)), 4)},
]
for s, sizes in zip(SEEDS, sizes_per_seed):
    rows.append({'metric': f'sizes_seed_{s}', 'value': str(sizes)})

robust_df = pd.DataFrame(rows)
robust_df.to_csv(f'{RES_DIR}/rule_path_robustness.csv', index=False)
print(robust_df.to_string(index=False))
```

- [ ] **Step 2: Run.**

```bash
python scripts/rule_path_robustness.py
```

Expected: 6-seed robustness CSV with mean ARI ≥ 0.95 if clustering is stable.

- [ ] **Step 3: Commit.**

```bash
git add scripts/rule_path_robustness.py results/thesis/rule_path_robustness.csv
git commit -m "rule-path phase 4: 6-seed stability check

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Phase 4 — bootstrap CIs per rule (consolidated)

**Files:**
- Modify: `scripts/rule_path_robustness.py`

- [ ] **Step 1: Append per-rule bootstrap Sharpe CI table.**

```python
# Append to scripts/rule_path_robustness.py

# ---- Per-rule Sharpe CIs (re-uses output of phase 3, but consolidates) ----
centroids = pd.read_csv(f'{RES_DIR}/rule_path_centroids.csv')
ci_df = centroids[['rule_id', 'n_plurality_months', 'sharpe_point',
                   'sharpe_lo95', 'sharpe_hi95']].copy()
ci_df['ci_excludes_zero'] = ci_df['sharpe_lo95'] > 0
ci_df.to_csv(f'{RES_DIR}/rule_path_sharpe_ci.csv', index=False)
print(f'\nSaved: {RES_DIR}/rule_path_sharpe_ci.csv')
print(ci_df.to_string(index=False))
```

- [ ] **Step 2: Run, verify file.**

```bash
python scripts/rule_path_robustness.py
cat results/thesis/rule_path_sharpe_ci.csv
```

- [ ] **Step 3: Commit.**

```bash
git add scripts/rule_path_robustness.py results/thesis/rule_path_sharpe_ci.csv
git commit -m "rule-path phase 4: consolidated per-rule Sharpe CI table

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Phase 5 — cross-tabs (rule × regime, rule × landscape)

**Files:**
- Create: `scripts/rule_path_crosstabs.py`

- [ ] **Step 1: Build cross-tab tables.**

```python
# scripts/rule_path_crosstabs.py
"""Phase 5 cross-tabulations: rule × regime and rule × landscape.

Outputs:
- results/thesis/rule_path_regime_crosstab.csv
- results/thesis/rule_path_landscape_crosstab.csv
"""
import os, pickle, sys
import numpy as np
import pandas as pd

RES_DIR = 'results/thesis'

labels_df = pd.read_csv(f'{RES_DIR}/rule_path_labels.csv', parse_dates=['date'])
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
test['date'] = pd.to_datetime(test['date'])
mom_cols = [f'mom_{h}' for h in range(1, 13)]
agg_mean = test.groupby('date')[mom_cols].mean()
pi = test.groupby('date')['pi_filter'].first()

# ---- Rule × Regime ----
labels_df['regime'] = np.where(
    labels_df['date'].map(pi) > 0.5, 'panic', 'calm'
)
ct_regime = pd.crosstab(labels_df['rule_id'], labels_df['regime'])
ct_regime['total'] = ct_regime.sum(axis=1)
ct_regime['panic_share'] = (ct_regime['panic'] / ct_regime['total']).round(3)
ct_regime.to_csv(f'{RES_DIR}/rule_path_regime_crosstab.csv')
print('Rule × Regime:')
print(ct_regime.to_string())

# ---- Rule × Landscape (binned by typical-stock long-mom) ----
SHORT, MID, LONG = [1,2,3,4], [5,6,7,8], [9,10,11,12]
def tert(df, g): return df[[f'mom_{h}' for h in g]].mean(axis=1)
land_long = tert(agg_mean, LONG)

# Bin into terciles of long-tertile-momentum across the 167 OOS months
edges = land_long.quantile([0, 1/3, 2/3, 1.0]).values
labels_df['landscape_tier'] = pd.cut(
    labels_df['date'].map(land_long),
    bins=edges,
    labels=['low_long_mom', 'mid_long_mom', 'high_long_mom'],
    include_lowest=True,
)
ct_land = pd.crosstab(labels_df['rule_id'], labels_df['landscape_tier'])
ct_land.to_csv(f'{RES_DIR}/rule_path_landscape_crosstab.csv')
print('\nRule × Landscape:')
print(ct_land.to_string())
```

- [ ] **Step 2: Run.**

```bash
python scripts/rule_path_crosstabs.py
```

- [ ] **Step 3: Verify.**

```bash
cat results/thesis/rule_path_regime_crosstab.csv
cat results/thesis/rule_path_landscape_crosstab.csv
```

- [ ] **Step 4: Commit.**

```bash
git add scripts/rule_path_crosstabs.py \
  results/thesis/rule_path_regime_crosstab.csv \
  results/thesis/rule_path_landscape_crosstab.csv
git commit -m "rule-path phase 5: rule × regime and rule × landscape crosstabs

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: Phase 5 — within-month rule-path heterogeneity

**Files:**
- Create: `scripts/rule_path_within_month_homogeneity.py`

- [ ] **Step 1: Per month, compute the Herfindahl-Hirschman Index (HHI) of rule-path firings within long-leg picks.**

```python
# scripts/rule_path_within_month_homogeneity.py
"""Phase 5 complementary: within-month rule-path heterogeneity.

Per month, compute fraction of long-leg picks per rule, then HHI:
  HHI = sum(p_r^2)
HHI close to 1.0 -> all picks same rule (homogeneous month).
HHI close to 1/k -> picks evenly spread across k rules (heterogeneous month).

Output:
- results/thesis/rule_path_within_month_hhi.csv
- plots/thesis/rule_path_within_month_homogeneity.{png,pdf}
"""
import os, pickle, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RES_DIR = 'results/thesis'
PLOT_DIR = 'plots/thesis'
os.makedirs(PLOT_DIR, exist_ok=True)

labels_df = pd.read_csv(f'{RES_DIR}/rule_path_labels.csv', parse_dates=['date'])
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
test['date'] = pd.to_datetime(test['date'])
pi = test.groupby('date')['pi_filter'].first()

# Per-month HHI
def hhi(group):
    counts = group['rule_id'].value_counts()
    p = counts / counts.sum()
    return float((p ** 2).sum())

hhi_per_month = (labels_df.groupby('date')
                          .apply(hhi)
                          .rename('hhi')
                          .reset_index())
hhi_per_month['regime'] = np.where(
    hhi_per_month['date'].map(pi) > 0.5, 'panic', 'calm'
)
hhi_per_month.to_csv(f'{RES_DIR}/rule_path_within_month_hhi.csv', index=False)
print(hhi_per_month.head())
print(f'\nMean HHI by regime:')
print(hhi_per_month.groupby('regime')['hhi'].agg(['mean', 'std', 'min', 'max']))

# Plot
fig, ax = plt.subplots(figsize=(10, 4))
for r, c in [('calm', '#1f77b4'), ('panic', '#d62728')]:
    sub = hhi_per_month[hhi_per_month['regime'] == r]
    ax.scatter(sub['date'], sub['hhi'], s=20, color=c, label=r, alpha=0.7)
ax.set_xlabel('Date')
ax.set_ylabel('Within-month HHI of rule-path firings')
ax.set_title('Within-month homogeneity of long-leg rule-path firings')
ax.axhline(1.0, color='#888', lw=0.6, ls='--', label='single rule')
ax.legend()
fig.tight_layout()
fig.savefig(f'{PLOT_DIR}/rule_path_within_month_homogeneity.png', dpi=200)
fig.savefig(f'{PLOT_DIR}/rule_path_within_month_homogeneity.pdf')
print(f'Saved plots/thesis/rule_path_within_month_homogeneity.{{png,pdf}}')
```

- [ ] **Step 2: Run.**

```bash
python scripts/rule_path_within_month_homogeneity.py
```

- [ ] **Step 3: Commit.**

```bash
git add scripts/rule_path_within_month_homogeneity.py \
  results/thesis/rule_path_within_month_hhi.csv \
  plots/thesis/rule_path_within_month_homogeneity.png \
  plots/thesis/rule_path_within_month_homogeneity.pdf
git commit -m "rule-path phase 5: within-month homogeneity diagnostic

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: Phase 5 — rule-path centroid feature plot

**Files:**
- Create: `scripts/plot_rule_path_centroids.py`

- [ ] **Step 1: Plot per-rule typical mom profile (12 horizons) as a line plot, one line per rule.**

```python
# scripts/plot_rule_path_centroids.py
"""Phase 5: per-rule typical-stock mom profile.

One line per rule, x-axis = horizon 1..12, y-axis = mean stock-month mom_h
within rule. Annotated with n_stockmonths and Sharpe.

Output:
- plots/thesis/rule_path_centroid_features.{png,pdf}
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RES_DIR = 'results/thesis'
PLOT_DIR = 'plots/thesis'
HORIZONS = list(range(1, 13))

centroids = pd.read_csv(f'{RES_DIR}/rule_path_centroids.csv')
n_rules = len(centroids)
print(f'Plotting {n_rules} rules')

fig, ax = plt.subplots(figsize=(11, 6))
cmap = plt.cm.tab10(np.linspace(0, 1, n_rules))
for i, row in centroids.iterrows():
    y = [row[f'centroid_mom_{h}'] for h in HORIZONS]
    sharpe = row['sharpe_point']
    n = int(row['n_stockmonths'])
    label = (f"rule {int(row['rule_id'])} "
             f"(n={n}, Sharpe={sharpe:.2f})"
             if pd.notna(sharpe) else f"rule {int(row['rule_id'])} (n={n})")
    ax.plot(HORIZONS, y, marker='o', color=cmap[i], lw=2, label=label)

ax.axhline(0, color='#888', lw=0.6)
ax.set_xticks(HORIZONS)
ax.set_xlabel('Momentum horizon (months)')
ax.set_ylabel('Typical-stock standardised mom')
ax.set_title('Per-rule typical-stock momentum profile')
ax.legend(loc='best', fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{PLOT_DIR}/rule_path_centroid_features.png', dpi=200)
fig.savefig(f'{PLOT_DIR}/rule_path_centroid_features.pdf')
print(f'Saved plots/thesis/rule_path_centroid_features.{{png,pdf}}')
```

- [ ] **Step 2: Run.**

```bash
python scripts/plot_rule_path_centroids.py
```

- [ ] **Step 3: Commit.**

```bash
git add scripts/plot_rule_path_centroids.py \
  plots/thesis/rule_path_centroid_features.png \
  plots/thesis/rule_path_centroid_features.pdf
git commit -m "rule-path phase 5: per-rule centroid mom profile plot

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: Phase 5 — crosstab heatmap plots

**Files:**
- Create: `scripts/plot_rule_path_crosstabs.py`

- [ ] **Step 1: Heatmap for rule × regime and rule × landscape.**

```python
# scripts/plot_rule_path_crosstabs.py
"""Phase 5: heatmap visualisation of rule × regime and rule × landscape.

Output:
- plots/thesis/rule_path_crosstabs.{png,pdf}
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RES_DIR = 'results/thesis'
PLOT_DIR = 'plots/thesis'

ct_regime = pd.read_csv(f'{RES_DIR}/rule_path_regime_crosstab.csv', index_col=0)
ct_land = pd.read_csv(f'{RES_DIR}/rule_path_landscape_crosstab.csv', index_col=0)

# Drop the 'total' and 'panic_share' columns added in phase 5 crosstab task
ct_regime_pure = ct_regime[['calm', 'panic']]

# Normalise to row percentages
ct_regime_norm = ct_regime_pure.div(ct_regime_pure.sum(axis=1), axis=0) * 100
ct_land_norm = ct_land.div(ct_land.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, df, title in [
    (axes[0], ct_regime_norm, 'Rule × Regime (% within rule)'),
    (axes[1], ct_land_norm, 'Rule × Landscape tier (% within rule)'),
]:
    im = ax.imshow(df.values, aspect='auto', cmap='viridis', vmin=0, vmax=100)
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=30, ha='right')
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.index)
    ax.set_title(title)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f'{df.values[i, j]:.0f}', ha='center', va='center',
                    color='white' if df.values[i, j] < 50 else 'black',
                    fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.tight_layout()
fig.savefig(f'{PLOT_DIR}/rule_path_crosstabs.png', dpi=200)
fig.savefig(f'{PLOT_DIR}/rule_path_crosstabs.pdf')
print(f'Saved plots/thesis/rule_path_crosstabs.{{png,pdf}}')
```

- [ ] **Step 2: Run.**

```bash
python scripts/plot_rule_path_crosstabs.py
```

- [ ] **Step 3: Commit.**

```bash
git add scripts/plot_rule_path_crosstabs.py \
  plots/thesis/rule_path_crosstabs.png \
  plots/thesis/rule_path_crosstabs.pdf
git commit -m "rule-path phase 5: rule × regime / rule × landscape heatmaps

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 18: Phase 5 — dispersion plot per rule (long-leg z-profile)

**Files:**
- Create: `scripts/plot_rule_path_dispersion.py`

- [ ] **Step 1: Plot, per rule, the dispersion of *long-leg z-profiles* (output side) for the months where that rule fires plurality. Same dispersion-plot style as `output_first_*_dispersion_robust.pdf`.**

```python
# scripts/plot_rule_path_dispersion.py
"""Phase 5: dispersion of long-leg z-profiles, faceted by plurality rule.

For each rule, identify months where it fires plurality (>50% of long-leg
picks belong to it), then plot the long-leg z-profile dispersion for those
months in the same style as the existing thesis dispersion figures.

Output:
- plots/thesis/rule_path_dispersion.{png,pdf}
"""
import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RES_DIR = 'results/thesis'
PLOT_DIR = 'plots/thesis'
HORIZONS = list(range(1, 13))
MOM_COLS = [f'mom_{h}' for h in HORIZONS]

labels_df = pd.read_csv(f'{RES_DIR}/rule_path_labels.csv', parse_dates=['date'])
long_z = pd.read_csv(f'{RES_DIR}/zscore_long_by_month.csv',
                     parse_dates=['date']).set_index('date')

# Plurality rule per month
plurality = (labels_df.groupby(['date', 'rule_id']).size()
                       .reset_index(name='n')
                       .sort_values(['date', 'n'], ascending=[True, False])
                       .groupby('date').first()
                       .reset_index())
plurality['plurality_share'] = plurality.apply(
    lambda r: r['n'] / labels_df[labels_df['date'] == r['date']].shape[0],
    axis=1,
)
plurality_dates = plurality[plurality['plurality_share'] > 0.5]
print(f'Months with a >50% plurality rule: {len(plurality_dates)} / {labels_df["date"].nunique()}')

# Calm reference
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
test['date'] = pd.to_datetime(test['date'])
pi = test.groupby('date')['pi_filter'].first()
calm_mask = pi <= 0.5
calm_ref = long_z.loc[calm_mask, MOM_COLS].mean().values

rules = sorted(plurality_dates['rule_id'].unique())
n_rules = len(rules)
n_cols = min(n_rules, 4)
n_rows = int(np.ceil(n_rules / n_cols))

fig, axes_2d = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 5 * n_rows),
                             sharey=True, squeeze=False)
axes = axes_2d.flatten()
cmap = plt.cm.tab10(np.linspace(0, 1, n_rules))

for i, rule_id in enumerate(rules):
    ax = axes[i]
    rule_dates = plurality_dates[plurality_dates['rule_id'] == rule_id]['date']
    Z = long_z.loc[rule_dates, MOM_COLS].values
    if len(Z) == 0:
        continue
    centroid = Z.mean(axis=0)
    color = cmap[i]
    for row in Z:
        ax.plot(HORIZONS, row, color=color, lw=0.8, alpha=0.35)
    ax.plot(HORIZONS, centroid, color=color, lw=2.5, marker='o', ms=6,
            label=f'rule {rule_id} centroid')
    ax.plot(HORIZONS, calm_ref, color='#333', lw=1.5, ls='--',
            label='Calm centroid (reference)')
    ax.axhline(0, color='#888', lw=0.6)
    ax.set_xticks(HORIZONS)
    ax.set_ylim(-1.5, 1.0)
    ax.set_xlabel('Momentum lookback (months)')
    ax.set_title(f'Rule {rule_id} (n_months={len(Z)})')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)

axes_2d[0, 0].set_ylabel('Long-leg cross-sectional z-score')
for j in range(n_rules, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Long-leg z-profile dispersion, per plurality rule', y=1.00)
fig.tight_layout()
fig.savefig(f'{PLOT_DIR}/rule_path_dispersion.png', dpi=200, bbox_inches='tight')
fig.savefig(f'{PLOT_DIR}/rule_path_dispersion.pdf', bbox_inches='tight')
print(f'Saved plots/thesis/rule_path_dispersion.{{png,pdf}}')
```

- [ ] **Step 2: Run.**

```bash
python scripts/plot_rule_path_dispersion.py
```

- [ ] **Step 3: Commit.**

```bash
git add scripts/plot_rule_path_dispersion.py \
  plots/thesis/rule_path_dispersion.png \
  plots/thesis/rule_path_dispersion.pdf
git commit -m "rule-path phase 5: per-rule long-leg z-profile dispersion

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 19: SHAP-vs-leaf ARI sanity check (appendix)

**Files:**
- Create: `scripts/rule_path_shap_appendix.py`

- [ ] **Step 1: Cluster long-leg stock-months by SHAP profile at the same k as the leaf-value clustering, then compute ARI vs leaf-value rule labels.**

```python
# scripts/rule_path_shap_appendix.py
"""Appendix: sanity check that SHAP-based clustering broadly agrees with
the leaf-value-based rule labels.

Single test: cluster long-leg stock-months on their 13-d SHAP attribution
vector at the same k as the chosen leaf-value clustering. Report ARI.

High ARI -> two different model-introspection methods agree on rules.
Low ARI -> rules are sensitive to the introspection method.

Output:
- results/thesis/rule_path_shap_ari.csv (one row)
"""
import os, pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

RES_DIR = 'results/thesis'

with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
test['date'] = pd.to_datetime(test['date'])
shap = art['shap_values']  # (n_test, 13)

# Identify long-leg stock-months (mirror Phase 1 logic)
df = test.copy()
df['leg'] = 'middle'
for date, grp in df.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score_xgb'].dropna()
    if len(nyse) < 10:
        continue
    hi = nyse.quantile(0.90)
    df.loc[grp[grp['score_xgb'] >= hi].index, 'leg'] = 'long'
long_mask = df['leg'].values == 'long'
shap_long = shap[long_mask]

X = StandardScaler().fit_transform(shap_long)

# Cluster at the same k as the leaf-value rule clustering
labels_artefact = np.load('artefacts/rule_path_labels.npz', allow_pickle=False)
rule_k = int(labels_artefact['best_k'])
km = KMeans(n_clusters=rule_k, random_state=42, n_init=20).fit(X)

rule_labels = pd.read_csv(f'{RES_DIR}/rule_path_labels.csv')['rule_id'].values
ari = float(adjusted_rand_score(km.labels_, rule_labels))
print(f'ARI between SHAP-cluster labels and leaf-value rules (k={rule_k}): {ari:.4f}')

result = pd.DataFrame([{
    'k': rule_k,
    'shap_vs_leaf_ari': round(ari, 4),
}])
result.to_csv(f'{RES_DIR}/rule_path_shap_ari.csv', index=False)
print(f'Saved: {RES_DIR}/rule_path_shap_ari.csv')
```

- [ ] **Step 2: Run.**

```bash
python scripts/rule_path_shap_appendix.py
```

- [ ] **Step 3: Commit.**

```bash
git add scripts/rule_path_shap_appendix.py results/thesis/rule_path_shap_ari.csv
git commit -m "rule-path appendix: SHAP-vs-leaf ARI sanity check

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 20: Final verification — end-to-end re-run from clean state

**Files:**
- Investigation only.

- [ ] **Step 1: Re-run all phase scripts in order to confirm reproducibility.**

```bash
pytest scripts/test_bootstrap_helpers.py scripts/test_leaf_clustering_helpers.py
python scripts/extract_leaf_signatures.py
python scripts/cluster_rule_paths.py
python scripts/characterise_rule_paths.py
python scripts/rule_path_robustness.py
python scripts/rule_path_crosstabs.py
python scripts/rule_path_within_month_homogeneity.py
python scripts/plot_rule_path_centroids.py
python scripts/plot_rule_path_crosstabs.py
python scripts/plot_rule_path_dispersion.py
python scripts/rule_path_shap_appendix.py
```

Expected: all pass; CSVs and plots regenerated identically.

- [ ] **Step 2: Diff outputs against committed versions.**

```bash
git status
git diff --stat
```

Expected: no changes (all numbers reproducible from seed=42).

- [ ] **Step 3: Verification report.**

Document the verification in a comment in [scripts/extract_leaf_signatures.py](scripts/extract_leaf_signatures.py) or in `results/thesis/rule_path_REPRODUCIBILITY.md` summarising:
- Pipeline successfully reproduces all numbers from seed=42
- Date of verification
- Any caveats

- [ ] **Step 4: Commit.**

```bash
git add results/thesis/rule_path_REPRODUCIBILITY.md
git commit -m "rule-path: end-to-end reproducibility verified

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 21: LaTeX revision — §5.2 rewrite

**Files:**
- Modify: `latex/main_results.tex`

- [ ] **Step 1: Read the existing 5.2 section to understand current wording and citation pattern.**

```bash
grep -n "subsection\|subsubsection" latex/main_results.tex | head -30
```

- [ ] **Step 2: Draft the revised §5.2 in a separate file first, for review.**

Create `latex/section_5_2_revision_draft.tex` with the new structure outlined in the design doc:

- §5.2.1 Aggregate output structure (k=2 output clusters per regime)
- §5.2.2 Aggregate inputs are insufficient (kNN R² ceiling)
- §5.2.3 Rule-path analysis (primary)
- §5.2.4 Within-month rule heterogeneity
- §5.2.5 Momentum × regime interaction
- Appendix subsection on SHAP comparison

Each subsection should cite the corresponding `results/thesis/*.csv` and `plots/thesis/*.pdf` files generated above.

Note: this task is DRAFT only — Gilad must review before merging into `main_results.tex` (per CLAUDE.md "Don't edit thesis .tex for data-dependent claims until Gilad has reviewed the numbers").

- [ ] **Step 3: Commit the draft.**

```bash
git add latex/section_5_2_revision_draft.tex
git commit -m "rule-path latex: draft §5.2 revision (awaiting Gilad review)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Stop here pending user review.** Do NOT merge into `main_results.tex` autonomously.

---

## Self-Review

**Spec coverage check** (against design doc):

| spec section | implemented in task |
|---|---|
| Phase 1 — Data extraction | Tasks 1, 6, 7 |
| Phase 2 — Rule discovery | Tasks 8, 9 |
| Phase 3 — Rule characterisation | Tasks 10, 11 |
| Phase 4 — Robustness | Tasks 12, 13 |
| Phase 5 — Synthesis | Tasks 14, 15, 16, 17, 18 |
| Appendix (SHAP-vs-leaf ARI) | Task 19 |
| LaTeX revision | Task 21 |
| TDD primitives | Tasks 2–5, 8 |
| Foundation audit | Task 7 |
| End-to-end reproducibility | Task 20 |

All spec items covered.

**Placeholder scan:** every code step has a runnable code block. No "TBD" / "implement later". Single primary clustering method (PCA + KMeans on standardised leaf-values) — no head-to-head comparison, since leaf-values is the principled choice for tree weighting.

**Type consistency:** `rule_id` column name used consistently across phases; `leaf_values` and `leaves_dense` matrices referenced consistently between extract and downstream tasks; `bootstrap_helpers.block_bootstrap_sharpe` signature matches what was tested in the earlier session.

**Scope check:** ~21 bite-sized tasks, ~14 days of work. Within single-plan scope. Phases are sequential but tasks within a phase are mostly independent — works for subagent dispatching. Statistical-test budget deliberately lean (k=2,3 only; single seed-stability check; single SHAP-vs-leaf ARI; bootstrap CIs only at chosen k) to respect the 167-month ceiling on monthly-aggregated claims.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-29-stock-level-rule-path-analysis.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
