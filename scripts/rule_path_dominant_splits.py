"""Phase 3 (continued): dominant tree splits per rule.

For each rule, sample up to 500 of its stock-months. For each (stock, tree)
pair, trace the path from root to leaf and record every (feature, threshold,
direction) decision. Tally across all sampled paths within the rule. The
top-frequency decisions are the rule's logic.

Outputs:
  results/thesis/rule_path_dominant_splits.csv
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RES_DIR = 'results/thesis'
LABELS_CSV = f'{RES_DIR}/rule_path_labels.csv'
LEAF_PATH = 'artefacts/leaf_signatures.npz'
TREES_PATH = 'artefacts/pi_verify_trees_seeds50.pkl'
OUT_CSV = f'{RES_DIR}/rule_path_dominant_splits.csv'

MAX_STOCKMONTHS_PER_RULE = 500
TOP_N_SPLITS = 10


def trace_path_to_leaf(tree_dict, leaf_nid):
    """Return a list of (feature, threshold, direction) for the path from
    the root of tree_dict to leaf_nid. Direction is 'yes' (left, x<thr) or
    'no' (right, x>=thr).
    """
    # Build parent map and edge labels via BFS
    parent = {0: None}
    edge = {}  # child_nid -> (feature, threshold, direction)
    stack = [0]
    while stack:
        nid = stack.pop()
        node = tree_dict[nid]
        if node.get('leaf', False):
            continue
        for d in ('yes', 'no'):
            cnid = node[d]
            parent[cnid] = nid
            edge[cnid] = (node['feature'], node['threshold'], d)
            stack.append(cnid)
    # Walk from leaf back to root, collecting edges
    path = []
    cur = leaf_nid
    while parent.get(cur) is not None:
        path.append(edge[cur])
        cur = parent[cur]
    return path[::-1]


def main():
    print('Loading rule labels, leaf signatures, and trees...', flush=True)
    labels_df = pd.read_csv(LABELS_CSV, parse_dates=['date'])
    leaf_data = np.load(LEAF_PATH, allow_pickle=False)
    leaves_raw = leaf_data['leaves_raw']  # shape (n_long, n_trees), int16
    label_dates = leaf_data['date']
    label_permnos = leaf_data['permno']

    with open(TREES_PATH, 'rb') as f:
        trees_wrapped, seeds = pickle.load(f)
    n_trees = len(trees_wrapped)

    # Sanity check: labels_df should align with leaves_raw rows
    assert len(labels_df) == leaves_raw.shape[0], \
        f'Label count {len(labels_df)} != leaves_raw rows {leaves_raw.shape[0]}'

    rules = sorted(labels_df['rule_id'].unique())
    print(f'{len(rules)} rules', flush=True)

    rng = np.random.default_rng(42)
    split_rows = []
    for r in rules:
        rule_idx = np.where(labels_df['rule_id'].values == r)[0]
        if len(rule_idx) > MAX_STOCKMONTHS_PER_RULE:
            sampled = rng.choice(rule_idx, MAX_STOCKMONTHS_PER_RULE, replace=False)
        else:
            sampled = rule_idx
        print(f'  rule {r}: tracing {len(sampled)} stock-months × {n_trees} '
              f'trees...', flush=True)

        counts = {}  # (feature, threshold_round, direction) -> count
        total_paths = 0
        for sm in sampled:
            for ti in range(n_trees):
                leaf_nid = int(leaves_raw[sm, ti])
                path = trace_path_to_leaf(trees_wrapped[ti]['tree'], leaf_nid)
                for feat, thr, direction in path:
                    key = (feat, round(float(thr), 3), direction)
                    counts[key] = counts.get(key, 0) + 1
                total_paths += 1

        # Top splits by frequency
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:TOP_N_SPLITS]
        for (feat, thr, direction), c in top:
            split_rows.append({
                'rule_id': int(r),
                'feature': feat,
                'threshold': thr,
                'direction': direction,
                'frequency_pct': round(100 * c / total_paths, 1),
            })

    splits_df = pd.DataFrame(split_rows)
    splits_df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved: {OUT_CSV}', flush=True)
    print(f'\nTop splits per rule:')
    for r in rules:
        sub = splits_df[splits_df['rule_id'] == r]
        print(f'\nRule {r}:')
        print(sub.to_string(index=False))


if __name__ == '__main__':
    main()
