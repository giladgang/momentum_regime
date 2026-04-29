"""Foundation audit: verify leaf signatures align with thesis 5.3.1 numbers.

Three checks:
  1. Long-leg count is sane.
  2. 73.9% of trees contain a pi_filter split (matches thesis 5.3.1).
  3. Leaf-value column statistics are non-degenerate.

If any check fails, prints a clear diagnostic and exits non-zero so
downstream phases don't proceed on corrupt data.
"""
import os
import pickle
import sys

import numpy as np

LEAF_PATH = 'artefacts/leaf_signatures.npz'
TREES_PATH = 'artefacts/pi_verify_trees_seeds50.pkl'

EXPECTED_PI_SPLIT_RATE = 0.739
PI_SPLIT_TOLERANCE = 0.01


def main():
    print('Foundation audit: leaf signatures', flush=True)
    print('=' * 50, flush=True)

    # Load leaf signatures
    d = np.load(LEAF_PATH, allow_pickle=False)
    n_long, n_trees = d['leaf_values'].shape
    print(f'leaf_values shape: ({n_long}, {n_trees})', flush=True)
    print(f'leaves_dense shape: {d["leaves_dense"].shape}', flush=True)
    print(f'leaves_raw shape: {d["leaves_raw"].shape}', flush=True)

    failures = []

    # Check 1: long-leg count
    if not (50_000 < n_long < 90_000):
        failures.append(
            f'long-leg count {n_long} outside expected band (50k-90k)'
        )
    else:
        print(f'\n[OK] long-leg count {n_long} in expected band', flush=True)

    # Check 2: pi_filter split rate
    with open(TREES_PATH, 'rb') as f:
        trees, _ = pickle.load(f)
    n_with_pi = sum(1 for t in trees if t['has_pi'])
    rate = n_with_pi / len(trees)
    if abs(rate - EXPECTED_PI_SPLIT_RATE) > PI_SPLIT_TOLERANCE:
        failures.append(
            f'pi_filter split rate {rate:.3f} differs from thesis 5.3.1 '
            f'(expected {EXPECTED_PI_SPLIT_RATE} +/- {PI_SPLIT_TOLERANCE})'
        )
    else:
        print(f'[OK] pi_filter split rate {rate:.3f} matches thesis 5.3.1 '
              f'({EXPECTED_PI_SPLIT_RATE})', flush=True)

    # Check 3: leaf-value column stats are non-degenerate
    col_std = d['leaf_values'].std(axis=0)
    n_zero_var = int((col_std == 0).sum())
    if n_zero_var > n_trees * 0.05:
        failures.append(
            f'{n_zero_var} of {n_trees} trees ({100*n_zero_var/n_trees:.1f}%) '
            f'have zero per-stock variance — likely a bug'
        )
    else:
        print(f'[OK] {n_zero_var} zero-variance trees out of {n_trees} '
              f'(<5%, expected)', flush=True)
    print(f'     leaf-value stats: '
          f'min std={col_std.min():.5f}, '
          f'median std={np.median(col_std):.5f}, '
          f'max std={col_std.max():.5f}', flush=True)

    # Report
    print('\n' + '=' * 50, flush=True)
    if failures:
        print('AUDIT FAILED:', flush=True)
        for f in failures:
            print(f'  - {f}', flush=True)
        sys.exit(1)
    else:
        print('AUDIT PASSED — Phase 1 outputs verified.', flush=True)


if __name__ == '__main__':
    main()
