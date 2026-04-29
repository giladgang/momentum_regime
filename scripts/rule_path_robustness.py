"""Phase 4: robustness checks for the rule-path clustering.

Note: seed-stability gating is already done in Phase 2 (cluster_rule_paths.py)
as part of the joint silhouette+ARI selection. This script is the
post-hoc reporting + Sharpe-CI consolidation.

Outputs:
  results/thesis/rule_path_robustness.csv
  results/thesis/rule_path_sharpe_ci.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RES_DIR = 'results/thesis'
SWEEP_CSV = f'{RES_DIR}/rule_path_clustering_sweep.csv'
CENTROIDS_CSV = f'{RES_DIR}/rule_path_centroids.csv'
ROBUST_CSV = f'{RES_DIR}/rule_path_robustness.csv'
CI_CSV = f'{RES_DIR}/rule_path_sharpe_ci.csv'


def main():
    print('Loading clustering sweep + centroid results...', flush=True)
    sweep = pd.read_csv(SWEEP_CSV)
    centroids = pd.read_csv(CENTROIDS_CSV)

    # ---- Robustness summary table ----
    rows = []
    for _, r in sweep.iterrows():
        rows.append({
            'metric': f'k={int(r["k"])}_mean_silhouette',
            'value': r['mean_silhouette'],
        })
        rows.append({
            'metric': f'k={int(r["k"])}_mean_ari',
            'value': r['mean_ari'],
        })
        rows.append({
            'metric': f'k={int(r["k"])}_stable',
            'value': r['stable_at_threshold'],
        })
        rows.append({
            'metric': f'k={int(r["k"])}_sizes_seed42',
            'value': r['sizes_seed42'],
        })
    robust = pd.DataFrame(rows)
    robust.to_csv(ROBUST_CSV, index=False)
    print(f'Saved: {ROBUST_CSV}', flush=True)
    print(robust.to_string(index=False))

    # ---- Per-rule Sharpe CI table ----
    ci_df = centroids[[
        'rule_id', 'n_stockmonths', 'n_plurality_months',
        'mean_pi', 'panic_share',
        'plurality_sharpe', 'plurality_sharpe_lo95', 'plurality_sharpe_hi95',
    ]].copy()
    ci_df['ci_excludes_zero'] = (ci_df['plurality_sharpe_lo95'] > 0)
    ci_df.to_csv(CI_CSV, index=False)
    print(f'\nSaved: {CI_CSV}', flush=True)
    print(ci_df.to_string(index=False))


if __name__ == '__main__':
    main()
