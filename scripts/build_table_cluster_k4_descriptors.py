"""
build_table_cluster_k4_descriptors.py
======================================
Generates tables/table_cluster_k4_descriptors.tex for thesis §5.2.
Reads results/thesis/cluster_k4_descriptor_table.csv and emits a booktabs table
in inverse layout (clusters as columns, features as rows). The features
displayed are the ones referenced in the §5.2 cluster analysis: regime
(current and prior panic frequencies), cross-section trend, dispersion,
skewness, trailing strategy Sharpe, long-leg z-curve mean, and per-cluster
Sharpe.

Run:
    python scripts/build_table_cluster_k4_descriptors.py

Output:
    tables/table_cluster_k4_descriptors.tex
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RESULTS_THESIS_DIR, TABLES_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DESCRIPTORS_PATH = os.path.join(RESULTS_THESIS_DIR, 'cluster_k4_descriptor_table.csv')
TEX_PATH         = os.path.join(TABLES_DIR, 'table_cluster_k4_descriptors.tex')

CLUSTER_LABELS = {
    0: 'calm',
    1: 'transition',
    2: 'post-panic',
    3: 'deep crisis',
}


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
def _fmt_2dp(x):
    if np.isnan(x):
        return '---'
    return f'$-${abs(x):.2f}' if x < 0 else f'{x:.2f}'


def _fmt_signed_2dp(x):
    """Always show sign for emphasis (e.g. z-curve mean)."""
    if np.isnan(x):
        return '---'
    sign = '$-$' if x < 0 else '$+$'
    return f'{sign}{abs(x):.2f}'


def _fmt_pct(x, decimals=0):
    if np.isnan(x):
        return '---'
    pct = x * 100
    sign = '$-$' if pct < 0 else '$+$'
    return f'{sign}{abs(pct):.{decimals}f}\\%'


def _fmt_skew(x):
    if np.isnan(x):
        return '---'
    return f'{x:.1f}'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(DESCRIPTORS_PATH):
        raise FileNotFoundError(f'{DESCRIPTORS_PATH} not found.')

    df = pd.read_csv(DESCRIPTORS_PATH).sort_values('cluster').reset_index(drop=True)
    print(f'Loaded {len(df)} clusters')

    # Group rows by what they describe: regime, cross-section state,
    # strategy state, picks, cluster outcome. Midrules separate groups.
    groups = []

    groups.append([
        (r'Months ($n$)',
         [str(int(r['n_months'])) for _, r in df.iterrows()]),
    ])

    groups.append([
        (r'$\bar{\pi}_t^{\text{filter}}$ (current panic prob.)',
         [f"{r['pi_panic_mean']:.2f}" for _, r in df.iterrows()]),
        (r'Cross-section avg stock return (12-mo)',
         [_fmt_pct(r['mom_overall_mean']) for _, r in df.iterrows()]),
    ])

    groups.append([
        (r'Cross-section std of 9--12 mo momentum',
         [_fmt_2dp(r['cs_disp_long_mean']) for _, r in df.iterrows()]),
        (r'Cross-section skewness of 1--4 mo momentum',
         [_fmt_skew(r['cs_skew_short_mean']) for _, r in df.iterrows()]),
    ])

    # Compute long-leg and short-leg avg pick momentum (across 12 horizons).
    long_pick_cols = [f'pick_mom_{i}_mean' for i in range(1, 13)]
    long_avg_mom = df[long_pick_cols].mean(axis=1)

    sl_path = os.path.join(RESULTS_THESIS_DIR, 'cluster_k4_short_leg_descriptors.csv')
    sl = pd.read_csv(sl_path).sort_values('cluster').reset_index(drop=True)
    short_pick_cols = [f'short_pick_mom_{i}_mean' for i in range(1, 13)]
    short_avg_mom = sl[short_pick_cols].mean(axis=1)

    groups.append([
        (r'Long-leg avg z-score $\bar{z}$',
         [_fmt_signed_2dp(r['z_centroid_mean']) for _, r in df.iterrows()]),
        (r'Long-leg avg trailing return',
         [_fmt_pct(v) for v in long_avg_mom]),
        (r'Short-leg avg trailing return',
         [_fmt_pct(v) for v in short_avg_mom]),
    ])

    groups.append([
        (r'Cluster Sharpe',
         [_fmt_2dp(r['sharpe']) for _, r in df.iterrows()]),
    ])

    cluster_headers = []
    for _, r in df.iterrows():
        k = int(r['cluster'])
        label = CLUSTER_LABELS[k]
        cluster_headers.append(f"{k+1} ({label})")

    header_line = 'Feature & ' + ' & '.join(cluster_headers) + r' \\'

    body_lines = []
    for i, group in enumerate(groups):
        for label, vals in group:
            body_lines.append(f'    {label} & ' + ' & '.join(vals) + r' \\')
        if i < len(groups) - 1:
            body_lines.append(r'    \midrule')

    caption = (
        r'$K=4$ cluster descriptors, 2011--2025 test period, '
        r'$n_{\text{total}} = 167$ months. '
        r'Each column is one cluster; rows are grouped (top to bottom): '
        r'cluster size; regime-routing inputs (current $\pi_t^{\text{filter}}$ '
        r"and cross-section average stock return, the two features that "
        r'jointly determine which selection mode the model commits to); '
        r'descriptive cross-section state (cross-stock standard deviation '
        r'of 9--12 month momentum, cross-stock skewness of 1--4 month '
        r'momentum); the picks themselves (long-leg average z-score $\bar{z}$, '
        r'and the average trailing return of the long and short legs, all '
        r'averaged across the 12 momentum horizons); and the cluster Sharpe '
        r'(annualised, block bootstrap with block size 6, 5{,}000 reps). '
        r'The 9--12 month and 1--4 month windows are chosen as the ones '
        r'with the largest cross-cluster spread for each measure.'
    )

    tex_lines = [
        r'% Auto-generated by scripts/build_table_cluster_k4_descriptors.py -- DO NOT HAND EDIT.',
        r'\begin{table}[H]',
        r'\centering',
        r'\small',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{l r r r r}',
        r'\toprule',
        header_line,
        r'\midrule',
    ]
    tex_lines.extend(body_lines)
    tex_lines += [
        r'\bottomrule',
        r'\end{tabular}%',
        r'}',
        f'\\caption{{{caption}}}',
        r'\label{tab:cluster_k4_descriptors}',
        r'\end{table}',
    ]

    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(TEX_PATH, 'w') as f:
        f.write('\n'.join(tex_lines) + '\n')

    print(f'Wrote {TEX_PATH}')


if __name__ == '__main__':
    main()
