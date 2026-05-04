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
    0: 'calm cont.',
    1: 'mild cont.',
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
        (r'$\pi$ 6-mo prior frequency',
         [f"{r['pi_panic_freq_6mo_mean']:.2f}" for _, r in df.iterrows()]),
        (r'$\pi$ 12-mo prior frequency',
         [f"{r['pi_panic_freq_12mo_mean']:.2f}" for _, r in df.iterrows()]),
    ])

    groups.append([
        (r'Cross-section avg stock return',
         [_fmt_pct(r['mom_overall_mean']) for _, r in df.iterrows()]),
        (r'Cross-section disp.\ (long, $h{=}9$--$12$)',
         [_fmt_2dp(r['cs_disp_long_mean']) for _, r in df.iterrows()]),
        (r'Cross-section skew.\ (short, $h{=}1$--$4$)',
         [_fmt_skew(r['cs_skew_short_mean']) for _, r in df.iterrows()]),
    ])

    groups.append([
        (r'Strategy trailing 12-mo Sharpe',
         [_fmt_2dp(r['past_sharpe_12mo_mean']) for _, r in df.iterrows()]),
    ])

    groups.append([
        (r'Long-leg avg z-score $\bar{z}$',
         [_fmt_signed_2dp(r['z_centroid_mean']) for _, r in df.iterrows()]),
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
        r'cluster size; regime context (current $\pi_t^{\text{filter}}$ and '
        r'prior 6-/12-month panic frequencies); cross-section state '
        r'(average-stock trailing return; dispersion at the long horizon band; '
        r'skewness at the short horizon band); the strategy\textquoteright s trailing 12-month '
        r"Sharpe entering the cluster's months; long-leg average z-score "
        r'$\bar{z}$ (the model\textquoteright s picks, averaged across the 12 momentum '
        r'horizons); and the cluster Sharpe (annualised, block bootstrap with '
        r'block size 6, 5{,}000 reps). '
        r'The horizon shown for dispersion and skewness is the one with the '
        r"largest cross-cluster spread, averaged across the cluster\textquoteright s months. "
        r'Intuitively, cross-section dispersion measures how much stocks differ '
        r'from each other in their trailing returns; cross-section skewness '
        r'measures whether a few outlier stocks dominate the cross-sectional '
        r'average return.'
    )

    tex_lines = [
        r'% Auto-generated by scripts/build_table_cluster_k4_descriptors.py -- DO NOT HAND EDIT.',
        r'\begin{table}[H]',
        r'\centering',
        r'\small',
        r'\begin{tabular}{l r r r r}',
        r'\toprule',
        header_line,
        r'\midrule',
    ]
    tex_lines.extend(body_lines)
    tex_lines += [
        r'\bottomrule',
        r'\end{tabular}',
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
