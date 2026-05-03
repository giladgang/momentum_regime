"""
build_table_cluster_k4_short_leg.py
=====================================
Generates tables/table_cluster_k4_short_leg.tex for thesis §5.2.
Reads results/thesis/cluster_k4_short_leg_descriptors.csv and emits a
booktabs table summarising per-cluster short-leg z-curve structure:
month-1 z-score, mean z-score across all 12 horizons, and basket size.

This table is produced because the short-leg z-curves are NOT simply the
negative mirror of the long-leg z-curves: corr(short_z, -long_z) ranges from
-0.84 (cluster 3, deep crisis) to +0.94 (cluster 0, calm continuation),
indicating meaningful structural asymmetry that warrants thesis commentary.

Run:
    python scripts/build_table_cluster_k4_short_leg.py

Output:
    tables/table_cluster_k4_short_leg.tex
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
INPUT_PATH        = os.path.join(RESULTS_THESIS_DIR, 'cluster_k4_short_leg_descriptors.csv')
LONG_DESC_PATH    = os.path.join(RESULTS_THESIS_DIR, 'cluster_k4_descriptor_table.csv')
TEX_PATH          = os.path.join(TABLES_DIR, 'table_cluster_k4_short_leg.tex')

CLUSTER_LABELS = {
    0: 'calm continuation',
    1: 'mild continuation',
    2: 'post-panic recovery',
    3: 'deep crisis',
}

EXPECTED_COLS = {'cluster', 'n_months', 'short_basket_size'} | \
    {f'short_z_centroid_mom_{h}' for h in range(1, 13)}

EXPECTED_LONG_COLS = {'cluster'} | {f'z_centroid_mom_{h}' for h in range(1, 13)}


def _fmt_z(x):
    """Format z-score to 3 d.p. with explicit sign."""
    if np.isnan(x):
        return '---'
    sign = '$-$' if x < 0 else r'\phantom{$-$}'
    return f'{sign}{abs(x):.3f}'


def _fmt_corr(x):
    """Format correlation to 3 d.p. with explicit sign."""
    if np.isnan(x):
        return '---'
    sign = '$-$' if x < 0 else r'\phantom{$-$}'
    return f'{sign}{abs(x):.3f}'


def main():
    # ------------------------------------------------------------------
    # Load and validate inputs
    # ------------------------------------------------------------------
    for path in [INPUT_PATH, LONG_DESC_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'{path} not found. Run cluster_k4_short_leg_descriptors.py first.'
            )

    df = pd.read_csv(INPUT_PATH)
    long_desc = pd.read_csv(LONG_DESC_PATH)

    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f'cluster_k4_short_leg_descriptors.csv missing columns: {missing}')

    missing_long = EXPECTED_LONG_COLS - set(long_desc.columns)
    if missing_long:
        raise ValueError(f'cluster_k4_descriptor_table.csv missing columns: {missing_long}')

    df = df.sort_values('cluster').reset_index(drop=True)
    long_desc = long_desc.sort_values('cluster').reset_index(drop=True)
    horizons = list(range(1, 13))
    print(f'Loaded {len(df)} clusters')

    # ------------------------------------------------------------------
    # Compute per-cluster summary statistics
    # ------------------------------------------------------------------
    rows_data = []
    for _, row in df.iterrows():
        k = int(row['cluster'])
        long_row = long_desc[long_desc['cluster'] == k].iloc[0]

        short_zvec = np.array([row[f'short_z_centroid_mom_{h}'] for h in horizons])
        long_zvec  = np.array([long_row[f'z_centroid_mom_{h}'] for h in horizons])

        z_mom1   = row['short_z_centroid_mom_1']
        z_mean   = short_zvec.mean()
        basket   = row['short_basket_size']
        corr_neg = np.corrcoef(short_zvec, -long_zvec)[0, 1]

        rows_data.append({
            'cluster':  k,
            'label':    CLUSTER_LABELS[k],
            'n_months': int(row['n_months']),
            'z_mom1':   z_mom1,
            'z_mean':   z_mean,
            'basket':   basket,
            'corr_neg': corr_neg,
        })

    # ------------------------------------------------------------------
    # Build LaTeX rows
    # ------------------------------------------------------------------
    data_rows = []
    for r in rows_data:
        data_rows.append(
            f"    {r['cluster']+1} ({r['label']}) & {r['n_months']}"
            f" & {_fmt_z(r['z_mom1'])}"
            f" & {_fmt_z(r['z_mean'])}"
            f" & {r['basket']:.0f}"
            f" & {_fmt_corr(r['corr_neg'])} \\\\"
        )

    # ------------------------------------------------------------------
    # Assemble .tex
    # ------------------------------------------------------------------
    caption = (
        r'Short-leg z-curve summary by K=4 cluster, 2011--2024 test period. '
        r'$z_1$ is the cluster-mean cross-sectional z-score of short-leg picks '
        r'at the 1-month horizon; $\bar{z}$ averages across all 12 horizons. '
        r'Basket size is mean stocks per month. '
        r'$\rho(-z^L)$ is the correlation between the short-leg z-curve centroid '
        r'and the negated long-leg z-curve centroid; values below 0.95 indicate '
        r'structural asymmetry.'
    )

    tex_lines = [
        r'% Auto-generated by scripts/build_table_cluster_k4_short_leg.py -- DO NOT HAND EDIT.',
        r'\begin{table}[H]',
        r'\centering',
        r'\small',
        r'\begin{tabular}{l r r r r r}',
        r'\toprule',
        r'Cluster & $n$ & $z_1$ & $\bar{z}$ & Basket & $\rho(-z^L)$ \\',
        r'\midrule',
    ]
    tex_lines.extend(data_rows)
    tex_lines += [
        r'\bottomrule',
        r'\end{tabular}',
        f'\\caption{{{caption}}}',
        r'\label{tab:cluster_k4_short_leg}',
        r'\end{table}',
    ]

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(TEX_PATH, 'w') as f:
        f.write('\n'.join(tex_lines) + '\n')

    print(f'Wrote {TEX_PATH}')

    # Print summary for review
    print(f'\nPer-cluster short-leg summary:')
    print(f"{'Cluster':>8s} {'z_mom1':>8s} {'z_mean':>8s} {'basket':>8s} {'corr(-zL)':>10s}")
    for r in rows_data:
        print(f"  {r['cluster']:>6d} {r['z_mom1']:>+8.4f} {r['z_mean']:>+8.4f} "
              f"{r['basket']:>8.1f} {r['corr_neg']:>+10.4f}")


if __name__ == '__main__':
    main()
