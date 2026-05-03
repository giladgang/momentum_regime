"""
build_table_cluster_k4_features_appendix.py
============================================
Generates tables/table_cluster_k4_features_appendix.tex for the thesis appendix.
Reads results/thesis/cluster_k4_descriptor_table.csv (57 columns, 4 rows) and
emits a booktabs tall table: 37 feature rows x 4 cluster columns, grouped into
three sections separated by \\midrule.

Run:
    python scripts/build_table_cluster_k4_features_appendix.py

Output:
    tables/table_cluster_k4_features_appendix.tex
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
TEX_PATH         = os.path.join(TABLES_DIR, 'table_cluster_k4_features_appendix.tex')

# ---------------------------------------------------------------------------
# Cluster column headers (same labelling as main-body descriptor table)
# ---------------------------------------------------------------------------
CLUSTER_LABELS = {
    0: r'0 (calm cont.)',
    1: r'1 (mild cont.)',
    2: r'2 (post-panic)',
    3: r'3 (deep crisis)',
}

# ---------------------------------------------------------------------------
# Feature row definitions: (csv_column, display_label, format_tag)
#   format_tag  "pct1"  -> multiply by 100, 1 d.p., + sign, append \%
#               "disp2" -> 2 d.p., no sign prefix
#               "skew1" -> 1 d.p., no sign prefix
#               "freq2" -> 2 d.p., no sign prefix
#               "sr2"   -> 2 d.p., sign alignment
#               "z2"    -> 2 d.p., +/- sign, phantom for alignment
# ---------------------------------------------------------------------------
SECTION_1 = [
    ('mom_overall_mean',        r'Cross-section mom (overall mean)',                          'pct1'),
    ('cs_mom_short_mean',       r'Cross-section mom (short tertile, $h=1$--$4$)',             'pct1'),
    ('cs_mom_mid_mean',         r'Cross-section mom (mid tertile, $h=5$--$8$)',               'pct1'),
    ('cs_mom_long_mean',        r'Cross-section mom (long tertile, $h=9$--$12$)',             'pct1'),
    ('cs_disp_short_mean',      r'Cross-section dispersion (short)',                          'disp2'),
    ('cs_disp_mid_mean',        r'Cross-section dispersion (mid)',                            'disp2'),
    ('cs_disp_long_mean',       r'Cross-section dispersion (long)',                           'disp2'),
    ('cs_skew_short_mean',      r'Cross-section skewness (short)',                            'skew1'),
    ('cs_skew_mid_mean',        r'Cross-section skewness (mid)',                              'skew1'),
    ('cs_skew_long_mean',       r'Cross-section skewness (long)',                             'skew1'),
    ('pi_panic_freq_6mo_mean',  r'$\pi^{\text{filter}}$ frequency, prior 6 months',          'freq2'),
    ('pi_panic_freq_12mo_mean', r'$\pi^{\text{filter}}$ frequency, prior 12 months',         'freq2'),
    ('past_sharpe_12mo_mean',   r'Strategy Sharpe, prior 12 months',                         'sr2'),
]

SECTION_2 = [
    (f'pick_mom_{h}_mean', rf'Long-leg pick mom, $h={h}$', 'pct1')
    for h in range(1, 13)
]

SECTION_3 = [
    (f'z_centroid_mom_{h}', rf'Long-leg z-curve, $h={h}$', 'z2')
    for h in range(1, 13)
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _sign_phantom(x, formatted_abs):
    """Return signed formatted string with \\phantom{$-$} for positive values."""
    if x < 0:
        return f'$-${formatted_abs}'
    else:
        return f'\\phantom{{$-$}}{formatted_abs}'


def _fmt(x, tag):
    """Format a scalar value according to format_tag."""
    if pd.isna(x):
        return '---'

    if tag == 'pct1':
        pct = x * 100.0
        # Treat values that round to 0.0 as unsigned zero (avoid "-0.0%")
        if abs(pct) < 0.05:
            return f'\\phantom{{$-$}}0.0\\%'
        formatted_abs = f'{abs(pct):.1f}\\%'
        if pct < 0:
            return f'$-${abs(pct):.1f}\\%'
        else:
            return f'\\phantom{{$-$}}{abs(pct):.1f}\\%'

    elif tag == 'disp2':
        return f'{x:.2f}'

    elif tag == 'skew1':
        return f'{x:.1f}'

    elif tag == 'freq2':
        return f'{x:.2f}'

    elif tag == 'sr2':
        formatted_abs = f'{abs(x):.2f}'
        return _sign_phantom(x, formatted_abs)

    elif tag == 'z2':
        formatted_abs = f'{abs(x):.2f}'
        return _sign_phantom(x, formatted_abs)

    else:
        raise ValueError(f'Unknown format tag: {tag}')


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def _build_row(label, col, tag, df_by_cluster):
    """Build a single LaTeX table row string."""
    cells = []
    for k in [0, 1, 2, 3]:
        val = df_by_cluster.loc[k, col]
        cells.append(_fmt(val, tag))
    return f'    {label} & {" & ".join(cells)} \\\\'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # Load and validate
    # ------------------------------------------------------------------
    if not os.path.exists(DESCRIPTORS_PATH):
        raise FileNotFoundError(
            f'{DESCRIPTORS_PATH} not found. '
            'Run the cluster descriptor generation scripts first.'
        )

    df = pd.read_csv(DESCRIPTORS_PATH)
    df = df.sort_values('cluster').set_index('cluster')

    # Check all required columns are present
    all_features = SECTION_1 + SECTION_2 + SECTION_3
    required_cols = {col for col, _, _ in all_features}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f'CSV missing columns: {sorted(missing)}')

    print(f'Loaded {len(df)} cluster rows; {len(df.columns)} columns.')

    # ------------------------------------------------------------------
    # Sanity checks (printed to stdout for verification)
    # ------------------------------------------------------------------
    mom_c0 = df.loc[0, 'mom_overall_mean']
    mom_c3 = df.loc[3, 'mom_overall_mean']
    skew_long_c1 = df.loc[1, 'cs_skew_long_mean']

    print(f'SANITY CHECK: cluster 0 mom_overall_mean  = {mom_c0:.4f}  (expect ~+0.1384, ~+14%)')
    print(f'SANITY CHECK: cluster 3 mom_overall_mean  = {mom_c3:.4f}  (expect ~-0.0926, ~-9%)')
    print(f'SANITY CHECK: cluster 1 cs_skew_long_mean = {skew_long_c1:.4f}  (expect ~8.1, highest)')

    assert abs(mom_c0 - 0.1384) < 0.001, f'Unexpected cluster 0 mom_overall: {mom_c0}'
    assert abs(mom_c3 - (-0.0926)) < 0.001, f'Unexpected cluster 3 mom_overall: {mom_c3}'
    assert abs(skew_long_c1 - 8.11) < 0.1, f'Unexpected cluster 1 cs_skew_long: {skew_long_c1}'
    print('All sanity checks passed.')

    # ------------------------------------------------------------------
    # Build LaTeX rows for each section
    # ------------------------------------------------------------------
    rows_s1 = [_build_row(label, col, tag, df) for col, label, tag in SECTION_1]
    rows_s2 = [_build_row(label, col, tag, df) for col, label, tag in SECTION_2]
    rows_s3 = [_build_row(label, col, tag, df) for col, label, tag in SECTION_3]

    total_rows = len(rows_s1) + len(rows_s2) + len(rows_s3)
    print(f'Table dimensions: {total_rows} rows x 4 cluster columns  (expected 37 x 4)')
    assert total_rows == 37, f'Expected 37 rows, got {total_rows}'

    # ------------------------------------------------------------------
    # Column header line
    # ------------------------------------------------------------------
    col_header_parts = [
        r'Feature',
        r'\multicolumn{1}{c}{Cluster 1}',
        r'\multicolumn{1}{c}{Cluster 2}',
        r'\multicolumn{1}{c}{Cluster 3}',
        r'\multicolumn{1}{c}{Cluster 4}',
    ]
    col_header = ' & '.join(col_header_parts) + r' \\'

    sub_header_parts = [
        r'',
        r'\multicolumn{1}{c}{\textit{calm cont.}}',
        r'\multicolumn{1}{c}{\textit{mild cont.}}',
        r'\multicolumn{1}{c}{\textit{post-panic}}',
        r'\multicolumn{1}{c}{\textit{deep crisis}}',
    ]
    sub_header = ' & '.join(sub_header_parts) + r' \\'

    # ------------------------------------------------------------------
    # Section label rows (span all 5 columns)
    # ------------------------------------------------------------------
    def section_header(text):
        return f'    \\multicolumn{{5}}{{l}}{{\\textit{{{text}}}}} \\\\'

    # ------------------------------------------------------------------
    # Caption
    # ------------------------------------------------------------------
    caption = (
        r'Per-cluster feature signature for the $K=4$ partition, 2011--2024 test period. '
        r'Features are grouped into three blocks: '
        r'(1) cross-section context features (regime composition, momentum landscape, '
        r'dispersion, skewness, recent strategy performance) computed at the month level '
        r'and averaged across each cluster\'s months; '
        r'(2) long-leg pick raw momentum at each horizon, averaged across each cluster\'s months; '
        r'(3) long-leg pick cross-sectional z-score (deviation from the cross-section mean '
        r'at each horizon, in units of cross-section std) averaged across cluster months. '
        r'Per-cluster Sharpes and other strategy outcomes appear in '
        r'Table~\ref{tab:cluster_k4_descriptors}; '
        r'this table is the full descriptive feature signature.'
    )

    # ------------------------------------------------------------------
    # Assemble .tex
    # ------------------------------------------------------------------
    tex_lines = [
        r'% Auto-generated by scripts/build_table_cluster_k4_features_appendix.py -- DO NOT HAND EDIT.',
        r'\begin{table}[H]',
        r'\centering',
        r'\small',
        r'\begin{tabular}{l r r r r}',
        r'\toprule',
        col_header,
        sub_header,
        r'\midrule',
        section_header('Cross-section context'),
    ]
    tex_lines.extend(rows_s1)
    tex_lines += [
        r'\midrule',
        section_header('Long-leg pick raw momentum by horizon'),
    ]
    tex_lines.extend(rows_s2)
    tex_lines += [
        r'\midrule',
        section_header('Long-leg pick cross-sectional z-score by horizon'),
    ]
    tex_lines.extend(rows_s3)
    tex_lines += [
        r'\bottomrule',
        r'\end{tabular}',
        f'\\caption{{{caption}}}',
        r'\label{tab:cluster_k4_features_appendix}',
        r'\end{table}',
    ]

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(TEX_PATH, 'w') as f:
        f.write('\n'.join(tex_lines) + '\n')

    print(f'Wrote {TEX_PATH}')


if __name__ == '__main__':
    main()
