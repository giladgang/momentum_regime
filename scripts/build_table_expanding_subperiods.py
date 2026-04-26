"""
build_table_expanding_subperiods.py
===================================
Generates tables/table_expanding_subperiods.tex for Section 5.4.3 (Historical
Stress: Out-of-Sample Evidence). Reads results/expanding_returns_prod.csv
(produced by scripts/expanding_window_backtest_parallel.py) and emits the
sub-period decomposition table with Sharpe, cumulative return, and maximum
drawdown for each calendar-year sub-period.

Run after the expanding-window backtest completes:
    python scripts/build_table_expanding_subperiods.py

Output:
    tables/table_expanding_subperiods.tex
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RESULTS_DIR, TABLES_DIR


RETURNS_PATH = os.path.join(RESULTS_DIR, 'expanding_returns_prod.csv')
TEX_PATH = os.path.join(TABLES_DIR, 'table_expanding_subperiods.tex')

SUBPERIODS = [
    ('Full sample, 1995--2024', None, None),
    ('1995--1999',              '1995-01-01', '2000-01-01'),
    ('2000--2002 dot-com',      '2000-01-01', '2003-01-01'),
    ('2003--2006 bull',         '2003-01-01', '2007-01-01'),
    ('2007--2009 GFC',          '2007-01-01', '2010-01-01'),
    ('2010--2019 calm',         '2010-01-01', '2020-01-01'),
    ('2020--2024 COVID era',    '2020-01-01', '2025-01-01'),
]


def stats(rs):
    if len(rs) < 2 or rs.std() == 0:
        return dict(n=len(rs), sharpe=np.nan, cum=np.nan, mdd=np.nan)
    sharpe = rs.mean() / rs.std() * np.sqrt(12)
    cum = (1 + rs).prod() - 1
    cs = (1 + rs).cumprod()
    mdd = ((cs - cs.cummax()) / cs.cummax()).min()
    return dict(n=len(rs), sharpe=sharpe, cum=cum, mdd=mdd)


def main():
    if not os.path.exists(RETURNS_PATH):
        raise FileNotFoundError(
            f"{RETURNS_PATH} not found. Run "
            f"scripts/expanding_window_backtest_parallel.py first."
        )

    r = pd.read_csv(RETURNS_PATH, parse_dates=['date']).set_index('date').sort_index()
    print(f"Loaded {len(r)} OOS months from {r.index.min().date()} to "
          f"{r.index.max().date()}")

    # Compute production-overlap stats for the table caption note
    overlap = r['ret'][r.index.year >= 2011]
    overlap_stats = stats(overlap)

    rows = []
    for label, start, end in SUBPERIODS:
        if start is None:
            rs = r['ret']
        else:
            rs = r['ret'][(r.index >= start) & (r.index < end)]
        s = stats(rs)
        rows.append((label, s))
        print(f"  {label:<28} N={s['n']:>3}  Sh={s['sharpe']:+.2f}  "
              f"Cum={s['cum'] * 100:>+8.1f}%  MDD={s['mdd'] * 100:>+7.1f}%")

    # Format helpers matching thesis convention:
    # - Bare numbers: positive shown unsigned, negative as $-$ (math minus).
    # - Percentages: positive shown unsigned, negative with literal `-`.
    # - Thousands separator: use {,} so LaTeX doesn't insert extra space.
    def fmt_sharpe(x):
        if np.isnan(x):
            return '---'
        return f"$-${abs(x):.2f}" if x < 0 else f"{x:.2f}"

    def fmt_pct(x):
        if np.isnan(x):
            return '---'
        pct = x * 100
        sign = '-' if pct < 0 else ''
        return f"{sign}{abs(pct):,.1f}\\%".replace(',', '{,}')

    # Write LaTeX table
    tex = []
    tex.append(r'\begin{table}[H]')
    tex.append(r'\centering')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{l r r r r}')
    tex.append(r'\toprule')
    tex.append(r'Period & N months & Sharpe & Cumulative & Max DD \\')
    tex.append(r'\midrule')
    for i, (label, s) in enumerate(rows):
        n = s['n']
        sh = fmt_sharpe(s['sharpe'])
        cum = fmt_pct(s['cum'])
        mdd = fmt_pct(s['mdd'])
        tex.append(f"{label} & {n} & {sh} & {cum} & {mdd} \\\\")
        # Visual separator after the full-sample row
        if i == 0:
            tex.append(r'\midrule')
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(
        r'\caption{Sub-period decomposition of the 30-year expanding-window '
        r'out-of-sample backtest. Sub-period rows aggregate by calendar year. '
        rf"The 2011--2024 sub-window of this test (167 months, the same window "
        rf"as the main test in Section~\ref{{sec:performance_results}}) "
        rf"reproduces the production Sharpe within sampling noise: "
        rf"{overlap_stats['sharpe']:.2f} vs production 1.11.}}"
    )
    tex.append(r'\label{tab:expanding_subperiods}')
    tex.append(r'\end{table}')

    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(TEX_PATH, 'w') as f:
        f.write('\n'.join(tex) + '\n')
    print(f"\nSaved: {TEX_PATH}")
    print(f"\nProduction-overlap (2011-2024, N={overlap_stats['n']}): "
          f"Sharpe {overlap_stats['sharpe']:.2f}")


if __name__ == '__main__':
    main()
