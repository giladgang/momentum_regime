"""
oos_panic_subtypes.py
=====================
Out-of-sample panic-subtype decomposition for M2.

Tests whether the thesis's core panic sub-type finding (Section 5.2.3 of
main_results.tex) -- panic months split into crash (market down, M2 bleeds)
vs recovery (market up, M2 strong) -- holds on the OOS 2000-2010 sample.

Inputs:
    results/oos_returns_prod_1990_1999.csv   -- M2 OOS returns
    results/oos_pi_filter_prod_1990_1999.csv -- OOS-estimated pi_filter
    data/panel.parquet                       -- market return (vwretd)

Outputs:
    results/oos_panic_subtypes.csv
    tables/table_oos_panic_subtypes.tex
"""

import argparse
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RESULTS_DIR, TABLES_DIR

_ap = argparse.ArgumentParser()
_ap.add_argument('--returns', default='oos_returns_prod_1990_1999.csv',
                 help='M2 OOS returns CSV in results/')
_ap.add_argument('--pi', default='oos_pi_filter_prod_1990_1999.csv',
                 help='OOS pi_filter CSV in results/')
_ap.add_argument('--tag', default='1990_1999',
                 help='Output tag')
ARGS = _ap.parse_args()


def annualised_sharpe(r):
    r = pd.Series(r).dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return r.mean() / r.std() * np.sqrt(12)


def main():
    print("=" * 70)
    print("  OOS PANIC SUBTYPE DECOMPOSITION")
    print("=" * 70)

    # Load M2 OOS returns
    m2 = pd.read_csv(
        os.path.join(RESULTS_DIR, ARGS.returns),
        index_col='date', parse_dates=True,
    )['ret']
    m2.index = pd.to_datetime(m2.index)

    # Load OOS pi_filter
    pi = pd.read_csv(
        os.path.join(RESULTS_DIR, ARGS.pi),
        parse_dates=['date'],
    ).set_index('date')['pi_filter']
    pi.index = pd.to_datetime(pi.index)

    # Load market returns
    panel = pd.read_parquet('data/panel.parquet')
    panel['date'] = pd.to_datetime(panel['date'])
    mkt = (panel[['date', 'vwretd']]
           .dropna()
           .drop_duplicates('date')
           .sort_values('date')
           .set_index('date')['vwretd'])

    # Align on M2 dates (m2 dates are month-end)
    df = pd.DataFrame({'ret': m2})
    df['pi'] = pi.reindex(df.index, method='nearest').values
    df['mkt'] = mkt.reindex(df.index, method='nearest').values
    df = df.dropna()

    print(f"\nAligned OOS months: {len(df)} "
          f"({df.index.min().date()} to {df.index.max().date()})")
    print(f"  pi range: {df['pi'].min():.2f} to {df['pi'].max():.2f}")
    print(f"  mkt range: {df['mkt'].min() * 100:+.1f}% to {df['mkt'].max() * 100:+.1f}%")

    # Classify regime and panic subtype
    df['regime'] = np.where(df['pi'] >= 0.5, 'Panic', 'Calm')
    df['panic_subtype'] = np.where(
        df['regime'] == 'Calm',
        '-',
        np.where(df['mkt'] < 0, 'Crash', 'Recovery'),
    )

    # Overall counts
    print("\nRegime classification counts:")
    print(df['regime'].value_counts().to_string())
    print("\nPanic subtype counts:")
    print(df['panic_subtype'].value_counts().to_string())

    # Compute bucket stats
    print("\n" + "=" * 90)
    print("  OOS M2 PERFORMANCE BY REGIME AND PANIC SUBTYPE")
    print("=" * 90)
    print(f"{'Bucket':<20} {'N':>4} {'Mean':>10} {'Cum':>10} {'Sharpe':>10}")
    print("-" * 90)

    buckets = []

    def add_bucket(label, mask):
        sub = df[mask]
        if len(sub) == 0:
            return
        m = sub['ret'].mean()
        c = (1 + sub['ret']).prod() - 1
        s = annualised_sharpe(sub['ret'].values)
        buckets.append({
            'bucket': label,
            'n': len(sub),
            'mean_pct': m * 100,
            'cum_pct': c * 100,
            'sharpe': s,
        })
        print(f"  {label:<18} {len(sub):>4} "
              f"{m * 100:>+9.2f}% {c * 100:>+9.1f}% {s:>10.2f}")

    add_bucket('Overall',          np.ones(len(df), dtype=bool))
    add_bucket('Calm',              df['regime'] == 'Calm')
    add_bucket('Panic (all)',       df['regime'] == 'Panic')
    add_bucket('Panic crash',       df['panic_subtype'] == 'Crash')
    add_bucket('Panic recovery',    df['panic_subtype'] == 'Recovery')

    out = pd.DataFrame(buckets)
    out_path = os.path.join(RESULTS_DIR, f'oos_panic_subtypes_{ARGS.tag}.csv')
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Compare against thesis's main-test numbers (Section 5.2.3)
    print("\n" + "=" * 70)
    print("  COMPARISON: OOS vs MAIN TEST (2011-2025)")
    print("=" * 70)
    print(f"{'Bucket':<20} {'OOS mean':>10} {'Main mean':>12} {'OOS Sharpe':>12} {'Main Sharpe':>12}")
    main_test_numbers = {
        'Calm':           {'mean': 1.3 / 100, 'sharpe': 0.82},
        'Panic (all)':    {'mean': 3.09 / 100, 'sharpe': 1.56},
        'Panic crash':    {'mean': -0.49 / 100, 'sharpe': None},
        'Panic recovery': {'mean': 4.42 / 100, 'sharpe': 2.35},
    }
    for b in buckets:
        label = b['bucket']
        if label in main_test_numbers:
            mt = main_test_numbers[label]
            mt_mean = f"{mt['mean'] * 100:+.2f}%"
            mt_sharpe = f"{mt['sharpe']:.2f}" if mt['sharpe'] is not None else '---'
            oos_mean = f"{b['mean_pct']:+.2f}%"
            oos_sharpe = f"{b['sharpe']:.2f}"
            print(f"  {label:<18} {oos_mean:>10} {mt_mean:>12} "
                  f"{oos_sharpe:>12} {mt_sharpe:>12}")

    # LaTeX table
    tex = []
    tex.append(r'\begin{table}[H]')
    tex.append(r'\centering')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{l r r r r}')
    tex.append(r'\toprule')
    tex.append(r'Bucket & N & Monthly mean & Cumulative & Sharpe \\')
    tex.append(r'\midrule')
    for b in buckets:
        tex.append(
            f"{b['bucket']} & {b['n']} & "
            f"{b['mean_pct']:+.2f}\\% & {b['cum_pct']:+.1f}\\% & "
            f"{b['sharpe']:.2f} \\\\"
        )
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(
        r'\caption{Out-of-sample panic-subtype decomposition (2000--2010, M2 '
        r"trained on 1990--1999). Panic months ($\pi_t^{\text{filter}} \geq "
        r'0.5$, estimated by the OOS-fitted HMM) are split by concurrent market '
        r'return into crash (market down) and recovery (market up) buckets. '
        r'Compare with Section~\ref{sec:panic} main-test decomposition.}'
    )
    tex.append(r'\label{tab:oos_panic_subtypes}')
    tex.append(r'\end{table}')

    tex_path = os.path.join(TABLES_DIR, f'table_oos_panic_subtypes_{ARGS.tag}.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex) + '\n')
    print(f"\nSaved: {tex_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
