"""
historical_superbear_stress.py
==============================
Option C: Synthetic back-to-back super-bear stress test. Concatenates the
dot-com bust (32 months) and the GFC crash+rebound (27 months) into a single
59-month contiguous block representing a hypothetical scenario where the two
major historical bears occurred consecutively. Inserts this block into the
2011-2025 test period at every valid insertion point to measure distribution
of outcomes.

This is strictly more severe than any single historical bear -- it tests the
"prolonged economic deterioration" scenario referenced in the Limitations
subsection with actual historical returns (not simulated -3.51% flat).

Inputs:
    results/oos_returns_prod_1990_1999.csv  -- from historical_oos_production
    artefacts/cs_artefacts_data.pkl         -- main 2011-2025 M2 returns

Outputs:
    results/historical_superbear_results.csv
    tables/table_historical_superbear.tex
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, TABLES_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

OOS_RETURNS_PATH = os.path.join(RESULTS_DIR, 'oos_returns_prod_1990_1999.csv')
ARTEFACTS_PATH = 'artefacts/cs_artefacts_data.pkl'

DOTCOM = ('2000-03-01', '2002-10-31')
GFC    = ('2007-10-01', '2009-12-31')


def annualised_sharpe(r):
    r = pd.Series(r).dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return r.mean() / r.std() * np.sqrt(12)


def max_drawdown(r):
    cum = (1 + pd.Series(r).dropna()).cumprod()
    if len(cum) == 0:
        return np.nan
    return ((cum - cum.cummax()) / cum.cummax()).min()


def load_main_test_returns():
    with open(ARTEFACTS_PATH, 'rb') as f:
        art = pickle.load(f)
    strats = art['strategies_lo']
    xgb_keys = [k for k in strats if 'xgb' in k.lower()]
    r = strats[xgb_keys[0]] if xgb_keys else None
    if r is None:
        raise RuntimeError(f"No XGB key found. Keys: {list(strats.keys())}")
    print(f"  Using strategy key: {xgb_keys[0]}")
    r = pd.Series(r).dropna()
    r.index = pd.to_datetime(r.index)
    return r


def main():
    print("=" * 70)
    print("  HISTORICAL SUPER-BEAR STRESS TEST (Option C)")
    print("=" * 70)

    # Load OOS returns
    print("\n[1/3] Loading OOS returns and extracting bear blocks ...")
    oos = pd.read_csv(OOS_RETURNS_PATH, index_col='date', parse_dates=True)['ret']

    dotcom_mask = (oos.index >= DOTCOM[0]) & (oos.index <= DOTCOM[1])
    gfc_mask = (oos.index >= GFC[0]) & (oos.index <= GFC[1])

    dotcom = oos[dotcom_mask].values
    gfc = oos[gfc_mask].values

    print(f"  Dot-com: {len(dotcom)} months, mean={dotcom.mean() * 100:+.2f}%")
    print(f"  GFC:     {len(gfc)} months, mean={gfc.mean() * 100:+.2f}%")

    # Concatenate: dot-com immediately followed by GFC
    superbear = np.concatenate([dotcom, gfc])
    cum_sb = (1 + pd.Series(superbear)).prod() - 1
    print(f"  Super-bear (concatenated): {len(superbear)} months, "
          f"cumulative {cum_sb * 100:+.1f}%, "
          f"MDD {max_drawdown(superbear) * 100:+.1f}%")

    # Also try reversed order (GFC then dot-com)
    superbear_rev = np.concatenate([gfc, dotcom])

    # Load main test returns
    print("\n[2/3] Loading main test returns ...")
    main_r = load_main_test_returns()
    main_arr = main_r.values
    T = len(main_arr)
    print(f"  Main test: {T} months, baseline Sharpe {annualised_sharpe(main_arr):.2f}")

    # Insert super-bear
    print("\n[3/3] Inserting super-bear at every valid position ...")
    rows = []
    for label, block in [('Dot-com -> GFC', superbear),
                         ('GFC -> Dot-com', superbear_rev)]:
        N = len(block)
        if N > T:
            print(f"  {label}: SKIP (block {N} > main {T})")
            continue
        outcomes = []
        for insert_at in range(T - N + 1):
            simulated = main_arr.copy()
            simulated[insert_at:insert_at + N] = block
            outcomes.append((annualised_sharpe(simulated), max_drawdown(simulated)))
        sharpes = np.array([o[0] for o in outcomes])
        mdds = np.array([o[1] for o in outcomes])
        row = {
            'scenario': label,
            'block_months': N,
            'n_insertions': len(outcomes),
            'sharpe_median': np.nanmedian(sharpes),
            'sharpe_p05': np.nanpercentile(sharpes, 5),
            'sharpe_min': np.nanmin(sharpes),
            'sharpe_max': np.nanmax(sharpes),
            'mdd_min': np.nanmin(mdds),
            'mdd_median': np.nanmedian(mdds),
        }
        rows.append(row)
        print(f"  {label}: block={N} months, n={row['n_insertions']}, "
              f"Sharpe median={row['sharpe_median']:.2f}, "
              f"5%tile={row['sharpe_p05']:.2f}, "
              f"min={row['sharpe_min']:.2f}, "
              f"MDD min={row['mdd_min'] * 100:+.1f}%")

    df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, 'historical_superbear_results.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # LaTeX table
    tex = []
    tex.append(r'\begin{table}[H]')
    tex.append(r'\centering')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{l r r r r r r}')
    tex.append(r'\toprule')
    tex.append(r'Scenario & Months & $n$ & Median Sharpe & 5th-\%ile & Min Sharpe & Worst MDD \\')
    tex.append(r'\midrule')
    for _, row in df.iterrows():
        tex.append(
            f"{row['scenario']} & {int(row['block_months'])} & "
            f"{int(row['n_insertions'])} & "
            f"{row['sharpe_median']:.2f} & {row['sharpe_p05']:.2f} & "
            f"{row['sharpe_min']:.2f} & {row['mdd_min'] * 100:+.1f}\\% \\\\"
        )
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(
        r'\caption{Synthetic back-to-back super-bear stress test. The dot-com bust '
        r'(32 months, 2000--2002) and the GFC crash+rebound (27 months, 2007--2009) '
        r'are concatenated into a 59-month continuous bear block, then inserted at '
        r'every valid position in the 2011--2025 test period. This is strictly more '
        r'severe than any single historical bear observed in the sample.}'
    )
    tex.append(r'\label{tab:historical_superbear}')
    tex.append(r'\end{table}')

    tex_path = os.path.join(TABLES_DIR, 'table_historical_superbear.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex) + '\n')
    print(f"Saved: {tex_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
