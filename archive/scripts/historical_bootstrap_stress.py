"""
historical_bootstrap_stress.py
==============================
Option B: Historical-returns stress test. Upgrades the Section 5.6.3
simulation (flat -3.51% insertion) to use actual historical M2 bear-market
monthly returns, sampled in contiguous blocks.

Inputs (from historical_oos_production.py):
    results/oos_returns_prod_1990_1999.csv  -- per-month OOS L/S returns

Method:
    1. Load 2000-2010 OOS returns.
    2. Extract bear-period return segments (dot-com 2000-03 to 2002-10;
       GFC crash + rebound 2007-10 to 2009-12).
    3. Also load the main test-period returns (2011-2025) from the
       production artefacts.
    4. For each recession length N in {6, 12, 18, 24}, for each bear source
       (dot-com, GFC, pooled), for each sliding N-month contiguous block
       from the source, for each valid insertion point in 2011-2025:
         - Replace N consecutive months with the historical block.
         - Compute resulting Sharpe and max drawdown.
    5. Report worst-case Sharpe and MDD per (N, source) cell, plus
       percentile distribution.

Outputs:
    results/historical_stress_results.csv
    tables/table_historical_stress.tex

Usage:
    python scripts/historical_bootstrap_stress.py
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_FEE, RESULTS_DIR, TABLES_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


# ── Inputs ────────────────────────────────────────────────────────────────────

OOS_RETURNS_PATH = os.path.join(RESULTS_DIR, 'oos_returns_prod_1990_1999.csv')
ARTEFACTS_PATH = 'artefacts/cs_artefacts_data.pkl'

BEAR_SOURCES = {
    'Dot-com': ('2000-03-01', '2002-10-31'),
    'GFC':     ('2007-10-01', '2009-12-31'),
}

RECESSION_LENGTHS = [6, 12, 18, 24]


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    """Load M2's 2011-2025 L/S returns from production artefacts."""
    with open(ARTEFACTS_PATH, 'rb') as f:
        art = pickle.load(f)
    strats = art['strategies_lo']
    for key in ['M2_xgb', 'xgb_mom_pi', 'xgb', 'M2']:
        if key in strats:
            r = strats[key]
            break
    else:
        # fall back to any key containing 'xgb'
        xgb_keys = [k for k in strats if 'xgb' in k.lower()]
        if not xgb_keys:
            raise RuntimeError(f"Could not find XGB returns in artefacts; keys: {list(strats.keys())}")
        r = strats[xgb_keys[0]]
        print(f"  Using strategy key: {xgb_keys[0]}")
    r = pd.Series(r).dropna()
    r.index = pd.to_datetime(r.index)
    return r


def extract_bear_segments(oos_returns, sources):
    segments = {}
    for name, (start, end) in sources.items():
        mask = (oos_returns.index >= start) & (oos_returns.index <= end)
        seg = oos_returns[mask]
        segments[name] = seg.values
        print(f"  {name}: {len(seg)} months, "
              f"mean={seg.mean() * 100:+.2f}%, std={seg.std() * 100:.2f}%")
    # Pooled
    pooled = np.concatenate([segments[k] for k in segments])
    segments['Pooled'] = pooled
    print(f"  Pooled: {len(pooled)} months, "
          f"mean={pooled.mean() * 100:+.2f}%, std={pooled.std() * 100:.2f}%")
    return segments


def stress_one_config(main_returns, bear_segment, N):
    """
    For recession length N, slide every N-block from bear_segment into every
    valid insertion point in main_returns. Return list of (Sharpe, MDD) tuples.
    """
    main_arr = main_returns.values.copy()
    T = len(main_arr)
    seg = np.asarray(bear_segment)
    S = len(seg)

    if S < N:
        return []

    outcomes = []
    for block_start in range(S - N + 1):
        block = seg[block_start:block_start + N]
        for insert_at in range(T - N + 1):
            simulated = main_arr.copy()
            simulated[insert_at:insert_at + N] = block
            sh = annualised_sharpe(simulated)
            dd = max_drawdown(simulated)
            outcomes.append((sh, dd))
    return outcomes


def summarise_outcomes(outcomes):
    if not outcomes:
        return dict(n=0, sharpe_median=np.nan, sharpe_p05=np.nan, sharpe_min=np.nan,
                    mdd_median=np.nan, mdd_p05=np.nan, mdd_min=np.nan)
    sharpes = np.array([o[0] for o in outcomes])
    mdds = np.array([o[1] for o in outcomes])
    return dict(
        n=len(outcomes),
        sharpe_median=np.nanmedian(sharpes),
        sharpe_p05=np.nanpercentile(sharpes, 5),
        sharpe_min=np.nanmin(sharpes),
        mdd_median=np.nanmedian(mdds),
        mdd_p05=np.nanpercentile(mdds, 5),
        mdd_min=np.nanmin(mdds),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  HISTORICAL-RETURNS STRESS TEST (Option B)")
    print("=" * 70)

    # Load OOS returns
    print("\n[1/4] Loading OOS returns ...")
    if not os.path.exists(OOS_RETURNS_PATH):
        raise FileNotFoundError(
            f"Missing {OOS_RETURNS_PATH}. Run historical_oos_production.py first."
        )
    oos = pd.read_csv(OOS_RETURNS_PATH, index_col='date', parse_dates=True)['ret']
    print(f"  OOS returns: {len(oos)} months, "
          f"{oos.index.min().date()} to {oos.index.max().date()}")

    # Extract bear segments
    print("\n[2/4] Extracting bear-period segments ...")
    segments = extract_bear_segments(oos, BEAR_SOURCES)

    # Load main test returns
    print("\n[3/4] Loading main test returns (2011-2025) ...")
    main_r = load_main_test_returns()
    print(f"  Main test: {len(main_r)} months, "
          f"{main_r.index.min().date()} to {main_r.index.max().date()}")
    print(f"  Baseline Sharpe: {annualised_sharpe(main_r.values):.2f}")
    print(f"  Baseline MDD:    {max_drawdown(main_r.values) * 100:+.1f}%")

    # Stress test
    print("\n[4/4] Running stress test ...")
    rows = []
    for source_name in list(segments.keys()):
        seg = segments[source_name]
        for N in RECESSION_LENGTHS:
            if len(seg) < N:
                print(f"  {source_name:<10} N={N:>2}: SKIP (segment {len(seg)} < {N})")
                continue
            outcomes = stress_one_config(main_r, seg, N)
            s = summarise_outcomes(outcomes)
            rows.append({'source': source_name, 'N_months': N, **s})
            print(f"  {source_name:<10} N={N:>2}: "
                  f"n={s['n']:>5}, Sharpe median={s['sharpe_median']:.2f}, "
                  f"5%tile={s['sharpe_p05']:.2f}, min={s['sharpe_min']:.2f} | "
                  f"MDD min={s['mdd_min'] * 100:+.1f}%")

    # Save results
    out_path = os.path.join(RESULTS_DIR, 'historical_stress_results.csv')
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Build LaTeX table
    tex = []
    tex.append(r'\begin{table}[H]')
    tex.append(r'\centering')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{l r r r r r}')
    tex.append(r'\toprule')
    tex.append(r'Source & Months & Median Sharpe & 5th-\%ile Sharpe & Min Sharpe & Worst-case MDD \\')
    tex.append(r'\midrule')
    for _, row in df.iterrows():
        tex.append(
            f"{row['source']} & {int(row['N_months'])} & "
            f"{row['sharpe_median']:.2f} & {row['sharpe_p05']:.2f} & "
            f"{row['sharpe_min']:.2f} & {row['mdd_min'] * 100:+.1f}\\% \\\\"
        )
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(
        r'\caption{Historical-returns stress test. At each recession length, every '
        r'contiguous $N$-month block from the source is inserted at every valid '
        r'insertion point in the 2011--2025 test period, replacing the realised '
        r'returns. Reports distribution of resulting Sharpe ratios and minimum MDD '
        r'across all (block, insertion) combinations.}'
    )
    tex.append(r'\label{tab:historical_stress}')
    tex.append(r'\end{table}')

    tex_path = os.path.join(TABLES_DIR, 'table_historical_stress.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex) + '\n')
    print(f"Saved: {tex_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
