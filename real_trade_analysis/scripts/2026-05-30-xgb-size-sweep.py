"""
2026-05-30-xgb-size-sweep.py
============================
EXPLORATORY (experiments/ -- throwaway).

Follow-up to 2026-05-30-xgb-russell1000.py: sweep the size cutoff to see where
the XGB long-short edge starts to fade between the full universe (Sharpe ~1.08)
and a mega-cap Top-1000 universe (Sharpe ~0.60).

Universes (largest -> smallest):
  Full          -- reconstructed from the prior run's saved returns CSV (no retrain)
  NYSE-median   -- Fama-French "large-cap": me >= monthly NYSE-listed median me
  Top-3000      -- ~Russell 3000 proxy
  Top-2000
  Top-1000      -- ~Russell 1000 proxy
  Top-500       -- ~S&P 500 proxy

Reuses build_features / apply_size_screen / run_xgb_ls / metrics helpers from
the russell1000 experiment module (its main() is __main__-guarded, so importing
it runs nothing). Same isolation guarantees: reads only data inputs, writes only
its own CSV next to this file.
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

# Load the sibling experiment module by path (filename starts with a date, so a
# normal import statement won't work).
_spec = importlib.util.spec_from_file_location(
    "xgb_russell1000",
    os.path.join(_ROOT, 'real_trade_analysis', 'scripts', '2026-05-30-xgb-russell1000.py'))
exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp)

from src.utils import metrics

_REGIME_PATH = os.path.join(_ROOT, 'data', 'panel_with_regimes.parquet')
_FULL_RET    = os.path.join(_ROOT, 'real_trade_analysis', 'results',
                            '2026-05-30-xgb-russell1000_returns_full.csv')


def _pi_by_month():
    """Monthly pi_filter indexed by date, for the calm/panic split."""
    r = pd.read_parquet(_REGIME_PATH)[['date', 'pi_filter']]
    r['date'] = pd.to_datetime(r['date'])
    return r.drop_duplicates('date').set_index('date')['pi_filter']


def row_from_returns(returns, label):
    """Build a results row (metrics + regime split) from a monthly return series."""
    r = returns.dropna()
    pim = _pi_by_month().reindex(r.index)
    calm, panic = r[pim < 0.5], r[pim >= 0.5]
    sh = lambda x: x.mean() / x.std() * np.sqrt(12) if len(x) > 1 and x.std() > 0 else np.nan
    ann, vol, sharpe, mdd = metrics(r)
    return {'label': label, 'months': len(r),
            'n_calm': int((pim < 0.5).sum()), 'n_panic': int((pim >= 0.5).sum()),
            'ann_ret': ann, 'vol': vol, 'sharpe': sharpe, 'mdd': mdd,
            'calm_sharpe': sh(calm), 'panic_sharpe': sh(panic)}


def apply_nyse_median(df):
    """Fama-French large-cap cut: keep stocks with me >= the monthly NYSE median.

    The breakpoint is the median market equity among NYSE-listed (exchcd==1)
    stocks that month; AMEX/NASDAQ names above it are kept too (standard FF).
    """
    sized = df[df['me'].notna()].copy()
    nyse_med = (sized[sized['exchcd'] == 1]
                .groupby('date')['me'].median().rename('_bp'))
    sized = sized.merge(nyse_med, on='date', how='inner')
    return sized[sized['me'] >= sized['_bp']].drop(columns='_bp').reset_index(drop=True)


def main():
    df = exp.build_features()
    print(f"Eligible stock-months: {len(df):,}  "
          f"({df['date'].min().date()} -> {df['date'].max().date()})\n")

    results = []

    # Full universe: reuse the prior run's exact returns (no retrain).
    full_ret = pd.read_csv(_FULL_RET, parse_dates=['date']).set_index('date')['ret']
    results.append(row_from_returns(full_ret, 'Full universe'))
    print("Full universe: loaded from saved returns (no retrain).")

    # NYSE-median large-cap cut.
    nyse = apply_nyse_median(df)
    avg_n = round(len(nyse[nyse['date'] >= exp.TRAIN_END]) /
                  nyse[nyse['date'] >= exp.TRAIN_END]['date'].nunique())
    results.append(exp.run_xgb_ls(nyse, f'NYSE-median (~{avg_n}/mo)'))

    # Top-N proxies.
    for n in (3000, 2000, 1000, 500):
        sub = exp.apply_size_screen(df, n)
        results.append(exp.run_xgb_ls(sub, f'Top-{n}'))

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  XGB LONG-SHORT (mom + pi):  SIZE-CUTOFF SWEEP  (test 2011-01..2024-11)")
    print("=" * 80)
    print(f"  {'Universe':<24} {'Mo':>4} {'AnnRet':>8} {'Vol':>7} {'Sharpe':>7} "
          f"{'MDD':>8} {'CalmSh':>7} {'PanicSh':>8}")
    print("  " + "-" * 76)
    for r in results:
        print(f"  {r['label']:<24} {r['months']:>4} {r['ann_ret']:>7.1%} {r['vol']:>6.1%} "
              f"{r['sharpe']:>7.2f} {r['mdd']:>7.1%} {r['calm_sharpe']:>7.2f} {r['panic_sharpe']:>8.2f}")
    print("=" * 80)

    out = os.path.join(_ROOT, 'real_trade_analysis', 'results', '2026-05-30-xgb-size-sweep_summary.csv')
    pd.DataFrame([{k: v for k, v in r.items() if k != 'returns'} for r in results]).to_csv(out, index=False)
    print(f"  saved {out}")


if __name__ == '__main__':
    main()
