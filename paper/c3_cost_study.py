"""How much COST does widening the band in cluster-3 months save?

Cost is the OUTCOME here, not a nuisance inside an IR. Measuring cost directly
avoids the noise problem that makes net-IR comparisons hopeless (cost moves IR
by <=0.03 against a 0.054 noise floor; but cost itself is measured to <1bp).

Design:
  - PIT month clusters (k=4, expanding refit), label 3 = deepest-loser book.
  - Baseline = monthly rebalance = no-op band (E=10 == top-100 of top-1000).
  - Treatment = band widens to E_c3 ONLY in cluster-3 months; 8 levels.
  - Reference = static band at the same E in EVERY month (upper bound on the
    cost saving available from banding at that width).
  - Costing = DNMV / Novy-Marx-Velikov effective half-spread: cost = sum|dw|*hs.

Reports per strategy x level: annualized turnover, cost bp/yr, cost saved vs
monthly (bp/yr and %), and what it costs in gross/net return (the constraint).

Usage: .venv/bin/python -m paper.c3_cost_study
Output: paper/results/banding_study/c3_cost.{md,csv}
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)
warnings.filterwarnings('ignore')

from paper import config as C                                   # noqa: E402
from paper import execution as X                                # noqa: E402
from paper import banding_study as B                            # noqa: E402
from paper.src import clusters as CL                            # noqa: E402
from paper.src import signals                                   # noqa: E402
from paper.cluster_bands import _pit_month_labels               # noqa: E402

C3 = 3                                   # deepest-loser cluster label (0..3)
LEVELS = [10, 15, 20, 25, 30, 40, 60, 100]   # 10 = no widening (= baseline)


def _stats(r, led, sp_pack):
    """(ann turnover, cost bp/yr, gross %/yr, net %/yr) for one book."""
    cost, _, _ = B.price_ledger(led, sp_pack)
    cost = cost.reindex(r.index).fillna(0.0)
    net = B._net(r['gross'], cost)
    return (float(r['turnover'].mean() * 12),
            float(cost.mean() * 12 * 1e4),
            float(r['gross'].mean() * 12 * 100),
            float(net.mean() * 12 * 100))


def main():
    sp_pack = B.load_spreads()
    stocks = pd.read_parquet(C.STOCKS_PARQUET)
    sz = CL.build_stock_zcurves(stocks)
    rows = []
    for s in C.BS_STRATEGIES:
        x = signals.build(s)
        x = x[x['date'] >= C.SWEEP_START].reset_index(drop=True)
        bz = CL.book_zcurve(sz, x)
        mlab = _pit_month_labels(x, bz)
        x = x.assign(cluster=x['date'].map(mlab).fillna(0).astype(int))
        share_c3 = float((x.groupby('date')['cluster'].first() == C3).mean())

        # baseline: no-op band every month
        r0, l0 = X.simulate(x, ('nmv_band', (10, 10)), score_col='score')
        to0, c0, g0, n0 = _stats(r0, l0, sp_pack)

        for E in LEVELS:
            # widen ONLY in cluster-3 months
            arg = tuple(float(E if c == C3 else 10) for c in range(4))
            r1, l1 = X.simulate(x, ('cluster_band', arg), score_col='score')
            to1, c1, g1, n1 = _stats(r1, l1, sp_pack)
            # reference: same width in EVERY month
            r2, l2 = X.simulate(x, ('nmv_band', (E, E)), score_col='score')
            to2, c2, g2, n2 = _stats(r2, l2, sp_pack)
            rows.append({
                'strategy': s, 'E_c3': E, 'pct_mo_c3': round(share_c3, 3),
                'to_base': round(to0, 3), 'to_c3band': round(to1, 3),
                'to_always': round(to2, 3),
                'cost_base_bpyr': round(c0, 1), 'cost_c3band_bpyr': round(c1, 1),
                'cost_always_bpyr': round(c2, 1),
                'save_c3_bpyr': round(c0 - c1, 1),
                'save_c3_pct': round((c0 - c1) / c0 * 100, 1) if c0 else 0.0,
                'save_always_pct': round((c0 - c2) / c0 * 100, 1) if c0 else 0.0,
                'gross_base': round(g0, 2), 'gross_c3band': round(g1, 2),
                'net_base': round(n0, 2), 'net_c3band': round(n1, 2),
                'net_always': round(n2, 2)})
        print(f"[c3] {s}: base cost {c0:.1f}bp/yr, c3 months {share_c3:.0%} | "
              f"widest E=100 saves {rows[-1]['save_c3_pct']:.1f}% "
              f"(always-band saves {rows[-1]['save_always_pct']:.1f}%)",
              flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(C.BANDING_DIR, 'c3_cost.csv'), index=False)
    pd.set_option('display.width', 250)
    show = ['strategy', 'E_c3', 'to_base', 'to_c3band', 'cost_base_bpyr',
            'cost_c3band_bpyr', 'save_c3_bpyr', 'save_c3_pct',
            'save_always_pct', 'net_base', 'net_c3band', 'net_always']
    lines = [
        '# Cost saved by widening the band in cluster-3 months\n',
        'Cost is the OUTCOME, measured directly (DNMV/NMV effective half-spread,'
        ' cost = sum|dw|*hs). Baseline = no-op band (E=10) every month.'
        ' c3band = widen to E_c3 ONLY in deepest-loser-cluster months.'
        ' always = same width every month (the ceiling on banding savings).\n',
        R[show].to_string(index=False), '',
        'save_c3_pct  = cost cut by widening in c3 months only.',
        'save_always_pct = cost cut by banding at that width in EVERY month.',
        'The gap between them is the price of only banding 24% of the time.']
    with open(os.path.join(C.BANDING_DIR, 'c3_cost.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
