"""Train the band adjustment AS A FUNCTION OF cluster state, walk-forward.

Distinct from cluster_bands.py (which used FIXED econ/GP per-cluster ratios):
here the per-cluster band widths are LEARNED from prior data each year --
argmax net-IR over a per-cluster band grid (each of the 4 clusters in
{10,20,40} -> 81 vectors), applied out-of-sample. This is the literal
"train band adjustment on cluster state" idea. Two selections:
- trained_argmax: raw argmax of the 81 vectors on prior data.
- trained_reg: 1-SE complexity-regularized (prefer fewer distinct widths) --
  neutralizes the 81-vs-5 search asymmetry vs static.
Compared to walk-forward-selected static. All 7 strategies, paired bootstrap
CI + BH-FDR. PIT expanding-refit month clusters.

Usage: .venv/bin/python -m paper.cluster_trained
Output: paper/results/banding_study/cluster_trained.{md,csv}
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from paper import config as C                                   # noqa: E402
from paper import execution as X                                # noqa: E402
from paper import banding_study as B                            # noqa: E402
from paper.src import metrics as M                              # noqa: E402
from paper.src import signals                                   # noqa: E402
from paper.src import clusters as CL                            # noqa: E402
from paper.gp_bands import _active                              # noqa: E402
from paper.cluster_bands import (_pit_month_labels, _boot_p)    # noqa: E402

LEVELS = [10, 20, 40]                       # per-cluster band grid
E_BASE = [10, 15, 20, 30, 40]
START_OOS = 2001
VECS = list(itertools.product(LEVELS, repeat=CL.K))    # 81 per-cluster vectors


def _walk_trained(vec_active, static_active, start_oos):
    """Each year: static = best-of-5 diagonal; trained_argmax = best-of-81
    vectors; trained_reg = simplest 81-vector within 1 Sharpe-SE of the best.
    All selected on prior data, stitched OOS."""
    V = pd.DataFrame(vec_active)             # months x 81 vectors (str keys)
    S = pd.DataFrame(static_active)          # months x 5
    yrs = [y for y in sorted({d.year for d in V.index}) if y >= start_oos]
    rec = {'static': [], 'trained_argmax': [], 'trained_reg': []}
    for y in yrs:
        trV, teV = V[V.index.year < y], V[V.index.year == y]
        trS, teS = S[S.index.year < y], S[S.index.year == y]
        if len(trV) < 60 or len(teV) == 0:
            continue

        def ir(fr, c):
            a = fr[c].dropna()
            return (a.mean() / a.std() * np.sqrt(12)
                    if len(a) > 12 and a.std() > 0 else -np.inf)
        rec['static'].append(teS[max(S.columns, key=lambda c: ir(trS, c))])
        irs = {c: ir(trV, c) for c in V.columns}
        best = max(irs.values())
        rec['trained_argmax'].append(teV[max(irs, key=irs.get)])
        ny = max(len(trV) / 12, 1)
        se = np.sqrt((1 + 0.5 * best ** 2) / ny)
        within = [c for c in V.columns if irs[c] >= best - se]

        def ndistinct(c):
            return len(set(eval(c)))
        rec['trained_reg'].append(teV[min(within,
                                  key=lambda c: (ndistinct(c), -irs[c]))])
    return {a: pd.concat(v).sort_index() for a, v in rec.items() if v}


def main():
    sp_pack = B.load_spreads()
    stocks = pd.read_parquet(C.STOCKS_PARQUET,
                             columns=['permno', 'date'] + CL.MOMS)
    stocks['date'] = pd.to_datetime(stocks['date'])
    stock_z = CL.build_stock_zcurves(stocks)
    rows = []
    for s in C.BS_STRATEGIES:
        x = signals.build(s)
        x = x[x['date'] >= C.SWEEP_START].reset_index(drop=True)
        start = START_OOS if s != 'xgb' else 2013
        book_z = CL.book_zcurve(stock_z, x)
        mlab = _pit_month_labels(x, book_z)
        x = x.assign(cluster=x['date'].map(mlab).fillna(0).astype(int))
        b, bled = X.simulate(x, ('benchmark', None), score_col='score')
        bench_net = B._net(b['gross'], B.price_ledger(bled, sp_pack)[0])

        def act(pol):
            return _active(x, pol, sp_pack, bench_net)
        vec_active = {str(v): act(('cluster_band', v)) for v in VECS}
        static_active = {e: act(('nmv_band', (e, e))) for e in E_BASE}
        st = _walk_trained(vec_active, static_active, start)
        z = pd.Series(0.0, index=st['static'].index)
        base = M.ir(st['static'], z)
        for a in ['trained_argmax', 'trained_reg']:
            d = (st[a] - st['static']).dropna()
            lo, hi = X.paired_block_bootstrap(
                st[a].reindex(d.index).astype(float),
                st['static'].reindex(d.index).astype(float))
            rows.append({'strategy': s, 'arm': a, 'oos_ir': M.ir(st[a], z),
                         'minus_static': M.ir(st[a], z) - base,
                         'excl0': bool(lo > 0 or hi < 0),
                         'p_boot': _boot_p(st[a].reindex(d.index),
                                           st['static'].reindex(d.index))})
        print(f'[ct] {s}: static {base:+.3f} | argmax '
              f'{M.ir(st["trained_argmax"], z) - base:+.3f} | reg '
              f'{M.ir(st["trained_reg"], z) - base:+.3f}', flush=True)
    R = pd.DataFrame(rows)
    p = R['p_boot'].sort_values()
    thr = (np.arange(1, len(p) + 1) / len(p)) * 0.05
    R['bh_sig'] = False
    R.loc[p.index, 'bh_sig'] = (p.values <= thr)
    R.round(4).to_csv(os.path.join(C.BANDING_DIR, 'cluster_trained.csv'),
                      index=False)
    lines = ['# Trained-per-cluster band adjustment vs static (honest WF)\n',
             'Per-cluster band LEARNED from prior data (argmax over 81 vectors,'
             ' clusters in {10,20,40}), applied OOS. trained_reg = 1-SE '
             'complexity-regularized (matches static search). PIT month '
             'clusters. BH-FDR q=0.05.\n',
             R.round(3).to_string(index=False), '',
             f'trained beats static: argmax '
             f'{int(((R.arm=="trained_argmax")&(R.minus_static>0)).sum())}/7, '
             f'reg {int(((R.arm=="trained_reg")&(R.minus_static>0)).sum())}/7. '
             f'CI excl 0: {int(R.excl0.sum())}/{len(R)}. '
             f'positive+BH-sig: '
             f'{int(((R.minus_static>0)&R.bh_sig).sum())}/{len(R)}.']
    with open(os.path.join(C.BANDING_DIR, 'cluster_trained.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
