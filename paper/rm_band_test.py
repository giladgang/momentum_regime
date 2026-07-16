"""Band width = f(regime x past-momentum), honest walk-forward, all 7.

The precise "condition banding on regime AND past momentum" test. 4-state band
(calm/panic x high/low-momentum). Three fits vs walk-forward-selected static:
  econ    : panic low-momentum (beaten-down) names get a WIDER band (hold);
            one global scale E -> arg (E,E,E, min(2E,100)). Matched to static.
  argmax  : learn the 4-cell band from prior data (81 vectors in {10,20,40}).
  reg     : 1-SE complexity-regularized (matches static search freedom).
Paired block-bootstrap CI + BH-FDR.

Usage: .venv/bin/python -m paper.rm_band_test
Output: paper/results/banding_study/rm_band.{md,csv}
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
from paper.gp_bands import _active                              # noqa: E402
from paper.cluster_bands import _walk, _boot_p, E_BASE          # noqa: E402
from paper.cluster_trained import _walk_trained                 # noqa: E402

START_OOS = 2001
VECS = list(itertools.product([10, 20, 40], repeat=4))   # 81 rm_band vectors


def main():
    sp_pack = B.load_spreads()
    rows = []
    for s in C.BS_STRATEGIES:
        x = signals.build(s)
        x = x[x['date'] >= C.SWEEP_START].reset_index(drop=True)
        start = START_OOS if s != 'xgb' else 2013
        b, bled = X.simulate(x, ('benchmark', None), score_col='score')
        bench_net = B._net(b['gross'], B.price_ledger(bled, sp_pack)[0])

        def act(pol):
            return _active(x, pol, sp_pack, bench_net)
        econ = _walk({'a': pd.DataFrame({e: act(
            ('rm_band', (e, e, e, float(np.clip(2 * e, 5, 100)))))
            for e in E_BASE})}, start)['a']
        vec_active = {str(v): act(('rm_band', v)) for v in VECS}
        static_active = {e: act(('nmv_band', (e, e))) for e in E_BASE}
        st = _walk_trained(vec_active, static_active, start)
        st['econ'] = econ.reindex(st['static'].index)
        z = pd.Series(0.0, index=st['static'].index)
        base = M.ir(st['static'], z)
        for a in ['econ', 'trained_argmax', 'trained_reg']:
            d = (st[a] - st['static']).dropna()
            lo, hi = X.paired_block_bootstrap(
                st[a].reindex(d.index).astype(float),
                st['static'].reindex(d.index).astype(float))
            rows.append({'strategy': s, 'arm': a, 'oos_ir': M.ir(st[a], z),
                         'minus_static': M.ir(st[a], z) - base,
                         'excl0': bool(lo > 0 or hi < 0),
                         'p_boot': _boot_p(st[a].reindex(d.index),
                                           st['static'].reindex(d.index))})
        print(f"[rm] {s}: static {base:+.3f} | econ "
              f"{M.ir(st['econ'], z)-base:+.3f} | argmax "
              f"{M.ir(st['trained_argmax'], z)-base:+.3f} | reg "
              f"{M.ir(st['trained_reg'], z)-base:+.3f}", flush=True)
    R = pd.DataFrame(rows)
    p = R['p_boot'].sort_values()
    thr = (np.arange(1, len(p) + 1) / len(p)) * 0.05
    R['bh_sig'] = False
    R.loc[p.index, 'bh_sig'] = (p.values <= thr)
    R.round(4).to_csv(os.path.join(C.BANDING_DIR, 'rm_band.csv'), index=False)
    lines = ['# Band = f(regime x past-momentum), honest WF, all 7\n',
             'econ = panic low-mom wider (one scale); argmax/reg = learn the '
             '4-cell band (81 vectors). vs WF-selected static. BH-FDR q=0.05.\n',
             R.round(3).to_string(index=False), '',
             f'Positive vs static: {int((R.minus_static>0).sum())}/{len(R)}. '
             f'CI excl 0: {int(R.excl0.sum())}/{len(R)}. '
             f'positive+BH-sig: '
             f'{int(((R.minus_static>0)&R.bh_sig).sum())}/{len(R)}.']
    with open(os.path.join(C.BANDING_DIR, 'rm_band.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
