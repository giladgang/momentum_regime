"""Beaten-down-hold cost rule vs static banding, honest walk-forward, 7 strat.

Gilad's idea: in the panic market conditions we found, trade the beaten-down
(loser) names LESS -- operationalize "hold the fallen names through recovery"
as a turnover reducer. These rules are PARAMETER-FREE (freeze rules), so unlike
the band grids there is nothing to overfit: if a 0-param rule beats the
1-param walk-forward-selected static band OOS, the effect is real.

Arms:
  monthly | static (WF-select E) | panic_no_sell (hold ALL in panic, 0p) |
  panic_hold_losers (hold below-median-mom held names in panic, 0p) |
  panic_hold_losers_wf (thr in {0.3,0.5,0.7} WF-selected, 1p matched to static)

Walk-forward OOS from 2001 (xgb 2013); paired block-bootstrap CI + BH-FDR.

Usage: .venv/bin/python -m paper.hold_losers
Output: paper/results/banding_study/hold_losers.{md,csv}
"""
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

START_OOS = 2001
COND = ['panic_no_sell', 'panic_hold_losers', 'panic_hold_losers_wf']


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
        static = _walk({'static': pd.DataFrame(
            {e: act(('nmv_band', (e, e))) for e in E_BASE})}, start)['static']
        idx = static.index
        z = pd.Series(0.0, index=idx)
        base = M.ir(static, z)
        series = {
            'monthly': act(('nmv_band', (10, 10))).reindex(idx),
            'panic_no_sell': act(('panic_no_sell', None)).reindex(idx),
            'panic_hold_losers': act(('panic_hold_losers', 0.5)).reindex(idx),
            'panic_hold_losers_wf': _walk({'a': pd.DataFrame(
                {t: act(('panic_hold_losers', t)) for t in (0.3, 0.5, 0.7)})},
                start)['a'].reindex(idx),
        }
        for a in COND:
            d = (series[a] - static).dropna()
            lo, hi = X.paired_block_bootstrap(
                series[a].reindex(d.index).astype(float),
                static.reindex(d.index).astype(float))
            rows.append({'strategy': s, 'arm': a, 'oos_ir': M.ir(series[a], z),
                         'minus_static': M.ir(series[a], z) - base,
                         'excl0': bool(lo > 0 or hi < 0),
                         'p_boot': _boot_p(series[a].reindex(d.index),
                                           static.reindex(d.index))})
        print(f'[hl] {s}: static {base:+.3f} monthly '
              f'{M.ir(series["monthly"], z):+.3f} | ' + ' | '.join(
                  f'{a.replace("panic_", "")} '
                  f'{M.ir(series[a], z) - base:+.3f}' for a in COND),
              flush=True)
    R = pd.DataFrame(rows)
    p = R['p_boot'].sort_values()
    thr = (np.arange(1, len(p) + 1) / len(p)) * 0.05
    R['bh_sig'] = False
    R.loc[p.index, 'bh_sig'] = (p.values <= thr)
    R.round(4).to_csv(os.path.join(C.BANDING_DIR, 'hold_losers.csv'),
                      index=False)
    npos = int((R['minus_static'] > 0).sum())
    lines = ['# Beaten-down-hold cost rules vs static banding (honest WF)\n',
             'Parameter-free freeze rules (panic_no_sell, panic_hold_losers) '
             'and a 1-param thr-selected variant, vs walk-forward-selected '
             'static. OOS from 2001 (xgb 2013). BH-FDR q=0.05.\n',
             R.round(3).to_string(index=False), '',
             f'Positive vs static: {npos}/{len(R)}. CI excludes 0: '
             f'{int(R["excl0"].sum())}/{len(R)}. Survive BH-FDR: '
             f'{int(R["bh_sig"].sum())}/{len(R)} '
             f'(positive+sig: {int(((R["minus_static"] > 0) & R["bh_sig"]).sum())}).']
    with open(os.path.join(C.BANDING_DIR, 'hold_losers.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
