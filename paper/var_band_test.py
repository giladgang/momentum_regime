"""Continuous band optimized as a function of the HMM + market factors.

E_t = clip(E_base * exp(beta * stress_t), 5, 100). Two stress signals:
  hmm   : 2*(pi_t - 0.5)  -- the continuous HMM panic probability (PIT).
  multi : mean of panel z-factors (DD_z, VOL_z, DISP_z, CS_z) -- market stress.
Walk-forward select (E_base, beta) on prior data (argmax + 1-SE-prefer-static
regularized); beta=0 recovers the rigid static band. vs WF-selected static,
all 7 strategies, paired block-bootstrap CI + BH-FDR. This is the continuous /
multi-factor "optimize the band" version, distinct from the discrete-state
lookups already tested.

Usage: .venv/bin/python -m paper.var_band_test
Output: paper/results/banding_study/var_band.{md,csv}
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
from paper.cluster_bands import _boot_p                         # noqa: E402

START_OOS = 2001
E_BASE = [10, 20, 40]
BETA = [-1.0, -0.5, 0.0, 0.5, 1.0]           # 0 = rigid static band
FZ = ['DD_z', 'VOL_z', 'DISP_z', 'CS_z']


def _walk_var(cand, is_simple, start_oos):
    """cand: {key: active DataFrame-col}. Returns stitched static / argmax /
    reg. static = best beta=0 (simple) candidate; reg prefers simple within 1
    Sharpe-SE of the best."""
    A = pd.DataFrame(cand)
    yrs = [y for y in sorted({d.year for d in A.index}) if y >= start_oos]
    simple = [c for c in A.columns if is_simple[c]]
    rec = {'static': [], 'argmax': [], 'reg': []}
    for y in yrs:
        tr, te = A[A.index.year < y], A[A.index.year == y]
        if len(tr) < 60 or len(te) == 0:
            continue

        def ir(c):
            a = tr[c].dropna()
            return (a.mean() / a.std() * np.sqrt(12)
                    if len(a) > 12 and a.std() > 0 else -np.inf)
        irs = {c: ir(c) for c in A.columns}
        best = max(irs.values())
        ny = max(len(tr) / 12, 1)
        se = np.sqrt((1 + 0.5 * best ** 2) / ny)
        within = [c for c in A.columns if irs[c] >= best - se]
        sw = [c for c in within if is_simple[c]]
        rec['static'].append(te[max(simple, key=lambda c: irs[c])])
        rec['argmax'].append(te[max(A.columns, key=lambda c: irs[c])])
        rec['reg'].append(te[max(sw or within, key=lambda c: irs[c])])
    return {k: pd.concat(v).sort_index() for k, v in rec.items() if v}


def main():
    sp_pack = B.load_spreads()
    panel = pd.read_parquet(C.PANEL_PARQUET, columns=['date'] + FZ)
    panel['date'] = pd.to_datetime(panel['date'])
    panel['multi'] = panel[FZ].mean(axis=1)
    rows = []
    for s in C.BS_STRATEGIES:
        x = signals.build(s)
        x = x[x['date'] >= C.SWEEP_START].reset_index(drop=True)
        start = START_OOS if s != 'xgb' else 2013
        b, bled = X.simulate(x, ('benchmark', None), score_col='score')
        bench_net = B._net(b['gross'], B.price_ledger(bled, sp_pack)[0])
        pim = x.groupby('date')['pi'].first()
        stress = {
            'hmm': (2 * (pim - 0.5)),
            'multi': panel.set_index('date')['multi'].reindex(pim.index)
            .fillna(0.0)}

        def act_eband(e_series):
            xx = x.assign(eband=x['date'].map(e_series))
            return _active(xx, ('var_band', None), sp_pack, bench_net)
        for sig in ['hmm', 'multi']:
            st_series = stress[sig]
            cand, is_simple = {}, {}
            for eb, beta in itertools.product(E_BASE, BETA):
                key = f'{eb}|{beta}'
                e_t = np.clip(eb * np.exp(beta * st_series), 5, 100)
                cand[key] = act_eband(e_t)
                is_simple[key] = (beta == 0.0)
            stw = _walk_var(cand, is_simple, start)
            z = pd.Series(0.0, index=stw['static'].index)
            base = M.ir(stw['static'], z)
            for arm in ['argmax', 'reg']:
                d = (stw[arm] - stw['static']).dropna()
                lo, hi = X.paired_block_bootstrap(
                    stw[arm].reindex(d.index).astype(float),
                    stw['static'].reindex(d.index).astype(float))
                rows.append({'strategy': s, 'stress': sig, 'arm': arm,
                             'minus_static': M.ir(stw[arm], z) - base,
                             'excl0': bool(lo > 0 or hi < 0),
                             'p_boot': _boot_p(stw[arm].reindex(d.index),
                                               stw['static'].reindex(d.index))})
            print(f"[vb] {s}/{sig}: static {base:+.3f} | argmax "
                  f"{M.ir(stw['argmax'], z)-base:+.3f} | reg "
                  f"{M.ir(stw['reg'], z)-base:+.3f}", flush=True)
    R = pd.DataFrame(rows)
    p = R['p_boot'].sort_values()
    thr = (np.arange(1, len(p) + 1) / len(p)) * 0.05
    R['bh_sig'] = False
    R.loc[p.index, 'bh_sig'] = (p.values <= thr)
    R.round(4).to_csv(os.path.join(C.BANDING_DIR, 'var_band.csv'), index=False)
    lines = ['# Continuous band E_t = E_base*exp(beta*stress) — HMM + market\n',
             'stress = hmm (2*(pi-0.5)) or multi (mean DD_z/VOL_z/DISP_z/CS_z).'
             ' beta=0 = rigid static. WF-selected. BH-FDR q=0.05.\n',
             R.round(3).to_string(index=False), '',
             f'Positive vs static: {int((R.minus_static>0).sum())}/{len(R)}. '
             f'CI excl 0: {int(R.excl0.sum())}/{len(R)}. '
             f'positive+BH-sig: '
             f'{int(((R.minus_static>0)&R.bh_sig).sum())}/{len(R)}.']
    with open(os.path.join(C.BANDING_DIR, 'var_band.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
