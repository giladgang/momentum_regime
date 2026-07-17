"""Learned flexible band: E_t = clip(E_base * exp(z_t . w), 5, 100), where w is
a LEARNED weight vector over the full market-factor set (not a fixed equal
weight). Policy search (black-box) over (E_base, w) maximizing TRAIN net-IR;
evaluated on a disjoint OOS test window. Two regularizations: lambda=0
(unregularized, overfit-prone) and lambda>0 (L2 shrink toward static w=0).
This is the flexible / learned-weight optimization the earlier fixed-composite
single-slope test did not do.

Factors (standardized on TRAIN, PIT): pi + panel z-features. 60/40 chronological
train/test per strategy. Paired block-bootstrap CI + BH-FDR vs static (w=0).

Usage: .venv/bin/python -m paper.ml_band
Output: paper/results/banding_study/ml_band.{md,csv}
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
from paper.cluster_bands import _boot_p                         # noqa: E402

FZ = ['DD_z', 'VOL_z', 'DISP_z', 'CS_z', 'LVIX_z', 'TERM_z', 'SKEW_z',
      'REL_N_z']
N_RANDOM = 900
_RNG = np.random.RandomState(0)


def _ir(a):
    a = a.dropna()
    return a.mean() / a.std() * np.sqrt(12) if len(a) > 12 and a.std() > 0 \
        else -np.inf


def main():
    sp_pack = B.load_spreads()
    panel = pd.read_parquet(C.PANEL_PARQUET, columns=['date'] + FZ)
    panel['date'] = pd.to_datetime(panel['date'])
    rows = []
    for s in C.BS_STRATEGIES:
        x = signals.build(s)
        x = x[x['date'] >= C.SWEEP_START].reset_index(drop=True)
        b, bled = X.simulate(x, ('benchmark', None), score_col='score')
        bench_net = B._net(b['gross'], B.price_ledger(bled, sp_pack)[0])
        months = pd.DatetimeIndex(sorted(x['date'].unique()))
        split = months[int(len(months) * 0.6)]
        # factor matrix per month: pi (centered) + panel z, standardized on TRAIN
        pim = x.groupby('date')['pi'].first()
        F = pd.DataFrame({'pi': 2 * (pim - 0.5)}).join(
            panel.set_index('date')[FZ]).reindex(months).fillna(0.0)
        tr_m = months < split
        F = (F - F[tr_m].mean()) / F[tr_m].std(ddof=0).replace(0, 1)
        Z = F.astype(float).values
        cols = list(F.columns)

        def band_active(e_base, w):
            e_t = np.clip(e_base * np.exp(Z @ w), 5, 100)
            eb = pd.Series(e_t, index=months)
            xx = x.assign(eband=x['date'].map(eb))
            return _active(xx, ('var_band', None), sp_pack, bench_net)

        def split_ir(act):
            i = act.index
            return _ir(act[i < split]), _ir(act[i >= split])

        # static baseline: w=0, best E_base on train
        best_static = max(
            [(band_active(e, np.zeros(len(cols)))) for e in [10, 20, 30, 40]],
            key=lambda a: split_ir(a)[0])
        static_test = split_ir(best_static)[1]

        # random policy search over (E_base, w); evaluate train IR, pick per lambda
        cands = []
        for _ in range(N_RANDOM):
            e_base = _RNG.uniform(10, 50)
            w = _RNG.uniform(-1.2, 1.2, len(cols))
            act = band_active(e_base, w)
            tr_ir, te_ir = split_ir(act)
            cands.append((e_base, w, tr_ir, te_ir, act, float((w ** 2).mean())))
        for lam, tag in [(0.0, 'learned_unreg'), (0.3, 'learned_reg')]:
            best = max(cands, key=lambda c: c[2] - lam * c[5])
            e_base, w, tr_ir, te_ir, act, wpen = best
            test_series = act[act.index >= split]
            static_series = best_static[best_static.index >= split]
            d = (test_series - static_series).dropna()
            lo, hi = X.paired_block_bootstrap(
                test_series.reindex(d.index).astype(float),
                static_series.reindex(d.index).astype(float))
            top_w = ', '.join(f'{cols[j]}:{w[j]:+.2f}' for j in
                              np.argsort(-np.abs(w))[:3])
            rows.append({'strategy': s, 'arm': tag,
                         'test_ir': te_ir, 'static_test_ir': static_test,
                         'minus_static': te_ir - static_test,
                         'excl0': bool(lo > 0 or hi < 0),
                         'p_boot': _boot_p(test_series.reindex(d.index),
                                           static_series.reindex(d.index)),
                         'top_factors': top_w})
        print(f"[ml] {s}: static_test {static_test:+.3f} | unreg "
              f"{rows[-2]['minus_static']:+.3f} | reg "
              f"{rows[-1]['minus_static']:+.3f} (w: {rows[-1]['top_factors']})",
              flush=True)
    R = pd.DataFrame(rows)
    p = R['p_boot'].sort_values()
    thr = (np.arange(1, len(p) + 1) / len(p)) * 0.05
    R['bh_sig'] = False
    R.loc[p.index, 'bh_sig'] = (p.values <= thr)
    R.round(4).to_csv(os.path.join(C.BANDING_DIR, 'ml_band.csv'), index=False)
    lines = ['# Learned flexible band E_t=E_base*exp(z.w), learned weights\n',
             f'Policy search ({N_RANDOM} draws) over (E_base, w) on 60% train, '
             'evaluated on 40% OOS test. lambda=0 unreg, 0.3 L2-shrink. '
             'Factors: pi + 8 panel z-features. BH-FDR q=0.05.\n',
             R.round(3).to_string(index=False), '',
             f'Positive vs static (test): {int((R.minus_static>0).sum())}/{len(R)}'
             f'. CI excl 0: {int(R.excl0.sum())}/{len(R)}. '
             f'positive+BH-sig: {int(((R.minus_static>0)&R.bh_sig).sum())}/{len(R)}.']
    with open(os.path.join(C.BANDING_DIR, 'ml_band.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
