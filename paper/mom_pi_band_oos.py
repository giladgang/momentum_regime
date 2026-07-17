"""Rigorous OOS test: does a momentum/pi-shaped per-stock band cut cost on the
MODERN test set, without in-sample cherry-picking?

Fixes the two flaws of paper/mom_pi_band.py:
 (1) era-confound: that run pooled 1992-2024, where panic months coincide with
     the high-spread 1990s/2008 era. Here cost/turnover are evaluated ONLY on
     2011-2024 months (the modern, relevant cost regime). The momentum->spread
     map is still PIT (expanding from full history).
 (2) in-sample selection: that run reported the MAX over (kappa,beta,E_base) at
     matched turnover -- cherry-picking. Here the shape arms are FIXED by theory
     (kappa=1/3 = Constantinides), and each arm's advantage is measured at FIXED
     operating points (uniform E in {20,30,40}), then averaged -- no selection.

Arms (each a fixed band shape, per-stock eband = E_base * shape):
  uniform      : shape = 1
  mom (k=1/3)  : shape = pred_spread^(1/3)          -- momentum only
  mom (k=2/3)  : shape = pred_spread^(2/3)          -- stronger momentum
  pi (b=0.5)   : shape = exp(0.5*(pi-0.5))          -- regime tilt only
  both         : shape = pred_spread^(1/3)*exp(0.5*(pi-0.5))

Advantage = uniform_cost - shaped_cost at MATCHED turnover, averaged over the
three operating points, on 2011-2024. Positive => shaping cuts cost OOS.

Usage: .venv/bin/python -m paper.mom_pi_band_oos
Output: paper/results/banding_study/mom_pi_band_oos.{md,csv}
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
from paper.src import signals                                   # noqa: E402
from paper.mom_pi_band import _pit_mom_spread                   # noqa: E402

TEST_START = pd.Timestamp('2011-01-01')
TEST_END = pd.Timestamp('2024-11-30')
E_BASE = [10, 15, 20, 25, 30, 40, 60, 100]
OP_POINTS = [20, 30, 40]                          # uniform bands = operating pts


def _cost_to_masked(r, led, sp_pack, mask_dates):
    """(cost bp/yr, turnover/yr) evaluated ONLY on months in mask_dates."""
    cost, _, _ = B.price_ledger(led, sp_pack)
    cost = cost.reindex(r.index).fillna(0.0)
    m = r.index.isin(mask_dates)
    return (float(cost[m].mean() * 12 * 1e4), float(r['turnover'][m].mean() * 12))


def _frontier(x, sp_pack, shape, e_grid, mask):
    pts = []
    for eb in e_grid:
        xx = x.assign(eband=np.clip(eb * shape.values, 5, 100))
        r, led = X.simulate(xx, ('stock_var_band', None), score_col='score')
        c, to = _cost_to_masked(r, led, sp_pack, mask)
        pts.append((to, c))
    return np.array(sorted(pts))


def main():
    sp_pack = B.load_spreads()
    sp = sp_pack[0]
    rows = []
    for s in C.BS_STRATEGIES:
        x = signals.build(s)
        x = x[(x['date'] >= C.SWEEP_START)
              & (x['date'] <= TEST_END)].reset_index(drop=True)
        mask = x['date'][(x['date'] >= TEST_START)].unique()   # eval months
        pi = x['date'].map(x.groupby('date')['pi'].first())
        pred = _pit_mom_spread(x, sp)

        one = pd.Series(1.0, index=x.index)
        arms = {
            'mom_k33': pred ** (1.0 / 3),
            'mom_k67': pred ** (2.0 / 3),
            'pi_b50': np.exp(0.5 * (pi - 0.5)),
            'both': (pred ** (1.0 / 3)) * np.exp(0.5 * (pi - 0.5))}

        uni = _frontier(x, sp_pack, one, E_BASE, mask)
        uni_to, uni_cost = uni[:, 0], uni[:, 1]
        c0 = float(uni_cost.max())

        # operating-point turnovers = uniform E in OP_POINTS
        op_to = []
        for E in OP_POINTS:
            xx = x.assign(eband=float(E))
            r, led = X.simulate(xx, ('stock_var_band', None), score_col='score')
            _, to = _cost_to_masked(r, led, sp_pack, mask)
            op_to.append(to)

        for arm, shape in arms.items():
            fr = _frontier(x, sp_pack, shape, E_BASE, mask)
            fr_to, fr_cost = fr[:, 0], fr[:, 1]
            advs = []
            for t in op_to:                       # advantage at each op point
                if t < fr_to.min() or t < uni_to.min():
                    continue
                shaped_c = float(np.interp(t, fr_to, fr_cost))
                uni_c = float(np.interp(t, uni_to, uni_cost))
                advs.append(uni_c - shaped_c)     # positive = shaped cheaper
            adv = float(np.mean(advs)) if advs else 0.0
            rows.append({
                'strategy': s, 'arm': arm, 'cost_base_bpyr': round(c0, 2),
                'adv_bpyr': round(adv, 4),
                'adv_pct': round(adv / c0 * 100, 2) if c0 else 0.0,
                'n_op': len(advs)})
        best = max((r for r in rows if r['strategy'] == s),
                   key=lambda r: r['adv_pct'])
        print(f"[oos] {s}: base {c0:.1f}bp/yr (2011-24) | mom_k33 "
              f"{[r['adv_pct'] for r in rows if r['strategy']==s and r['arm']=='mom_k33'][0]:+.2f}%"
              f" | pi_b50 "
              f"{[r['adv_pct'] for r in rows if r['strategy']==s and r['arm']=='pi_b50'][0]:+.2f}%"
              f" | best={best['arm']} {best['adv_pct']:+.2f}%", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(C.BANDING_DIR, 'mom_pi_band_oos.csv'), index=False)
    pd.set_option('display.width', 260)
    piv = R.pivot(index='strategy', columns='arm', values='adv_pct')
    lines = [
        '# Momentum/pi-shaped band vs uniform, cost, 2011-2024 (no cherry-pick)\n',
        'Per-stock eband = E_base*shape; shape fixed by theory (kappa=1/3 ='
        ' Constantinides). Advantage = uniform_cost - shaped_cost at MATCHED'
        ' turnover, averaged over uniform E in {20,30,40}. Cost evaluated on'
        ' 2011-2024 only (era-confound removed); momentum->spread map is PIT.\n',
        '## Advantage over uniform (% of cost), by arm\n',
        piv.round(2).to_string(), '',
        'mom_k33/k67 = momentum-shaped (Constantinides). pi_b50 = regime tilt.'
        ' both = momentum x regime. Positive => cuts cost vs uniform at equal'
        ' turnover. Compare momentum arms (the 29%-dispersion channel) vs pi'
        ' (already shown null OOS by clean-room). Absolute bp/yr in the csv.']
    with open(os.path.join(C.BANDING_DIR, 'mom_pi_band_oos.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))


if __name__ == '__main__':
    main()
