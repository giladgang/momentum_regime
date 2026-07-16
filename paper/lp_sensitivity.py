"""Liquidity-provision saving as a function of realized execution quality.

Turns the best-case LP bound into a defensible range. On panic beaten-down
buys, the net cost as a multiple f of the quoted half-spread:
  f=+1  pay the full half-spread (cross the ask) = naive, no benefit
  f= 0  cross at mid (pay nothing)
  f=-1  earn the full half-spread (perfect passive provision) = best case
Realistic passive provision nets the "realized spread" ~30-50% of quoted
(adverse selection eats the rest) AND you only fill a fraction passively,
crossing the remainder -> realistic effective f ~ [-0.2, +0.2] (near mid).
saving(f) = (1 - f) * (naive cost of the panic beaten-down buys).

Usage: .venv/bin/python -m paper.lp_sensitivity
Output: paper/results/banding_study/lp_sensitivity.{md,csv}
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

F_GRID = [1.0, 0.5, 0.2, 0.0, -0.2, -0.5, -1.0]   # +1 naive ... -1 best case


def main():
    sp, month_med, _ = B.load_spreads()
    rows = []
    for s in C.BS_STRATEGIES:
        x = signals.build(s)
        x = x[x['date'] >= C.SWEEP_START].reset_index(drop=True)
        r, led = X.simulate(x, ('nmv_band', (10, 10)), score_col='score')
        b, _ = X.simulate(x, ('benchmark', None), score_col='score')
        m = led.copy()
        m['ym'] = m['date'].dt.to_period('M')
        m = m.merge(sp[['permno', 'ym', 'hs']], on=['permno', 'ym'], how='left')
        m['hs'] = (m['hs'].fillna(m['ym'].map(month_med))
                   .fillna(float(month_med.median())))
        m = m.merge(x[['date', 'permno', 'mom_12']], on=['date', 'permno'],
                    how='left')
        m['med'] = m['date'].map(x.groupby('date')['mom_12'].median())
        is_lp = ((m['pi'] >= 0.5) & (m['dw'] > 0) & (m['mom_12'] <= m['med']))
        m['base'] = m['dw'].abs() * m['hs'] / 1e4          # decimal cost
        naive = m.groupby('date')['base'].sum().reindex(r.index).fillna(0)
        lp_base = (m.loc[is_lp].groupby('date')['base'].sum()
                   .reindex(r.index).fillna(0))            # cost of LP trades
        row = {'strategy': s}
        for f in F_GRID:
            cost_f = naive - (1 - f) * lp_base             # cost at exec factor f
            net = (r['gross'] - cost_f).astype(float)
            act = net - b['gross'].astype(float)
            row[f'save_pct_f{f:+.1f}'] = float(
                ((1 - f) * lp_base).mean() * 12 * 100)
            row[f'ir_f{f:+.1f}'] = float(act.mean() / act.std() * np.sqrt(12))
        rows.append(row)
        print(f"[lps] {s}: saving %/yr  naive(f=+1) 0.00  mid(f=0) "
              f"{row['save_pct_f+0.0']:.3f}  best(f=-1) {row['save_pct_f-1.0']:.3f}"
              f"  | realistic(f=0) IR {row['ir_f+0.0']:.3f}", flush=True)
    R = pd.DataFrame(rows)
    R.round(4).to_csv(os.path.join(C.BANDING_DIR, 'lp_sensitivity.csv'),
                      index=False)
    save_cols = [f'save_pct_f{f:+.1f}' for f in F_GRID]
    lines = ['# Liquidity-provision saving (%/yr) vs realized execution factor f\n',
             'f=+1 naive (pay half-spread) ... f=0 mid (cross at mid) ... f=-1 '
             'best (earn half-spread). Realistic passive provision ~ f in '
             '[-0.2,+0.2] (near mid): net realized spread ~30-50% of quoted, '
             'times a partial passive fill rate.\n',
             R[['strategy'] + save_cols].round(3).to_string(index=False), '',
             'Realistic (f~0, the mid column) is the honest central estimate; '
             'f=-1 is the optimistic bound already reported.']
    with open(os.path.join(C.BANDING_DIR, 'lp_sensitivity.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
