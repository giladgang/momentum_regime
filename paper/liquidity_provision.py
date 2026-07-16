"""Rebound-enabled liquidity provision: the spread lever, not the turnover lever.

The thesis result (beaten-down names bought in panic REBOUND) removes execution
urgency and de-risks providing liquidity on exactly those buys. So on
panic-regime buys of below-median-momentum names, you can be the passive side
and EARN the half-spread instead of crossing the (widest) panic spread.

For each strategy's realized trades (monthly book), recompute cost under:
  naive  : pay +hs on every trade (standard model).
  mid    : panic beaten-down buys cross at mid -> pay 0; else +hs (conservative).
  lp     : panic beaten-down buys EARN -hs (full liquidity provision); else +hs.
Reports cost bp/mo, the saving, its concentration in panic, and net-IR lift.
Cost-accounting analysis (fill assumption optimistic) -> BEST-CASE bound.

Usage: .venv/bin/python -m paper.liquidity_provision
Output: paper/results/banding_study/liquidity_provision.{md,csv}
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


def main():
    sp, month_med, _ = B.load_spreads()
    rows = []
    for s in C.BS_STRATEGIES:
        x = signals.build(s)
        x = x[x['date'] >= C.SWEEP_START].reset_index(drop=True)
        # monthly book = full trade set (max liquidity-provision opportunity)
        r, led = X.simulate(x, ('nmv_band', (10, 10)), score_col='score')
        b, _ = X.simulate(x, ('benchmark', None), score_col='score')
        gross_act = (r['gross'] - b['gross']).astype(float)

        m = led.copy()
        m['ym'] = m['date'].dt.to_period('M')
        m = m.merge(sp[['permno', 'ym', 'hs']], on=['permno', 'ym'],
                    how='left')
        # load_spreads already returns hs in BASIS POINTS; fill in bp too
        m['hs'] = (m['hs'].fillna(m['ym'].map(month_med))
                   .fillna(float(month_med.median())))
        m = m.merge(x[['date', 'permno', 'mom_12']], on=['date', 'permno'],
                    how='left')
        med = x.groupby('date')['mom_12'].median()
        m['med'] = m['date'].map(med)
        is_lp = ((m['pi'] >= 0.5) & (m['dw'] > 0)
                 & (m['mom_12'] <= m['med']))
        base = m['dw'].abs() * m['hs'] / 1e4
        m['c_naive'] = base
        m['c_mid'] = np.where(is_lp, 0.0, base)
        m['c_lp'] = np.where(is_lp, -base, base)
        cost = {k: m.groupby('date')[f'c_{k}'].sum().reindex(r.index).fillna(0)
                for k in ['naive', 'mid', 'lp']}

        def net_ir(cser):
            net = (r['gross'] - cser).astype(float)
            act = net - b['gross'].astype(float)
            return float(act.mean() / act.std() * np.sqrt(12))
        cn = cost['naive'].mean() * 1e4       # bp/mo
        cl = cost['lp'].mean() * 1e4
        lp_frac = float(is_lp.sum() / len(m)) if len(m) else 0.0
        # share of the naive cost that sits on the LP (panic beaten-down) trades
        lp_cost_share = float(base[is_lp].sum() / base.sum()) if base.sum() else 0
        rows.append({
            'strategy': s, 'cost_naive_bp': cn, 'cost_mid_bp': cost['mid'].mean() * 1e4,
            'cost_lp_bp': cl, 'saving_bp': cn - cl,
            'saving_ann_pct': (cn - cl) * 12 / 100,
            'lp_trade_frac': lp_frac, 'lp_cost_share': lp_cost_share,
            'net_ir_naive': net_ir(cost['naive']),
            'net_ir_mid': net_ir(cost['mid']),
            'net_ir_lp': net_ir(cost['lp'])})
        print(f"[lp] {s}: cost {cn:.2f}->{cl:.2f}bp/mo (save {cn-cl:.2f}, "
              f"{(cn-cl)*12/100:.2f}%/yr) | net IR "
              f"{rows[-1]['net_ir_naive']:.3f}->{rows[-1]['net_ir_lp']:.3f} | "
              f"LP trades {lp_frac:.0%}, {lp_cost_share:.0%} of cost",
              flush=True)
    R = pd.DataFrame(rows)
    R.round(4).to_csv(os.path.join(C.BANDING_DIR, 'liquidity_provision.csv'),
                      index=False)
    lines = ['# Rebound-enabled liquidity provision (spread lever) — best-case\n',
             'On panic-regime buys of below-median-momentum names, EARN the '
             'half-spread (passive fill) instead of paying it. Monthly book '
             '(full trades). Cost-accounting bound (fill assumption '
             'optimistic).\n',
             R.round(3).to_string(index=False), '',
             'saving_ann_pct = annualized cost reduction; lp_cost_share = share'
             ' of naive cost sitting on the panic beaten-down buys (where the '
             'rebound de-risks providing liquidity). Biggest for strategies '
             'that BUY losers in panic (contrarian/regime), ~0 for momentum '
             '(buys winners).']
    with open(os.path.join(C.BANDING_DIR, 'liquidity_provision.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
