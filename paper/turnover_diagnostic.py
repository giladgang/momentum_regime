"""When does each strategy trade most, and where does the cost concentrate?

Diagnostic (not a conditioning test). For each strategy's monthly book,
decompose turnover and net-of-spread cost by:
  - regime: calm (pi<0.5) vs panic (pi>=0.5)
  - regime TRANSITION: months where the panic/calm state flips vs stable
  - trade cause: entry / exit_rank / exit_univ / weight (from the ledger)
Reports mean turnover, mean cost (bp/mo), and each slice's SHARE of total cost
(cost = turnover x spread, so panic concentration exceeds turnover
concentration because panic spreads are wider).

Usage: .venv/bin/python -m paper.turnover_diagnostic
Output: paper/results/banding_study/turnover_diagnostic.{md,csv}
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
from paper.src import signals                                   # noqa: E402


def main():
    sp, month_med, _ = B.load_spreads()
    rows, cause_rows = [], []
    for s in C.BS_STRATEGIES:
        x = signals.build(s)
        x = x[x['date'] >= C.SWEEP_START].reset_index(drop=True)
        r, led = X.simulate(x, ('nmv_band', (10, 10)), score_col='score')
        # per-trade cost
        m = led.copy()
        m['ym'] = m['date'].dt.to_period('M')
        m = m.merge(sp[['permno', 'ym', 'hs']], on=['permno', 'ym'], how='left')
        m['hs'] = (m['hs'].fillna(m['ym'].map(month_med))
                   .fillna(float(month_med.median())))
        m['c'] = m['dw'].abs() * m['hs'] / 1e4
        cost = m.groupby('date')['c'].sum().reindex(r.index).fillna(0)
        # regime + transition flags per month
        pi = r['pi']
        panic = pi >= 0.5
        trans = panic.ne(panic.shift(1)).fillna(False)
        to = r['turnover']

        def share(mask):
            return float(cost[mask].sum() / cost.sum()) if cost.sum() else 0.0

        def mo(x_, mask):
            return float(x_[mask].mean()) if mask.sum() else 0.0
        rows.append({
            'strategy': s, 'n_mo': len(r),
            'to_mo': float(to.mean()), 'cost_bp_mo': float(cost.mean() * 1e4),
            'pct_panic_mo': float(panic.mean()),
            'to_calm': mo(to, ~panic), 'to_panic': mo(to, panic),
            'cost_calm_bp': mo(cost, ~panic) * 1e4,
            'cost_panic_bp': mo(cost, panic) * 1e4,
            'pct_cost_panic': share(panic),
            'pct_mo_transition': float(trans.mean()),
            'to_stable': mo(to, ~trans), 'to_transition': mo(to, trans),
            'cost_transition_bp': mo(cost, trans) * 1e4,
            'pct_cost_transition': share(trans)})
        # cost by trade cause
        cc = m.groupby('cause')['c'].sum()
        cc = (cc / cc.sum()).to_dict()
        cause_rows.append({'strategy': s,
                           **{k: cc.get(k, 0.0) for k in
                              ['entry', 'exit_rank', 'exit_univ', 'weight']}})
        print(f"[to] {s}: cost {rows[-1]['cost_bp_mo']:.2f}bp/mo | "
              f"panic {rows[-1]['pct_cost_panic']:.0%} of cost "
              f"({rows[-1]['pct_panic_mo']:.0%} of months) | "
              f"transitions {rows[-1]['pct_cost_transition']:.0%} of cost "
              f"({rows[-1]['pct_mo_transition']:.0%} of months)", flush=True)
    R = pd.DataFrame(rows)
    CR = pd.DataFrame(cause_rows)
    R.round(4).to_csv(os.path.join(C.BANDING_DIR, 'turnover_diagnostic.csv'),
                      index=False)
    pd.set_option('display.width', 220)
    lines = [
        '# When does each strategy trade most / cost concentrate?\n',
        'Monthly book. panic = pi>=0.5; transition = panic/calm state flips '
        'vs prior month. cost = turnover x spread.\n',
        '## Regime & transition concentration\n',
        R[['strategy', 'cost_bp_mo', 'pct_panic_mo', 'to_calm', 'to_panic',
           'pct_cost_panic', 'pct_mo_transition', 'to_transition',
           'pct_cost_transition']].round(3).to_string(index=False), '',
        '## Cost by trade cause (share)\n',
        CR.round(3).to_string(index=False), '',
        'pct_cost_panic vs pct_panic_mo: if >, cost is DISPROPORTIONATELY in '
        'panic (wider spreads + more trading). Same for transitions.']
    with open(os.path.join(C.BANDING_DIR, 'turnover_diagnostic.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
