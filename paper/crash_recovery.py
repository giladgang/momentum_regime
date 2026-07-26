"""Crash/recovery decomposition of the applied model's active return.

Promoted from the 2026-07-15 scratch analysis (spec §1). Splits each month
into calm / panic_crash / panic_recovery on the market's own path: panic
(pi >= 0.5) months are RECOVERY when the market drawdown is healing
(bench_ret > 0 this month, equivalently dd_t > dd_{t-1}), else CRASH.

IMPORTANT — this is a CONTEMPORANEOUS, DESCRIPTIVE split, NOT the tradeable
signal. bench_ret[t] is the market return earned over the SAME held month as
the active return, so "recovery carries the edge" is a correct statement
about co-timing. It is NOT what the banding sweep's regime3_band trades on:
that uses `state3` = regime_labels.label_states on PANEL vwretd, which is
only known at FORMATION (last month's direction) and is therefore a lagged,
weak proxy for the phase. In short: the edge shows up in recovery months, but
you cannot know at decision time whether next month is crash or recovery —
which is exactly why regime-conditioning adds only a modest OOS increment and
the practical lesson is patience, not faster reaction. Do not "align" this to
state3; they answer different questions (descriptive vs implementable).

Usage: .venv/bin/python -m paper.crash_recovery
Output: paper/results/banding_study/crash_recovery.csv
"""
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from paper import config as C                                   # noqa: E402


def decompose(walk):
    """Split the model's active return into calm / panic-crash / panic-recovery
    and report each bucket's mean/t-stat and share of the total edge. `walk` has
    columns date, strat_ret, bench_ret, pi (the production monthly returns)."""
    m = walk.sort_values('date').copy()
    m['active'] = m['strat_ret'] - m['bench_ret']
    panic = m['pi'] >= 0.5                              # panic = high regime probability
    # within panic, RECOVERY if the market rose this month (drawdown healing), else CRASH.
    # NB contemporaneous: bench_ret[t] is earned over the same held month (see module docstring).
    m['bucket'] = np.where(~panic, 'calm',
                           np.where(m['bench_ret'] > 0,
                                    'panic_recovery', 'panic_crash'))
    tot = m['active'].sum()
    rows = []
    for b in ['calm', 'panic_crash', 'panic_recovery']:
        g = m[m['bucket'] == b]
        n = len(g)
        std = g['active'].std()
        rows.append({
            'bucket': b, 'n_mo': n,
            'strat_mo': g['strat_ret'].mean(),
            'bench_mo': g['bench_ret'].mean(),
            'active_mo': g['active'].mean(),
            # simple one-sample t-stat of the bucket's monthly active return
            'active_t': (g['active'].mean() / std * np.sqrt(n)
                         if n > 1 and std > 0 else np.nan),
            'sum_active': g['active'].sum(),
            'share_of_total_active': g['active'].sum() / tot,   # can exceed 1 (crash bucket is negative)
        })
    return pd.DataFrame(rows)


def main():
    walk = pd.read_csv(C.RETURNS_CSV, parse_dates=['date'])
    m = walk[(walk['rule'] == 'rule_r') & (walk['combo'] == 'DD')]
    out = decompose(m)
    os.makedirs(C.BANDING_DIR, exist_ok=True)
    path = os.path.join(C.BANDING_DIR, 'crash_recovery.csv')
    out.round(6).to_csv(path, index=False)
    print(out.round(4).to_string(index=False))
    print('Saved:', path)


if __name__ == '__main__':
    main()
