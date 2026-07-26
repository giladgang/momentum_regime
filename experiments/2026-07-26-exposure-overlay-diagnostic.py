"""Exposure-overlay diagnostic (the D&M lever, NOT the band lever).

Applies contemporaneous exposure overlays to the applied model's ACTIVE return
(strat - bench) and reports annualized return / vol / IR / max-drawdown. This is
a DESCRIPTIVE ceiling (uses full-sample, and the 'crash-off' arm peeks at the
same-month benchmark sign), not a tradeable walk-forward result.

Finding: de-risking in panic (lambda=1-pi) DESTROYS the edge (IR 0.21 -> -0.04)
because the strategy's active edge lives in the panic-recovery months; scaling
exposure UP in stress (lambda=pi) raises IR to 0.40 with lower vol/drawdown.
That IR gain is the Daniel & Moskowitz (2016) dynamic-momentum result, applied
to the MIRROR side (this model profits in the panic our unconditional-momentum
cousins crash in). It is NOT a novel overlay, and the low drawdown is a
2011-2025-window artifact (no sustained bear).

Run: .venv/bin/python experiments/2026-07-26-exposure-overlay-diagnostic.py
"""
import sys
sys.path.insert(0, '/Users/giladgang/momentum_regime')
import numpy as np
import pandas as pd

w = pd.read_csv('paper/results/walk_returns.csv', parse_dates=['date'])
w = w[(w['rule'] == 'rule_r') & (w['combo'] == 'DD')].sort_values('date')
a = (w['strat_ret'] - w['bench_ret']).values
pi = w['pi'].values
bench = w['bench_ret'].values


def stats(x):
    x = np.asarray(x)
    mu, sd = x.mean(), x.std(ddof=1)
    c = np.cumprod(1 + x)
    dd = (c / np.maximum.accumulate(c) - 1).min()
    return mu * 12 * 100, sd * np.sqrt(12) * 100, mu / sd * np.sqrt(12), dd * 100


print('Contemporaneous exposure overlays on the ACTIVE return (descriptive, NOT walk-forward):')
print(f'{"overlay":42} {"annRet%":>8} {"annVol%":>8} {"IR":>6} {"maxDD%":>8}')
for name, lam in [
    ('static (lambda=1)', np.ones_like(pi)),
    ('de-risk in panic (lambda=1-pi)', 1 - pi),
    ('lean-in (lambda=1+pi)', 1 + pi),
    ('concentrate in panic (lambda=pi)', pi),
    ('crash-off ORACLE (lambda=0 when bench<0 & pi>=.5)',
     np.where((bench < 0) & (pi >= .5), 0.0, 1.0)),
]:
    r, v, ir, dd = stats(lam * a)
    print(f'{name:42} {r:8.2f} {v:8.2f} {ir:6.3f} {dd:8.1f}')
print(f'\nrecovery months (pi>=.5 & bench>0): {int(((pi>=.5)&(bench>0)).sum())} | '
      f'crash months (pi>=.5 & bench<0): {int(((pi>=.5)&(bench<0)).sum())}')
