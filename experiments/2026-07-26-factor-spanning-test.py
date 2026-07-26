"""Factor spanning test: is the applied model's ACTIVE return explained by the
factors that would most naturally 'explain it away' — momentum, a D&M/Barroso
vol-scaled momentum, and a regime-split momentum (so a regression can freely
reproduce a calm=continuation / panic=reversal sign flip)?

Finding (179 months, 2011-2025, Newey-West lag-6): the active return is NOT a
repackaging (near-zero loadings on all momentum controls, R2 ~ 0.01), i.e. the
mechanism is orthogonal to the usual suspects -- BUT the alpha is only ~2.7%/yr
at t~0.8, i.e. NOT statistically significant. A ~0.21-IR long-only bet over 15
years simply lacks the power to distinguish its alpha from zero. The
significance lives in the thesis LONG-SHORT strategy, not this applied
long-only active return.

NOTE: FF factors in data/ff_factors.parquet are already DECIMAL and yyyymm is an
int (YYYYMM) -- parse with format='%Y%m' (naive pd.to_datetime misreads it).

Run: .venv/bin/python experiments/2026-07-26-factor-spanning-test.py
"""
import sys
sys.path.insert(0, '/Users/giladgang/momentum_regime')
import numpy as np
import pandas as pd

w = pd.read_csv('paper/results/walk_returns.csv', parse_dates=['date'])
w = w[(w['rule'] == 'rule_r') & (w['combo'] == 'DD')].sort_values('date').reset_index(drop=True)
w['active'] = w['strat_ret'] - w['bench_ret']
w['ym'] = w['date'].dt.to_period('M')

ff = pd.read_parquet('data/ff_factors.parquet')
ff['ym'] = pd.to_datetime(ff['yyyymm'].astype(str), format='%Y%m').dt.to_period('M')
ff = ff.drop_duplicates('ym').sort_values('ym')
umd = ff.set_index('ym')['UMD']
ff['vol6'] = umd.rolling(6).std().shift(1).values          # PIT trailing-6m UMD vol

d = w.merge(ff, on='ym', how='left')
d['UMDsc'] = d['UMD'] / d['vol6']                          # D&M/Barroso vol-scaled momentum
d['UMD_calm'] = d['UMD'] * (d['pi'] < 0.5)
d['UMD_panic'] = d['UMD'] * (d['pi'] >= 0.5)
d = d.dropna(subset=['Mkt-RF', 'UMDsc']).reset_index(drop=True)
y = d['active'].values


def run(cols, label):
    X = np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtX_inv = np.linalg.inv(X.T @ X)
    L = 6
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, L + 1):
        wl = 1 - l / (L + 1)
        G = Xe[l:].T @ Xe[:-l]
        S += wl * (G + G.T)
    t = b / np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))
    print(f'{label:42} alpha={b[0]*12*100:6.2f}%/yr  t={t[0]:5.2f}   R2={1-e.var()/y.var():.2f}')


print(f'n months: {len(d)} | dep = model ACTIVE return (strat-bench), NW-HAC lag6')
run(['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'], 'vs FF5')
run(['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD'], 'vs FF5 + UMD')
run(['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD', 'UMDsc'], 'vs FF5 + UMD + D&M vol-scaled mom')
run(['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD', 'UMDsc', 'UMD_calm', 'UMD_panic'],
    'vs FF5 + UMD + D&M + regime-split mom')
