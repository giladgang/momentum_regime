"""
dm_managed_oos.py
=================
Out-of-sample Daniel-Moskowitz (2016) managed momentum.

Replicates the D&M implementation from new_ls_analyses.py but applied to
the historical OOS test period (2000-2010). Compares D&M, unscaled WML,
and M2 across the same sub-periods to answer: does D&M's exposure-scaling
mechanism survive the dot-com bust where M2 struggled?

Outputs:
    results/dm_oos_subperiods.csv
    tables/table_dm_oos_comparison.tex
"""

import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RESULTS_DIR, TABLES_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument('--train-end', default='2000-01-01',
                help='End of training window (exclusive)')
ap.add_argument('--test-end', default='2011-01-01',
                help='End of test window (exclusive)')
ap.add_argument('--m2-returns', default='oos_returns_prod_1990_1999.csv',
                help='M2 OOS returns CSV in results/ for the matching config')
ap.add_argument('--tag', default='1990_1999',
                help='Output tag, e.g. 1990_1999 or 1990_2004')
args = ap.parse_args()

FEE = 0.001
TRAIN_END = args.train_end
TEST_END = args.test_end

SUBPERIODS = [
    ('Dot-com bust',   '2000-03-01', '2002-10-31'),
    ('2003-2007 bull', '2002-11-01', '2007-09-30'),
    ('GFC crash',      '2007-10-01', '2009-02-28'),
    ('GFC rebound',    '2009-03-01', '2009-12-31'),
    ('Post-GFC calm',  '2010-01-01', '2010-12-31'),
]


def annualised_sharpe(r):
    r = pd.Series(r).dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return r.mean() / r.std() * np.sqrt(12)


def max_drawdown(r):
    cum = (1 + pd.Series(r).dropna()).cumprod()
    return ((cum - cum.cummax()) / cum.cummax()).min() if len(cum) else np.nan


def cumulative(r):
    return (1 + pd.Series(r).dropna()).prod() - 1


print("=" * 70)
print(f"  D&M MANAGED MOMENTUM OOS TEST")
print(f"  Train: 1990-01-01 to {TRAIN_END} | Test: {TRAIN_END} to {TEST_END}")
print(f"  Comparing against M2 from: {args.m2_returns}")
print("=" * 70)

# ── Build WML (winner-minus-loser) momentum portfolio ──

print("\n[1/4] Building WML momentum portfolio ...")
stocks = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks['date'] = pd.to_datetime(stocks['date'])
stocks = stocks[
    stocks['shrcd'].isin([10, 11])
    & stocks['exchcd'].isin([1, 2, 3])
    & (stocks['prc'].abs() > 1)
].sort_values(['permno', 'date']).reset_index(drop=True)
stocks['_lr'] = np.log1p(stocks['ret_adj'].clip(lower=-0.999))
# D&M momentum: 11-month return skipping 1 month (mom_12_2)
stocks['_lr_s2'] = stocks.groupby('permno')['_lr'].shift(2)
roll = (stocks.groupby('permno', sort=False)['_lr_s2']
        .rolling(11, min_periods=11)
        .sum()
        .reset_index(level='permno', drop=True)
        .sort_index())
stocks['mom_12_2'] = np.expm1(roll)
stocks['ret_fwd'] = stocks.groupby('permno')['ret_adj'].transform(lambda x: x.shift(-1))
df_wml = stocks.dropna(subset=['mom_12_2', 'ret_fwd']).copy()

prev_lw, prev_sw = {}, {}
wml_monthly = []
for date, grp in df_wml.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['mom_12_2'].dropna()
    if len(nyse) < 10:
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    longs = grp[grp['mom_12_2'] >= hi]
    shorts = grp[grp['mom_12_2'] <= lo]
    if longs['me'].sum() == 0 or shorts['me'].sum() == 0:
        continue
    lme = longs['me'].sum()
    new_lw = (longs.set_index('permno')['me'] / lme).to_dict()
    r_long = (longs['ret_fwd'] * longs['me']).sum() / lme
    sme = shorts['me'].sum()
    new_sw = (shorts.set_index('permno')['me'] / sme).to_dict()
    r_short = (shorts['ret_fwd'] * shorts['me']).sum() / sme
    tl = sum(abs(new_lw.get(p, 0) - prev_lw.get(p, 0))
             for p in set(new_lw) | set(prev_lw)) / 2
    ts = sum(abs(new_sw.get(p, 0) - prev_sw.get(p, 0))
             for p in set(new_sw) | set(prev_sw)) / 2
    wml_monthly.append({'date': date, 'r_wml': r_long - r_short, 'turnover': tl + ts})
    prev_lw, prev_sw = new_lw, new_sw

wml_all = pd.DataFrame(wml_monthly).set_index('date')
wml_all.index = pd.to_datetime(wml_all.index)
print(f"  WML series: {len(wml_all)} months, {wml_all.index.min().date()} to {wml_all.index.max().date()}")


# ── Build bear indicator from market return ──

print("\n[2/4] Building bear indicator (24-mo cumulative market return < 0) ...")
panel = pd.read_parquet('data/panel.parquet')
panel['date'] = pd.to_datetime(panel['date'])
mkt_ts = (panel[['date', 'vwretd']]
          .dropna()
          .drop_duplicates('date')
          .sort_values('date')
          .set_index('date')['vwretd'])
mkt_cum_24 = mkt_ts.rolling(24).apply(lambda x: (1 + x).prod() - 1, raw=True)
bear = (mkt_cum_24 < 0).astype(int).rename('bear')
wml_all = wml_all.join(bear, how='left')
wml_all['bear'] = wml_all['bear'].fillna(0)
wml_all['sigma2'] = wml_all['r_wml'].rolling(6, min_periods=6).var()


# ── Train OLS on 1990-1999, apply to 2000-2010 ──

print("\n[3/4] Fitting D&M OLS on training window and applying OOS ...")

train_wml = wml_all[wml_all.index < TRAIN_END]
test_wml = wml_all[(wml_all.index >= TRAIN_END) & (wml_all.index < TEST_END)].copy()

# Drop rows with missing predictors
train_wml = train_wml.dropna(subset=['r_wml', 'bear'])
test_wml = test_wml.dropna(subset=['r_wml', 'bear', 'sigma2'])

X_bear_tr = sm.add_constant(train_wml['bear'])
ols = sm.OLS(train_wml['r_wml'], X_bear_tr).fit()
print(f"  OLS fit on {len(train_wml)} training months")
print(f"  Coefficients: const={ols.params['const']:.4f}, bear={ols.params['bear']:.4f}")

X_bear_te = sm.add_constant(test_wml['bear'])
mu = ols.predict(X_bear_te)
sigma2 = test_wml['sigma2']
w_dm = (0.5 * mu / sigma2).clip(-0.5, 1.5)
print(f"  D&M weight summary: min={w_dm.min():.2f}, max={w_dm.max():.2f}, mean={w_dm.mean():.2f}")

r_dm = pd.Series(
    w_dm.values * (test_wml['r_wml'].values - FEE * test_wml['turnover'].values),
    index=test_wml.index,
)
r_wml_net = test_wml['r_wml'] - FEE * test_wml['turnover']


# ── Load M2 OOS returns for comparison ──

m2 = pd.read_csv(
    os.path.join(RESULTS_DIR, args.m2_returns),
    index_col='date', parse_dates=True,
)['ret']


# ── Sub-period decomposition ──

print("\n[4/4] Sub-period decomposition ...")

def decompose(r, label):
    rows = [{
        'strategy': label,
        'subperiod': 'Overall',
        'n': len(r),
        'sharpe': annualised_sharpe(r),
        'cum': cumulative(r),
        'mdd': max_drawdown(r),
    }]
    for name, start, end in SUBPERIODS:
        idx = pd.DatetimeIndex(r.index)
        mask = (idx >= start) & (idx <= end)
        rsp = r[mask]
        if len(rsp) == 0:
            continue
        rows.append({
            'strategy': label,
            'subperiod': name,
            'n': len(rsp),
            'sharpe': annualised_sharpe(rsp),
            'cum': cumulative(rsp),
            'mdd': max_drawdown(rsp),
        })
    return rows


rows = []
rows += decompose(m2, 'M2 (mom+pi)')
rows += decompose(r_dm, 'D&M managed')
rows += decompose(r_wml_net, 'Unscaled WML')

df = pd.DataFrame(rows)

print("\n" + "=" * 90)
print(f"{'Strategy':<20} {'Subperiod':<18} {'N':>4} {'Sharpe':>8} {'Cum':>8} {'MDD':>8}")
print("=" * 90)
for _, row in df.iterrows():
    print(f"{row['strategy']:<20} {row['subperiod']:<18} {row['n']:>4} "
          f"{row['sharpe']:>8.2f} {row['cum'] * 100:>+7.1f}% "
          f"{row['mdd'] * 100:>+7.1f}%")

out_path = os.path.join(RESULTS_DIR, f'dm_oos_subperiods_{args.tag}.csv')
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")


# ── LaTeX table: Sharpe pivot ──

pivot = df.pivot(index='subperiod', columns='strategy', values='sharpe')
row_order = ['Overall'] + [s[0] for s in SUBPERIODS]
pivot = pivot.reindex([r for r in row_order if r in pivot.index])
col_order = ['M2 (mom+pi)', 'D&M managed', 'Unscaled WML']
pivot = pivot[[c for c in col_order if c in pivot.columns]]

tex = []
tex.append(r'\begin{table}[H]')
tex.append(r'\centering')
tex.append(r'\small')
tex.append(r'\begin{tabular}{l r r r}')
tex.append(r'\toprule')
tex.append(r'Subperiod & M2 (mom+$\pi$) & D\&M managed & Unscaled WML \\')
tex.append(r'\midrule')
for sp, row in pivot.iterrows():
    vals = ' & '.join(
        f"{v:.2f}" if not np.isnan(v) else '---'
        for v in row.values
    )
    tex.append(f"{sp} & {vals} \\\\")
tex.append(r'\bottomrule')
tex.append(r'\end{tabular}')
tex.append(
    r'\caption{Out-of-sample Sharpe ratio by sub-period (2000--2010 test, '
    r'M2 trained on 1990--1999). D\&M is the \citet{DanielMoskowitz2016} '
    r'managed-momentum exposure-scaling strategy applied to unscaled WML '
    r'using a bear-market indicator predicted by 24-month cumulative '
    r'market returns.}'
)
tex.append(rf'\label{{tab:dm_oos_comparison_{args.tag}}}')
tex.append(r'\end{table}')

tex_path = os.path.join(TABLES_DIR, f'table_dm_oos_comparison_{args.tag}.tex')
with open(tex_path, 'w') as f:
    f.write('\n'.join(tex) + '\n')
print(f"Saved: {tex_path}")

print("\nDone.")
