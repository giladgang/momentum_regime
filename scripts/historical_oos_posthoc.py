"""
historical_oos_posthoc.py
=========================
Post-hoc benchmark computation on retrained-window OOS runs. Builds fixed
12-month and fixed 1-month momentum portfolios under the same L/S construction
used throughout the thesis (NYSE breakpoints, value-weighted, 10 bps cost) and
decomposes returns by sub-period.

Section 5.4.3 cites the fixed 12-month momentum cumulative return of -62.6%
during the 2009 momentum-crash rebound (Mar-Dec 2009) — this is the script
that computed it. Output cell appears in results/oos_subperiod_full.csv.

Operates on retrained-window OOS data:
    results/oos_returns_prod_1990_1999.csv
    results/oos_returns_prod_1990_2004.csv

Outputs:
    results/oos_subperiod_full.csv          -- benchmark and M2 sub-period stats
    tables/table_oos_subperiods_m2.tex
    tables/table_oos_benchmark_comparison.tex
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RESULTS_DIR, TABLES_DIR, MOM_FEATURES

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

FEE = 0.001
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
    if len(cum) == 0:
        return np.nan
    return ((cum - cum.cummax()) / cum.cummax()).min()


def cumulative(r):
    return (1 + pd.Series(r).dropna()).prod() - 1


def block_bootstrap_sharpe(r, block=6, n_boot=5000, seed=42):
    """
    Block bootstrap Sharpe CI. For short series (n<20), uses block=3 to allow
    meaningful resampling.
    """
    r = pd.Series(r).dropna().values
    n = len(r)
    if n < 4:
        return np.nan, np.nan, np.nan
    block = min(block, max(2, n // 3))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_max = max(1, n - block + 1)
    boot_sharpes = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:n]
        if sample.std() > 0:
            boot_sharpes[b] = sample.mean() / sample.std() * np.sqrt(12)
        else:
            boot_sharpes[b] = 0.0
    return (np.nanpercentile(boot_sharpes, 2.5),
            np.nanpercentile(boot_sharpes, 50),
            np.nanpercentile(boot_sharpes, 97.5))


def subperiod_slice(r, start, end):
    idx = pd.DatetimeIndex(r.index)
    mask = (idx >= start) & (idx <= end)
    return r[mask]


# ── Load M2 OOS returns ──

print("=" * 70)
print("  POST-HOC ANALYSIS: Historical OOS sub-period decomposition")
print("=" * 70)

m2_a = pd.read_csv(os.path.join(RESULTS_DIR, 'oos_returns_prod_1990_1999.csv'),
                   index_col='date', parse_dates=True)['ret']
m2_ap = pd.read_csv(os.path.join(RESULTS_DIR, 'oos_returns_prod_1990_2004.csv'),
                    index_col='date', parse_dates=True)['ret']

print(f"\nA  (1990-1999 train): {len(m2_a)} months, "
      f"{m2_a.index.min().date()} to {m2_a.index.max().date()}")
print(f"A' (1990-2004 train): {len(m2_ap)} months, "
      f"{m2_ap.index.min().date()} to {m2_ap.index.max().date()}")


# ── Build fixed momentum and market benchmarks for 2000-2010 ──

print("\n[1/4] Building benchmark portfolios for 2000-2010 ...")

stocks = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks['date'] = pd.to_datetime(stocks['date'])
stocks = stocks[
    stocks['shrcd'].isin([10, 11])
    & stocks['exchcd'].isin([1, 2, 3])
    & (stocks['prc'].abs() > 1)
].sort_values(['permno', 'date']).reset_index(drop=True)

stocks['_lr'] = np.log1p(stocks['ret_adj'].clip(lower=-0.999))
stocks['_lr_s1'] = stocks.groupby('permno')['_lr'].shift(1)
for lb in (1, 12):
    rs = (stocks.groupby('permno', sort=False)['_lr_s1']
          .rolling(lb, min_periods=lb)
          .sum()
          .reset_index(level='permno', drop=True)
          .sort_index())
    stocks[f'mom_{lb}'] = np.expm1(rs)
stocks.drop(columns=['_lr', '_lr_s1'], inplace=True)
stocks['ret_fwd'] = stocks.groupby('permno')['ret_adj'].transform(lambda x: x.shift(-1))


def build_ls(df_test, score_col, fee=FEE):
    prev_lw, prev_sw = {}, {}
    monthly = []
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0:
            continue
        lw = (longs.set_index('permno')['me'] / longs['me'].sum()).to_dict()
        sw = (shorts.set_index('permno')['me'] / shorts['me'].sum()).to_dict()
        tl = sum(abs(lw.get(p, 0) - prev_lw.get(p, 0))
                 for p in set(lw) | set(prev_lw)) / 2
        ts = sum(abs(sw.get(p, 0) - prev_sw.get(p, 0))
                 for p in set(sw) | set(prev_sw)) / 2
        r_l = (longs['ret_fwd'] * longs['me']).sum() / longs['me'].sum()
        r_s = (shorts['ret_fwd'] * shorts['me']).sum() / shorts['me'].sum()
        monthly.append({'date': date, 'ret': r_l - r_s - fee * (tl + ts)})
        prev_lw, prev_sw = lw, sw
    return (pd.DataFrame(monthly).set_index('date')['ret']
            if monthly else pd.Series(dtype=float))


# 2000-2010 test period (matches A)
test_2000_2010 = stocks[(stocks['date'] >= '2000-01-01')
                        & (stocks['date'] < '2011-01-01')].copy()
test_2005_2010 = stocks[(stocks['date'] >= '2005-01-01')
                        & (stocks['date'] < '2011-01-01')].copy()

print("  Building fixed 12-mo momentum L/S (2000-2010) ...")
mom12_2000_2010 = build_ls(test_2000_2010.dropna(subset=['mom_12', 'ret_fwd']), 'mom_12')
print(f"    {len(mom12_2000_2010)} months, Sharpe {annualised_sharpe(mom12_2000_2010):.2f}")

print("  Building fixed 1-mo momentum L/S (2000-2010) ...")
mom1_2000_2010 = build_ls(test_2000_2010.dropna(subset=['mom_1', 'ret_fwd']), 'mom_1')
print(f"    {len(mom1_2000_2010)} months, Sharpe {annualised_sharpe(mom1_2000_2010):.2f}")

# Market benchmark: value-weighted all stocks
print("  Building market value-weighted buy-hold (2000-2010) ...")
mkt_monthly = []
for date, grp in test_2000_2010.dropna(subset=['ret_fwd']).groupby('date'):
    if grp['me'].sum() == 0:
        continue
    r = (grp['ret_fwd'] * grp['me']).sum() / grp['me'].sum()
    mkt_monthly.append({'date': date, 'ret': r})
mkt_2000_2010 = pd.DataFrame(mkt_monthly).set_index('date')['ret']
print(f"    {len(mkt_2000_2010)} months, Sharpe {annualised_sharpe(mkt_2000_2010):.2f}")


# ── Sub-period decomposition with bootstrap CIs ──

print("\n[2/4] Computing sub-period metrics with bootstrap CIs ...")

def decompose_with_ci(r, label_prefix, n_boot=5000):
    rows = []
    overall = {
        'strategy': label_prefix,
        'subperiod': 'Overall',
        'start': r.index.min().strftime('%Y-%m-%d'),
        'end': r.index.max().strftime('%Y-%m-%d'),
        'n': len(r),
        'sharpe': annualised_sharpe(r),
        'cum': cumulative(r),
        'mdd': max_drawdown(r),
    }
    lo, med, hi = block_bootstrap_sharpe(r, n_boot=n_boot)
    overall.update(sharpe_lo=lo, sharpe_med=med, sharpe_hi=hi)
    rows.append(overall)
    for sp_name, start, end in SUBPERIODS:
        rsp = subperiod_slice(r, start, end)
        if len(rsp) == 0:
            continue
        lo, med, hi = block_bootstrap_sharpe(rsp, n_boot=n_boot)
        rows.append({
            'strategy': label_prefix,
            'subperiod': sp_name,
            'start': start,
            'end': end,
            'n': len(rsp),
            'sharpe': annualised_sharpe(rsp),
            'cum': cumulative(rsp),
            'mdd': max_drawdown(rsp),
            'sharpe_lo': lo,
            'sharpe_med': med,
            'sharpe_hi': hi,
        })
    return rows


all_rows = []
all_rows += decompose_with_ci(m2_a,              'M2 (1990-1999 train)')
all_rows += decompose_with_ci(m2_ap,             'M2 (1990-2004 train)')
all_rows += decompose_with_ci(mom12_2000_2010,   'Fixed 12-mo mom')
all_rows += decompose_with_ci(mom1_2000_2010,    'Fixed 1-mo mom')
all_rows += decompose_with_ci(mkt_2000_2010,     'Market (VW buy-hold)')

df = pd.DataFrame(all_rows)

# Print tidy table
print("\n" + "=" * 110)
print(f"{'Strategy':<24} {'Subperiod':<18} {'N':>4} {'Sharpe':>8} {'[2.5%,97.5%]':>16} {'Cum':>8} {'MDD':>8}")
print("=" * 110)
for _, row in df.iterrows():
    ci = f"[{row['sharpe_lo']:.2f}, {row['sharpe_hi']:.2f}]"
    print(f"{row['strategy']:<24} {row['subperiod']:<18} {row['n']:>4} "
          f"{row['sharpe']:>8.2f} {ci:>16} {row['cum'] * 100:>+7.1f}% "
          f"{row['mdd'] * 100:>+7.1f}%")

out_path = os.path.join(RESULTS_DIR, 'oos_subperiod_full.csv')
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")


# ── LaTeX table: M2 with CIs + fixed-mom + market ──

print("\n[3/4] Writing LaTeX tables ...")

# Table 1: M2 sub-period with bootstrap CIs (1990-1999 train, primary config)
def fmt_ci(row):
    return f"[{row['sharpe_lo']:.2f}, {row['sharpe_hi']:.2f}]"

m2_rows_primary = df[df['strategy'] == 'M2 (1990-1999 train)'].copy()

tex = []
tex.append(r'\begin{table}[H]')
tex.append(r'\centering')
tex.append(r'\small')
tex.append(r'\begin{tabular}{l r r r r r}')
tex.append(r'\toprule')
tex.append(r'Subperiod & N & Sharpe & 95\% CI & Cumulative & Max DD \\')
tex.append(r'\midrule')
for _, row in m2_rows_primary.iterrows():
    label = 'Overall' if row['subperiod'] == 'Overall' else row['subperiod']
    tex.append(
        f"{label} & {row['n']} & {row['sharpe']:.2f} & {fmt_ci(row)} & "
        f"{row['cum'] * 100:+.1f}\\% & {row['mdd'] * 100:+.1f}\\% \\\\"
    )
tex.append(r'\bottomrule')
tex.append(r'\end{tabular}')
tex.append(
    r'\caption{Historical out-of-sample sub-period decomposition of M2 '
    r'(XGBoost, mom+$\pi$) trained on 1990--1999 and tested on 2000--2010. '
    r'95\% confidence intervals are block-bootstrapped (block=6 months, '
    r'5{,}000 resamples). Small sub-period samples ($n < 20$) use smaller '
    r'blocks, giving wider intervals.}'
)
tex.append(r'\label{tab:oos_subperiods_m2}')
tex.append(r'\end{table}')

tex_path = os.path.join(TABLES_DIR, 'table_oos_subperiods_m2.tex')
with open(tex_path, 'w') as f:
    f.write('\n'.join(tex) + '\n')
print(f"  Saved: {tex_path}")


# Table 2: Benchmark comparison across sub-periods (Sharpe only, compact)
bench_rows = df[df['strategy'].isin([
    'M2 (1990-1999 train)',
    'Fixed 12-mo mom',
    'Fixed 1-mo mom',
    'Market (VW buy-hold)',
])].copy()

pivot = bench_rows.pivot(index='subperiod', columns='strategy', values='sharpe')
# Reorder rows to match SUBPERIODS + Overall at top
row_order = ['Overall'] + [s[0] for s in SUBPERIODS]
pivot = pivot.reindex(row_order)
col_order = ['M2 (1990-1999 train)', 'Fixed 12-mo mom', 'Fixed 1-mo mom', 'Market (VW buy-hold)']
pivot = pivot[col_order]

tex = []
tex.append(r'\begin{table}[H]')
tex.append(r'\centering')
tex.append(r'\small')
tex.append(r'\begin{tabular}{l r r r r}')
tex.append(r'\toprule')
tex.append(r'Subperiod & M2 & Fixed 12-mo mom & Fixed 1-mo mom & Market VW \\')
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
    r'\caption{Out-of-sample Sharpe ratio by sub-period (2000--2010 test). '
    r'M2 is trained on 1990--1999 (no crisis in training). Fixed momentum '
    r'portfolios and market buy-hold use the same L/S construction (top/bottom '
    r'decile, NYSE breakpoints, value-weighted) for fair comparison; the market '
    r'is value-weighted buy-and-hold.}'
)
tex.append(r'\label{tab:oos_benchmark_comparison}')
tex.append(r'\end{table}')

tex_path = os.path.join(TABLES_DIR, 'table_oos_benchmark_comparison.tex')
with open(tex_path, 'w') as f:
    f.write('\n'.join(tex) + '\n')
print(f"  Saved: {tex_path}")


print("\n[4/4] Done.")
