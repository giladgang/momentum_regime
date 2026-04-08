"""
dm_replication.py
=================
Faithful replication of Daniel & Moskowitz (2016) momentum crash management,
then comparison with the HMM pi_filter as a drop-in replacement.

D&M Methodology (Section III of their paper):
1. Construct WML (Winner-Minus-Loser) long-short momentum portfolio
   - Standard 2-12 momentum (skip most recent month)
   - Long top decile, short bottom decile, value-weighted, NYSE breakpoints
2. Bear market indicator: B_t = 1 if cumulative market return over past 24 months < 0
3. Forecast conditional mean:  mu_t = a + b * B_t  (estimated in-sample)
4. Conditional variance: trailing 6-month realized variance of WML returns
5. Optimal weight: w_t = (1/2) * mu_t / sigma_t^2, clipped to [-0.5, 1.5]
6. Managed return: r_managed_t = w_t * r_WML_t

Then replicate with pi_filter replacing B_t.

Train: pre-2011  |  Test: 2011-2025
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 1. Load data ─────────────────────────────────────────────────────────────

print("[ 1/5 ] Loading data ...")

stocks = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks['date'] = pd.to_datetime(stocks['date'])
stocks = stocks.sort_values(['permno', 'date']).reset_index(drop=True)

# Eligibility
stocks = stocks[stocks['shrcd'].isin([10, 11])]
stocks = stocks[stocks['exchcd'].isin([1, 2, 3])]
stocks = stocks[stocks['prc'].abs() > 1.0]
stocks = stocks.reset_index(drop=True)

# Regime data
regimes = pd.read_parquet('data/panel_with_regimes.parquet')[['date', 'vwretd', 'pi_filter']]
regimes['date'] = pd.to_datetime(regimes['date'])

stocks = stocks.merge(regimes[['date', 'pi_filter']], on='date', how='left')
stocks['pi_filter'] = stocks['pi_filter'].ffill()

print(f"  Stocks: {len(stocks):,} rows  |  {stocks['permno'].nunique():,} permnos")

# ── 2. Compute 2-12 momentum (skip most recent month, as in D&M) ────────────

print("[ 2/5 ] Computing 2-12 momentum ...")

stocks['_log_ret'] = np.log1p(stocks['ret_adj'].clip(lower=-0.999))
stocks['_log_ret_s1'] = stocks.groupby('permno')['_log_ret'].shift(1)  # skip t

# 11-month cumulative return from t-12 to t-2 (i.e., shift(2) then roll 11)
stocks['_log_ret_s2'] = stocks.groupby('permno')['_log_ret'].shift(2)

# Rolling 11-month sum starting from t-2 (months t-12 through t-2)
roll_sum_12_2 = (
    stocks.groupby('permno', sort=False)['_log_ret_s2']
    .rolling(11, min_periods=11)
    .sum()
    .reset_index(level='permno', drop=True)
    .sort_index()
)
stocks['mom_12_2'] = np.expm1(roll_sum_12_2)

# Forward return
stocks['ret_fwd'] = stocks.groupby('permno')['ret_adj'].transform(lambda x: x.shift(-1))

# Drop rows without momentum or forward returns
df = stocks.dropna(subset=['mom_12_2', 'ret_fwd']).copy()
df = df.reset_index(drop=True)

print(f"  Usable rows: {len(df):,}  |  {df['date'].min().date()} -> {df['date'].max().date()}")

# ── 3. Construct WML portfolio returns ───────────────────────────────────────

print("[ 3/5 ] Constructing WML portfolio ...")

def wml_returns(data):
    """
    Each month: NYSE P10/P90 breakpoints on mom_12_2.
    Long >= P90, Short <= P10, value-weighted by me.
    Returns monthly long-short returns.
    """
    monthly = []
    for date, grp in data.groupby('date'):
        nyse = grp[grp['exchcd'] == 1]['mom_12_2'].dropna()
        if len(nyse) < 10:
            continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs  = grp[grp['mom_12_2'] >= hi]
        shorts = grp[grp['mom_12_2'] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0:
            continue
        r_long  = (longs['ret_fwd']  * longs['me']).sum()  / longs['me'].sum()
        r_short = (shorts['ret_fwd'] * shorts['me']).sum() / shorts['me'].sum()
        monthly.append({
            'date': date,
            'r_wml': r_long - r_short,
            'r_long': r_long,
            'r_short': r_short,
            'pi_filter': grp['pi_filter'].iloc[0]
        })
    return pd.DataFrame(monthly).set_index('date')

wml = wml_returns(df)
print(f"  WML: {len(wml)} months  |  {wml.index.min().date()} -> {wml.index.max().date()}")

# ── 4. Bear market indicator (D&M definition) ───────────────────────────────

# Cumulative market return over past 24 months
mkt = regimes.set_index('date')['vwretd'].sort_index()

# Rolling 24-month cumulative return: prod(1 + r_t) - 1
mkt_cum_24 = mkt.rolling(24).apply(lambda x: (1 + x).prod() - 1, raw=True)
bear_indicator = (mkt_cum_24 < 0).astype(int)
bear_indicator.name = 'bear'

# Merge bear indicator into WML
wml = wml.join(bear_indicator, how='left')
wml['bear'] = wml['bear'].fillna(0).astype(int)

print(f"  Bear months in full sample: {wml['bear'].sum()} / {len(wml)}")

# ── 5. Train/test split ─────────────────────────────────────────────────────

train_wml = wml[wml.index < '2011-01-01'].copy()
test_wml  = wml[wml.index >= '2011-01-01'].copy()

print(f"  Train: {len(train_wml)} months  |  Test: {len(test_wml)} months")

# ── 6. D&M Optimal Scaling ──────────────────────────────────────────────────

print("[ 4/5 ] D&M optimal scaling ...")

import statsmodels.api as sm

# --- 6a. Forecast conditional mean: mu_t = a + b * B_t ---
# Estimated on training data
X_bear_train = sm.add_constant(train_wml['bear'])
ols_dm = sm.OLS(train_wml['r_wml'], X_bear_train).fit()
print(f"\n  D&M Forecast regression (train):")
print(f"    Intercept: {ols_dm.params['const']:.4f}  (t={ols_dm.tvalues['const']:.2f})")
print(f"    Bear coef: {ols_dm.params['bear']:.4f}  (t={ols_dm.tvalues['bear']:.2f})")

# Out-of-sample conditional mean
X_bear_test = sm.add_constant(test_wml['bear'])
test_wml['mu_dm'] = ols_dm.predict(X_bear_test)

# --- 6b. Conditional variance: trailing 6-month realized variance ---
# Use full WML series for rolling variance (no look-ahead: variance at t uses t-5..t)
wml['sigma2_trailing'] = wml['r_wml'].rolling(6, min_periods=6).var()
test_wml['sigma2_dm'] = wml.loc[test_wml.index, 'sigma2_trailing']

# --- 6c. Optimal weight: w_t = (1/2) * mu_t / sigma2_t, clipped ---
test_wml['w_dm'] = (0.5 * test_wml['mu_dm'] / test_wml['sigma2_dm']).clip(-0.5, 1.5)

# Managed returns
test_wml['r_dm_managed'] = test_wml['w_dm'] * test_wml['r_wml']

# ── 7. Pi-filter replacement ────────────────────────────────────────────────

print("\n  Pi-filter forecast regression (train) ...")

# --- 7a. Forecast conditional mean: mu_t = a + b * pi_filter ---
X_pi_train = sm.add_constant(train_wml['pi_filter'])
ols_pi = sm.OLS(train_wml['r_wml'], X_pi_train).fit()
print(f"    Intercept:  {ols_pi.params['const']:.4f}  (t={ols_pi.tvalues['const']:.2f})")
print(f"    Pi coef:    {ols_pi.params['pi_filter']:.4f}  (t={ols_pi.tvalues['pi_filter']:.2f})")

# Out-of-sample conditional mean
X_pi_test = sm.add_constant(test_wml['pi_filter'])
test_wml['mu_pi'] = ols_pi.predict(X_pi_test)

# Same conditional variance (WML's own trailing vol)
test_wml['w_pi'] = (0.5 * test_wml['mu_pi'] / test_wml['sigma2_dm']).clip(-0.5, 1.5)
test_wml['r_pi_managed'] = test_wml['w_pi'] * test_wml['r_wml']

# --- 7b. Also try pi_filter as bear replacement: B_t = 1 if pi > 0.5 ---
test_wml['bear_pi'] = (test_wml['pi_filter'] > 0.5).astype(int)
X_bpi_train = sm.add_constant((train_wml['pi_filter'] > 0.5).astype(int).rename('bear_pi'))
ols_bpi = sm.OLS(train_wml['r_wml'], X_bpi_train).fit()
print(f"\n  Pi-binary forecast regression (train):")
print(f"    Intercept:   {ols_bpi.params['const']:.4f}  (t={ols_bpi.tvalues['const']:.2f})")
print(f"    Bear_pi:     {ols_bpi.params['bear_pi']:.4f}  (t={ols_bpi.tvalues['bear_pi']:.2f})")

X_bpi_test = sm.add_constant(test_wml['bear_pi'])
test_wml['mu_bpi'] = ols_bpi.predict(X_bpi_test)
test_wml['w_bpi'] = (0.5 * test_wml['mu_bpi'] / test_wml['sigma2_dm']).clip(-0.5, 1.5)
test_wml['r_bpi_managed'] = test_wml['w_bpi'] * test_wml['r_wml']

# ── 8. Barroso & Santa-Clara (2015) constant-volatility overlay ─────────────

print("\n  Barroso & Santa-Clara constant-vol overlay ...")

# Target vol = in-sample annualised vol of WML
target_vol = train_wml['r_wml'].std() * np.sqrt(12)
print(f"    Target vol (train ann.): {target_vol:.4f}")

# w_t = target_vol / (sqrt(12) * sigma_t)  where sigma_t is trailing 6-mo realised vol
test_wml['sigma_trailing'] = np.sqrt(test_wml['sigma2_dm'])
test_wml['w_bsc'] = (target_vol / (np.sqrt(12) * test_wml['sigma_trailing'])).clip(0, 1.5)
test_wml['r_bsc_managed'] = test_wml['w_bsc'] * test_wml['r_wml']

# ── 9. Metrics ───────────────────────────────────────────────────────────────

print("\n[ 5/5 ] Results ...\n")

def metrics(r, name=''):
    r = r.dropna()
    n = len(r)
    ann_ret = (1 + r).prod() ** (12 / n) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe  = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum     = (1 + r).cumprod()
    mdd     = ((cum - cum.cummax()) / cum.cummax()).min()
    return {'Strategy': name, 'Ann. Return': ann_ret, 'Ann. Vol': ann_vol,
            'Sharpe': sharpe, 'Max DD': mdd, 'N': n}

results = pd.DataFrame([
    metrics(test_wml['r_wml'],          'Unscaled WML'),
    metrics(test_wml['r_dm_managed'],   'D&M (Bear indicator)'),
    metrics(test_wml['r_bsc_managed'],  'B&SC (Constant vol)'),
    metrics(test_wml['r_pi_managed'],   'Pi-filter (continuous)'),
    metrics(test_wml['r_bpi_managed'],  'Pi-filter (binary >0.5)'),
])

print("=" * 85)
print("  D&M (2016) Replication: Long-Short Momentum Crash Management")
print("  Test period: 2011-2025")
print("=" * 85)
print(results.to_string(index=False, float_format='{:.4f}'.format))
print()

# ── 10. Regime-conditional performance ───────────────────────────────────────

print("\n  Regime-conditional Sharpe ratios:")
print("  (Bear = D&M bear indicator; Panic = pi_filter > 0.5)")

for label, mask_col, mask_val in [
    ('D&M Bear=1',   'bear', 1),
    ('D&M Bear=0',   'bear', 0),
    ('HMM Panic',    'bear_pi', 1),
    ('HMM Calm',     'bear_pi', 0),
]:
    subset = test_wml[test_wml[mask_col] == mask_val]
    n = len(subset)
    if n < 3:
        print(f"    {label:15s}: N={n} (too few)")
        continue
    strats = {
        'Unscaled': subset['r_wml'],
        'D&M':      subset['r_dm_managed'],
        'B&SC':     subset['r_bsc_managed'],
        'Pi-cont':  subset['r_pi_managed'],
        'Pi-bin':   subset['r_bpi_managed'],
    }
    sharpes = {k: v.mean() / v.std() * np.sqrt(12) if v.std() > 0 else 0
               for k, v in strats.items()}
    line = f"    {label:15s} (N={n:3d}): " + "  ".join(f"{k}={v:+.2f}" for k, v in sharpes.items())
    print(line)

# ── 11. Weight analysis ─────────────────────────────────────────────────────

print("\n  Average scaling weights:")
for col, name in [('w_dm', 'D&M'), ('w_bsc', 'B&SC'), ('w_pi', 'Pi-cont'), ('w_bpi', 'Pi-bin')]:
    w = test_wml[col].dropna()
    print(f"    {name:12s}: mean={w.mean():.3f}  std={w.std():.3f}  "
          f"min={w.min():.3f}  max={w.max():.3f}  "
          f"frac_negative={( w < 0).mean():.3f}")

# ── 12. Cumulative performance plot ──────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

# Top panel: cumulative returns
ax = axes[0]
for col, label, ls, c in [
    ('r_wml',          'Unscaled WML',           '-',  'grey'),
    ('r_dm_managed',   'D&M (Bear indicator)',    '-',  'blue'),
    ('r_bsc_managed',  'B&SC (Constant vol)',     '--', 'green'),
    ('r_pi_managed',   'Pi-filter (continuous)',   '-',  'red'),
    ('r_bpi_managed',  'Pi-filter (binary >0.5)', '--', 'orange'),
]:
    cum = (1 + test_wml[col].fillna(0)).cumprod()
    ax.plot(cum.index, cum, label=label, linestyle=ls, color=c, linewidth=1.5)

ax.set_ylabel('Cumulative Return (Growth of $1)')
ax.set_title('D&M (2016) Momentum Crash Management: Original vs Pi-filter\nLong-Short 2-12 Momentum, 2011-2025')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Bottom panel: scaling weights
ax2 = axes[1]
ax2.plot(test_wml.index, test_wml['w_dm'],  label='D&M weight',  color='blue', alpha=0.7)
ax2.plot(test_wml.index, test_wml['w_pi'],  label='Pi weight',   color='red',  alpha=0.7)
ax2.axhline(1, color='grey', linestyle='--', linewidth=0.8, label='Full exposure')
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_ylabel('Scaling Weight')
ax2.set_xlabel('Date')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig('dm_replication.png', dpi=150)
plt.close(fig)
print("\n  Plot saved: dm_replication.png")

# ── 13. Statistical test: are managed returns significantly different? ───────

from scipy import stats

print("\n  Paired t-test: D&M managed vs Pi-filter managed")
valid = test_wml[['r_dm_managed', 'r_pi_managed']].dropna()
diff = valid['r_pi_managed'] - valid['r_dm_managed']
t_stat, p_val = stats.ttest_1samp(diff, 0)
print(f"    Mean diff (Pi - D&M): {diff.mean():.5f}")
print(f"    t-stat: {t_stat:.3f}  p-value: {p_val:.4f}")

# Newey-West HAC t-test
print("\n  Newey-West HAC test (6 lags):")
for col, name in [('r_wml', 'Unscaled'), ('r_dm_managed', 'D&M'),
                   ('r_bsc_managed', 'B&SC'), ('r_pi_managed', 'Pi-cont'),
                   ('r_bpi_managed', 'Pi-bin')]:
    y = test_wml[col].dropna()
    X = sm.add_constant(pd.Series(1, index=y.index, name='const'))
    res = sm.OLS(y, X[['const']]).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
    print(f"    {name:12s}: mean={res.params['const']:.5f}  "
          f"NW t={res.tvalues['const']:.3f}  p={res.pvalues['const']:.4f}")

print("\nDone.")
