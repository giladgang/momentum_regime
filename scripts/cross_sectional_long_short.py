"""
cross_sectional_long_short.py
=============================
Exact replica of cross_sectional_model.py but with LONG-SHORT portfolios.
Uses saved artifacts (pre-trained models, pre-computed scores) to ensure
identical model outputs. Only the portfolio construction changes.

Long  : top decile by score (value-weighted, NYSE breakpoints)
Short : bottom decile by score (value-weighted, NYSE breakpoints)
"""

import numpy as np
import pandas as pd
import pickle
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load pre-computed data (identical to cross_sectional_model.py output)
# ══════════════════════════════════════════════════════════════════════════════

print("[ 1/4 ] Loading artifacts ...")

with open('cs_artefacts_data.pkl', 'rb') as f:
    artefacts = pickle.load(f)

test  = artefacts['test'].copy()
r_mkt = artefacts['r_mkt']

# Verify data integrity
print(f"  Test: {len(test):,} stock-months  |  {test['date'].nunique()} months  "
      f"|  {test['date'].min().date()} -> {test['date'].max().date()}")
print(f"  Scores: {[c for c in test.columns if 'score' in c]}")

# Pi_filter for regime-conditional analysis
pi_monthly = test.groupby('date')['pi_filter'].first()
calm_dates  = pi_monthly[pi_monthly < 0.5].index
panic_dates = pi_monthly[pi_monthly >= 0.5].index
print(f"  Calm months: {len(calm_dates)}  |  Panic months: {len(panic_dates)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Portfolio construction (long-short, identical rules to long_only_port)
# ══════════════════════════════════════════════════════════════════════════════

TRADING_FEE = 0.001  # 10 bps one-way, same as main pipeline
QUARTERLY_MONTHS = {3, 6, 9, 12}

def long_short_port(df_test, score_col, fee=TRADING_FEE, rebal_months=None):
    """
    Top-decile long, bottom-decile short, value-weighted, NYSE breakpoints.
    Identical filtering and breakpoint logic as long_only_port in
    cross_sectional_model.py. Transaction costs applied to both legs.
    """
    if rebal_months is None:
        rebal_months = set(range(1, 13))

    monthly = []
    prev_long_w  = {}
    prev_short_w = {}

    for date, grp in df_test.groupby('date'):
        if date.month not in rebal_months:
            # Hold month: keep previous holdings, no fee
            if not prev_long_w or not prev_short_w:
                continue
            held_l = grp[grp['permno'].isin(prev_long_w)]
            held_s = grp[grp['permno'].isin(prev_short_w)]
            if held_l.empty or held_s.empty:
                continue
            if held_l['me'].sum() == 0 or held_s['me'].sum() == 0:
                continue
            r_long  = (held_l['ret_fwd'] * held_l['me']).sum() / held_l['me'].sum()
            r_short = (held_s['ret_fwd'] * held_s['me']).sum() / held_s['me'].sum()
            # Drift weights
            prev_long_w  = (held_l.set_index('permno')['me'] / held_l['me'].sum()).to_dict()
            prev_short_w = (held_s.set_index('permno')['me'] / held_s['me'].sum()).to_dict()
            monthly.append({'date': date, 'ret': r_long - r_short})
            continue

        # Rebalancing month
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs  = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0:
            continue

        # Long leg
        long_me = longs['me'].sum()
        new_long_w = (longs.set_index('permno')['me'] / long_me).to_dict()
        r_long = (longs['ret_fwd'] * longs['me']).sum() / long_me

        # Short leg
        short_me = shorts['me'].sum()
        new_short_w = (shorts.set_index('permno')['me'] / short_me).to_dict()
        r_short = (shorts['ret_fwd'] * shorts['me']).sum() / short_me

        # Turnover (both legs)
        all_long = set(new_long_w) | set(prev_long_w)
        turn_long = sum(abs(new_long_w.get(p, 0) - prev_long_w.get(p, 0))
                        for p in all_long) / 2
        all_short = set(new_short_w) | set(prev_short_w)
        turn_short = sum(abs(new_short_w.get(p, 0) - prev_short_w.get(p, 0))
                         for p in all_short) / 2

        cost = fee * (turn_long + turn_short)
        monthly.append({'date': date, 'ret': r_long - r_short - cost})
        prev_long_w  = new_long_w
        prev_short_w = new_short_w

    return pd.DataFrame(monthly).set_index('date')['ret']


def metrics(r):
    r = pd.Series(r).dropna()
    n = len(r)
    ann_ret = (1 + r).prod() ** (12 / n) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe  = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum     = (1 + r).cumprod()
    mdd     = ((cum - cum.cummax()) / cum.cummax()).min()
    return {'Ann. Ret': ann_ret, 'Ann. Vol': ann_vol, 'Sharpe': sharpe, 'MDD': mdd, 'N': n}


def capm(r_port, r_mkt):
    common = r_port.index.intersection(r_mkt.index)
    rp, rm = r_port.loc[common].values, r_mkt.loc[common].values
    slope, intercept, _, _, _ = stats.linregress(rm, rp)
    resid = rp - (intercept + slope * rm)
    se = np.std(resid, ddof=2) / np.sqrt(len(rp))
    t = intercept / se
    p = 2 * (1 - stats.t.cdf(abs(t), len(rp) - 2))
    return {'alpha_a': intercept * 12, 't_alpha': t, 'p_alpha': p, 'beta': slope}


# ══════════════════════════════════════════════════════════════════════════════
# 3. Run all strategies (same as cross_sectional_model.py)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[ 2/4 ] Computing long-short portfolios ...")

strategies = {
    'Mom-12':       'score_mom12',
    'Mom-1':        'score_mom1',
    'M0 (Formula)': 'score_formula',
    'M1 (LR)':      'score_lr',
    'M2 (XGBoost)': 'score_xgb',
}

all_results = []

for name, score_col in strategies.items():
    # Monthly rebalancing
    r_m = long_short_port(test, score_col)
    m_m = metrics(r_m)
    c_m = capm(r_m, r_mkt)

    # Quarterly rebalancing
    r_q = long_short_port(test, score_col, rebal_months=QUARTERLY_MONTHS)
    m_q = metrics(r_q)

    # Regime-conditional Sharpe
    r_calm  = r_m[r_m.index.isin(calm_dates)]
    r_panic = r_m[r_m.index.isin(panic_dates)]
    sr_calm  = r_calm.mean()  / r_calm.std()  * np.sqrt(12) if r_calm.std()  > 0 else 0
    sr_panic = r_panic.mean() / r_panic.std() * np.sqrt(12) if r_panic.std() > 0 else 0

    all_results.append({
        'Strategy': name,
        'Sharpe_M': m_m['Sharpe'], 'Ret_M': m_m['Ann. Ret'], 'Vol_M': m_m['Ann. Vol'],
        'MDD_M': m_m['MDD'],
        'Sharpe_Q': m_q['Sharpe'], 'Ret_Q': m_q['Ann. Ret'],
        'Alpha': c_m['alpha_a'], 't_alpha': c_m['t_alpha'], 'p_alpha': c_m['p_alpha'],
        'Beta': c_m['beta'],
        'SR_calm': sr_calm, 'SR_panic': sr_panic,
        'Wealth': float((1 + r_m).prod()),
    })

    print(f"  {name:<16} Sharpe={m_m['Sharpe']:.3f}  Ret={m_m['Ann. Ret']:.1%}  "
          f"Alpha={c_m['alpha_a']:.1%}  Beta={c_m['beta']:.2f}  "
          f"SR_calm={sr_calm:.3f}  SR_panic={sr_panic:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Full results output
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*120}")
print("COMPLETE LONG-SHORT RESULTS (exact pipeline replica)")
print(f"{'='*120}")

# Table 1: Main performance
print(f"\n--- Table 1: Portfolio Performance (Monthly Rebalancing) ---")
print(f"{'Strategy':<16} {'Sharpe':>7} {'Ann.Ret':>8} {'Ann.Vol':>8} {'MDD':>8} "
      f"{'Alpha':>8} {'t(a)':>6} {'p(a)':>6} {'Beta':>6} {'Wealth':>8}")
print(f"{'-'*100}")
for r in all_results:
    sig = '**' if r['p_alpha'] < 0.05 else '  '
    print(f"{r['Strategy']:<16} {r['Sharpe_M']:>7.3f} {r['Ret_M']:>7.1%} {r['Vol_M']:>7.1%} "
          f"{r['MDD_M']:>7.1%} {r['Alpha']:>6.1%}{sig} {r['t_alpha']:>6.2f} "
          f"{r['p_alpha']:>6.3f} {r['Beta']:>6.2f} ${r['Wealth']:>7.2f}")

# Table 2: Quarterly rebalancing
print(f"\n--- Table 2: Monthly vs Quarterly Rebalancing ---")
print(f"{'Strategy':<16} {'Monthly':>9} {'Quarterly':>10}")
print(f"{'-'*40}")
for r in all_results:
    print(f"{r['Strategy']:<16} {r['Sharpe_M']:>9.3f} {r['Sharpe_Q']:>10.3f}")

# Table 3: Regime-conditional
print(f"\n--- Table 3: Regime-Conditional Sharpe Ratios ---")
print(f"{'Strategy':<16} {'Overall':>8} {'Calm':>8} {'Panic':>8} {'N_calm':>7} {'N_panic':>8}")
print(f"{'-'*60}")
for r in all_results:
    print(f"{r['Strategy']:<16} {r['Sharpe_M']:>8.3f} {r['SR_calm']:>8.3f} "
          f"{r['SR_panic']:>8.3f} {len(calm_dates):>7} {len(panic_dates):>8}")

# Table 4: Sub-period
print(f"\n--- Table 4: Sub-Period Sharpe Ratios ---")
periods = [('2011-2015', '2011-01-01', '2016-01-01'),
           ('2016-2020', '2016-01-01', '2021-01-01'),
           ('2021-2025', '2021-01-01', '2026-01-01')]
print(f"{'Strategy':<16} {'2011-2015':>10} {'2016-2020':>10} {'2021-2025':>10}")
print(f"{'-'*50}")
for r_dict in all_results:
    name = r_dict['Strategy']
    score_col = strategies[name]
    r = long_short_port(test, score_col)
    srs = []
    for _, start, end in periods:
        rp = r[(r.index >= start) & (r.index < end)]
        sr = rp.mean() / rp.std() * np.sqrt(12) if len(rp) > 3 and rp.std() > 0 else 0
        srs.append(sr)
    print(f"{name:<16} {srs[0]:>10.3f} {srs[1]:>10.3f} {srs[2]:>10.3f}")

# Newey-West t-test
print(f"\n--- Table 5: Statistical Significance (Newey-West HAC, 6 lags) ---")

import statsmodels.api as sm

print(f"{'Strategy':<16} {'Mean Ret':>10} {'NW t-stat':>10} {'p-value':>8}")
print(f"{'-'*50}")
for name, score_col in strategies.items():
    r = long_short_port(test, score_col)
    y = r.dropna()
    X = sm.add_constant(pd.Series(1, index=y.index, name='const'))
    res = sm.OLS(y, X[['const']]).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
    print(f"{name:<16} {res.params['const']:>10.5f} {res.tvalues['const']:>10.3f} "
          f"{res.pvalues['const']:>8.4f}")

print(f"\n{'='*120}")
print("Done.")
