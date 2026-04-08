"""
test_long_short.py
==================
Compare long-only vs long-short portfolios for all methods.
Uses saved model artifacts -- does not retrain anything.
"""

import numpy as np
import pandas as pd
import pickle

# ── Load artifacts ───────────────────────────────────────────────────────────

print("Loading data ...")
with open('cs_artefacts_data.pkl', 'rb') as f:
    artefacts = pickle.load(f)

test = artefacts['test'].copy()

print(f"Test set: {test['date'].nunique()} months, {len(test):,} stock-months")

# ── Portfolio functions ──────────────────────────────────────────────────────

TRADING_FEE = 0.0010

def long_only_port(df, score_col, fee=TRADING_FEE):
    """Top-decile long-only, value-weighted, NYSE breakpoints."""
    monthly = []
    prev_weights = {}
    for date, grp in df.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        hi = nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        if longs['me'].sum() == 0:
            continue
        total_me = longs['me'].sum()
        new_weights = (longs.set_index('permno')['me'] / total_me).to_dict()
        all_permnos = set(new_weights) | set(prev_weights)
        turnover = sum(abs(new_weights.get(p, 0) - prev_weights.get(p, 0))
                       for p in all_permnos) / 2
        r_gross = (longs['ret_fwd'] * longs['me']).sum() / total_me
        monthly.append({'date': date, 'ret': r_gross - fee * turnover})
        prev_weights = new_weights
    return pd.DataFrame(monthly).set_index('date')['ret']


def long_short_port(df, score_col, fee=TRADING_FEE):
    """Top-decile long, bottom-decile short, value-weighted, NYSE breakpoints."""
    monthly = []
    prev_long_w = {}
    prev_short_w = {}
    for date, grp in df.groupby('date'):
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

        prev_long_w = new_long_w
        prev_short_w = new_short_w

    return pd.DataFrame(monthly).set_index('date')['ret']


def metrics(r):
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return {}
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe  = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum     = (1 + r).cumprod()
    mdd     = ((cum - cum.cummax()) / cum.cummax()).min()
    return {'Ann. Ret': ann_ret, 'Ann. Vol': ann_vol, 'Sharpe': sharpe, 'MDD': mdd}


# All scores are already in the test dataframe
print(f"  Scores available: {[c for c in test.columns if 'score' in c]}")

# ── Run portfolios ──────────────────────────────────────────────────────────

strategies = {
    'Mom-12': 'score_mom12',
    'Mom-1':  'score_mom1',
    'M0 (Formula)': 'score_formula',
    'M1 (LR)': 'score_lr',
    'M2 (XGB)': 'score_xgb',
}

print("\nComputing long-only and long-short portfolios ...")

results = []
for name, score_col in strategies.items():
    r_lo = long_only_port(test, score_col)
    r_ls = long_short_port(test, score_col)

    m_lo = metrics(r_lo)
    m_ls = metrics(r_ls)

    results.append({
        'Strategy': name,
        'Type': 'Long-Only',
        **m_lo
    })
    results.append({
        'Strategy': name,
        'Type': 'Long-Short',
        **m_ls
    })

# ── Display results ──────────────────────────────────────────────────────────

df = pd.DataFrame(results)

print(f"\n{'='*85}")
print("LONG-ONLY vs LONG-SHORT PORTFOLIO COMPARISON")
print(f"{'='*85}")
print(f"{'Strategy':<16} {'Type':<12} {'Sharpe':>8} {'Ann.Ret':>9} {'Ann.Vol':>9} {'MDD':>9}")
print(f"{'-'*85}")

for _, row in df.iterrows():
    print(f"{row['Strategy']:<16} {row['Type']:<12} "
          f"{row['Sharpe']:>8.3f} {row['Ann. Ret']:>8.1%} {row['Ann. Vol']:>8.1%} {row['MDD']:>8.1%}")

# ── Side-by-side summary ────────────────────────────────────────────────────

print(f"\n{'='*85}")
print("SIDE-BY-SIDE: Long-Only Sharpe vs Long-Short Sharpe")
print(f"{'='*85}")
print(f"{'Strategy':<16} {'LO Sharpe':>10} {'LS Sharpe':>10} {'Diff':>10} {'Better':>10}")
print(f"{'-'*85}")

for name in strategies:
    lo = df[(df['Strategy'] == name) & (df['Type'] == 'Long-Only')].iloc[0]
    ls = df[(df['Strategy'] == name) & (df['Type'] == 'Long-Short')].iloc[0]
    diff = ls['Sharpe'] - lo['Sharpe']
    better = 'L/S' if diff > 0 else 'L/O'
    print(f"{name:<16} {lo['Sharpe']:>10.3f} {ls['Sharpe']:>10.3f} {diff:>+10.3f} {better:>10}")

# ── Regime-conditional breakdown ─────────────────────────────────────────────

print(f"\n{'='*85}")
print("REGIME-CONDITIONAL SHARPE (Long-Short only)")
print(f"{'='*85}")

pi_monthly = test.groupby('date')['pi_filter'].first()

for name, score_col in strategies.items():
    r_ls = long_short_port(test, score_col)
    calm_dates  = pi_monthly[pi_monthly < 0.5].index
    panic_dates = pi_monthly[pi_monthly >= 0.5].index

    r_calm  = r_ls[r_ls.index.isin(calm_dates)]
    r_panic = r_ls[r_ls.index.isin(panic_dates)]

    m_calm  = metrics(r_calm)
    m_panic = metrics(r_panic)

    sr_calm  = m_calm.get('Sharpe', 0)
    sr_panic = m_panic.get('Sharpe', 0)

    print(f"  {name:<16} Calm Sharpe: {sr_calm:>7.3f}   Panic Sharpe: {sr_panic:>7.3f}")
