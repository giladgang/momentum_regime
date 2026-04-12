"""Compute CAPM beta for each strategy, including GHM benchmarks."""
import numpy as np
import pandas as pd
import pickle, sys, os
sys.path.insert(0, 'scripts')

# Load artefacts
with open('cs_artefacts_data.pkl', 'rb') as f:
    artefacts = pickle.load(f)

strats_lo = artefacts['strategies_lo']
r_mkt = artefacts['r_mkt']
test = artefacts['test'].copy()
train = artefacts['train'].copy()

# ── Recompute GHM returns ──
panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
mkt_ret = panel[['date', 'vwretd']].dropna().drop_duplicates('date').sort_values('date').reset_index(drop=True)
mkt_ret['mkt_fast'] = mkt_ret['vwretd']
mkt_ret['mkt_slow'] = mkt_ret['vwretd'].rolling(12, min_periods=12).mean()

def classify_cycle(row):
    if pd.isna(row['mkt_slow']): return np.nan
    if row['mkt_slow'] >= 0 and row['mkt_fast'] >= 0: return 'Bull'
    elif row['mkt_slow'] >= 0 and row['mkt_fast'] < 0: return 'Correction'
    elif row['mkt_slow'] < 0 and row['mkt_fast'] < 0: return 'Bear'
    else: return 'Rebound'

mkt_ret['cycle'] = mkt_ret.apply(classify_cycle, axis=1)
cycle_map = mkt_ret[['date', 'cycle']].dropna()
train = train.merge(cycle_map, on='date', how='left')
test = test.merge(cycle_map, on='date', how='left')

TRADING_FEE = 0.001
def long_only_port(df_test, score_col, fee=TRADING_FEE):
    monthly, prev_weights = [], {}
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10: continue
        hi = nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        if longs['me'].sum() == 0: continue
        total_me = longs['me'].sum()
        new_w = (longs.set_index('permno')['me'] / total_me).to_dict()
        turnover = sum(abs(new_w.get(p, 0) - prev_weights.get(p, 0))
                       for p in set(new_w) | set(prev_weights)) / 2
        r_gross = (longs['ret_fwd'] * longs['me']).sum() / total_me
        monthly.append({'date': date, 'ret': r_gross - fee * turnover})
        prev_weights = new_w
    if not monthly: return pd.Series(dtype=float)
    return pd.DataFrame(monthly).set_index('date')['ret']

def compute_blended_score(data, a):
    rank_slow = data.groupby('date')['mom_12'].rank(pct=True)
    rank_fast = data.groupby('date')['mom_1'].rank(pct=True)
    return (1 - a) * rank_slow + a * rank_fast

def compute_dyn_score(data, a_bu, a_co, a_be, a_re):
    rank_slow = data.groupby('date')['mom_12'].rank(pct=True)
    rank_fast = data.groupby('date')['mom_1'].rank(pct=True)
    a = pd.Series(np.nan, index=data.index)
    a[data['cycle'] == 'Bull'] = a_bu
    a[data['cycle'] == 'Correction'] = a_co
    a[data['cycle'] == 'Bear'] = a_be
    a[data['cycle'] == 'Rebound'] = a_re
    a = a.fillna(0.5)
    return (1 - a) * rank_slow + a * rank_fast

ghm_returns = {}
for name, a in [('GHM SLOW', 0.0), ('GHM MED', 0.5), ('GHM FAST', 1.0)]:
    test[f'score_{name}'] = compute_blended_score(test, a).values
    ghm_returns[name] = long_only_port(test, f'score_{name}')

# DYN grid search
best_sharpe, best_pair = -999, (0.5, 0.5)
grid = np.arange(0.0, 1.05, 0.1)
for a_co in grid:
    for a_re in grid:
        train['_dyn'] = compute_dyn_score(train, 0.5, a_co, 0.5, a_re).values
        r = long_only_port(train, '_dyn')
        if len(r) < 12: continue
        sh = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
        if sh > best_sharpe:
            best_sharpe = sh
            best_pair = (a_co, a_re)

a_co_hat, a_re_hat = best_pair
test['score_dyn'] = compute_dyn_score(test, 0.5, a_co_hat, 0.5, a_re_hat).values
ghm_returns['GHM DYN'] = long_only_port(test, 'score_dyn')

# All strategies
all_strats = {
    'Market':          r_mkt,
    'Fixed 12-mo mom': strats_lo['Fixed 12-mo mom'],
    'Fixed 1-mo mom':  strats_lo['Fixed 1-mo mom'],
    'M0: Formula':     strats_lo['Method 0: Formula'],
    'M1: LR':          strats_lo['Method 1: LR'],
    'M2: XGB':         strats_lo['Method 2: XGB'],
    'GHM SLOW':        ghm_returns['GHM SLOW'],
    'GHM MED':         ghm_returns['GHM MED'],
    'GHM FAST':        ghm_returns['GHM FAST'],
    'GHM DYN':         ghm_returns['GHM DYN'],
}

print(f"{'Strategy':<20s}  {'Beta':>6s}  {'Alpha (ann)':>12s}")
print("-" * 45)

for name, r in all_strats.items():
    if name == 'Market':
        print(f"{'Market':<20s}  {'1.000':>6s}  {'--':>12s}")
        continue

    # Align dates
    common = r.index.intersection(r_mkt.index)
    y = r.loc[common].values.astype(float)
    x = r_mkt.loc[common].values.astype(float)

    # OLS: r_portfolio = alpha + beta * r_market
    X = np.column_stack([np.ones(len(x)), x])
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha_monthly = coeffs[0]
    beta = coeffs[1]
    alpha_ann = alpha_monthly * 12

    print(f"{name:<20s}  {beta:>6.3f}  {alpha_ann:>11.1%}")
