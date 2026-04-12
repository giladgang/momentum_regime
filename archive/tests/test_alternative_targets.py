"""
test_alternative_targets.py
===========================
Test alternative XGBoost training targets:
1. Baseline: raw return r
2. Mean-variance utility: r - (gamma/2) * sigma^2  (gamma=2 as representative)
3. Log mean-variance utility: log(1 + r) - (gamma/2) * sigma^2
4. Sharpe-like (r / sigma)
5. Sharpe-like with variance (r / sigma^2)
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("  ALTERNATIVE XGB TARGETS")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA (same pipeline as risk_aversion_analysis.py)
# ═══════════════════════════════════════════════════════════════════════════════

print("Loading data ...")
stocks = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks['date'] = pd.to_datetime(stocks['date'])
stocks = stocks.sort_values(['permno', 'date']).reset_index(drop=True)
stocks = stocks[stocks['shrcd'].isin([10, 11])]
stocks = stocks[stocks['exchcd'].isin([1, 2, 3])]
stocks = stocks[stocks['prc'].abs() > 1.0]
stocks = stocks.reset_index(drop=True)

regimes = pd.read_parquet('data/panel_with_regimes.parquet')[['date', 'pi_filter']]
regimes['date'] = pd.to_datetime(regimes['date'])
stocks = stocks.merge(regimes, on='date', how='left')
stocks['pi_filter'] = stocks['pi_filter'].ffill()

MOM_LBS = list(range(1, 13))
stocks['_log_ret'] = np.log1p(stocks['ret_adj'].clip(lower=-0.999))
stocks['_log_ret_s1'] = stocks.groupby('permno')['_log_ret'].shift(1)
for lb in MOM_LBS:
    roll_sum = (
        stocks.groupby('permno', sort=False)['_log_ret_s1']
        .rolling(lb, min_periods=lb).sum()
        .reset_index(level='permno', drop=True).sort_index()
    )
    stocks[f'mom_{lb}'] = np.expm1(roll_sum)
stocks.drop(columns=['_log_ret', '_log_ret_s1'], inplace=True)

stocks['log_me'] = np.log(
    stocks.groupby('permno')['me'].transform(lambda x: x.shift(1)).replace(0, np.nan)
)
stocks['ret_fwd'] = stocks.groupby('permno')['ret_adj'].transform(lambda x: x.shift(-1))

# Trailing realized variance and std (12-month rolling)
stocks['trail_var'] = (
    stocks.groupby('permno')['ret_adj']
    .transform(lambda x: x.shift(1).rolling(12, min_periods=6).var())
)
stocks['trail_std'] = np.sqrt(stocks['trail_var'])

MOM_FEATURES = [f'mom_{lb}' for lb in MOM_LBS]
FEATURES = MOM_FEATURES + [
    'pi_filter', 'bm', 'roe', 'earnings_growth',
    'leverage', 'asset_growth', 'gross_profit_a', 'log_me',
]

CORE_FEATURES = MOM_FEATURES + ['pi_filter', 'log_me']
df = stocks.dropna(subset=['ret_fwd', 'trail_var', 'trail_std'] + CORE_FEATURES).copy()
df = df.reset_index(drop=True)

train = df[df['date'] < '2011-01-01'].copy()
test  = df[df['date'] >= '2011-01-01'].copy()

X_train = train[FEATURES].values.astype(float)
X_test  = test[FEATURES].values.astype(float)

print(f"  Train: {len(train):,}  |  Test: {len(test):,}")

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

TRADING_FEE = 0.001

def long_only_port(df_test, score_col, fee=TRADING_FEE):
    monthly, prev_weights = [], {}
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        hi = nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        if longs['me'].sum() == 0:
            continue
        total_me = longs['me'].sum()
        new_w = (longs.set_index('permno')['me'] / total_me).to_dict()
        turnover = sum(abs(new_w.get(p, 0) - prev_weights.get(p, 0))
                       for p in set(new_w) | set(prev_weights)) / 2
        r_gross = (longs['ret_fwd'] * longs['me']).sum() / total_me
        monthly.append({'date': date, 'ret': r_gross - fee * turnover})
        prev_weights = new_w
    if not monthly:
        return pd.Series(dtype=float)
    return pd.DataFrame(monthly).set_index('date')['ret']

def compute_metrics(r):
    r = pd.Series(r).dropna()
    if len(r) < 12:
        return 0, 0, 0, 0
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return ann_ret, ann_vol, sharpe, mdd

# ═══════════════════════════════════════════════════════════════════════════════
#  DEFINE TARGETS
# ═══════════════════════════════════════════════════════════════════════════════

r_tr = train['ret_fwd'].values.astype(float)
var_tr = train['trail_var'].values.astype(float)
std_tr = train['trail_std'].values.astype(float)

TARGETS = {}

# Baseline
TARGETS['Baseline (raw return)'] = r_tr

# Mean-variance utility: r - (gamma/2) * sigma^2
for gamma in [0.5, 1, 2, 5, 10]:
    TARGETS[f'Mean-var (gamma={gamma})'] = r_tr - (gamma / 2) * var_tr

# Log mean-variance: log(1+r) - (gamma/2) * sigma^2
for gamma in [0.5, 1, 2, 5, 10]:
    TARGETS[f'Log mean-var (gamma={gamma})'] = np.log1p(r_tr.clip(min=-0.999)) - (gamma / 2) * var_tr

# Sharpe-like: r / sigma^a
for a in [0.5, 1, 1.5, 2]:
    label = f'Sharpe-like (r / sigma^{a})'
    TARGETS[label] = r_tr / np.clip(std_tr ** a, 1e-4, None)

# ═══════════════════════════════════════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════════════════════════════════════

all_results = {}

for name, y_target in TARGETS.items():
    print(f"\n{'─' * 60}")
    print(f"  {name}")

    model = XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method='hist', random_state=42, verbosity=0
    )
    model.fit(X_train, y_target)

    score_col = 'score_tmp'
    test_copy = test.copy()
    test_copy[score_col] = model.predict(X_test)

    r = long_only_port(test_copy, score_col)
    if len(r) > 12:
        m = compute_metrics(r)
        all_results[name] = m
        print(f"  Ann.Ret={m[0]:.1%}  Vol={m[1]:.1%}  SR={m[2]:.3f}  MDD={m[3]:.1%}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  SUMMARY: ALTERNATIVE XGB TARGETS")
print("=" * 80)

print(f"\n  {'Target':<35s}  {'Sharpe':>8s}  {'Ann.Ret':>8s}  {'Vol':>8s}  {'MDD':>8s}")
print("  " + "─" * 75)

for name, m in all_results.items():
    print(f"  {name:<35s}  {m[2]:>8.3f}  {m[0]:>7.1%}  {m[1]:>7.1%}  {m[3]:>7.1%}")

print("\n" + "=" * 80)
