"""
test_pi_polynomial.py
=====================
Test whether adding polynomial terms of pi_filter (pi^2, pi^3, etc.)
to the logistic regression helps capture the nonlinear regime effect.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings, time
warnings.filterwarnings('ignore')

print("=" * 80)
print("  PI_FILTER POLYNOMIAL TEST: LR with pi, pi^2, pi^3, ...")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════════

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])

stocks = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks['date'] = pd.to_datetime(stocks['date'])
stocks = stocks[stocks['shrcd'].isin([10, 11])]
stocks = stocks[stocks['exchcd'].isin([1, 2, 3])]
stocks = stocks[stocks['prc'].abs() > 1.0]
stocks = stocks.sort_values(['permno', 'date']).reset_index(drop=True)

# Momentum
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

# Merge pi_filter
stocks = stocks.merge(panel[['date', 'pi_filter']], on='date', how='left')
stocks['pi_filter'] = stocks['pi_filter'].ffill()

MOM_FEATURES = [f'mom_{lb}' for lb in MOM_LBS]
FUND_FEATURES = ['bm', 'roe', 'earnings_growth', 'leverage', 'asset_growth',
                 'gross_profit_a', 'log_me']
FEATURES_BASE = MOM_FEATURES + ['pi_filter'] + FUND_FEATURES
CORE_DROP = MOM_FEATURES + ['log_me', 'ret_fwd']

TRADING_FEE = 0.001

print(f"  Stocks: {len(stocks):,} rows")

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

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
#  RUN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

df = stocks.dropna(subset=CORE_DROP).copy().reset_index(drop=True)
train = df[df['date'] < '2011-01-01'].copy()
test  = df[df['date'] >= '2011-01-01'].copy()

# Binary target
train['above_med'] = train.groupby('date')['ret_fwd'].transform(
    lambda x: (x >= x.median()).astype(int))
y_cls = train['above_med'].values

# Test configurations: baseline, +pi^2, +pi^2+pi^3, interactions
CONFIGS = {
    'Baseline (pi)':          [],
    '+ pi^2':                 [2],
    '+ pi^2 + pi^3':          [2, 3],
    '+ pi^2 + pi^3 + pi^4':  [2, 3, 4],
    '+ pi * mom interactions': 'interactions',
}

all_results = {}

for name, poly_powers in CONFIGS.items():
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    t0 = time.time()

    # Build feature matrices
    train_feats = train[FEATURES_BASE].copy()
    test_feats = test[FEATURES_BASE].copy()

    if poly_powers == 'interactions':
        # Add pi * each momentum feature
        for mom in MOM_FEATURES:
            train_feats[f'pi_x_{mom}'] = train['pi_filter'] * train[mom]
            test_feats[f'pi_x_{mom}'] = test['pi_filter'] * test[mom]
    elif poly_powers:
        for p in poly_powers:
            train_feats[f'pi_pow{p}'] = train['pi_filter'] ** p
            test_feats[f'pi_pow{p}'] = test['pi_filter'] ** p

    X_tr = train_feats.values.astype(float)
    X_te = test_feats.values.astype(float)

    # Impute NaN
    for j in range(X_tr.shape[1]):
        col_median = np.nanmedian(X_tr[:, j])
        X_tr[np.isnan(X_tr[:, j]), j] = col_median
        X_te[np.isnan(X_te[:, j]), j] = col_median

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_tr_s, y_cls)
    test_copy = test.copy()
    test_copy['score_lr'] = lr.predict_proba(X_te_s)[:, 1]

    r_lr = long_only_port(test_copy, 'score_lr')
    if len(r_lr) > 12:
        m = compute_metrics(r_lr)
        all_results[name] = m
        print(f"  LR: Ann.Ret={m[0]:.1%}  Vol={m[1]:.1%}  SR={m[2]:.3f}  MDD={m[3]:.1%}")

    # Also get pi_filter coefficient(s)
    feat_names = list(train_feats.columns)
    pi_idx = feat_names.index('pi_filter')
    print(f"  pi_filter coef: {lr.coef_[0][pi_idx]:.4f}")
    for fn in feat_names:
        if fn.startswith('pi_pow') or fn.startswith('pi_x_'):
            idx = feat_names.index(fn)
            print(f"  {fn} coef: {lr.coef_[0][idx]:.4f}")

    elapsed = time.time() - t0
    print(f"  ({elapsed:.0f}s)")


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  SUMMARY: PI_FILTER POLYNOMIAL LR TEST")
print("=" * 80)

print(f"\n  {'Config':<35s}  {'Sharpe':>8s}  {'Ann.Ret':>8s}  {'Vol':>8s}")
print("  " + "─" * 65)

for name, m in all_results.items():
    print(f"  {name:<35s}  {m[2]:>8.3f}  {m[0]:>7.1%}  {m[1]:>7.1%}")

print("\n" + "=" * 80)
