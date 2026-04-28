"""
test_regime_signal_ablation.py
==============================
Key ablation: does the HMM regime signal (pi_filter) add value over the
GHM market-state variable, or over no regime signal at all?

Runs XGBoost with three regime signal variants:
  1. pi_filter        (HMM filtered panic probability)
  2. GHM cycle        (Bull/Correction/Bear/Rebound encoded as numeric)
  3. No regime signal (momentum + fundamentals only)

Same features, same XGBoost hyperparameters, same portfolio construction.
"""

import numpy as np
import pandas as pd
import sys, os
from xgboost import XGBRegressor
import warnings, time
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (PORTFOLIO_TYPE, TRADING_FEE as CFG_TRADING_FEE, TRAIN_END,
                    N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, COLSAMPLE)

print("=" * 80)
print("  REGIME SIGNAL ABLATION: pi_filter vs GHM cycle vs None")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[ 1 ] Loading data ...")

stocks = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks['date'] = pd.to_datetime(stocks['date'])
stocks = stocks.sort_values(['permno', 'date']).reset_index(drop=True)
stocks = stocks[stocks['shrcd'].isin([10, 11])]
stocks = stocks[stocks['exchcd'].isin([1, 2, 3])]
stocks = stocks[stocks['prc'].abs() > 1.0]
stocks = stocks.reset_index(drop=True)

# pi_filter from HMM
regimes = pd.read_parquet('data/panel_with_regimes.parquet')[['date', 'pi_filter', 'vwretd']]
regimes['date'] = pd.to_datetime(regimes['date'])

stocks = stocks.merge(regimes[['date', 'pi_filter']], on='date', how='left')
stocks['pi_filter'] = stocks['pi_filter'].ffill()

# GHM market cycle
mkt_ret = regimes[['date', 'vwretd']].dropna().sort_values('date').reset_index(drop=True)
mkt_ret['mkt_fast'] = mkt_ret['vwretd']
mkt_ret['mkt_slow'] = mkt_ret['vwretd'].rolling(12, min_periods=12).mean()

def classify_cycle(row):
    if pd.isna(row['mkt_slow']):
        return np.nan
    if row['mkt_slow'] >= 0 and row['mkt_fast'] >= 0:
        return 0  # Bull
    elif row['mkt_slow'] >= 0 and row['mkt_fast'] < 0:
        return 1  # Correction
    elif row['mkt_slow'] < 0 and row['mkt_fast'] < 0:
        return 2  # Bear
    else:
        return 3  # Rebound

mkt_ret['ghm_cycle'] = mkt_ret.apply(classify_cycle, axis=1)
stocks = stocks.merge(mkt_ret[['date', 'ghm_cycle']], on='date', how='left')
stocks['ghm_cycle'] = stocks['ghm_cycle'].ffill()

print(f"  Stocks: {len(stocks):,} rows")

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

print("[ 2 ] Computing momentum signals ...")

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

MOM_FEATURES = [f'mom_{lb}' for lb in MOM_LBS]

# Three feature sets (momentum + regime signal only, no fundamentals)
FEATURES_PI = MOM_FEATURES + ['pi_filter']
FEATURES_GHM = MOM_FEATURES + ['ghm_cycle']
FEATURES_NONE = MOM_FEATURES

CORE_DROP = MOM_FEATURES + ['log_me', 'ret_fwd']
df = stocks.dropna(subset=CORE_DROP).copy().reset_index(drop=True)

train = df[df['date'] < TRAIN_END].copy()
test  = df[df['date'] >= TRAIN_END].copy()

print(f"  Train: {len(train):,}  |  Test: {len(test):,}")

# ═══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

TRADING_FEE = CFG_TRADING_FEE

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

def long_short_port(df_test, score_col, fee=TRADING_FEE):
    monthly = []
    prev_lw, prev_sw = {}, {}
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs  = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0:
            continue
        lme = longs['me'].sum()
        new_lw = (longs.set_index('permno')['me'] / lme).to_dict()
        r_long = (longs['ret_fwd'] * longs['me']).sum() / lme
        sme = shorts['me'].sum()
        new_sw = (shorts.set_index('permno')['me'] / sme).to_dict()
        r_short = (shorts['ret_fwd'] * shorts['me']).sum() / sme
        tl = sum(abs(new_lw.get(p, 0) - prev_lw.get(p, 0)) for p in set(new_lw) | set(prev_lw)) / 2
        ts = sum(abs(new_sw.get(p, 0) - prev_sw.get(p, 0)) for p in set(new_sw) | set(prev_sw)) / 2
        monthly.append({'date': date, 'ret': r_long - r_short - fee * (tl + ts)})
        prev_lw, prev_sw = new_lw, new_sw
    if not monthly:
        return pd.Series(dtype=float)
    return pd.DataFrame(monthly).set_index('date')['ret']


def build_port(df_test, score_col, fee=TRADING_FEE):
    if PORTFOLIO_TYPE == 'long_short':
        return long_short_port(df_test, score_col, fee=fee)
    else:
        return long_only_port(df_test, score_col, fee=fee)

def metrics(r):
    r = pd.Series(r).dropna()
    if len(r) < 12:
        return {'ann_ret': 0, 'ann_vol': 0, 'sharpe': 0, 'mdd': 0}
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {'ann_ret': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe, 'mdd': mdd}

# ═══════════════════════════════════════════════════════════════════════════════
#  RUN THREE VARIANTS
# ═══════════════════════════════════════════════════════════════════════════════

variants = {
    'XGB + pi_filter (HMM)': FEATURES_PI,
    'XGB + GHM cycle':       FEATURES_GHM,
    'XGB (no regime signal)': FEATURES_NONE,
}

results = {}

for name, features in variants.items():
    print(f"\n{'─' * 60}")
    print(f"  Running: {name}")
    print(f"  Features ({len(features)}): {', '.join(features)}")
    t0 = time.time()

    X_train = train[features].values.astype(float)
    y_train = train['ret_fwd'].values.astype(float)
    X_test  = test[features].values.astype(float)

    xgb = XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method='hist', random_state=42, verbosity=0
    )
    xgb.fit(X_train, y_train)

    test[f'score_{name}'] = xgb.predict(X_test)
    r = build_port(test, f'score_{name}')
    m = metrics(r)
    results[name] = m

    elapsed = time.time() - t0
    print(f"  Ann.Ret: {m['ann_ret']:.1%}  |  Ann.Vol: {m['ann_vol']:.1%}  "
          f"|  Sharpe: {m['sharpe']:.3f}  |  Max DD: {m['mdd']:.1%}  |  ({elapsed:.0f}s)")

    # Feature importance (top 5)
    fi = pd.Series(xgb.feature_importances_, index=features).sort_values(ascending=False)
    print(f"  Top 5 features: {', '.join(f'{f}={v:.3f}' for f, v in fi.head(5).items())}")

# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  SUMMARY: REGIME SIGNAL ABLATION")
print("=" * 80)
print(f"\n  {'Variant':<30s}  {'Ann.Ret':>8s}  {'Ann.Vol':>8s}  {'Sharpe':>8s}  {'Max DD':>8s}")
print("  " + "─" * 70)

for name, m in results.items():
    print(f"  {name:<30s}  {m['ann_ret']:>7.1%}  {m['ann_vol']:>7.1%}  "
          f"{m['sharpe']:>8.3f}  {m['mdd']:>7.1%}")

# Relative comparison
if 'XGB + pi_filter (HMM)' in results and 'XGB + GHM cycle' in results:
    sr_pi = results['XGB + pi_filter (HMM)']['sharpe']
    sr_ghm = results['XGB + GHM cycle']['sharpe']
    sr_none = results['XGB (no regime signal)']['sharpe']
    print(f"\n  Sharpe difference (HMM vs GHM):    {sr_pi - sr_ghm:+.3f}")
    print(f"  Sharpe difference (HMM vs None):   {sr_pi - sr_none:+.3f}")
    print(f"  Sharpe difference (GHM vs None):   {sr_ghm - sr_none:+.3f}")

print("\n" + "=" * 80)
