"""
fundamentals_test.py
====================
Standalone test: retrain XGB with fundamental features added.
Does NOT modify config.py, pipeline artefacts, or any production files.

Saves results to: results/fundamentals_test_results.csv

Usage:
    python -u scripts/fundamentals_test.py
"""

import numpy as np
import pandas as pd
import pickle, sys, os, time
from xgboost import XGBRegressor
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE,
                    COLSAMPLE, XGB_SEEDS, TRADING_FEE, TRAIN_END)

# ── Load data ──
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

# Momentum
print("Computing momentum ...")
stocks['_log_ret'] = np.log1p(stocks['ret_adj'].clip(lower=-0.999))
stocks['_log_ret_s1'] = stocks.groupby('permno')['_log_ret'].shift(1)
for lb in range(1, 13):
    roll_sum = (
        stocks.groupby('permno', sort=False)['_log_ret_s1']
        .rolling(lb, min_periods=lb).sum()
        .reset_index(level='permno', drop=True).sort_index()
    )
    stocks[f'mom_{lb}'] = np.expm1(roll_sum)
stocks.drop(columns=['_log_ret', '_log_ret_s1'], inplace=True)

# Size
stocks['log_me'] = np.log(
    stocks.groupby('permno')['me'].transform(lambda x: x.shift(1)).replace(0, np.nan)
)

# Target
stocks['ret_fwd'] = stocks.groupby('permno')['ret_adj'].transform(lambda x: x.shift(-1))

# Feature sets
MOM_FEATURES = [f'mom_{lb}' for lb in range(1, 13)]
FUND_FEATURES = ['bm', 'roe', 'earnings_growth', 'leverage', 'asset_growth',
                 'gross_profit_a', 'log_me']

FEATURES_BASE = MOM_FEATURES + ['pi_filter']
FEATURES_FUND = MOM_FEATURES + ['pi_filter'] + FUND_FEATURES

# Drop rows missing core features (XGB handles NaN fundamentals natively)
CORE = MOM_FEATURES + ['pi_filter', 'log_me']
df = stocks.dropna(subset=['ret_fwd'] + CORE).copy().reset_index(drop=True)

train = df[df['date'] < TRAIN_END].copy()
test = df[df['date'] >= TRAIN_END].copy()

print(f"  Train: {len(train):,}  |  Test: {len(test):,}")


def long_short_port(df_test, score_col, fee=TRADING_FEE):
    monthly = []
    prev_lw, prev_sw = {}, {}
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
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
        monthly.append({'date': date, 'ret': r_long - r_short - fee * (tl + ts)})
        prev_lw, prev_sw = new_lw, new_sw
    if not monthly:
        return pd.Series(dtype=float)
    return pd.DataFrame(monthly).set_index('date')['ret']


def train_and_eval(name, features, train_df, test_df):
    X_tr = train_df[features].values.astype(float)
    X_te = test_df[features].values.astype(float)
    y_tr = train_df['ret_fwd'].values.astype(float)

    print(f"\n  {name} ({len(features)} features, {len(XGB_SEEDS)} seeds) ...")
    t0 = time.time()
    preds = np.zeros(len(X_te))
    last_model = None
    for xs in XGB_SEEDS:
        model = XGBRegressor(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE, tree_method='hist',
            random_state=xs, verbosity=0
        )
        model.fit(X_tr, y_tr)
        preds += model.predict(X_te)
        last_model = model
    preds /= len(XGB_SEEDS)

    score_col = f'score_{name}'
    test_df[score_col] = preds
    r = long_short_port(test_df, score_col)

    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0

    # SHAP
    explainer = shap.TreeExplainer(last_model)
    sv = explainer.shap_values(X_te)
    abs_shap = np.abs(sv).mean(axis=0)
    total = abs_shap.sum()

    pi_share = abs_shap[features.index('pi_filter')] / total * 100 if 'pi_filter' in features else 0.0
    mom_share = sum(abs_shap[features.index(f)] for f in MOM_FEATURES if f in features) / total * 100
    fund_share = sum(abs_shap[features.index(f)] for f in FUND_FEATURES if f in features) / total * 100

    elapsed = time.time() - t0
    print(f"    Sharpe={sharpe:.2f}  Ret={ann_ret:.1%}  Vol={ann_vol:.1%}")
    print(f"    SHAP: Mom={mom_share:.0f}%  Pi={pi_share:.0f}%  Fund={fund_share:.0f}%")
    print(f"    Time: {elapsed:.0f}s")

    # Per-feature SHAP
    feat_shap = {features[i]: abs_shap[i] / total * 100 for i in range(len(features))}

    return {
        'name': name, 'n_features': len(features),
        'sharpe': round(sharpe, 3), 'ann_ret': round(ann_ret, 4),
        'ann_vol': round(ann_vol, 4),
        'mom_share': round(mom_share, 1), 'pi_share': round(pi_share, 1),
        'fund_share': round(fund_share, 1),
        **{f'shap_{k}': round(v, 2) for k, v in feat_shap.items()}
    }


# ── Run comparisons ──
print("\n" + "=" * 60)
print("  FUNDAMENTALS ABLATION TEST")
print("=" * 60)

results = []

# 1. Baseline: mom + pi (13 features) -- same as production
results.append(train_and_eval('baseline_mom_pi', FEATURES_BASE, train, test))

# 2. Full: mom + pi + fundamentals (20 features)
results.append(train_and_eval('full_mom_pi_fund', FEATURES_FUND, train, test))

# 3. Fundamentals only (no momentum, no pi) -- 7 features
results.append(train_and_eval('fund_only', FUND_FEATURES, train, test))

# 4. Fundamentals + pi (no momentum) -- 8 features
results.append(train_and_eval('fund_pi', FUND_FEATURES + ['pi_filter'], train, test))

# 5. Mom + fundamentals (no pi) -- 19 features
results.append(train_and_eval('mom_fund_no_pi', MOM_FEATURES + FUND_FEATURES, train, test))

df_results = pd.DataFrame(results)
df_results.to_csv('results/fundamentals_test_results.csv', index=False)

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"\n{'Config':<25s}  {'Feats':>5s}  {'Sharpe':>7s}  {'Ret':>7s}  {'Mom%':>5s}  {'Pi%':>5s}  {'Fund%':>5s}")
print("-" * 70)
for r in results:
    print(f"{r['name']:<25s}  {r['n_features']:>5d}  {r['sharpe']:>7.2f}  {r['ann_ret']:>6.1%}  "
          f"{r['mom_share']:>5.1f}  {r['pi_share']:>5.1f}  {r['fund_share']:>5.1f}")

print(f"\nSaved: results/fundamentals_test_results.csv")
