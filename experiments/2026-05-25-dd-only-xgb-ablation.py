"""
2026-05-25-dd-only-xgb-ablation.py
==================================
Tests XGBoost with mom_1..mom_12 + DD_z (single raw stress indicator) instead of
the HMM-distilled pi_filter. Same hyperparameters, seeds, and portfolio
construction as the main pipeline.

Closes a loophole in the regime-signal ablation: the existing Table 5
compares HMM-distilled pi (Sharpe 1.11) vs all 4 raw indicators (0.26) vs
no signal (0.43). Critics could argue "the 4 raw indicators overfit; would
1 well-chosen raw indicator (DD) do as well as the HMM?". This script
answers that.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (XGB_SEEDS, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE,
                    SUBSAMPLE, COLSAMPLE, MOM_FEATURES, TRADING_FEE,
                    PANEL_WITH_REGIMES_PATH, ARTEFACTS_PATH)
from xgboost import XGBRegressor


print("Loading artefacts ...")
with open(ARTEFACTS_PATH, 'rb') as f:
    art = pickle.load(f)
train = art['train']
test = art['test']
y_train = art['y_train']

print("Merging DD_z (single raw stress indicator) ...")
panel = pd.read_parquet(PANEL_WITH_REGIMES_PATH)[['date', 'DD_z']].dropna()
panel['date'] = pd.to_datetime(panel['date'])
train = train.merge(panel, on='date', how='left')
test = test.merge(panel, on='date', how='left')
med = train['DD_z'].median()
train['DD_z'] = train['DD_z'].fillna(med)
test['DD_z'] = test['DD_z'].fillna(med)

FEATURES = MOM_FEATURES + ['DD_z']
print(f"  Features ({len(FEATURES)}): {FEATURES}")

X_train = train[FEATURES].values.astype(float)
X_test = test[FEATURES].values.astype(float)

print(f"Training XGBoost ensemble ({len(XGB_SEEDS)} seeds) ...")
preds = np.zeros(len(X_test))
for i, seed in enumerate(XGB_SEEDS):
    xgb = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                       learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                       colsample_bytree=COLSAMPLE, tree_method='hist',
                       random_state=seed, verbosity=0)
    xgb.fit(X_train, y_train)
    preds += xgb.predict(X_test)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(XGB_SEEDS)} seeds done")
preds /= len(XGB_SEEDS)

test = test.copy()
test['score_dd'] = preds


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
    return pd.DataFrame(monthly).set_index('date')['ret']


def metrics(r):
    r = pd.Series(r).dropna()
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return ann_ret, ann_vol, sharpe, mdd, cum.iloc[-1]


def nw_t(r, maxlags=6):
    vals = pd.Series(r).dropna().values.astype(float)
    X = np.ones((len(vals), 1))
    model = sm.OLS(vals, X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return model.tvalues[0], model.pvalues[0]


print("Building portfolio ...")
r = long_short_port(test, 'score_dd')
ar, av, sh, mdd, final = metrics(r)
t, p = nw_t(r)

print("\n" + "=" * 70)
print("  XGB with mom_1..12 + DD_z only (no HMM)")
print("=" * 70)
print(f"  Ann. Return:    {ar:.2%}")
print(f"  Ann. Vol:       {av:.2%}")
print(f"  Sharpe:         {sh:.3f}")
print(f"  Max DD:         {mdd:.1%}")
print(f"  NW t-stat:      {t:.2f}")
print(f"  Months:         {len(r)}")
print("=" * 70)

print("\nFor comparison (from Table 5):")
print(f"  XGB + pi_filter (HMM):     Sharpe 1.107")
print(f"  XGB + 4 raw indicators:    Sharpe 0.257")
print(f"  XGB (no regime signal):    Sharpe 0.429")
print(f"  XGB + DD_z only (THIS):    Sharpe {sh:.3f}")
