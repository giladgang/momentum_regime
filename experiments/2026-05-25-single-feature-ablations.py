"""
2026-05-25-single-feature-ablations.py
=======================================
Test XGB with each of the 4 HMM input features SINGLY as the regime signal
(replacing pi_filter): DD_z, DISP_z, REL_N_z, CS_z. Same hyperparameters,
seeds, and portfolio construction as the main pipeline.

Extends 2026-05-25-dd-only-xgb-ablation.py to cover all four features.
"""

import numpy as np
import pandas as pd
import pickle, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (XGB_SEEDS, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE,
                    SUBSAMPLE, COLSAMPLE, MOM_FEATURES, TRADING_FEE,
                    PANEL_WITH_REGIMES_PATH, ARTEFACTS_PATH)
from xgboost import XGBRegressor


print("Loading artefacts ...")
with open(ARTEFACTS_PATH, 'rb') as f:
    art = pickle.load(f)
train_base = art['train']
test_base = art['test']
y_train = art['y_train']

panel = pd.read_parquet(PANEL_WITH_REGIMES_PATH)[
    ['date', 'DD_z', 'DISP_z', 'REL_N_z', 'CS_z']].dropna()
panel['date'] = pd.to_datetime(panel['date'])


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
    return ann_ret, ann_vol, sharpe, mdd


def train_and_score(feature_name):
    train = train_base.copy().merge(
        panel[['date', feature_name]], on='date', how='left')
    test = test_base.copy().merge(
        panel[['date', feature_name]], on='date', how='left')
    med = train[feature_name].median()
    train[feature_name] = train[feature_name].fillna(med)
    test[feature_name] = test[feature_name].fillna(med)
    FEATURES = MOM_FEATURES + [feature_name]
    X_train = train[FEATURES].values.astype(float)
    X_test = test[FEATURES].values.astype(float)
    preds = np.zeros(len(X_test))
    for seed in XGB_SEEDS:
        xgb = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                           learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                           colsample_bytree=COLSAMPLE, tree_method='hist',
                           random_state=seed, verbosity=0)
        xgb.fit(X_train, y_train)
        preds += xgb.predict(X_test)
    preds /= len(XGB_SEEDS)
    test = test.copy()
    test['score'] = preds
    return long_short_port(test, 'score')


results = {}
for feat in ['DD_z', 'DISP_z', 'REL_N_z', 'CS_z']:
    print(f"\nTraining XGB + {feat} only ({len(XGB_SEEDS)} seeds) ...")
    r = train_and_score(feat)
    ar, av, sh, mdd = metrics(r)
    results[feat] = {'ann_ret': ar, 'ann_vol': av, 'sharpe': sh, 'mdd': mdd, 'n': len(r)}
    print(f"  Sharpe = {sh:.3f}  Ann.Ret = {ar:.2%}  Max DD = {mdd:.1%}")

print("\n" + "=" * 78)
print("  Single-feature XGB ablations (mom_1..12 + one raw stress indicator)")
print("=" * 78)
print(f"  {'Feature':<10} {'Ann.Ret':>10} {'Ann.Vol':>10} {'Sharpe':>10} {'Max DD':>10}")
print("  " + "-" * 60)
for feat, m in results.items():
    print(f"  {feat:<10} {m['ann_ret']:>10.2%} {m['ann_vol']:>10.2%} "
          f"{m['sharpe']:>10.3f} {m['mdd']:>10.1%}")

print("\nFor comparison (from Table 5):")
print(f"  XGB + pi_filter (HMM):     Sharpe 1.107")
print(f"  XGB + 4 raw indicators:    Sharpe 0.257")
print(f"  XGB (no regime signal):    Sharpe 0.429")
