"""
random_forest_test.py
=====================
Standalone test: Random Forest as an alternative nonlinear model to XGBoost.

Same features as M2 (mom_1..mom_12, pi_filter), same train/test split,
same long-short portfolio construction. Tests whether the regime-momentum
interaction is specific to XGBoost or generalises to tree-based models
broadly.

Tests three RF configurations:
  1. Depth-matched: max_depth=4 (same as XGBoost)
  2. Classical RF: unrestricted depth, default RF hyperparameters
  3. Depth-matched with 100 trees (faster comparator)

Does NOT modify config.py, pipeline artefacts, or any production files.

Usage:
    python -u scripts/random_forest_test.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TRADING_FEE, TRAIN_END)

# ── Load data ──
print("Loading data ...")
stocks = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks['date'] = pd.to_datetime(stocks['date'])
stocks = stocks.sort_values(['permno', 'date']).reset_index(drop=True)
stocks = stocks[stocks['shrcd'].isin([10, 11])]
stocks = stocks[stocks['exchcd'].isin([1, 2, 3])]
stocks = stocks[stocks['prc'].abs() > 1.0].reset_index(drop=True)

regimes = pd.read_parquet('data/panel_with_regimes.parquet')[['date', 'pi_filter']]
regimes['date'] = pd.to_datetime(regimes['date'])
stocks = stocks.merge(regimes, on='date', how='left')
stocks['pi_filter'] = stocks['pi_filter'].ffill()

print("Computing momentum ...")
stocks['_lr'] = np.log1p(stocks['ret_adj'].clip(lower=-0.999))
stocks['_lr_s1'] = stocks.groupby('permno')['_lr'].shift(1)
for lb in range(1, 13):
    rs = (
        stocks.groupby('permno', sort=False)['_lr_s1']
        .rolling(lb, min_periods=lb).sum()
        .reset_index(level='permno', drop=True).sort_index()
    )
    stocks[f'mom_{lb}'] = np.expm1(rs)
stocks.drop(columns=['_lr', '_lr_s1'], inplace=True)
stocks['log_me'] = np.log(
    stocks.groupby('permno')['me'].transform(lambda x: x.shift(1)).replace(0, np.nan)
)
stocks['ret_fwd'] = stocks.groupby('permno')['ret_adj'].transform(lambda x: x.shift(-1))

MOM_FEATURES = [f'mom_{lb}' for lb in range(1, 13)]
FEATURES = MOM_FEATURES + ['pi_filter']

CORE = MOM_FEATURES + ['pi_filter', 'log_me']
df_all = stocks.dropna(subset=['ret_fwd'] + CORE).copy().reset_index(drop=True)
train = df_all[df_all['date'] < TRAIN_END].copy()
test = df_all[df_all['date'] >= TRAIN_END].copy()

X_tr = train[FEATURES].values.astype(float)
X_te = test[FEATURES].values.astype(float)
y_tr = train['ret_fwd'].values.astype(float)

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


def metrics(r):
    r = pd.Series(r).dropna()
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return ann_ret, ann_vol, sharpe, mdd


def train_rf_ensemble(config_name, n_estimators, max_depth, n_seeds, max_features=None):
    """Train ensemble of RF with given config. Returns portfolio Sharpe and SHAP shares."""
    print(f"\n  {config_name}: n_estimators={n_estimators}, max_depth={max_depth}, "
          f"seeds={n_seeds}, max_features={max_features}")
    t0 = time.time()
    seeds = list(range(1, n_seeds + 1))
    preds = np.zeros(len(X_te))
    # Keep the first 5 models for ensemble SHAP (full n_seeds SHAP would be too expensive)
    shap_models = []
    for i, xs in enumerate(seeds):
        kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=xs,
            n_jobs=-1,
        )
        if max_features is not None:
            kwargs['max_features'] = max_features
        rf = RandomForestRegressor(**kwargs)
        rf.fit(X_tr, y_tr)
        preds += rf.predict(X_te)
        if i < 5:
            shap_models.append(rf)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{n_seeds} seeds ({elapsed:.0f}s)", flush=True)
    preds /= n_seeds

    score_col = f'score_rf_{config_name.replace(" ", "_")}'
    test[score_col] = preds
    r = long_short_port(test, score_col)
    ann_ret, ann_vol, sharpe, mdd = metrics(r)

    # Ensemble SHAP over first 5 seeds, on a 30K sample of the test set
    print(f"    Computing SHAP on {len(shap_models)}-seed ensemble ...")
    sample_idx = np.random.RandomState(42).choice(len(X_te), size=min(30000, len(X_te)), replace=False)
    X_sample = X_te[sample_idx]
    sv_sum = np.zeros_like(X_sample, dtype=float)
    for m in shap_models:
        sv_sum += shap.TreeExplainer(m).shap_values(X_sample)
    sv = sv_sum / len(shap_models)
    abs_shap = np.abs(sv).mean(axis=0)
    total = abs_shap.sum()
    pi_idx = FEATURES.index('pi_filter')
    pi_share = abs_shap[pi_idx] / total * 100
    mom_share = sum(abs_shap[FEATURES.index(f)] for f in MOM_FEATURES) / total * 100

    elapsed = time.time() - t0
    print(f"    Sharpe={sharpe:.3f}  Ret={ann_ret:.1%}  Vol={ann_vol:.1%}  MDD={mdd:.1%}")
    print(f"    SHAP: Mom={mom_share:.0f}%  Pi={pi_share:.0f}%")
    print(f"    Total time: {elapsed/60:.1f} min")

    return {
        'config': config_name,
        'n_estimators': n_estimators,
        'max_depth': max_depth if max_depth else 'unrestricted',
        'n_seeds': n_seeds,
        'sharpe': round(sharpe, 3),
        'ann_ret': round(ann_ret, 4),
        'ann_vol': round(ann_vol, 4),
        'mdd': round(mdd, 4),
        'mom_share': round(mom_share, 1),
        'pi_share': round(pi_share, 1),
        'time_min': round(elapsed / 60, 1),
    }


# ── Run comparisons ──
print("\n" + "=" * 60)
print("  RANDOM FOREST ALTERNATIVE MODEL TEST")
print("=" * 60)

results = []

# 1. Depth-matched to XGBoost (direct comparison)
results.append(train_rf_ensemble(
    'depth4_50seeds',
    n_estimators=500,
    max_depth=4,
    n_seeds=50,
    max_features=0.8,
))

# 2. Classical RF: deeper trees, feature subsampling at each split (sqrt)
# Capped at depth 12 to keep runtime reasonable on 1.16M training rows.
results.append(train_rf_ensemble(
    'classical_sqrt',
    n_estimators=200,
    max_depth=12,
    n_seeds=20,
    max_features='sqrt',
))

# 3. Depth-matched quick version (fewer trees, fewer seeds, for a sanity check)
results.append(train_rf_ensemble(
    'depth4_100trees',
    n_estimators=100,
    max_depth=4,
    n_seeds=20,
    max_features=0.8,
))

df_results = pd.DataFrame(results)
df_results.to_csv('results/random_forest_results.csv', index=False)

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"\n{'Config':<25s}  {'Depth':>10s}  {'Sharpe':>7s}  {'Ret':>7s}  {'Mom%':>5s}  {'Pi%':>5s}")
print("-" * 70)
for r in results:
    print(f"{r['config']:<25s}  {str(r['max_depth']):>10s}  {r['sharpe']:>7.2f}  "
          f"{r['ann_ret']:>6.1%}  {r['mom_share']:>5.1f}  {r['pi_share']:>5.1f}")

print(f"\nBaseline XGBoost: Sharpe 1.11, Mom 54%, Pi 46%")
print(f"\nSaved: results/random_forest_results.csv")
