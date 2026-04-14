"""
ridge_baseline_test.py
======================
Test whether the M1/M2 divergence is driven by linearity vs nonlinearity,
or by classification vs regression targets.

Adds a Ridge regression (M1b) that predicts continuous forward returns
(same target as XGBoost M2) using a linear model (same functional form as M1).
This isolates the linearity/nonlinearity question from the target question.

Uses saved artefacts from cross_sectional_model.py to avoid re-running
the full pipeline.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_FEE, MOM_FEATURES

# ── Load saved artefacts ─────────────────────────────────────────────────────

print("Loading artefacts ...")
with open('cs_artefacts_data.pkl', 'rb') as f:
    artefacts = pickle.load(f)

test     = artefacts['test'].copy()
train    = artefacts['train'].copy()
X_train  = artefacts['X_train']
X_test   = artefacts['X_test']
y_train  = artefacts['y_train']
FEATURES = artefacts['FEATURES']

print(f"  Train: {len(train):,} rows  |  Test: {len(test):,} rows")
print(f"  Features: {FEATURES}")

# ── Portfolio construction (copied from cross_sectional_model.py) ────────────

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

def metrics(r):
    r = pd.Series(r).dropna()
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe  = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum     = (1 + r).cumprod()
    mdd     = ((cum - cum.cummax()) / cum.cummax()).min()
    return ann_ret, ann_vol, sharpe, mdd

# ── Preprocessing (same as M1: impute + scale) ──────────────────────────────

imputer = SimpleImputer(strategy='median')
scaler  = StandardScaler()
X_tr_s  = scaler.fit_transform(imputer.fit_transform(X_train))
X_te_s  = scaler.transform(imputer.transform(X_test))

# ── Ridge Regression: predicting continuous returns (same target as M2) ──────

print("\n=== Ridge Regression (continuous return target, same as M2) ===")

alphas_to_test = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

for alpha in alphas_to_test:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_tr_s, y_train)
    test['score_ridge'] = ridge.predict(X_te_s)

    r_ridge = long_short_port(test, 'score_ridge')
    ar, av, sh, mdd = metrics(r_ridge)

    # Regime-conditional
    test_dates = test[['date', 'pi_filter']].drop_duplicates('date').set_index('date')
    pi = test_dates.reindex(r_ridge.index)['pi_filter']
    calm  = r_ridge[pi < 0.5]
    panic = r_ridge[pi >= 0.5]
    sr_calm  = calm.mean() / calm.std() * np.sqrt(12) if len(calm) > 1 and calm.std() > 0 else float('nan')
    sr_panic = panic.mean() / panic.std() * np.sqrt(12) if len(panic) > 1 and panic.std() > 0 else float('nan')

    print(f"  alpha={alpha:>7.1f}  |  Sharpe={sh:>6.2f}  Ann.Ret={ar:>7.1%}  "
          f"Ann.Vol={av:>7.1%}  MDD={mdd:>7.1%}  |  Calm={sr_calm:>6.2f}  Panic={sr_panic:>6.2f}")

    # Print coefficients for the best alpha
    if alpha == 1.0:
        print(f"\n  Ridge (alpha=1.0) coefficients:")
        coefs = pd.Series(ridge.coef_, index=FEATURES).sort_values(ascending=False)
        for feat, c in coefs.items():
            print(f"    {feat:<15s} {c:>+8.5f}")

# ── OLS (no regularization) for comparison ───────────────────────────────────

print("\n=== OLS (no regularization) ===")
from sklearn.linear_model import LinearRegression
ols = LinearRegression()
ols.fit(X_tr_s, y_train)
test['score_ols'] = ols.predict(X_te_s)
r_ols = long_short_port(test, 'score_ols')
ar, av, sh, mdd = metrics(r_ols)

pi = test_dates.reindex(r_ols.index)['pi_filter']
calm  = r_ols[pi < 0.5]
panic = r_ols[pi >= 0.5]
sr_calm  = calm.mean() / calm.std() * np.sqrt(12) if len(calm) > 1 and calm.std() > 0 else float('nan')
sr_panic = panic.mean() / panic.std() * np.sqrt(12) if len(panic) > 1 and panic.std() > 0 else float('nan')

print(f"  OLS       |  Sharpe={sh:>6.2f}  Ann.Ret={ar:>7.1%}  "
      f"Ann.Vol={av:>7.1%}  MDD={mdd:>7.1%}  |  Calm={sr_calm:>6.2f}  Panic={sr_panic:>6.2f}")

print(f"\n  OLS coefficients:")
coefs = pd.Series(ols.coef_, index=FEATURES).sort_values(ascending=False)
for feat, c in coefs.items():
    print(f"    {feat:<15s} {c:>+8.5f}")

# ── Recap: M1 (LR classification) and M2 (XGBoost) for comparison ───────────

print("\n=== Comparison with existing models ===")

strats = artefacts['strategies_lo']
for name in ['Method 1: LR', 'Method 2: XGB']:
    if name in strats:
        r = strats[name]
        ar, av, sh, mdd = metrics(r)
        pi = test_dates.reindex(r.index)['pi_filter']
        calm  = r[pi < 0.5]
        panic = r[pi >= 0.5]
        sr_calm  = calm.mean() / calm.std() * np.sqrt(12) if len(calm) > 1 and calm.std() > 0 else float('nan')
        sr_panic = panic.mean() / panic.std() * np.sqrt(12) if len(panic) > 1 and panic.std() > 0 else float('nan')
        print(f"  {name:<26s}  |  Sharpe={sh:>6.2f}  Ann.Ret={ar:>7.1%}  "
              f"Ann.Vol={av:>7.1%}  MDD={mdd:>7.1%}  |  Calm={sr_calm:>6.2f}  Panic={sr_panic:>6.2f}")

print("\nDone.")
