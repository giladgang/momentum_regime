"""
raw_features_test.py
====================
Standalone test: XGBoost with mom_1..mom_12 + raw stress indicators (DD_z, DISP_z, REL_N_z, CS_z)
instead of pi_filter. Same hyperparameters, seeds, and portfolio construction as the main pipeline.

Does NOT overwrite any existing artefacts or tables.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (XGB_SEEDS, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE,
                    SUBSAMPLE, COLSAMPLE, MOM_FEATURES, TRADING_FEE,
                    PANEL_WITH_REGIMES_PATH, ARTEFACTS_PATH, PORTFOLIO_TYPE)
from xgboost import XGBRegressor

# ── Load artefacts from main pipeline ────────────────────────────────────────

print("Loading artefacts from main pipeline ...")
with open(ARTEFACTS_PATH, 'rb') as f:
    art = pickle.load(f)

train = art['train']
test  = art['test']
y_train = art['y_train']
r_mkt = art['r_mkt']

# ── Merge raw HMM features ──────────────────────────────────────────────────

print("Merging raw HMM features ...")
RAW_HMM = ['DD_z', 'DISP_z', 'REL_N_z', 'CS_z']
panel = pd.read_parquet(PANEL_WITH_REGIMES_PATH)[['date'] + RAW_HMM].dropna()
panel['date'] = pd.to_datetime(panel['date'])

train = train.merge(panel, on='date', how='left')
test  = test.merge(panel, on='date', how='left')

for col in RAW_HMM:
    med = train[col].median()
    train[col] = train[col].fillna(med)
    test[col]  = test[col].fillna(med)

FEATURES = MOM_FEATURES + RAW_HMM
print(f"  Features ({len(FEATURES)}): {FEATURES}")

X_train = train[FEATURES].values.astype(float)
X_test  = test[FEATURES].values.astype(float)

# ── Train 50-seed XGBoost ensemble ───────────────────────────────────────────

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
test['score_raw'] = preds

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

# ── Metrics (same as main_results_analysis.py) ──────────────────────────────

def metrics(r):
    r = pd.Series(r).dropna()
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe  = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum     = (1 + r).cumprod()
    mdd     = ((cum - cum.cummax()) / cum.cummax()).min()
    final   = cum.iloc[-1]
    return ann_ret, ann_vol, sharpe, mdd, final

def block_bootstrap_sharpe(r, n_boot=10000, block_len=12):
    """Block bootstrap 95% CI on annualized Sharpe ratio."""
    r = np.array(r)
    T = len(r)
    n_blocks = int(np.ceil(T / block_len))
    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        starts = np.random.randint(0, T - block_len + 1, size=n_blocks)
        sample = np.concatenate([r[s:s+block_len] for s in starts])[:T]
        std = sample.std()
        sharpes[b] = sample.mean() / std * np.sqrt(12) if std > 0 else 0
    lo, hi = np.percentile(sharpes, [2.5, 97.5])
    return lo, hi

def newey_west_t(r_strat, r_bench, maxlags=6):
    """Newey-West t-stat for mean excess return vs benchmark."""
    r_strat = pd.Series(r_strat)
    r_bench = pd.Series(r_bench)
    common = r_strat.index.intersection(r_bench.index)
    excess = r_strat.loc[common].values - r_bench.loc[common].values
    excess = np.asarray(excess, dtype=float)
    excess = excess[~np.isnan(excess)]
    if len(excess) < 12:
        return np.nan, np.nan
    X = np.ones((len(excess), 1))
    model = sm.OLS(excess, X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return model.tvalues[0], model.pvalues[0]

def sig_stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''

# ── Compute results ──────────────────────────────────────────────────────────

print("Building portfolio ...")
r = long_short_port(test, 'score_raw')

ar, av, sh, mdd, final = metrics(r)
ci_lo, ci_hi = block_bootstrap_sharpe(r.values)
nw_t, nw_p = newey_west_t(r, r_mkt)

print("\n" + "=" * 70)
print("  XGB with mom_1..12 + raw stress indicators (no HMM)")
print("=" * 70)
print(f"  Ann. Return:    {ar:.1%}")
print(f"  Ann. Vol:       {av:.1%}")
print(f"  Sharpe:         {sh:.2f}")
print(f"  95% CI:         [{ci_lo:.2f}, {ci_hi:.2f}]")
print(f"  Max DD:         {mdd:.1%}")
print(f"  NW t-stat:      {nw_t:.2f}{sig_stars(nw_p)}")
print(f"  Final $:        {final:.1f}")
print(f"  Months:         {len(r)}")
print("=" * 70)

# LaTeX row ready to paste into table_performance.tex
nw_str = f"{nw_t:.2f}{sig_stars(nw_p)}"
print(f"\nLaTeX row:")
print(f"M2: XGB (mom+raw) & {ar:.1%} & {av:.1%} & {sh:.2f} & [{ci_lo:.2f},\\,{ci_hi:.2f}] & {mdd:.1%} & {nw_str} & {final:.1f} \\\\")
