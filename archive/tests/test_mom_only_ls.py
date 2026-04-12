"""
test_mom_only_ls.py
===================
Isolate the regime contribution: train XGBoost with ONLY momentum features
+ pi_filter (no fundamentals), run long-short, compare to full M2 and D&M.
"""

import numpy as np
import pandas as pd
import pickle
from xgboost import XGBRegressor
from scipy import stats

# ── Load artifacts ───────────────────────────────────────────────────────────

print("Loading data ...")
with open('cs_artefacts_data.pkl', 'rb') as f:
    artefacts = pickle.load(f)

test_full  = artefacts['test'].copy()
train_full = artefacts['train'].copy()
FEATURES   = artefacts['FEATURES']

MOM_FEATURES = [f'mom_{lb}' for lb in range(1, 13)]

# Feature sets to test
feature_sets = {
    'Mom only':         MOM_FEATURES,
    'Mom + pi':         MOM_FEATURES + ['pi_filter'],
    'Mom + fund':       MOM_FEATURES + ['bm', 'roe', 'earnings_growth',
                                         'leverage', 'asset_growth',
                                         'gross_profit_a', 'log_me'],
    'Full (Mom+fund+pi)': FEATURES,
}

TRADING_FEE = 0.001

# ── Portfolio functions ──────────────────────────────────────────────────────

def long_short_port(df, score_col, fee=TRADING_FEE):
    monthly = []
    prev_lw, prev_sw = {}, {}
    for date, grp in df.groupby('date'):
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
        tl = sum(abs(new_lw.get(p, 0) - prev_lw.get(p, 0))
                 for p in set(new_lw) | set(prev_lw)) / 2
        ts = sum(abs(new_sw.get(p, 0) - prev_sw.get(p, 0))
                 for p in set(new_sw) | set(prev_sw)) / 2
        monthly.append({'date': date, 'ret': r_long - r_short - fee * (tl + ts)})
        prev_lw, prev_sw = new_lw, new_sw
    return pd.DataFrame(monthly).set_index('date')['ret']


def long_only_port(df, score_col, fee=TRADING_FEE):
    monthly = []
    prev_weights = {}
    for date, grp in df.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        hi = nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        if longs['me'].sum() == 0:
            continue
        total_me = longs['me'].sum()
        new_weights = (longs.set_index('permno')['me'] / total_me).to_dict()
        all_p = set(new_weights) | set(prev_weights)
        turnover = sum(abs(new_weights.get(p, 0) - prev_weights.get(p, 0))
                       for p in all_p) / 2
        r_gross = (longs['ret_fwd'] * longs['me']).sum() / total_me
        monthly.append({'date': date, 'ret': r_gross - fee * turnover})
        prev_weights = new_weights
    return pd.DataFrame(monthly).set_index('date')['ret']


def metrics(r):
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return {}
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe  = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum     = (1 + r).cumprod()
    mdd     = ((cum - cum.cummax()) / cum.cummax()).min()
    return {'Ann. Ret': ann_ret, 'Ann. Vol': ann_vol, 'Sharpe': sharpe, 'MDD': mdd}


def capm_alpha(r_port, r_mkt):
    common = r_port.index.intersection(r_mkt.index)
    rp = r_port.loc[common].values
    rm = r_mkt.loc[common].values
    slope, intercept, _, _, _ = stats.linregress(rm, rp)
    resid = rp - (intercept + slope * rm)
    se = np.std(resid, ddof=2) / np.sqrt(len(rp))
    t = intercept / se
    p = 2 * (1 - stats.t.cdf(abs(t), len(rp) - 2))
    return intercept * 12, t, p, slope


# ── Train and evaluate each feature set ──────────────────────────────────────

r_mkt = artefacts['r_mkt']
y_train = train_full['ret_fwd'].values.astype(float)

results = []

for name, feats in feature_sets.items():
    print(f"\nTraining XGBoost with: {name} ({len(feats)} features) ...")

    X_tr = train_full[feats].values.astype(float)
    X_te = test_full[feats].values.astype(float)

    xgb = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        tree_method='hist', random_state=42, verbosity=0)
    xgb.fit(X_tr, y_train)

    test_full[f'score_{name}'] = xgb.predict(X_te)

    # Long-short
    r_ls = long_short_port(test_full, f'score_{name}')
    m_ls = metrics(r_ls)
    alpha_a, t_a, p_a, beta = capm_alpha(r_ls, r_mkt)

    # Long-only
    r_lo = long_only_port(test_full, f'score_{name}')
    m_lo = metrics(r_lo)

    results.append({
        'Features': name,
        'N_feat': len(feats),
        'LS_Sharpe': m_ls['Sharpe'],
        'LS_Ret': m_ls['Ann. Ret'],
        'LS_Vol': m_ls['Ann. Vol'],
        'LS_MDD': m_ls['MDD'],
        'LS_Alpha': alpha_a,
        'LS_t_alpha': t_a,
        'LS_p_alpha': p_a,
        'LS_Beta': beta,
        'LO_Sharpe': m_lo['Sharpe'],
        'LO_Ret': m_lo['Ann. Ret'],
    })

    print(f"  L/S: Sharpe={m_ls['Sharpe']:.3f}  Alpha={alpha_a:.1%} (t={t_a:.2f})  Beta={beta:.2f}")
    print(f"  L/O: Sharpe={m_lo['Sharpe']:.3f}  Ret={m_lo['Ann. Ret']:.1%}")

# ── Comparison table ─────────────────────────────────────────────────────────

print(f"\n{'='*100}")
print("DECOMPOSING THE CONTRIBUTION: Momentum vs Fundamentals vs Regime Signal")
print(f"{'='*100}")
print(f"\n--- LONG-SHORT ---")
print(f"{'Features':<22} {'#':>3} {'Sharpe':>8} {'Ann.Ret':>9} {'Ann.Vol':>9} {'MDD':>8} {'Alpha':>8} {'t(a)':>7} {'Beta':>7}")
print(f"{'-'*95}")

for r in results:
    sig = '**' if r['LS_p_alpha'] < 0.05 else '  '
    print(f"{r['Features']:<22} {r['N_feat']:>3} {r['LS_Sharpe']:>8.3f} {r['LS_Ret']:>8.1%} "
          f"{r['LS_Vol']:>8.1%} {r['LS_MDD']:>7.1%} {r['LS_Alpha']:>7.1%}{sig} {r['LS_t_alpha']:>6.2f} "
          f"{r['LS_Beta']:>7.2f}")

# Add D&M for reference
print(f"{'-'*95}")
print(f"{'D&M (Bear scaling)':<22} {'--':>3} {'0.531':>8} {'9.6%':>9} {'21.9%':>9} {'-39.2%':>8} "
      f"{'--':>8} {'--':>7} {'--':>7}   (scales WML, different approach)")

print(f"\n--- LONG-ONLY ---")
print(f"{'Features':<22} {'#':>3} {'Sharpe':>8} {'Ann.Ret':>9}")
print(f"{'-'*50}")
for r in results:
    print(f"{r['Features']:<22} {r['N_feat']:>3} {r['LO_Sharpe']:>8.3f} {r['LO_Ret']:>8.1%}")

print(f"\n{'='*100}")
print("KEY QUESTION: How much of M2's L/S advantage over D&M comes from pi_filter vs fundamentals?")
print(f"{'='*100}")

mom_only = [r for r in results if r['Features'] == 'Mom only'][0]
mom_pi   = [r for r in results if r['Features'] == 'Mom + pi'][0]
mom_fund = [r for r in results if r['Features'] == 'Mom + fund'][0]
full     = [r for r in results if r['Features'] == 'Full (Mom+fund+pi)'][0]

print(f"\n  Baseline (Mom only):            L/S Sharpe = {mom_only['LS_Sharpe']:.3f}")
print(f"  + pi_filter:                    L/S Sharpe = {mom_pi['LS_Sharpe']:.3f}  (delta = {mom_pi['LS_Sharpe']-mom_only['LS_Sharpe']:+.3f})")
print(f"  + fundamentals:                 L/S Sharpe = {mom_fund['LS_Sharpe']:.3f}  (delta = {mom_fund['LS_Sharpe']-mom_only['LS_Sharpe']:+.3f})")
print(f"  + both (full M2):               L/S Sharpe = {full['LS_Sharpe']:.3f}  (delta = {full['LS_Sharpe']-mom_only['LS_Sharpe']:+.3f})")
print(f"\n  D&M managed WML:                L/S Sharpe = 0.531")
