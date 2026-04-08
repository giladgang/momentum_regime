"""
test_risk_aversion_ls.py
========================
Risk-aversion analysis with LONG-SHORT portfolios.
Same as risk_aversion_analysis.py but with long-short instead of long-only.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import shap
import pickle, warnings
from scipy import stats
warnings.filterwarnings('ignore')

# ── 1. Load data ────────────────────────────────────────────────────────────

print("Loading data ...")
stocks = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks['date'] = pd.to_datetime(stocks['date'])
stocks = stocks.sort_values(['permno', 'date']).reset_index(drop=True)

stocks = stocks[stocks['shrcd'].isin([10, 11])]
stocks = stocks[stocks['exchcd'].isin([1, 2, 3])]
stocks = stocks[stocks['prc'].abs() > 1.0]
stocks = stocks.reset_index(drop=True)

regimes = pd.read_parquet('data/panel_with_regimes.parquet')[['date', 'vwretd', 'pi_filter']]
regimes['date'] = pd.to_datetime(regimes['date'])
stocks = stocks.merge(regimes[['date', 'pi_filter']], on='date', how='left')
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
stocks['trail_var'] = (
    stocks.groupby('permno')['ret_adj']
    .transform(lambda x: x.shift(1).rolling(12, min_periods=6).var())
)

MOM_FEATURES = [f'mom_{lb}' for lb in MOM_LBS]
FEATURES = MOM_FEATURES + [
    'pi_filter', 'bm', 'roe', 'earnings_growth',
    'leverage', 'asset_growth', 'gross_profit_a', 'log_me',
]

CORE_FEATURES = MOM_FEATURES + ['pi_filter', 'log_me']
df = stocks.dropna(subset=['ret_fwd', 'trail_var'] + CORE_FEATURES).copy()
df = df.reset_index(drop=True)

train = df[df['date'] < '2011-01-01'].copy()
test  = df[df['date'] >= '2011-01-01'].copy()

X_train = train[FEATURES].values.astype(float)
X_test  = test[FEATURES].values.astype(float)

# Market returns for CAPM
mkt = regimes[(regimes['date'] >= '2011-01-01') & regimes['vwretd'].notna()].copy()
r_mkt = mkt.set_index('date')['vwretd']

print(f"  Train: {len(train):,}  |  Test: {len(test):,}")

# ── 2. Portfolio helpers ────────────────────────────────────────────────────

TRADING_FEE = 0.001

def long_short_port(df_test, score_col, fee=TRADING_FEE):
    monthly = []
    prev_lw, prev_sw = {}, {}
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10: continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs  = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0: continue
        lme = longs['me'].sum()
        new_lw = (longs.set_index('permno')['me'] / lme).to_dict()
        r_long = (longs['ret_fwd'] * longs['me']).sum() / lme
        sme = shorts['me'].sum()
        new_sw = (shorts.set_index('permno')['me'] / sme).to_dict()
        r_short = (shorts['ret_fwd'] * shorts['me']).sum() / sme
        tl = sum(abs(new_lw.get(p,0) - prev_lw.get(p,0)) for p in set(new_lw)|set(prev_lw)) / 2
        ts = sum(abs(new_sw.get(p,0) - prev_sw.get(p,0)) for p in set(new_sw)|set(prev_sw)) / 2
        monthly.append({'date': date, 'ret': r_long - r_short - fee*(tl+ts)})
        prev_lw, prev_sw = new_lw, new_sw
    return pd.DataFrame(monthly).set_index('date')['ret']

def long_only_port(df_test, score_col, fee=TRADING_FEE):
    monthly = []
    prev_weights = {}
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10: continue
        hi = nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        if longs['me'].sum() == 0: continue
        total_me = longs['me'].sum()
        new_weights = (longs.set_index('permno')['me'] / total_me).to_dict()
        all_p = set(new_weights) | set(prev_weights)
        turnover = sum(abs(new_weights.get(p,0) - prev_weights.get(p,0)) for p in all_p) / 2
        r_gross = (longs['ret_fwd'] * longs['me']).sum() / total_me
        monthly.append({'date': date, 'ret': r_gross - fee * turnover})
        prev_weights = new_weights
    return pd.DataFrame(monthly).set_index('date')['ret']

def metrics(r):
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    return ann_ret, ann_vol, sharpe, mdd

def capm(r_port, r_mkt):
    common = r_port.index.intersection(r_mkt.index)
    rp, rm = r_port.loc[common].values, r_mkt.loc[common].values
    slope, intercept, _, _, _ = stats.linregress(rm, rp)
    resid = rp - (intercept + slope * rm)
    se = np.std(resid, ddof=2) / np.sqrt(len(rp))
    t = intercept / se
    p = 2 * (1 - stats.t.cdf(abs(t), len(rp)-2))
    return intercept * 12, t, p, slope

# ── 3. Sweep over gamma ────────────────────────────────────────────────────

GAMMAS = list(range(0, 11))

results_ls = []
results_lo = []
shap_by_gamma = {}

for gamma in GAMMAS:
    print(f"\n  gamma = {gamma} ...")

    y_train_g = train['ret_fwd'].values - (gamma / 2) * train['trail_var'].values

    model = XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method='hist', random_state=42, verbosity=0
    )
    model.fit(X_train, y_train_g)

    score_col = f'score_g{gamma}'
    test[score_col] = model.predict(X_test)

    # Long-short
    r_ls = long_short_port(test, score_col)
    ret_ls, vol_ls, sr_ls, mdd_ls = metrics(r_ls)
    alpha_ls, t_ls, p_ls, beta_ls = capm(r_ls, r_mkt)

    # Long-only
    r_lo = long_only_port(test, score_col)
    ret_lo, vol_lo, sr_lo, mdd_lo = metrics(r_lo)

    results_ls.append({
        'gamma': gamma, 'ann_ret': ret_ls, 'ann_vol': vol_ls,
        'sharpe': sr_ls, 'mdd': mdd_ls, 'alpha': alpha_ls,
        'beta': beta_ls, 't_alpha': t_ls, 'p_alpha': p_ls,
    })
    results_lo.append({
        'gamma': gamma, 'ann_ret': ret_lo, 'ann_vol': vol_lo,
        'sharpe': sr_lo, 'mdd': mdd_lo,
    })

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)
    mean_abs_shap = pd.Series(np.abs(shap_vals).mean(axis=0), index=FEATURES)
    mom_total = mean_abs_shap[MOM_FEATURES].sum()
    other = mean_abs_shap.drop(MOM_FEATURES)
    agg = pd.concat([pd.Series({'Momentum (agg)': mom_total}), other])
    agg = agg.sort_values(ascending=False)
    shap_by_gamma[gamma] = agg

    print(f"    L/S: Sharpe={sr_ls:.3f}  Alpha={alpha_ls:.1%} (t={t_ls:.2f})  Beta={beta_ls:.2f}")
    print(f"    L/O: Sharpe={sr_lo:.3f}  Ret={ret_lo:.1%}")

# ── 4. Results table ────────────────────────────────────────────────────────

df_ls = pd.DataFrame(results_ls)
df_lo = pd.DataFrame(results_lo)

print(f"\n{'='*110}")
print("RISK AVERSION: Long-Short vs Long-Only")
print(f"{'='*110}")
print(f"{'gamma':>5}  {'LS Sharpe':>9} {'LS Ret':>8} {'LS Vol':>8} {'LS MDD':>8} {'LS Alpha':>9} {'LS Beta':>8}  |  {'LO Sharpe':>9} {'LO Ret':>8}")
print(f"{'-'*105}")
for i in range(len(GAMMAS)):
    ls = df_ls.iloc[i]
    lo = df_lo.iloc[i]
    sig = '**' if ls['p_alpha'] < 0.05 else '  '
    print(f"{ls['gamma']:>5.0f}  {ls['sharpe']:>9.3f} {ls['ann_ret']:>7.1%} {ls['ann_vol']:>7.1%} "
          f"{ls['mdd']:>7.1%} {ls['alpha']:>7.1%}{sig} {ls['beta']:>8.2f}  |  "
          f"{lo['sharpe']:>9.3f} {lo['ann_ret']:>7.1%}")

# ── 5. SHAP importance table ───────────────────────────────────────────────

print(f"\n{'='*80}")
print("Pi_filter SHAP importance (% of total) vs gamma")
print(f"{'='*80}")

all_features_shap = shap_by_gamma[0].index.tolist()
shap_df = pd.DataFrame({g: shap_by_gamma[g].reindex(all_features_shap) for g in GAMMAS})
shap_norm = shap_df.div(shap_df.sum(axis=0), axis=1) * 100

print(f"{'gamma':>5}  {'Momentum':>10} {'pi_filter':>10} {'log_me':>8} {'bm':>8} {'roe':>8}")
print(f"{'-'*55}")
for g in GAMMAS:
    mom = shap_norm.loc['Momentum (agg)', g]
    pi  = shap_norm.loc['pi_filter', g]
    lme = shap_norm.loc['log_me', g]
    bm  = shap_norm.loc['bm', g]
    roe = shap_norm.loc['roe', g]
    print(f"{g:>5}  {mom:>9.1f}% {pi:>9.1f}% {lme:>7.1f}% {bm:>7.1f}% {roe:>7.1f}%")

print(f"\nDone.")
