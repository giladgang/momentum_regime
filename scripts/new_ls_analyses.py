"""
new_ls_analyses.py
==================
New analyses specific to the long-short portfolio:
1. D&M comparison (same data, same costs, stock selection vs exposure scaling)
2. Factor model alphas (CAPM through FF5+Mom)
3. M1 polynomial/interaction tests
4. IC rotation table: IC by horizon (1-12) in calm vs panic with t-stats
5. Spanning test: M2 on M1 (and reverse)
6. Sub-period IC stability
"""

import numpy as np
import pandas as pd
import pickle, warnings, os, sys
import statsmodels.api as sm
from scipy.stats import spearmanr, ttest_1samp
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TRADING_FEE, TRAIN_END, TABLES_DIR, CS_FEATURES, MOM_FEATURES,
                    N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, COLSAMPLE,
                    XGB_SEEDS, SUB_PERIODS)

os.makedirs(TABLES_DIR, exist_ok=True)

print("=" * 70)
print("  NEW LONG-SHORT ANALYSES")
print("=" * 70)

# ── Load data ──
with open('cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
train = art['train'].copy()
r_mkt = art['r_mkt']

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])

ff = pd.read_parquet('data/ff_factors.parquet')

FEE = TRADING_FEE
REDUCED = MOM_FEATURES + ['pi_filter']

def long_short_port(df_test, score_col, fee=FEE):
    monthly, prev_lw, prev_sw = [], {}, {}
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10: continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0: continue
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
    if not monthly: return pd.Series(dtype=float)
    return pd.DataFrame(monthly).set_index('date')['ret']

# Use production 50-seed ensemble scores from artefacts (no retraining)
print("\n[ 1 ] Using production XGB scores from artefacts ...")
X_tr = train[REDUCED].values.astype(float)
X_te = test[REDUCED].values.astype(float)
y_tr = train['ret_fwd'].values.astype(float)
test['score_m2'] = test['score_xgb']
r_m2 = long_short_port(test, 'score_m2')

# M1
imp = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(imp.fit_transform(X_tr))
X_te_s = scaler.transform(imp.transform(X_te))
train_c = train.copy()
train_c['above_med'] = train_c.groupby('date')['ret_fwd'].transform(lambda x: (x > x.median()).astype(int))
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_tr_s, train_c['above_med'].values)
test['score_m1'] = lr.predict_proba(X_te_s)[:, 1]
r_m1 = long_short_port(test, 'score_m1')

print(f"  M2 Sharpe: {r_m2.mean()/r_m2.std()*np.sqrt(12):.3f}")
print(f"  M1 Sharpe: {r_m1.mean()/r_m1.std()*np.sqrt(12):.3f}")

# ═══════════════════════════════════════════════════════════════════
# 1. D&M COMPARISON
# ═══════════════════════════════════════════════════════════════════

print("\n[ 2 ] D&M Comparison ...")

# Build WML
stocks_raw = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks_raw['date'] = pd.to_datetime(stocks_raw['date'])
stocks_raw = stocks_raw[stocks_raw['shrcd'].isin([10, 11]) & stocks_raw['exchcd'].isin([1, 2, 3]) & (stocks_raw['prc'].abs() > 1)]
stocks_raw = stocks_raw.sort_values(['permno', 'date']).reset_index(drop=True)
stocks_raw['_lr'] = np.log1p(stocks_raw['ret_adj'].clip(lower=-0.999))
stocks_raw['_lr_s2'] = stocks_raw.groupby('permno')['_lr'].shift(2)
roll = stocks_raw.groupby('permno', sort=False)['_lr_s2'].rolling(11, min_periods=11).sum().reset_index(level='permno', drop=True).sort_index()
stocks_raw['mom_12_2'] = np.expm1(roll)
stocks_raw['ret_fwd'] = stocks_raw.groupby('permno')['ret_adj'].transform(lambda x: x.shift(-1))
df_wml = stocks_raw.dropna(subset=['mom_12_2', 'ret_fwd']).copy()

prev_lw, prev_sw = {}, {}
wml_monthly = []
for date, grp in df_wml.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['mom_12_2'].dropna()
    if len(nyse) < 10: continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    longs = grp[grp['mom_12_2'] >= hi]; shorts = grp[grp['mom_12_2'] <= lo]
    if longs['me'].sum() == 0 or shorts['me'].sum() == 0: continue
    lme = longs['me'].sum(); new_lw = (longs.set_index('permno')['me'] / lme).to_dict()
    r_long = (longs['ret_fwd'] * longs['me']).sum() / lme
    sme = shorts['me'].sum(); new_sw = (shorts.set_index('permno')['me'] / sme).to_dict()
    r_short = (shorts['ret_fwd'] * shorts['me']).sum() / sme
    tl = sum(abs(new_lw.get(p, 0) - prev_lw.get(p, 0)) for p in set(new_lw) | set(prev_lw)) / 2
    ts = sum(abs(new_sw.get(p, 0) - prev_sw.get(p, 0)) for p in set(new_sw) | set(prev_sw)) / 2
    wml_monthly.append({'date': date, 'r_wml': r_long - r_short, 'turnover': tl + ts})
    prev_lw, prev_sw = new_lw, new_sw

wml_all = pd.DataFrame(wml_monthly).set_index('date')
mkt_ts = panel[['date', 'vwretd']].dropna().drop_duplicates('date').sort_values('date').set_index('date')['vwretd']
mkt_cum_24 = mkt_ts.rolling(24).apply(lambda x: (1 + x).prod() - 1, raw=True)
bear = (mkt_cum_24 < 0).astype(int).rename('bear')
wml_all = wml_all.join(bear, how='left')
wml_all['bear'] = wml_all['bear'].fillna(0)
wml_all['sigma2'] = wml_all['r_wml'].rolling(6, min_periods=6).var()

train_wml = wml_all[wml_all.index < TRAIN_END]
test_wml = wml_all[wml_all.index >= TRAIN_END].copy()

# D&M managed
X_bear_tr = sm.add_constant(train_wml['bear'])
ols = sm.OLS(train_wml['r_wml'], X_bear_tr).fit()
X_bear_te = sm.add_constant(test_wml['bear'])
mu = ols.predict(X_bear_te)
sigma2 = wml_all.loc[test_wml.index, 'sigma2']
w_dm = (0.5 * mu / sigma2).clip(-0.5, 1.5)
r_dm = pd.Series(w_dm.values * (test_wml['r_wml'].values - FEE * test_wml['turnover'].values), index=test_wml.index)
r_wml_net = test_wml['r_wml'] - FEE * test_wml['turnover']

print(f"  D&M Sharpe: {r_dm.mean()/r_dm.std()*np.sqrt(12):.3f}")
print(f"  WML Sharpe: {r_wml_net.mean()/r_wml_net.std()*np.sqrt(12):.3f}")

# ═══════════════════════════════════════════════════════════════════
# 2. FACTOR MODEL ALPHAS
# ═══════════════════════════════════════════════════════════════════

print("\n[ 3 ] Factor Model Alphas ...")

def get_alphas(r, name):
    r_df = r.to_frame('ret')
    r_df.index = r_df.index + pd.offsets.MonthEnd(0)
    merged = r_df.join(ff[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'UMD']], how='inner')
    y = merged['ret'].values

    results = {}
    all_factors = ['Mkt-RF', 'SMB', 'HML', 'UMD', 'RMW', 'CMA']
    for model_name, cols in [
        ('CAPM', ['Mkt-RF']),
        ('FF3', ['Mkt-RF', 'SMB', 'HML']),
        ('Carhart', ['Mkt-RF', 'SMB', 'HML', 'UMD']),
        ('FF5', ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']),
        ('FF6', ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD']),
    ]:
        X = sm.add_constant(merged[cols].values)
        res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
        betas = {}
        for j, c in enumerate(cols):
            betas[c] = res.params[j + 1]
        results[model_name] = {'alpha': res.params[0] * 12, 't': res.tvalues[0],
                               'p': res.pvalues[0], 'betas': betas, 'cols': cols}

    print(f"\n  {name}:")
    for m, r_dict in results.items():
        stars = '***' if r_dict['p'] < 0.01 else '**' if r_dict['p'] < 0.05 else '*' if r_dict['p'] < 0.10 else ''
        print(f"    {m:<10s}: alpha={r_dict['alpha']:>6.1%}  t={r_dict['t']:>5.2f}{stars}")
    return results

alpha_m2 = get_alphas(r_m2, 'M2 (mom+pi)')
alpha_dm = get_alphas(r_dm, 'D&M managed')
alpha_wml = get_alphas(r_wml_net, 'Unscaled WML')
alpha_m1 = get_alphas(r_m1, 'M1 (LR)')

# Export factor alpha table for M2
all_factors = ['Mkt-RF', 'SMB', 'HML', 'UMD', 'RMW', 'CMA']
tex = []
tex.append(r'\begin{table}[H]')
tex.append(r'\centering')
tex.append(r'\small')
tex.append(r'\begin{tabular}{l r r r r r r r r}')
tex.append(r'\toprule')
tex.append(r'Model & $\alpha$ (\%) & $t(\alpha)$ & Mkt-RF & SMB & HML & UMD & RMW & CMA \\')
tex.append(r'\midrule')
for mname, mdata in alpha_m2.items():
    alpha_ann = mdata['alpha'] * 100
    t_alpha = mdata['t']
    cells = [mname, f'{alpha_ann:.1f}', f'{t_alpha:.2f}']
    for f in all_factors:
        if f in mdata['betas']:
            cells.append(f'{mdata["betas"][f]:+.2f}')
        else:
            cells.append('')
    tex.append(' & '.join(cells) + r' \\')
tex.append(r'\bottomrule')
tex.append(r'\end{tabular}')
tex.append(r"\caption{Factor model regressions for M2 (XGBoost, mom+$\pi$). $\alpha$ is annualised. $t$-statistics use Newey--West standard errors (6 lags).}")
tex.append(r'\label{tab:factor_alphas}')
tex.append(r'\end{table}')

with open(os.path.join('tables', 'table_factor_alphas.tex'), 'w') as f:
    f.write('\n'.join(tex) + '\n')
print("  Saved: tables/table_factor_alphas.tex")

# ═══════════════════════════════════════════════════════════════════
# 3. M1 POLYNOMIAL/INTERACTION TESTS
# ═══════════════════════════════════════════════════════════════════

print("\n[ 4 ] M1 Polynomial/Interaction Tests ...")

pi_idx = REDUCED.index('pi_filter')

configs = [
    ('Baseline LR', None),
    ('+ pi^2,pi^3,pi^4', 'poly'),
    ('+ pi*mom_k interactions', 'interact'),
    ('+ pi^2 + interactions', 'both'),
]

print(f"\n  {'Config':<35s} {'Sharpe':>7s}")
print(f"  {'-'*44}")

for name, config in configs:
    if config is None:
        X_tr_aug, X_te_aug = X_tr_s, X_te_s
    elif config == 'poly':
        pi_tr = X_tr_s[:, pi_idx:pi_idx + 1]
        pi_te = X_te_s[:, pi_idx:pi_idx + 1]
        X_tr_aug = np.hstack([X_tr_s, pi_tr ** 2, pi_tr ** 3, pi_tr ** 4])
        X_te_aug = np.hstack([X_te_s, pi_te ** 2, pi_te ** 3, pi_te ** 4])
    elif config == 'interact':
        pi_tr = X_tr_s[:, pi_idx:pi_idx + 1]
        pi_te = X_te_s[:, pi_idx:pi_idx + 1]
        X_tr_aug = np.hstack([X_tr_s, X_tr_s[:, :12] * pi_tr])
        X_te_aug = np.hstack([X_te_s, X_te_s[:, :12] * pi_te])
    elif config == 'both':
        pi_tr = X_tr_s[:, pi_idx:pi_idx + 1]
        pi_te = X_te_s[:, pi_idx:pi_idx + 1]
        X_tr_aug = np.hstack([X_tr_s, pi_tr ** 2, X_tr_s[:, :12] * pi_tr])
        X_te_aug = np.hstack([X_te_s, pi_te ** 2, X_te_s[:, :12] * pi_te])

    lr_aug = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    lr_aug.fit(X_tr_aug, train_c['above_med'].values)
    test[f'score_{name}'] = lr_aug.predict_proba(X_te_aug)[:, 1]
    r = long_short_port(test, f'score_{name}')
    sh = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    print(f"  {name:<35s} {sh:>7.3f}")

# ═══════════════════════════════════════════════════════════════════
# 4. IC ROTATION TABLE
# ═══════════════════════════════════════════════════════════════════

print("\n[ 5 ] IC Rotation by Horizon ...")

pi_monthly = test.groupby('date')['pi_filter'].first()

print(f"\n  {'Horizon':<10s} {'IC All':>8s} {'IC Calm':>8s} {'IC Panic':>8s} {'t(All)':>7s} {'t(Calm)':>8s} {'t(Panic)':>9s}")
print(f"  {'-'*62}")

for lb in range(1, 13):
    col = f'mom_{lb}'
    ics_all, ics_calm, ics_panic = [], [], []
    for date, grp in test.groupby('date'):
        valid = grp[[col, 'ret_fwd']].dropna()
        if len(valid) < 30: continue
        rho, _ = spearmanr(valid[col], valid['ret_fwd'])
        pi = pi_monthly.get(date, 0.5)
        ics_all.append(rho)
        if pi < 0.5:
            ics_calm.append(rho)
        else:
            ics_panic.append(rho)

    mean_all = np.mean(ics_all) if ics_all else 0
    mean_calm = np.mean(ics_calm) if ics_calm else 0
    mean_panic = np.mean(ics_panic) if ics_panic else 0
    t_all = ttest_1samp(ics_all, 0)[0] if len(ics_all) > 1 else 0
    t_calm = ttest_1samp(ics_calm, 0)[0] if len(ics_calm) > 1 else 0
    t_panic = ttest_1samp(ics_panic, 0)[0] if len(ics_panic) > 1 else 0

    print(f"  mom_{lb:<5d} {mean_all:>8.4f} {mean_calm:>8.4f} {mean_panic:>8.4f} "
          f"{t_all:>7.2f} {t_calm:>8.2f} {t_panic:>9.2f}")

# ═══════════════════════════════════════════════════════════════════
# 5. SPANNING TEST: M2 on M1 (and reverse)
# ═══════════════════════════════════════════════════════════════════

print("\n[ 6 ] Spanning Tests ...")

common = r_m2.index.intersection(r_m1.index)
r_m2_c = r_m2.loc[common].values
r_m1_c = r_m1.loc[common].values

# M2 on M1
X_span = sm.add_constant(r_m1_c)
res_m2_on_m1 = sm.OLS(r_m2_c, X_span).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
print(f"\n  M2 regressed on M1:")
print(f"    Alpha: {res_m2_on_m1.params[0] * 12:.1%} (t={res_m2_on_m1.tvalues[0]:.2f}, p={res_m2_on_m1.pvalues[0]:.4f})")
print(f"    Beta on M1: {res_m2_on_m1.params[1]:.3f}")

# M1 on M2
X_span2 = sm.add_constant(r_m2_c)
res_m1_on_m2 = sm.OLS(r_m1_c, X_span2).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
print(f"\n  M1 regressed on M2:")
print(f"    Alpha: {res_m1_on_m2.params[0] * 12:.1%} (t={res_m1_on_m2.tvalues[0]:.2f}, p={res_m1_on_m2.pvalues[0]:.4f})")
print(f"    Beta on M2: {res_m1_on_m2.params[1]:.3f}")

# ═══════════════════════════════════════════════════════════════════
# 6. SUB-PERIOD IC STABILITY
# ═══════════════════════════════════════════════════════════════════

print("\n[ 7 ] Sub-period IC Stability ...")

for pname, pstart, pend in SUB_PERIODS:
    if pname == 'Full': continue
    sub_test = test[(test['date'] >= pstart) & (test['date'] < pend)]
    if len(sub_test) < 1000: continue

    pi_sub = sub_test.groupby('date')['pi_filter'].first()

    print(f"\n  {pname}:")
    print(f"  {'Horizon':<10s} {'IC Calm':>8s} {'IC Panic':>8s}")
    print(f"  {'-'*30}")

    for lb in [1, 6, 8, 11, 12]:
        col = f'mom_{lb}'
        ics_calm, ics_panic = [], []
        for date, grp in sub_test.groupby('date'):
            valid = grp[[col, 'ret_fwd']].dropna()
            if len(valid) < 30: continue
            rho, _ = spearmanr(valid[col], valid['ret_fwd'])
            pi = pi_sub.get(date, 0.5)
            if pi < 0.5:
                ics_calm.append(rho)
            else:
                ics_panic.append(rho)

        mean_calm = np.mean(ics_calm) if ics_calm else 0
        mean_panic = np.mean(ics_panic) if ics_panic else 0
        print(f"  mom_{lb:<5d} {mean_calm:>8.4f} {mean_panic:>8.4f}")

print("\n" + "=" * 70)
print("  ALL NEW L/S ANALYSES COMPLETE")
print("=" * 70)
