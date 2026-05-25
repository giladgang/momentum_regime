"""
2026-05-25-nw-residual-diagnostics.py
======================================
Diagnostic battery testing whether NW(6) is appropriate for the factor
regressions in scripts/new_ls_analyses.py (which produce Appendix H.3 /
Table 22 with t=4.81 for the XGB six-factor alpha).

Denis asked: ensure the underlying assumptions hold for the data.

Tests applied to residuals from each (strategy, model) pair:
  - Ljung-Box at lag 6 and lag 12 (no regressor adjustment, assumes
    homoskedasticity — included as baseline reference)
  - Breusch-Godfrey LM at lag 6 and 12 (standard form; assumes
    conditional homoskedasticity)
  - HC-robust Breusch-Godfrey (drops homoskedasticity; built manually)
  - Cumby-Huizinga (p=6, q=12; HAC-robust GMM test of the exact
    null "autocorrelations at lags 7..12 are zero given lags 1..6
    may be nonzero"). Implemented from scratch per Cumby-Huizinga
    (1992, Econometrica 60(1), 185–195).

Plus: NW lag-stability — t-stat on alpha at maxlags ∈ {3, 6, 12, 24}.
Plus: residual ACF plots for FF6 (one panel per strategy).

Strategies: XGB (production), LR, D&M, WML.
Models: CAPM, FF3, Carhart, FF5, FF6.

Outputs:
  - results/diagnostic/2026-05-25_nw_residual_diagnostics.csv
  - plots/diagnostic/2026-05-25_nw_residual_acf_ff6.png
  - Console summary table.
"""

from __future__ import annotations

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from config import TRADING_FEE, TRAIN_END, MOM_FEATURES

FEE = TRADING_FEE
REDUCED = MOM_FEATURES + ['pi_filter']

RESULTS_DIR = os.path.join(REPO, 'results', 'diagnostic')
PLOTS_DIR = os.path.join(REPO, 'plots', 'diagnostic')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────────────────
# Section 1: rebuild the 4 return series (XGB, LR, D&M, WML)
# ───────────────────────────────────────────────────────────────────────

print('=' * 70)
print('  NW RESIDUAL DIAGNOSTICS')
print('=' * 70)

print('\n[1] Loading artefacts ...')
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
train = art['train'].copy()

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])

ff = pd.read_parquet('data/ff_factors.parquet')

def long_short_port(df_test, score_col, fee=FEE):
    monthly, prev_lw, prev_sw = [], {}, {}
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

print('[2] Building return series ...')
# XGB: production score
test['score_m2'] = test['score_xgb']
r_xgb = long_short_port(test, 'score_m2')

# LR: refit on training set
X_tr = train[REDUCED].values.astype(float)
X_te = test[REDUCED].values.astype(float)
imp = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(imp.fit_transform(X_tr))
X_te_s = scaler.transform(imp.transform(X_te))
train_c = train.copy()
train_c['above_med'] = train_c.groupby('date')['ret_fwd'].transform(
    lambda x: (x > x.median()).astype(int))
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_tr_s, train_c['above_med'].values)
test['score_m1'] = lr.predict_proba(X_te_s)[:, 1]
r_lr = long_short_port(test, 'score_m1')

# WML: unconditional 12-2 momentum, decile-based
stocks_raw = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks_raw['date'] = pd.to_datetime(stocks_raw['date'])
stocks_raw = stocks_raw[
    stocks_raw['shrcd'].isin([10, 11])
    & stocks_raw['exchcd'].isin([1, 2, 3])
    & (stocks_raw['prc'].abs() > 1)
]
stocks_raw = stocks_raw.sort_values(['permno', 'date']).reset_index(drop=True)
stocks_raw['_lr'] = np.log1p(stocks_raw['ret_adj'].clip(lower=-0.999))
stocks_raw['_lr_s2'] = stocks_raw.groupby('permno')['_lr'].shift(2)
roll = (stocks_raw.groupby('permno', sort=False)['_lr_s2']
        .rolling(11, min_periods=11).sum()
        .reset_index(level='permno', drop=True).sort_index())
stocks_raw['mom_12_2'] = np.expm1(roll)
stocks_raw['ret_fwd'] = stocks_raw.groupby('permno')['ret_adj'].transform(
    lambda x: x.shift(-1))
df_wml = stocks_raw.dropna(subset=['mom_12_2', 'ret_fwd']).copy()

prev_lw, prev_sw = {}, {}
wml_monthly = []
for date, grp in df_wml.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['mom_12_2'].dropna()
    if len(nyse) < 10:
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    longs = grp[grp['mom_12_2'] >= hi]
    shorts = grp[grp['mom_12_2'] <= lo]
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
    wml_monthly.append({'date': date, 'r_wml': r_long - r_short, 'turnover': tl + ts})
    prev_lw, prev_sw = new_lw, new_sw

wml_all = pd.DataFrame(wml_monthly).set_index('date')
mkt_ts = (panel[['date', 'vwretd']].dropna().drop_duplicates('date')
          .sort_values('date').set_index('date')['vwretd'])
mkt_cum_24 = mkt_ts.rolling(24).apply(lambda x: (1 + x).prod() - 1, raw=True)
bear = (mkt_cum_24 < 0).astype(int).rename('bear')
wml_all = wml_all.join(bear, how='left')
wml_all['bear'] = wml_all['bear'].fillna(0)
wml_all['sigma2'] = wml_all['r_wml'].rolling(6, min_periods=6).var()

train_wml = wml_all[wml_all.index < TRAIN_END]
test_wml = wml_all[wml_all.index >= TRAIN_END].copy()

X_bear_tr = sm.add_constant(np.asarray(train_wml['bear'].values, dtype=np.float64))
ols = sm.OLS(np.asarray(train_wml['r_wml'].values, dtype=np.float64), X_bear_tr).fit()
X_bear_te = sm.add_constant(test_wml['bear'])
mu = ols.predict(X_bear_te)
sigma2 = wml_all.loc[test_wml.index, 'sigma2']
w_dm = (0.5 * mu / sigma2).clip(-0.5, 1.5)
r_dm = pd.Series(
    w_dm.values * (test_wml['r_wml'].values - FEE * test_wml['turnover'].values),
    index=test_wml.index)
r_wml_net = test_wml['r_wml'] - FEE * test_wml['turnover']

print(f"  XGB n={len(r_xgb)}, Sharpe={r_xgb.mean()/r_xgb.std()*np.sqrt(12):.2f}")
print(f"  LR  n={len(r_lr)},  Sharpe={r_lr.mean()/r_lr.std()*np.sqrt(12):.2f}")
print(f"  D&M n={len(r_dm)},  Sharpe={r_dm.mean()/r_dm.std()*np.sqrt(12):.2f}")
print(f"  WML n={len(r_wml_net)},  Sharpe={r_wml_net.mean()/r_wml_net.std()*np.sqrt(12):.2f}")

returns = {
    'XGB': r_xgb,
    'LR':  r_lr,
    'D&M': r_dm,
    'WML': r_wml_net,
}

# ───────────────────────────────────────────────────────────────────────
# Section 2: diagnostic test implementations
# ───────────────────────────────────────────────────────────────────────

def fit_factor_model(r, model_cols):
    """Return (results, residuals, X_with_const, y) for OLS fit r ~ const + cols."""
    r_df = r.to_frame('ret')
    r_df.index = r_df.index + pd.offsets.MonthEnd(0)
    merged = r_df.join(ff[model_cols], how='inner')
    y = np.asarray(merged['ret'].values, dtype=np.float64)
    X = sm.add_constant(np.asarray(merged[model_cols].values, dtype=np.float64))
    res_ols = sm.OLS(y, X).fit()
    return res_ols, res_ols.resid, X, y

def alpha_t_at_lag(y, X, maxlags):
    res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return float(res.tvalues[0])

def hc_breusch_godfrey(resid, X, nlags, hc_type='HC3'):
    """Heteroskedasticity-robust BG.

    Auxiliary regression: e_t = X_t γ + Σ_k δ_k e_{t-k} + u_t, with HC
    covariance. Joint Wald test on δ's. Returns (Wald stat, p-value).
    """
    e = np.asarray(resid, dtype=np.float64)
    T = len(e)
    if T <= nlags + X.shape[1] + 5:
        return np.nan, np.nan
    E_lag = np.zeros((T, nlags))
    for k in range(1, nlags + 1):
        E_lag[k:, k-1] = e[:-k]
    # Drop first nlags rows
    e_use = e[nlags:]
    X_use = np.column_stack([X[nlags:], E_lag[nlags:]])
    res = sm.OLS(e_use, X_use).fit(cov_type=hc_type)
    lag_start = X.shape[1]
    R = np.zeros((nlags, X_use.shape[1]))
    for k in range(nlags):
        R[k, lag_start + k] = 1.0
    wald = res.wald_test(R, scalar=True)
    return float(wald.statistic), float(wald.pvalue)

def cumby_huizinga(resid, p, q):
    """Cumby-Huizinga (1992) test.

    H0: E[e_t e_{t-k}] = 0 for k = p+1, ..., q
       (lags 1..p may be nonzero — unrestricted)
    Robust to heteroskedasticity and to autocorrelation at lags ≤ p.

    Construction:
      h_t = (e_t e_{t-p-1}, ..., e_t e_{t-q}) for t = q+1, ..., T
      Let h_bar = mean(h_t). Stat = T' * h_bar' S^{-1} h_bar ~ χ²(q-p)
      where S is a HAC estimate of Var(sqrt(T') h_bar) with Newey-West
      kernel, bandwidth = floor(4 (T'/100)^(2/9)).
    """
    e = np.asarray(resid, dtype=np.float64)
    T = len(e)
    m = q - p
    if T <= q + 10 or m < 1:
        return np.nan, np.nan
    H = np.zeros((T - q, m))
    for k_idx, k in enumerate(range(p + 1, q + 1)):
        H[:, k_idx] = e[q:T] * e[q - k:T - k]
    h_bar = H.mean(axis=0)
    T_eff = T - q
    bandwidth = max(1, int(np.floor(4 * (T_eff / 100) ** (2/9))))
    H_c = H - h_bar
    S = (H_c.T @ H_c) / T_eff
    for lag in range(1, bandwidth + 1):
        w = 1.0 - lag / (bandwidth + 1)
        G_lag = (H_c[lag:].T @ H_c[:-lag]) / T_eff
        S = S + w * (G_lag + G_lag.T)
    try:
        stat = T_eff * float(h_bar @ np.linalg.solve(S, h_bar))
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    pval = 1.0 - chi2.cdf(stat, df=m)
    return float(stat), float(pval)

# ───────────────────────────────────────────────────────────────────────
# Section 3: run the diagnostic battery on every (strategy, model) pair
# ───────────────────────────────────────────────────────────────────────

MODEL_COLS = {
    'CAPM':    ['Mkt-RF'],
    'FF3':     ['Mkt-RF', 'SMB', 'HML'],
    'Carhart': ['Mkt-RF', 'SMB', 'HML', 'UMD'],
    'FF5':     ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'],
    'FF6':     ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD'],
}

NW_LAGS = [3, 6, 12, 24]

print('\n[3] Running diagnostic battery ...')
rows = []
residuals_for_acf = {}

for strat_name, r in returns.items():
    for model_name, cols in MODEL_COLS.items():
        res_ols, resid, X, y = fit_factor_model(r, cols)
        T_resid = len(resid)
        alpha_ann = float(res_ols.params[0]) * 12

        # Ljung-Box at 6 and 12
        lb6 = sm.stats.diagnostic.acorr_ljungbox(
            resid, lags=[6], return_df=True).iloc[0]
        lb12 = sm.stats.diagnostic.acorr_ljungbox(
            resid, lags=[12], return_df=True).iloc[0]

        # Breusch-Godfrey at 6 and 12 (standard, homoskedastic)
        bg6 = sm.stats.diagnostic.acorr_breusch_godfrey(res_ols, nlags=6)
        bg12 = sm.stats.diagnostic.acorr_breusch_godfrey(res_ols, nlags=12)

        # HC-robust BG at 6 and 12
        hcbg6_stat, hcbg6_p = hc_breusch_godfrey(resid, X, nlags=6)
        hcbg12_stat, hcbg12_p = hc_breusch_godfrey(resid, X, nlags=12)

        # Cumby-Huizinga (p=6, q=12)
        ch_stat, ch_p = cumby_huizinga(resid, p=6, q=12)

        # NW lag stability
        nw_t = {L: alpha_t_at_lag(y, X, L) for L in NW_LAGS}

        rows.append({
            'strategy': strat_name,
            'model': model_name,
            'T': T_resid,
            'alpha_ann': alpha_ann,
            't_nw6': nw_t[6],
            't_nw3': nw_t[3],
            't_nw12': nw_t[12],
            't_nw24': nw_t[24],
            'LB6_stat': float(lb6['lb_stat']),
            'LB6_p': float(lb6['lb_pvalue']),
            'LB12_stat': float(lb12['lb_stat']),
            'LB12_p': float(lb12['lb_pvalue']),
            'BG6_stat': float(bg6[0]),
            'BG6_p': float(bg6[1]),
            'BG12_stat': float(bg12[0]),
            'BG12_p': float(bg12[1]),
            'HC_BG6_stat': hcbg6_stat,
            'HC_BG6_p': hcbg6_p,
            'HC_BG12_stat': hcbg12_stat,
            'HC_BG12_p': hcbg12_p,
            'CH_stat': ch_stat,
            'CH_p': ch_p,
        })

        if model_name == 'FF6':
            residuals_for_acf[strat_name] = pd.Series(resid)

df_out = pd.DataFrame(rows)
out_csv = os.path.join(RESULTS_DIR, '2026-05-25_nw_residual_diagnostics.csv')
df_out.to_csv(out_csv, index=False)
print(f"\n  Saved: {out_csv}")

# ───────────────────────────────────────────────────────────────────────
# Section 4: ACF plot grid for FF6 residuals
# ───────────────────────────────────────────────────────────────────────

print('\n[4] Generating ACF plots ...')
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, (strat, e) in zip(axes.flatten(), residuals_for_acf.items()):
    plot_acf(e.values, lags=24, ax=ax, title=f'{strat} — FF6 residuals')
    ax.axhline(0.0, color='black', linewidth=0.5)
plt.suptitle('Residual ACF (FF6 factor regression), lags 1-24', y=1.00)
plt.tight_layout()
acf_path = os.path.join(PLOTS_DIR, '2026-05-25_nw_residual_acf_ff6.png')
plt.savefig(acf_path, dpi=140, bbox_inches='tight')
plt.close()
print(f"  Saved: {acf_path}")

# ───────────────────────────────────────────────────────────────────────
# Section 5: console summary
# ───────────────────────────────────────────────────────────────────────

def fmt_p(p):
    if not np.isfinite(p):
        return '   --'
    return f'{p:.3f}'

print('\n' + '=' * 110)
print('  RESIDUAL AUTOCORRELATION DIAGNOSTICS — p-values (lower = reject H0 of no autocorrelation)')
print('=' * 110)
header = (f"  {'Strat':<5}  {'Model':<8}  {'T':<4}  "
          f"{'LB(6)':>7} {'LB(12)':>7}  "
          f"{'BG(6)':>7} {'BG(12)':>7}  "
          f"{'HC-BG(6)':>9} {'HC-BG(12)':>10}  "
          f"{'CH(p=6,q=12)':>14}")
print(header)
print('-' * 110)
for _, r in df_out.iterrows():
    print(f"  {r['strategy']:<5}  {r['model']:<8}  {int(r['T']):<4}  "
          f"{fmt_p(r['LB6_p']):>7} {fmt_p(r['LB12_p']):>7}  "
          f"{fmt_p(r['BG6_p']):>7} {fmt_p(r['BG12_p']):>7}  "
          f"{fmt_p(r['HC_BG6_p']):>9} {fmt_p(r['HC_BG12_p']):>10}  "
          f"{fmt_p(r['CH_p']):>14}")

print('\n' + '=' * 100)
print('  NW LAG-STABILITY — t-statistic on α at maxlags ∈ {3, 6, 12, 24}')
print('=' * 100)
header2 = (f"  {'Strat':<5}  {'Model':<8}  "
           f"{'α (%)':>7}  {'t(NW3)':>8} {'t(NW6)':>8} {'t(NW12)':>9} {'t(NW24)':>9}")
print(header2)
print('-' * 100)
for _, r in df_out.iterrows():
    print(f"  {r['strategy']:<5}  {r['model']:<8}  "
          f"{r['alpha_ann']*100:>7.2f}  {r['t_nw3']:>8.2f} {r['t_nw6']:>8.2f} "
          f"{r['t_nw12']:>9.2f} {r['t_nw24']:>9.2f}")

print('\n  Done.')
