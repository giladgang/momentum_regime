"""
2026-05-25-verify-nw-tstats.py
==============================
Re-compute every NW(6) t-statistic the thesis quotes and compare to the
reported value. Pinpoints any drift between the prose and the data.

Numbers checked:
  Table 3 (main_results.tex) "NW t" column — mean-return regressions
    XGB, Fixed 12-mo, Fixed 1-mo, DET, LR
  Table 22 (Appendix H.3) — factor-model alphas for XGB (5 models)
  Table 23 (Appendix H.3) — factor-model alphas for XGB+fund (5 models)
"""

from __future__ import annotations

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
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
NW_LAGS = 6


def nw_t_mean(r):
    """NW(6) t-stat on H0: E[r] = 0."""
    vals = np.asarray(pd.Series(r).dropna().values, dtype=np.float64)
    X = np.ones((len(vals), 1))
    res = sm.OLS(vals, X).fit(cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
    return float(res.tvalues[0])


def nw_t_alpha(r, ff_df, factors):
    """NW(6) t-stat on α from r = α + β'f + ε."""
    r_df = pd.Series(r).to_frame('ret')
    r_df.index = r_df.index + pd.offsets.MonthEnd(0)
    merged = r_df.join(ff_df[factors], how='inner')
    y = np.asarray(merged['ret'].values, dtype=np.float64)
    X = sm.add_constant(np.asarray(merged[factors].values, dtype=np.float64))
    res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
    return float(res.params[0]) * 12, float(res.tvalues[0])


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


print('=' * 78)
print('  Verifying NW(6) t-statistics quoted in the thesis')
print('=' * 78)

# ── Load data ───────────────────────────────────────────────────
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
train = art['train'].copy()
ff = pd.read_parquet('data/ff_factors.parquet')

# ── XGB: production scores ──────────────────────────────────────
test['score_xgb_'] = test['score_xgb']
r_xgb = long_short_port(test, 'score_xgb_')

# ── LR: refit ────────────────────────────────────────────────────
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
test['score_lr'] = lr.predict_proba(X_te_s)[:, 1]
r_lr = long_short_port(test, 'score_lr')

# ── DET: deterministic regime-adaptive formula ──────────────────
# score_t^s = product of past l_t monthly returns where l_t = round(12 - 11*pi)
panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
pi_ts = (panel[['date', 'pi_filter']].dropna()
         .drop_duplicates('date').set_index('date')['pi_filter'])
det_scores = []
mom_cols = [f'mom_{k}' for k in range(1, 13)]
for date, grp in test.groupby('date'):
    pi = float(pi_ts.get(date, np.nan))
    if not np.isfinite(pi):
        continue
    L = int(round(12 - 11 * pi))
    L = max(1, min(12, L))
    sub = grp.copy()
    sub['score_det'] = sub[f'mom_{L}']
    det_scores.append(sub)
test_det = pd.concat(det_scores, ignore_index=True)
r_det = long_short_port(test_det, 'score_det')

# ── Fixed 12-mo and 1-mo momentum on the same test panel ─────────
test['score_fixed12'] = test['mom_12']
r_fixed12 = long_short_port(test, 'score_fixed12')
test['score_fixed1'] = test['mom_1']
r_fixed1 = long_short_port(test, 'score_fixed1')

# ──────────────────────────────────────────────────────────────────
# Verify Table 3 NW t (mean-return regression)
# ──────────────────────────────────────────────────────────────────

print('\n── Table 3 (main_results) NW t  H0: E[r]=0 ──')
print(f"  {'Strategy':<18}{'Thesis t':>10}{'Recomputed':>12}{'Match':>8}")

thesis_t3 = {
    'XGB':           4.37,
    'Fixed 12-mo':   0.24,
    'Fixed 1-mo':    1.08,
    'DET':           0.43,
    'LR':           -0.04,
}
recomputed_t3 = {
    'XGB':         nw_t_mean(r_xgb),
    'Fixed 12-mo': nw_t_mean(r_fixed12),
    'Fixed 1-mo':  nw_t_mean(r_fixed1),
    'DET':         nw_t_mean(r_det),
    'LR':          nw_t_mean(r_lr),
}
for name, t_thesis in thesis_t3.items():
    t_recalc = recomputed_t3[name]
    diff = t_recalc - t_thesis
    ok = abs(diff) < 0.10
    print(f"  {name:<18}{t_thesis:>10.2f}{t_recalc:>12.2f}{('✓' if ok else f'Δ={diff:+.2f}'):>8}")

# ──────────────────────────────────────────────────────────────────
# Verify Table 22 factor alpha t-stats for XGB
# ──────────────────────────────────────────────────────────────────

print('\n── Table 22 (Appendix H.3) factor regressions — XGB ──')
print(f"  {'Model':<10}{'Thesis α':>10}{'Recalc α':>10}{'Thesis t':>10}{'Recalc t':>10}{'Match':>8}")

models = {
    'CAPM':    ['Mkt-RF'],
    'FF3':     ['Mkt-RF', 'SMB', 'HML'],
    'Carhart': ['Mkt-RF', 'SMB', 'HML', 'UMD'],
    'FF5':     ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'],
    'FF6':     ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD'],
}
thesis_t22 = {
    'CAPM':    (0.234, 4.08),
    'FF3':     (0.232, 4.48),
    'Carhart': (0.243, 4.75),
    'FF5':     (0.229, 4.62),
    'FF6':     (0.241, 4.81),
}
for model, factors in models.items():
    alpha_ann, t_recalc = nw_t_alpha(r_xgb, ff, factors)
    a_thesis, t_thesis = thesis_t22[model]
    ok_alpha = abs(alpha_ann - a_thesis) < 0.01
    ok_t = abs(t_recalc - t_thesis) < 0.10
    ok = '✓' if (ok_alpha and ok_t) else f'Δt={t_recalc - t_thesis:+.2f}'
    print(f"  {model:<10}{a_thesis*100:>9.1f}%{alpha_ann*100:>9.1f}%{t_thesis:>10.2f}{t_recalc:>10.2f}{ok:>8}")

print('\nDone.')
