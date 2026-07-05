"""
ols_baseline_row.py
===================
Computes the plain OLS baseline row for the thesis performance table
(tab:performance) and inserts it into tables/table_performance.tex.

The row is an ordinary least squares regression on the production feature
set (12 momentum horizons + pi_filter), predicting continuous forward
returns (the same target as XGB), using the production-preprocessed
feature matrices (X_tr_s / X_te_s) from the Step 2 artefacts. The score
is evaluated in the same long-short construction and with the same metric
definitions as every other row in the table.

scripts/main_results_analysis.py (Step 3) is the producer of
tables/table_performance.tex and is not modified (see its header note).
This script runs AFTER Step 3 and inserts/refreshes a single `OLS` row
directly below the LR row. The insertion is idempotent: rerunning
replaces the existing OLS row in place.

Outputs:
  - results/thesis/ols_monthly_returns.csv
  - ols_* metrics in results/PRODUCTION_METRICS.json (section m2_perf)
  - OLS row in tables/table_performance.tex
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TABLES_DIR, RESULTS_THESIS_DIR
from src.utils import load_artefacts, long_short_port, metrics
from scripts._canonical_metrics import set_many

PERFORMANCE_TABLE = os.path.join(TABLES_DIR, 'table_performance.tex')
RETURNS_CSV = os.path.join(RESULTS_THESIS_DIR, 'ols_monthly_returns.csv')


def newey_west_t(r, maxlags=6):
    """Newey-West t-stat for H0: mean return = 0 (as in Step 3)."""
    vals = pd.Series(r).dropna().values.astype(float)
    if len(vals) < 12:
        return np.nan, np.nan
    X = np.ones((len(vals), 1))
    model = sm.OLS(vals, X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return model.tvalues[0], model.pvalues[0]


def sig_stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''


def market_beta(r, r_mkt):
    """CAPM beta vs the market series (as in Step 3)."""
    common = r.index.intersection(r_mkt.index)
    r_a = r.loc[common].values
    m_a = r_mkt.loc[common].values
    valid = ~(np.isnan(r_a) | np.isnan(m_a))
    if valid.sum() <= 12:
        return np.nan
    cov_mat = np.cov(r_a[valid], m_a[valid])
    return cov_mat[0, 1] / cov_mat[1, 1]


def format_ols_row(ann_ret, ann_vol, sharpe, mdd, beta, nw_t, nw_p, final):
    """Format the OLS row exactly like the Step 3 writer formats its rows."""
    if np.isnan(nw_t):
        nw = '--'
    else:
        nw = f'{nw_t:.2f}{sig_stars(nw_p)}'
    if np.isnan(beta):
        beta_str = '--'
    elif beta < 0:
        beta_str = f'$-${abs(beta):.2f}'
    else:
        beta_str = f'{beta:.2f}'
    row = (f'OLS & {ann_ret:.1%} & {ann_vol:.1%} & {sharpe:.2f} & '
           f'{mdd:.1%} & {beta_str} & {nw} & {final:.1f} \\\\')
    # Escape unescaped % signs the same way Step 3's write_tex does
    return re.sub(r'(\d)%', r'\1\\%', row)


def insert_ols_row(tex, row_line):
    """Insert `row_line` directly below the LR row; replace an existing
    OLS row in place if one is already present. Raises if no LR row."""
    ols_pat = re.compile(r'^OLS\s*&.*?\\\\\s*$', re.MULTILINE)
    if ols_pat.search(tex):
        return ols_pat.sub(lambda _: row_line, tex, count=1)
    lr_pat = re.compile(r'^LR\s*&.*?\\\\', re.MULTILINE)
    m = lr_pat.search(tex)
    if not m:
        raise ValueError('LR row not found in table_performance.tex; '
                         'run Step 3 (main_results_analysis.py) first.')
    return tex[:m.end()] + '\n' + row_line + tex[m.end():]


def main():
    print('Loading artefacts ...')
    artefacts = load_artefacts()
    test = artefacts['test'].copy()
    X_tr_s = artefacts['X_tr_s']
    X_te_s = artefacts['X_te_s']
    y_train = artefacts['y_train']
    r_mkt = artefacts['r_mkt']

    print('Fitting OLS (continuous forward-return target) ...')
    ols = LinearRegression()
    ols.fit(X_tr_s, y_train)
    test['score_ols'] = ols.predict(X_te_s)

    r_ols = long_short_port(test, 'score_ols')
    ann_ret, ann_vol, sharpe, mdd = metrics(r_ols)
    final = (1 + r_ols.dropna()).cumprod().iloc[-1]
    beta = market_beta(r_ols, r_mkt)
    nw_t, nw_p = newey_west_t(r_ols)

    print(f'  OLS | Sharpe={sharpe:.2f}  Ann.Ret={ann_ret:.1%}  '
          f'Ann.Vol={ann_vol:.1%}  MDD={mdd:.1%}  beta={beta:.2f}  '
          f'NW t={nw_t:.2f} (p={nw_p:.3f})  Final$={final:.2f}')

    os.makedirs(RESULTS_THESIS_DIR, exist_ok=True)
    r_ols.rename('ret').to_csv(RETURNS_CSV)
    print(f'Saved: {RETURNS_CSV}')

    set_many('m2_perf', {
        'ols_ann_ret':      {'value': round(ann_ret * 100, 4), 'unit': 'percent'},
        'ols_ann_vol':      {'value': round(ann_vol * 100, 4), 'unit': 'percent'},
        'ols_sharpe':       {'value': round(sharpe, 4)},
        'ols_max_dd':       {'value': round(mdd * 100, 4), 'unit': 'percent'},
        'ols_beta':         {'value': round(beta, 4)},
        'ols_nw_t':         {'value': round(nw_t, 4), 'stars': sig_stars(nw_p) or None,
                             'p': round(nw_p, 4)},
        'ols_final_dollar': {'value': round(final, 4)},
    }, source=__file__)
    print('Registered ols_* metrics in PRODUCTION_METRICS.json (m2_perf)')

    row_line = format_ols_row(ann_ret, ann_vol, sharpe, mdd, beta, nw_t, nw_p, final)
    print(f'Row: {row_line}')
    with open(PERFORMANCE_TABLE) as f:
        tex = f.read()
    with open(PERFORMANCE_TABLE, 'w') as f:
        f.write(insert_ols_row(tex, row_line))
    print(f'Updated: {PERFORMANCE_TABLE}')


if __name__ == '__main__':
    main()
