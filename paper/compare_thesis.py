"""Thesis-results-section computations replicated for the applied config
(expanding window, long-only, top-1000), side-by-side with thesis values
pulled from results/PRODUCTION_METRICS.json.

Conventions copied from scripts/main_results_analysis.py /
scripts/new_ls_analyses.py: NW t = HAC(maxlags=6) t of mean; factor alphas =
OLS on FF factors, alpha x12, HAC(6); Ljung-Box on FF6 residuals, lags 6/12;
regime split at pi 0.5. One adaptation, disclosed: alphas here regress
EXCESS returns (r - RF) because the strategy is long-only (thesis L/S is
self-financing and regressed raw returns).

Output: paper/results/tables/thesis_comparison.md
Usage:  .venv/bin/python -m paper.compare_thesis
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from paper import config as C                                   # noqa: E402
from paper import walk_report as W                              # noqa: E402
from paper.src import metrics as M                              # noqa: E402

OUT = os.path.join(C.RESULTS, 'tables', 'thesis_comparison.md')


def newey_west_t(r, maxlags=6):
    vals = pd.Series(r).dropna().values.astype(float)
    if len(vals) < 12:
        return np.nan, np.nan
    res = sm.OLS(vals, np.ones((len(vals), 1))).fit(
        cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return float(res.tvalues[0]), float(res.pvalues[0])


def sig_stars(p):
    if p < 0.01:
        return '***'
    if p < 0.05:
        return '**'
    if p < 0.10:
        return '*'
    return ''


def get_alphas(r_excess, ff):
    df = r_excess.to_frame('ret')
    # series are FORMATION-indexed; the return is earned over month t+1,
    # so align with the NEXT month's factors
    df.index = df.index + pd.offsets.MonthEnd(1)
    m = df.join(ff, how='inner')
    y = np.asarray(m['ret'].values, dtype=np.float64)
    out = {}
    for name, cols in [('CAPM', ['Mkt-RF']),
                       ('FF3', ['Mkt-RF', 'SMB', 'HML']),
                       ('Carhart', ['Mkt-RF', 'SMB', 'HML', 'UMD']),
                       ('FF5', ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']),
                       ('FF6', ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD'])]:
        X = sm.add_constant(m[cols].values.astype(float))
        res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
        out[name] = {'alpha': float(res.params[0]) * 12,
                     't': float(res.tvalues[0]), 'p': float(res.pvalues[0]),
                     'resid': np.asarray(res.resid), 'n': len(y)}
    return out


def main():
    ff = pd.read_parquet('data/ff_factors.parquet')[
        ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD', 'RF']]
    pm = json.load(open('results/PRODUCTION_METRICS.json'))

    rets = pd.read_csv(C.RETURNS_CSV, parse_dates=['date'])
    sel = pd.read_csv(C.SELECTIONS_CSV)
    x, r_nopi, r_mom, bench, pi_m = W._comparator_series(sel)
    series = {}
    for ru, g in rets.groupby('rule'):
        series[f'pi_{ru}'] = g.sort_values('date').set_index('date')['strat_ret']
    series['no_pi'] = r_nopi
    series['mom_12_1'] = r_mom
    series['benchmark'] = bench

    rf = ff['RF'].copy()
    rf.index = rf.index + pd.offsets.MonthEnd(0)

    def excess(r):
        rr = r.copy()
        # subtract the RF of the realization month (t+1)
        rr.index = pd.to_datetime(rr.index) + pd.offsets.MonthEnd(1)
        rr = (rr - rf.reindex(rr.index)).dropna()
        rr.index = rr.index - pd.offsets.MonthEnd(1)   # back to formation
        return rr

    lines = ['# Thesis results-section computations, applied config\n',
             'Applied = expanding-window yearly walk, long-only top-decile VW,'
             ' point-in-time top-1000, GROSS, 2011-01..2025-11 (179 mo).\n'
             'Thesis refs = production L/S, full universe, net 10bps,'
             ' 2011-2024 (from PRODUCTION_METRICS.json).\n']

    # ── Panel A: performance (thesis tab:performance analog) ──
    rows = []
    for k, v in series.items():
        t_raw, p_raw = newey_west_t(v)
        a = M.active(v, bench) if k != 'benchmark' else None
        t_act, p_act = newey_west_t(a) if a is not None else (np.nan, np.nan)
        beta = M.rolling_beta(v, bench, window=len(v)).dropna()
        rows.append({
            'series': k, 'ann_ret': M.ann_ret(v), 'ann_vol': M.ann_vol(v),
            'sharpe': M.sharpe(v), 'max_dd': M.max_dd(v),
            'beta': float(beta.iloc[-1]) if len(beta) else 1.0,
            'final_$1': float((1 + v.dropna()).prod()),
            'nw_t_raw': f'{t_raw:.2f}{sig_stars(p_raw)}',
            'nw_t_active': (f'{t_act:.2f}{sig_stars(p_act)}'
                            if a is not None else '-'),
            'ir': M.ir(v, bench) if k != 'benchmark' else np.nan})
    perf = pd.DataFrame(rows)
    g = pm['m2_perf']
    lines += ['## A. Performance (thesis Table: performance)\n',
              perf.round(3).to_string(index=False), '',
              f"thesis refs: XGB L/S ann_ret {g['m2_ann_ret']['value']}% "
              f"vol {g['m2_ann_vol']['value']}% sharpe {g['m2_sharpe']['value']} "
              f"mdd {g['m2_max_dd']['value']}% beta {g['m2_beta']['value']} "
              f"$1->{g['m2_final_dollar']['value']} nw_t {g['m2_nw_t']['value']}*** | "
              f"12-1 L/S sharpe {g['fixed_12_sharpe']['value']} "
              f"mdd {g['fixed_12_max_dd']['value']}% | "
              f"market sharpe {g['market_sharpe']['value']}\n"]

    # ── Panel B: factor alphas (thesis Table: factor alphas) ──
    lines.append('## B. Factor alphas, EXCESS returns, alpha x12, NW(6) '
                 '(thesis: raw L/S returns)\n')
    ref = pm['factor_alphas']
    hdr = ['model'] + [k for k in series] + ['thesis_LS_ref']
    arow = {mname: {} for mname in ['CAPM', 'FF3', 'Carhart', 'FF5', 'FF6']}
    resid_store = {}
    for k, v in series.items():
        al = get_alphas(excess(v), ff.drop(columns=['RF']))
        for mname in arow:
            r_ = al[mname]
            arow[mname][k] = f"{r_['alpha']:+.1%} ({r_['t']:.2f}{sig_stars(r_['p'])})"
        resid_store[k] = al['FF6']['resid']
    tbl = []
    refmap = {'CAPM': 'capm_alpha', 'FF3': 'ff3_alpha', 'Carhart':
              'carhart_alpha', 'FF5': 'ff5_alpha', 'FF6': 'ff6_alpha'}
    for mname in arow:
        rr = ref[refmap[mname]]
        tbl.append({'model': mname, **arow[mname],
                    'thesis_LS_ref': f"+{rr['value']}% ({rr['t']})"})
    lines += [pd.DataFrame(tbl)[hdr].to_string(index=False), '']

    # ── Panel C: regime-conditional Sharpe (thesis Table: regime sharpe) ──
    rows = []
    for k, v in series.items():
        pi = pi_m.reindex(pd.to_datetime(v.index))
        rows.append({'series': k, 'full': M.sharpe(v),
                     'calm': M.sharpe(v[(pi < 0.5).values]),
                     'panic': M.sharpe(v[(pi >= 0.5).values])})
    rs = pm['regime_sharpe']
    lines += ['## C. Regime-conditional Sharpe (pi >= 0.5 = panic)\n',
              pd.DataFrame(rows).round(2).to_string(index=False), '',
              f"thesis refs: XGB L/S {rs['m2_full']['value']}/"
              f"{rs['m2_calm']['value']}/{rs['m2_panic']['value']} "
              f"(full/calm/panic) | market {rs['market_full']['value']}/"
              f"{rs['market_calm']['value']}/{rs['market_panic']['value']}\n"]

    # ── Panel D: Ljung-Box on FF6 residuals (thesis App. H.3) ──
    from statsmodels.stats.diagnostic import acorr_ljungbox
    rows = []
    for k in ['pi_rule_r', 'pi_argmax', 'no_pi']:
        lb = acorr_ljungbox(resid_store[k], lags=[6, 12], return_df=True)
        rows.append({'series': k,
                     'LB6': f"{lb.lb_stat.iloc[0]:.2f} (p={lb.lb_pvalue.iloc[0]:.3f})",
                     'LB12': f"{lb.lb_stat.iloc[1]:.2f} (p={lb.lb_pvalue.iloc[1]:.3f})"})
    rd = pm['residual_diagnostics']
    lines += ['## D. Ljung-Box, FF6 residuals\n',
              pd.DataFrame(rows).to_string(index=False), '',
              f"thesis ref (XGB/FF6): LB6 {rd['xgb_ff6_lb6_stat']['value']} "
              f"(p={rd['xgb_ff6_lb6_p']['value']}), "
              f"LB12 {rd['xgb_ff6_lb12_stat']['value']} "
              f"(p={rd['xgb_ff6_lb12_p']['value']})\n"]

    # ── Panel E: subperiod episodes within the applied window ──
    eps = [('COVID crash+recovery', '2020-01-01', '2020-12-31'),
           ('2022 bear market', '2022-01-01', '2022-12-31'),
           ('2025 tariff episode', '2025-03-01', '2025-06-30')]
    rows = []
    for name, s, e in eps:
        row = {'episode': name}
        for k, v in series.items():
            vv = v[(pd.to_datetime(v.index) >= s) & (pd.to_datetime(v.index) <= e)]
            row[k] = f'{float((1 + vv).prod() - 1):+.1%}'
        rows.append(row)
    lines += ['## E. Episode total returns (formation months in window)\n',
              pd.DataFrame(rows).to_string(index=False), '']

    text = '\n'.join(lines)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write(text + '\n')
    print(text)
    print('\nSaved:', OUT)


if __name__ == '__main__':
    main()
