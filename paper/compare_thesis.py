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


MODELS = [('CAPM', ['Mkt-RF']),
          ('FF3', ['Mkt-RF', 'SMB', 'HML']),
          ('Carhart', ['Mkt-RF', 'SMB', 'HML', 'UMD']),
          ('FF5', ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']),
          ('FF6', ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD'])]


def _fit(y, X, maxlags=6):
    res = sm.OLS(y, sm.add_constant(X)).fit(
        cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return {'alpha': float(res.params[0]) * 12, 't': float(res.tvalues[0]),
            'p': float(res.pvalues[0]), 'resid': np.asarray(res.resid),
            'n': int(len(y))}


def get_alphas(r_excess, ff):
    df = r_excess.to_frame('ret')
    # series are FORMATION-indexed; the return is earned over month t+1,
    # so align with the NEXT month's factors
    df.index = df.index + pd.offsets.MonthEnd(1)
    m = df.join(ff, how='inner')
    y = np.asarray(m['ret'].values, dtype=np.float64)
    return {name: _fit(y, m[cols].values.astype(float)) for name, cols in MODELS}


# Long-only analog of MODELS: market factor (already long-only) + the LONG
# legs of the factor sorts (Marc 2026-07-14). CAPM is identical to Panel B.
LO_MODELS = [
    ('CAPM',       ['Mkt-RF']),
    ('FF3-LO',     ['Mkt-RF', 'size_ex', 'value_ex']),
    ('Carhart-LO', ['Mkt-RF', 'size_ex', 'value_ex', 'winner_ex']),
    ('FF5-LO',     ['Mkt-RF', 'size_ex', 'value_ex', 'robust_ex', 'conservative_ex']),
    ('FF6-LO',     ['Mkt-RF', 'size_ex', 'value_ex', 'robust_ex',
                    'conservative_ex', 'winner_ex']),
]


def long_leg_design(ff, legs):
    """Excess-return design matrix: Mkt-RF + each long leg minus RF, month-end
    indexed to the return-realization month."""
    rf = ff['RF']
    leg_ex = legs.sub(rf.reindex(legs.index), axis=0)
    leg_ex.columns = [f'{c}_ex' for c in leg_ex.columns]
    return ff[['Mkt-RF']].join(leg_ex, how='inner').dropna()


def get_alphas_lo(r_excess, X_lo):
    df = r_excess.to_frame('ret')
    df.index = df.index + pd.offsets.MonthEnd(1)          # align to realization
    m = df.join(X_lo, how='inner')
    y = np.asarray(m['ret'].values, dtype=np.float64)
    return {name: _fit(y, m[cols].values.astype(float)) for name, cols in LO_MODELS}


def get_alphas_regime(r_excess, ff, pi_form, thresh=0.5):
    """Factor alphas restricted to calm vs panic months.

    pi_form is the regime probability indexed by FORMATION month (same index
    as r_excess). A month's realized return (earned over t+1) is labeled by the
    regime known at formation t -- so the split is tradable, decided one month
    before the return. HAC(6) SEs on the regime subsample follow the paper's
    convention; the panic subsample (~54 mo) is small, so t-stats are reported
    with n and read as indicative.
    """
    df = r_excess.to_frame('ret')
    df['panic'] = pi_form.reindex(df.index).values >= thresh
    df.index = df.index + pd.offsets.MonthEnd(1)          # align to factors
    m = df.join(ff, how='inner')
    out = {}
    for reg, sub in [('full', m), ('calm', m[~m['panic']]), ('panic', m[m['panic']])]:
        y = np.asarray(sub['ret'].values, dtype=np.float64)
        out[reg] = {name: _fit(y, sub[cols].values.astype(float)) for name, cols in MODELS}
    return out


def main():
    ff = pd.read_parquet('data/ff_factors.parquet')[
        ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD', 'RF']]
    if not os.path.exists(C.FF_LONG_LEGS):
        raise SystemExit(f'Missing {C.FF_LONG_LEGS}\n'
                         'Run once: .venv/bin/python -m paper.fetch_long_legs')
    legs = pd.read_parquet(C.FF_LONG_LEGS)
    X_lo = long_leg_design(ff, legs)
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

    # ── Panel B2: regime-conditional factor alphas ──
    # Marc's question (2026-07-14): does the panic edge survive the factors in
    # the months it occurs? Regress excess returns on FF factors WITHIN calm and
    # panic subsamples (label = pi at formation, so the split is tradable).
    lines.append('## B2. Regime-conditional factor alphas, EXCESS returns, '
                 'alpha x12, NW(6) -- calm vs panic subsample\n')
    lines.append('(panic = pi>=0.5 at formation; alpha [t] per subsample; '
                 'n months in the (n=) row)\n')
    for k in ['pi_rule_r', 'pi_argmax', 'no_pi', 'mom_12_1', 'benchmark']:
        rg = get_alphas_regime(excess(series[k]), ff.drop(columns=['RF']), pi_m)
        rows = []
        for mname, _ in MODELS:
            rows.append({'model': mname,
                         'calm': f"{rg['calm'][mname]['alpha']:+.1%} "
                                 f"({rg['calm'][mname]['t']:.2f}"
                                 f"{sig_stars(rg['calm'][mname]['p'])})",
                         'panic': f"{rg['panic'][mname]['alpha']:+.1%} "
                                  f"({rg['panic'][mname]['t']:.2f}"
                                  f"{sig_stars(rg['panic'][mname]['p'])})",
                         'full': f"{rg['full'][mname]['alpha']:+.1%} "
                                 f"({rg['full'][mname]['t']:.2f}"
                                 f"{sig_stars(rg['full'][mname]['p'])})"})
        n_calm = rg['calm']['FF6']['n']
        n_panic = rg['panic']['FF6']['n']
        n_full = rg['full']['FF6']['n']
        # reconciliation invariant: full-subsample alpha == Panel B unconditional
        chk = get_alphas(excess(series[k]), ff.drop(columns=['RF']))['FF6']['alpha']
        assert abs(rg['full']['FF6']['alpha'] - chk) < 1e-9, \
            f'{k}: regime full-sample FF6 alpha != unconditional'
        lines += [f'### {k}',
                  pd.DataFrame(rows)[['model', 'calm', 'panic', 'full']]
                  .to_string(index=False),
                  f'(n=)  calm {n_calm}   panic {n_panic}   full {n_full}\n']

    # ── Panel B3: LONG-ONLY factor legs (Marc 2026-07-14) ──
    # A long-only book benchmarked against the long legs of the factor sorts
    # (value/robust/conservative/small/winners, in excess form) + the market,
    # instead of the self-financing long-short factors of Panel B.
    lines.append('## B3. Long-ONLY factor-leg alphas, EXCESS returns, '
                 'alpha x12, NW(6)\n')
    lines.append('(RHS = Mkt-RF + long legs of the 2x3 sorts minus RF; '
                 'CAPM identical to Panel B; compare vs Panel B L/S-factor alphas)\n')
    lo_arow = {mname: {} for mname, _ in LO_MODELS}
    for k, v in series.items():
        al = get_alphas_lo(excess(v), X_lo)
        # reconciliation: long-only CAPM == Panel B CAPM (same regression)
        assert abs(al['CAPM']['alpha']
                   - get_alphas(excess(v), ff.drop(columns=['RF']))['CAPM']['alpha']
                   ) < 1e-9, f'{k}: CAPM mismatch across panels'
        for mname, _ in LO_MODELS:
            r_ = al[mname]
            lo_arow[mname][k] = f"{r_['alpha']:+.1%} ({r_['t']:.2f}{sig_stars(r_['p'])})"
    lo_tbl = [{'model': mname, **lo_arow[mname]} for mname, _ in LO_MODELS]
    lo_hdr = ['model'] + [k for k in series]
    lines += [pd.DataFrame(lo_tbl)[lo_hdr].to_string(index=False), '',
              'long legs (ann ret): '
              + ', '.join(f'{c} {(1 + legs[c]).prod() ** (12 / len(legs)) - 1:.1%}'
                          for c in legs.columns) + '\n']

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
