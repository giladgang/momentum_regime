"""S3 yearly walk + S4 report for the applied study.

stage_walk: for each trading year Y (2011..2025) select the combo per rule
from the fold cells (argmax primary, rule_r comparison), fit the 50-seed
canonical HMM on [PANEL_START, Dec Y-1], train 20-seed XGB ensembles WITH and
WITHOUT pi on top-1000 universe rows < Jan Y, score year-Y months, persist
the full monthly cross-section (the S5/TC enabler), and append gross monthly
strategy + benchmark returns.

stage_report: stitched per-rule series, comparator table on identical months
(strategy / no-pi / classic 12-1 / VW benchmark), regime splits,
concentration, selection paths, registered-expectation scoring E1-E5.
Full-table reporting: nothing dropped.

Panic labeling for the walk = crisis-sign rule (canonical eval convention;
train windows through >=2010 always contain >=20 crisis months).
"""
import os

import numpy as np
import pandas as pd

import config as repo_config
from paper import config as C
from paper.src import data_build, universe, model, portfolio
from paper.src import hmm as H
from paper.src import metrics as M

EXPECTED_MONTHS = {y: (11 if y == 2025 else 12) for y in C.EVAL_YEARS}


def _walk_year(year, combo, workers, hmm_seeds, xgb_seeds, n_iter, n_burnin):
    y0, y1 = f'{year}-01-01', f'{year + 1}-01-01'
    feats_z = [f + '_z' for f in combo.split('+')]
    assert feats_z[0] == 'DD_z', combo
    panel = data_build.load_panel()
    p = panel[(panel['date'] >= C.PANEL_START) & (panel['date'] < y1)]
    p = p.dropna(subset=feats_z).reset_index(drop=True)
    tr_p = p[p['date'] < y0]
    Z_tr = tr_p[feats_z].values.astype(float)
    Z_full = p[feats_z].values.astype(float)
    signs = H.crisis_signs(Z_tr, pd.to_datetime(tr_p['date']).values,
                           repo_config.CRISIS_WINDOWS)
    pi_avg = H.fit_pi(Z_tr, Z_full, signs, hmm_seeds, workers,
                      n_iter, n_burnin)
    pi_df = pd.DataFrame({'date': pd.to_datetime(p['date']), 'pi': pi_avg})

    s = data_build.load_stocks(columns=['date', 'permno', 'me', 'ret_fwd']
                               + repo_config.MOM_FEATURES)
    s = universe.top_n(s[s['date'] < y1], C.UNIVERSE_N)
    st = s.merge(pi_df, on='date', how='left').dropna(subset=['pi'])
    tr = st[st['date'] < y0]
    te = st[st['date'] >= y0].copy()
    te['score_pi'] = model.ensemble_scores(
        tr[model.FEATURES_PI].values.astype(float),
        tr['ret_fwd'].values.astype(float),
        te[model.FEATURES_PI].values.astype(float), xgb_seeds, n_jobs=workers)
    te['score_nopi'] = model.ensemble_scores(
        tr[model.FEATURES_NOPI].values.astype(float),
        tr['ret_fwd'].values.astype(float),
        te[model.FEATURES_NOPI].values.astype(float), xgb_seeds,
        n_jobs=workers)
    return te


def _xsec_path(year, combo):
    return os.path.join(C.XSEC_DIR,
                        f"xsec_{year}_{combo.replace('+', '_')}.parquet")


def _done_pairs():
    """(year, rule) pairs already complete in RETURNS_CSV; partial blocks
    are dropped from the file (resume-safety audit lesson)."""
    if not os.path.exists(C.RETURNS_CSV):
        return set()
    prev = pd.read_csv(C.RETURNS_CSV)
    counts = prev.groupby(['year', 'rule']).size()
    done, partial = set(), []
    for (y, ru), n in counts.items():
        if n >= EXPECTED_MONTHS.get(y, 12):
            done.add((y, ru))
        else:
            partial.append((y, ru))
    if partial:
        print(f'[walk] dropping partial blocks: {partial}', flush=True)
        keep = prev[~prev.set_index(['year', 'rule']).index.isin(partial)]
        keep.to_csv(C.RETURNS_CSV, index=False)
    return done


def stage_walk(workers, smoke=False):
    from paper.src import selection as S
    os.makedirs(C.XSEC_DIR, exist_ok=True)
    hmm_seeds, xgb_seeds = C.EVAL_HMM_SEEDS, C.EVAL_XGB_SEEDS
    n_iter, n_burnin = C.N_ITER, C.N_BURNIN
    years = C.EVAL_YEARS
    if smoke:
        hmm_seeds, xgb_seeds = C.EVAL_HMM_SEEDS[:3], C.EVAL_XGB_SEEDS[:3]
        n_iter, n_burnin, years = 400, 100, [2011]

    if os.path.exists(C.FOLD_CELLS_CSV):
        cells = pd.read_csv(C.FOLD_CELLS_CSV)
        n_expected = 25 * 21 * 3
        if not smoke:
            assert len(cells) >= n_expected, \
                f'fold grid incomplete: {len(cells)}/{n_expected} — run S2 first'
        sel = S.select_years(cells)
        sel.to_csv(C.SELECTIONS_CSV, index=False)
        print(sel.to_string(index=False), flush=True)
    else:
        assert smoke, 'fold_cells.csv missing — run S2 first'
        sel = pd.DataFrame([{'year': 2011, 'rule': 'argmax', 'combo': 'DD'},
                            {'year': 2011, 'rule': 'rule_r', 'combo': 'DD'}])

    done = _done_pairs()
    sel = sel[sel['year'].isin(years)]
    grouped = sel.groupby(['year', 'combo'])['rule'].apply(list).reset_index()
    for _, g in grouped.iterrows():
        year, combo = int(g['year']), g['combo']
        rules = [ru for ru in g['rule'] if (year, ru) not in done]
        if not rules:
            print(f'[walk] SKIP {year} {combo} (done)', flush=True)
            continue
        print(f'[walk] === {year} {combo} (rules {rules}) ===', flush=True)
        te = _walk_year(year, combo, workers, hmm_seeds, xgb_seeds,
                        n_iter, n_burnin)
        cols = ['date', 'permno', 'me', 'pi', 'score_pi', 'score_nopi',
                'mom_12', 'ret_fwd']
        tmp = _xsec_path(year, combo) + '.tmp'
        te[cols].to_parquet(tmp, index=False)
        os.replace(tmp, _xsec_path(year, combo))
        r, _ = portfolio.long_only_top(te, 'score_pi', C.DECILE_FRAC)
        b = portfolio.vw_benchmark(te)
        pi_m = (te[['date', 'pi']].drop_duplicates('date')
                .set_index('date')['pi'])
        block = pd.DataFrame({'date': r.index, 'strat_ret': r.values,
                              'bench_ret': b.reindex(r.index).values,
                              'pi': pi_m.reindex(r.index).values})
        block['year'] = year
        block['combo'] = combo
        for ru in rules:
            out = block.copy()
            out['rule'] = ru
            out[['date', 'year', 'rule', 'combo', 'strat_ret', 'bench_ret',
                 'pi']].to_csv(C.RETURNS_CSV, mode='a',
                               header=not os.path.exists(C.RETURNS_CSV),
                               index=False)
        print(f'[walk]   {len(r)} months  IR={M.ir(r, b):+.2f}  '
              f'active={12 * (r - b.reindex(r.index)).mean():+.1%}/yr',
              flush=True)
    print('[walk] DONE', flush=True)


def _comparator_series(sel):
    """No-pi / 12-1 / benchmark monthly series from the argmax xsec files."""
    frames = []
    for _, row in sel[sel['rule'] == 'argmax'].iterrows():
        path = _xsec_path(int(row['year']), row['combo'])
        if os.path.exists(path):
            frames.append(pd.read_parquet(path))
    x = pd.concat(frames, ignore_index=True)
    x['date'] = pd.to_datetime(x['date'])
    r_nopi, _ = portfolio.long_only_top(x, 'score_nopi', C.DECILE_FRAC)
    r_mom, _ = portfolio.long_only_top(x, 'mom_12', C.DECILE_FRAC)
    bench = portfolio.vw_benchmark(x)
    pi_m = x[['date', 'pi']].drop_duplicates('date').set_index('date')['pi']
    return x, r_nopi, r_mom, bench, pi_m


def _summary_row(name, r, b):
    up, dn = M.capture(r, b)
    beta = M.rolling_beta(r, b).dropna()
    return {'series': name, 'n_months': len(r.dropna()),
            'ann_ret': M.ann_ret(r), 'ann_vol': M.ann_vol(r),
            'sharpe': M.sharpe(r), 'max_dd': M.max_dd(r),
            'te': M.te(r, b), 'ir': M.ir(r, b),
            'up_capture': up, 'down_capture': dn,
            'beta36_mean': float(beta.mean()) if len(beta) else np.nan}


def stage_report(workers=1, smoke=False):
    os.makedirs(os.path.join(C.RESULTS, 'tables'), exist_ok=True)
    rets = pd.read_csv(C.RETURNS_CSV, parse_dates=['date'])
    sel = pd.read_csv(C.SELECTIONS_CSV)
    x, r_nopi, r_mom, bench, pi_m = _comparator_series(sel)

    series = {}
    for ru, g in rets.groupby('rule'):
        g = g.sort_values('date')
        series[f'xgb_pi_{ru}'] = g.set_index('date')['strat_ret']
    series['xgb_nopi'] = r_nopi
    series['mom_12_1'] = r_mom

    rows = [_summary_row(k, v, bench) for k, v in series.items()]
    rows.append({'series': 'vw_benchmark', 'n_months': len(bench),
                 'ann_ret': M.ann_ret(bench), 'ann_vol': M.ann_vol(bench),
                 'sharpe': M.sharpe(bench), 'max_dd': M.max_dd(bench),
                 'te': 0.0, 'ir': np.nan, 'up_capture': 1.0,
                 'down_capture': 1.0, 'beta36_mean': 1.0})
    headline = pd.DataFrame(rows)
    headline.to_csv(os.path.join(C.RESULTS, 'tables', 'headline.csv'),
                    index=False)

    # regime split: mean monthly active return in panic vs calm months
    reg_rows = []
    for k, v in series.items():
        a = M.active(v, bench)
        pi = pi_m.reindex(a.index)
        for label, mask in (('panic', pi >= 0.5), ('calm', pi < 0.5)):
            reg_rows.append({'series': k, 'regime': label,
                             'n': int(mask.sum()),
                             'mean_active_mo': float(a[mask].mean()),
                             'sharpe': M.sharpe(v[mask])})
    regime = pd.DataFrame(reg_rows)
    regime.to_csv(os.path.join(C.RESULTS, 'tables', 'regime_split.csv'),
                  index=False)

    # concentration (argmax strategy holdings, yearly medians)
    conc_rows = []
    for y, gy in x.groupby(x['date'].dt.year):
        _, h = portfolio.long_only_top(gy, 'score_pi', C.DECILE_FRAC)
        by_m = h.groupby('date')['weight']
        conc_rows.append({'year': int(y),
                          'max_weight': float(by_m.max().median()),
                          'eff_n': float((1 / (h.groupby('date')['weight']
                                          .apply(lambda w: (w ** 2).sum())))
                                         .median())})
    conc = pd.DataFrame(conc_rows)
    conc.to_csv(os.path.join(C.RESULTS, 'tables', 'concentration.csv'),
                index=False)

    paths = (sel.pivot(index='year', columns='rule', values='combo')
             .reset_index())
    paths.to_csv(os.path.join(C.RESULTS, 'tables', 'selection_path.csv'),
                 index=False)

    # registered expectations E1-E5 (spec 2026-07-13)
    arg = sel[sel['rule'] == 'argmax']
    rr = sel[sel['rule'] == 'rule_r']
    ir_pi = headline.set_index('series').loc['xgb_pi_argmax', 'ir']
    ir_nopi = headline.set_index('series').loc['xgb_nopi', 'ir']
    a_pi = M.active(series['xgb_pi_argmax'], bench)
    pi_al = pi_m.reindex(a_pi.index)
    e4_panic = float(a_pi[pi_al >= 0.5].mean())
    e4_calm = float(a_pi[pi_al < 0.5].mean())
    exps = [
        ('E1', f"argmax distinct combos = {arg['combo'].nunique()} (>=4)",
         arg['combo'].nunique() >= 4),
        ('E2', f'IR(pi)={ir_pi:.2f} > IR(nopi)={ir_nopi:.2f}',
         bool(ir_pi > ir_nopi)),
        ('E3', f'max gross IR = {max(ir_pi, ir_nopi):.2f} (<= 0.8)',
         bool(max(ir_pi, ir_nopi) <= 0.8)),
        ('E4', f'panic mean active {e4_panic:+.2%}/mo vs calm {e4_calm:+.2%}/mo',
         bool(e4_panic > e4_calm)),
        ('E5', f"rule_r distinct {rr['combo'].nunique()} <= "
               f"argmax {arg['combo'].nunique()}",
         bool(rr['combo'].nunique() <= arg['combo'].nunique())),
    ]

    lines = ['# Applied study: headline results (GROSS; costs land in S5)\n',
             'Convention: information through month-end t, positions formed '
             'at t close, return earned over t+1. Universe: point-in-time '
             f'top-{C.UNIVERSE_N} by market cap; long-only fully-invested '
             'top-decile, value-weighted. Full-table reporting.\n',
             '## Headline (monthly, stitched 2011-01..2025-11)\n',
             headline.round(3).to_string(index=False), '',
             '## Regime split (pi >= 0.5 = panic)\n',
             regime.round(4).to_string(index=False), '',
             '## Selection paths\n', paths.to_string(index=False), '',
             '## Concentration (argmax strategy)\n',
             conc.round(3).to_string(index=False), '',
             '## Registered expectations (spec 2026-07-13)\n']
    for tag, desc, hit in exps:
        lines.append(f"{tag}: {desc} -> {'HIT' if hit else 'MISS'}")
    text = '\n'.join(lines)
    with open(os.path.join(C.RESULTS, 'applied_summary.md'), 'w') as f:
        f.write(text + '\n')
    print(text)
    print(f"\nSaved: {os.path.join(C.RESULTS, 'applied_summary.md')}")
