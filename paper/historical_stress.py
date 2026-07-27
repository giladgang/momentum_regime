"""Pre-2011 historical-stress extension of the applied DD-only walk.

Runs the long-only top-1000 DD-only (rule_r) strategy on trading years
2000-2010 so the applied study has a genuine pre-2011 backtest covering the
dot-com bust, GFC and 2009 momentum crash. Panic = deeper drawdown, imposed
directly via signs=[-1.0] (as selection.py does for pre-crisis folds), so no
crisis-window calibration is needed and hmm.crisis_signs' >=20-month assert
is bypassed.

Point-in-time integrity: DD_z is recomputed PER FOLD from raw DD using only
data through Dec Y-1 (winsor 1/99 + standardize on train), NOT the panel's
fixed pre-2011 DD_z (which would leak 2000-2010 info into a year-2000 fit).

Separate output; does NOT touch the canonical 2011-2025 walk_returns.csv:
  paper/results/historical/hist_returns.csv   monthly strat/bench/mom12, 2000-2010

Usage:
  python -m paper.historical_stress --stage walk [--workers N] [--smoke]
  python -m paper.historical_stress --stage report   # stitch 2000-2025 + subperiods
"""
import argparse
import os

import numpy as np
import pandas as pd

import config as repo_config
from paper import config as C
from paper.src import data_build, universe, model, portfolio
from paper.src import hmm as H
from paper.src import metrics as M

HIST_DIR = os.path.join(C.RESULTS, 'historical')
HIST_RET = os.path.join(HIST_DIR, 'hist_returns.csv')
HIST_YEARS = list(range(2000, 2011))          # 2000..2010 (2011+ = canonical walk)

_CACHE = {}


def _panel():
    if 'panel' not in _CACHE:
        _CACHE['panel'] = data_build.load_panel()
    return _CACHE['panel']


def _stocks():
    if 'stocks' not in _CACHE:
        _CACHE['stocks'] = data_build.load_stocks(
            columns=['date', 'permno', 'me', 'ret_fwd'] + repo_config.MOM_FEATURES)
    return _CACHE['stocks']


def _dd_z_pit(p, y0):
    """Point-in-time standardized drawdown: winsor 1/99 + z on train (<y0)."""
    dd = p['DD'].astype(float)
    tr = dd[p['date'] < y0]
    lo, hi = tr.quantile(0.01), tr.quantile(0.99)
    trw = tr.clip(lo, hi)
    mu, sd = trw.mean(), trw.std()
    return ((dd.clip(lo, hi) - mu) / sd).values.reshape(-1, 1)


def _walk_year_dd(year, workers, hmm_seeds, xgb_seeds, n_iter, n_burnin):
    y0, y1 = f'{year}-01-01', f'{year + 1}-01-01'
    panel = _panel()
    p = panel[(panel['date'] >= C.PANEL_START) & (panel['date'] < y1)]
    p = p.dropna(subset=['DD']).reset_index(drop=True)
    z = _dd_z_pit(p, y0)
    tr_mask = (p['date'] < y0).values
    Z_tr, Z_full = z[tr_mask], z
    signs = np.array([-1.0])                      # deeper drawdown = panic
    pi_avg = H.fit_pi(Z_tr, Z_full, signs, hmm_seeds, workers, n_iter, n_burnin)
    pi_df = pd.DataFrame({'date': pd.to_datetime(p['date']), 'pi': pi_avg})

    s = _stocks()
    s = universe.top_n(s[s['date'] < y1], C.UNIVERSE_N)
    st = s.merge(pi_df, on='date', how='left').dropna(subset=['pi'])
    tr = st[st['date'] < y0]
    te = st[st['date'] >= y0].copy()
    if not len(te):
        return None
    te['score_pi'] = model.ensemble_scores(
        tr[model.FEATURES_PI].values.astype(float),
        tr['ret_fwd'].values.astype(float),
        te[model.FEATURES_PI].values.astype(float), xgb_seeds, n_jobs=workers)

    r, _ = portfolio.long_only_top(te, 'score_pi', C.DECILE_FRAC)
    b = portfolio.vw_benchmark(te)
    m12, _ = portfolio.long_only_top(te, 'mom_12', C.DECILE_FRAC)
    pi_m = te[['date', 'pi']].drop_duplicates('date').set_index('date')['pi']
    return pd.DataFrame({'date': r.index, 'strat_ret': r.values,
                         'bench_ret': b.reindex(r.index).values,
                         'mom12_ret': m12.reindex(r.index).values,
                         'pi': pi_m.reindex(r.index).values, 'year': year})


def _done_years():
    if not os.path.exists(HIST_RET):
        return set()
    prev = pd.read_csv(HIST_RET)
    cnt = prev.groupby('year').size()
    return {int(y) for y, n in cnt.items() if n >= 12}


def stage_walk(workers, smoke=False):
    os.makedirs(HIST_DIR, exist_ok=True)
    hmm_seeds, xgb_seeds = C.EVAL_HMM_SEEDS, C.EVAL_XGB_SEEDS
    n_iter, n_burnin, years = C.N_ITER, C.N_BURNIN, HIST_YEARS
    if smoke:
        hmm_seeds, xgb_seeds = C.EVAL_HMM_SEEDS[:3], C.EVAL_XGB_SEEDS[:3]
        n_iter, n_burnin, years = 400, 100, [2005]
    done = set() if smoke else _done_years()
    for year in years:
        if year in done:
            print(f'[hist] SKIP {year} (done)', flush=True)
            continue
        print(f'[hist] === {year} ===', flush=True)
        block = _walk_year_dd(year, workers, hmm_seeds, xgb_seeds,
                              n_iter, n_burnin)
        if block is None or not len(block):
            print(f'[hist]   {year} EMPTY', flush=True)
            continue
        r, b = block['strat_ret'], block['bench_ret']
        if not smoke:
            block.to_csv(HIST_RET, mode='a',
                         header=not os.path.exists(HIST_RET), index=False)
        act = 12 * (r - b).mean()
        print(f'[hist]   {len(block)} months  strat_sharpe={M.sharpe(r):+.2f} '
              f'active={act:+.1%}/yr  mean_pi={block["pi"].mean():.2f}',
              flush=True)
    print('[hist] WALK DONE', flush=True)


# Sub-periods for the stress table (formation months).
SUBPERIODS = [
    ('2000-2002 dot-com', '2000-01-01', '2002-12-31'),
    ('2003-2006 bull', '2003-01-01', '2006-12-31'),
    ('2007-2009 GFC', '2007-01-01', '2009-12-31'),
    ('2009 momentum crash', '2009-03-01', '2009-12-31'),
    ('2010-2019 calm', '2010-01-01', '2019-12-31'),
    ('2020-2025 COVID era', '2020-01-01', '2025-12-31'),
]


def stage_report():
    hist = pd.read_csv(HIST_RET, parse_dates=['date']).sort_values('date')
    walk = pd.read_csv(C.RETURNS_CSV, parse_dates=['date'])
    rr = walk[walk['rule'] == 'rule_r'][['date', 'strat_ret', 'bench_ret', 'pi']]
    rr = rr.assign(mom12_ret=np.nan)   # mom12 not saved in the canonical walk
    full = (pd.concat([hist[['date', 'strat_ret', 'bench_ret', 'mom12_ret', 'pi']],
                       rr], ignore_index=True)
            .drop_duplicates('date').sort_values('date').reset_index(drop=True))
    full.to_csv(os.path.join(HIST_DIR, 'stitched_2000_2025.csv'), index=False)

    def stats(df):
        r, b = df['strat_ret'], df['bench_ret']
        cum = float(np.prod(1 + r) - 1)
        cumb = float(np.prod(1 + b) - 1)
        return dict(n=len(df), strat_sharpe=round(M.sharpe(r), 2),
                    strat_cum=round(cum, 3), strat_mdd=round(M.max_dd(r), 3),
                    bench_sharpe=round(M.sharpe(b), 2), bench_cum=round(cumb, 3))

    rows = [{'period': 'Full 2000-2025', **stats(full)}]
    for name, s, e in SUBPERIODS:
        sub = full[(full['date'] >= s) & (full['date'] <= e)]
        if len(sub):
            rows.append({'period': name, **stats(sub)})
    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(HIST_DIR, 'subperiods.csv'), index=False)
    print(rep.to_string(index=False))
    print(f'\nstitched: {full["date"].min().date()}..{full["date"].max().date()} '
          f'({len(full)} months)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['walk', 'report'])
    ap.add_argument('--workers', type=int, default=7)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.stage == 'walk':
        stage_walk(a.workers, a.smoke)
    else:
        stage_report()


if __name__ == '__main__':
    main()
