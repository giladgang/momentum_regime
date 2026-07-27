"""Applied international replication: UK and JP, drawdown-only regime signal,
long-only top-decile VW within a mechanically chosen large-cap universe,
expanding yearly walk 2011-2025. Mirrors the US applied configuration.

Universe rule (disclosed): per region, the smallest N (multiple of 50) whose
average month-end market-cap coverage over the TRAINING period (<=2010) is at
least 90% of the listed total; capped at 1000 for comparability with the US.

Regime signal: univariate HMM on the region's market drawdown (panel DD_z,
standardized on <=2010 stats as imported, so point-in-time for eval years),
50 seeds, panic = deeper-drawdown state (signs=[-1]). XGB: mom_1..12 + pi,
20 seeds, thesis-frozen hyperparameters, trained on region stocks < Jan Y.

Outputs (paper/results/intl/):
  {uk,jp}_walk_returns.csv   date, strat_ret, bench_ret, mom12_ret, pi, year
  intl_summary.csv           per region: N, months, sharpe/annret/mdd/IR rows
Usage: python -m paper.intl_applied [--smoke]
"""
import argparse
import os

import numpy as np
import pandas as pd

import config as repo_config
from paper import config as C
from paper.src import model
from paper.src import hmm as H
from paper.src import metrics as M

OUT = os.path.join(C.RESULTS, 'intl')
MOMS = [f'mom_{h}' for h in range(1, 13)]
REPORT_END = '2025-11-30'          # match the US window
EVAL_YEARS = list(range(2011, 2026))


def load_region(region):
    s = pd.read_parquet(f'data/{region}_stock_panel.parquet',
                        columns=['gvkey', 'iid', 'date', 'me', 'ret_fwd'] + MOMS)
    s['date'] = pd.to_datetime(s['date'])
    s = s.dropna(subset=['me'])
    s['sid'] = pd.factorize(s['gvkey'].astype(str) + '_' + s['iid'].astype(str))[0]
    m = pd.read_parquet(f'data/{region}_market_panel.parquet',
                        columns=['date', 'DD_z'])
    m['date'] = pd.to_datetime(m['date'])
    return s, m.dropna(subset=['DD_z']).reset_index(drop=True)


def choose_n(s):
    tr = s[s['date'] <= '2010-12-31']
    tot = tr.groupby('date')['me'].sum()
    for n in range(50, 1001, 50):
        top = (tr.sort_values('me', ascending=False)
                 .groupby('date').head(n).groupby('date')['me'].sum())
        if float((top / tot).mean()) >= 0.90:
            return n
    return 1000


def top_n(s, n):
    return (s.sort_values('me', ascending=False)
             .groupby('date', group_keys=False).head(n))


def port(df, score, frac=0.10):
    rets = []
    for d, g in df.groupby('date', sort=True):
        k = max(int(len(g) * frac), 1)
        top = g.sort_values([score, 'sid'], ascending=[False, True]).head(k)
        w = top['me'] / top['me'].sum()
        rets.append((d, float((w * top['ret_fwd']).sum())))
    return pd.Series(dict(rets)).sort_index()


def bench(df):
    return (df.groupby('date')
              .apply(lambda g: float((g['me'] / g['me'].sum() * g['ret_fwd']).sum()))
              .sort_index())


def run_region(region, workers, hmm_seeds, xgb_seeds, n_iter, n_burnin, years):
    s, panel = load_region(region)
    n = choose_n(s)
    print(f'[{region}] universe N={n}', flush=True)
    uni = top_n(s.dropna(subset=['ret_fwd'] + MOMS), n)
    blocks = []
    for year in years:
        y0, y1 = f'{year}-01-01', f'{year + 1}-01-01'
        p = panel[panel['date'] < y1].reset_index(drop=True)
        Z = p[['DD_z']].values.astype(float)
        tr_mask = (p['date'] < y0).values
        pi = H.fit_pi(Z[tr_mask], Z, np.array([-1.0]), hmm_seeds, workers,
                      n_iter, n_burnin)
        pi_df = pd.DataFrame({'date': p['date'], 'pi': pi})
        st = uni[uni['date'] < y1].merge(pi_df, on='date', how='left')
        st = st.dropna(subset=['pi'])
        tr, te = st[st['date'] < y0], st[st['date'] >= y0].copy()
        if not len(te) or len(tr) < 1000:
            continue
        te['score'] = model.ensemble_scores(
            tr[MOMS + ['pi']].values.astype(float),
            tr['ret_fwd'].values.astype(float),
            te[MOMS + ['pi']].values.astype(float), xgb_seeds, n_jobs=workers)
        r = port(te, 'score'); b = bench(te); m12 = port(te, 'mom_12')
        pi_m = te[['date', 'pi']].drop_duplicates('date').set_index('date')['pi']
        blk = pd.DataFrame({'date': r.index, 'strat_ret': r.values,
                            'bench_ret': b.reindex(r.index).values,
                            'mom12_ret': m12.reindex(r.index).values,
                            'pi': pi_m.reindex(r.index).values, 'year': year})
        blocks.append(blk)
        print(f'[{region}] {year}: {len(blk)} mo  IR={M.ir(r, b.reindex(r.index)):+.2f}',
              flush=True)
    out = pd.concat(blocks, ignore_index=True)
    out.to_csv(os.path.join(OUT, f'{region}_walk_returns.csv'), index=False)
    return n, out


def summarize(region, n, out):
    d = out[out['date'] <= REPORT_END]
    rows = []
    b = d.set_index('date')['bench_ret']
    for name, col in [('benchmark', 'bench_ret'), ('mom_12_1', 'mom12_ret'),
                      ('strategy', 'strat_ret')]:
        r = d.set_index('date')[col]
        rows.append(dict(region=region, universe_n=n, series=name, n_months=len(r),
                         ann_ret=round(float((1 + r.mean()) ** 12 - 1), 4),
                         sharpe=round(M.sharpe(r), 2),
                         max_dd=round(M.max_dd(r), 3),
                         ir=(np.nan if name == 'benchmark'
                             else round(M.ir(r, b), 2))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--workers', type=int, default=7)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    hmm_seeds, xgb_seeds = C.EVAL_HMM_SEEDS, C.EVAL_XGB_SEEDS
    n_iter, n_burnin, years = C.N_ITER, C.N_BURNIN, EVAL_YEARS
    if a.smoke:
        hmm_seeds, xgb_seeds = C.EVAL_HMM_SEEDS[:3], C.EVAL_XGB_SEEDS[:3]
        n_iter, n_burnin, years = 400, 100, [2020]
    rows = []
    for region in ['uk', 'jp']:
        n, out = run_region(region, a.workers, hmm_seeds, xgb_seeds,
                            n_iter, n_burnin, years)
        rows += summarize(region, n, out)
    rep = pd.DataFrame(rows)
    if not a.smoke:
        rep.to_csv(os.path.join(OUT, 'intl_summary.csv'), index=False)
    print(rep.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
