"""Single-feature HMM ablation walk (thesis regime-signal ablation analog).

Runs the identical expanding-window walk with a NON-DD single-feature HMM
(e.g. REL_N only): same seeds, budgets, universe, portfolio. Panic state is
identified by the crisis-sign rule (dimension-general); state init anchors on
the feature itself (below-median), matching the canonical convention with
feats[0] as anchor.

Output: paper/results/ablations/abl_<feat>_returns.csv
Usage:  .venv/bin/python -m paper.ablation_walk --feat REL_N --workers 7
"""
import argparse
import os
import sys

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

import config as repo_config                                    # noqa: E402
from paper import config as C                                   # noqa: E402
from paper.src import data_build, universe, model, portfolio    # noqa: E402
from paper.src import hmm as H                                  # noqa: E402
from paper.src import metrics as M                              # noqa: E402

OUT_DIR = os.path.join(C.RESULTS, 'ablations')


def walk_year_feat(year, feat_z, workers, hmm_seeds, xgb_seeds,
                   n_iter, n_burnin):
    y0, y1 = f'{year}-01-01', f'{year + 1}-01-01'
    panel = data_build.load_panel()
    p = panel[(panel['date'] >= C.PANEL_START) & (panel['date'] < y1)]
    p = p.dropna(subset=[feat_z]).reset_index(drop=True)
    tr_p = p[p['date'] < y0]
    Z_tr = tr_p[[feat_z]].values.astype(float)
    Z_full = p[[feat_z]].values.astype(float)
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
    te['score'] = model.ensemble_scores(
        tr[model.FEATURES_PI].values.astype(float),
        tr['ret_fwd'].values.astype(float),
        te[model.FEATURES_PI].values.astype(float), xgb_seeds, n_jobs=workers)
    r, _ = portfolio.long_only_top(te, 'score', C.DECILE_FRAC)
    b = portfolio.vw_benchmark(te)
    pi_m = te[['date', 'pi']].drop_duplicates('date').set_index('date')['pi']
    out = pd.DataFrame({'date': r.index, 'strat_ret': r.values,
                        'bench_ret': b.reindex(r.index).values,
                        'pi': pi_m.reindex(r.index).values})
    out['year'] = year
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat', required=True, help='e.g. REL_N (no _z suffix)')
    ap.add_argument('--workers', type=int, default=7)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, f'abl_{a.feat.lower()}_returns.csv')
    done = set()
    if os.path.exists(out_csv):
        done = set(pd.read_csv(out_csv)['year'].unique())
    for year in C.EVAL_YEARS:
        if year in done:
            print(f'[abl] SKIP {year}', flush=True)
            continue
        print(f'[abl] === {year} {a.feat} ===', flush=True)
        block = walk_year_feat(year, a.feat + '_z', a.workers,
                               C.EVAL_HMM_SEEDS, C.EVAL_XGB_SEEDS,
                               C.N_ITER, C.N_BURNIN)
        block.to_csv(out_csv, mode='a', header=not os.path.exists(out_csv),
                     index=False)
        b = block.set_index('date')
        print(f"[abl]   IR={M.ir(b['strat_ret'], b['bench_ret']):+.2f}",
              flush=True)
    print('[abl] DONE', flush=True)


if __name__ == '__main__':
    main()
