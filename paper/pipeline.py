"""Applied-study pipeline runner. Stages: data, universe_check, folds, walk,
report. Resume-safe; spawn-pool parallelism over units. cwd is forced to the
repo root so all relative paths resolve.
"""
import argparse
import os
import sys
import time

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from paper import config as C                                   # noqa: E402


def _run_cell_unit(args):
    combo, fold, seed, n_iter, n_burnin, xgb_seeds = args
    from paper.src import selection as S
    try:
        row = S.run_cell(combo, fold, seed, n_iter, n_burnin, xgb_seeds,
                         n_jobs=1)
        return ('ok', row) if row is not None else ('empty', str((combo, fold)))
    except Exception as e:                                      # noqa: BLE001
        return ('failed', f'{combo}/{fold[0]}/s{seed}: {e!r}')


def stage_folds(workers, smoke=False):
    from paper.src import selection as S
    combos = list(pd.read_csv(C.POOL_CSV)['combo'])
    folds = C.BIENNIAL_FOLDS + C.ANNUAL_FOLDS
    seeds = C.SEL_HMM_SEEDS
    n_iter, n_burnin, xgb_seeds = C.N_ITER, C.N_BURNIN, C.SEL_XGB_SEEDS
    out = C.FOLD_CELLS_CSV
    if smoke:
        combos, folds, seeds = combos[:2], folds[:1], seeds[:1]
        n_iter, n_burnin, xgb_seeds = 300, 100, C.SEL_XGB_SEEDS[:2]
        out = out.replace('.csv', '_smoke.csv')
    done = S.completed_cells(out)
    units = [(c, f, s, n_iter, n_burnin, xgb_seeds)
             for c in combos for f in folds for s in seeds
             if (c, f[0], s) not in done]
    print(f'[folds] {len(units)} units to run '
          f'({len(combos)}x{len(folds)}x{len(seeds)}, done={len(done)})',
          flush=True)
    from multiprocessing import get_context
    ctx = get_context('spawn')
    t0, n_ok = time.time(), 0
    with ctx.Pool(workers) as pool:
        for status, payload in pool.imap_unordered(_run_cell_unit, units,
                                                   chunksize=1):
            if status == 'ok':
                S.append_row(out, payload)
                n_ok += 1
                if n_ok % 25 == 0:
                    rate = (time.time() - t0) / n_ok
                    eta = rate * (len(units) - n_ok) / 3600
                    print(f'  [{n_ok}/{len(units)}] {rate:.0f}s/unit '
                          f'ETA {eta:.1f}h', flush=True)
            else:
                print(f'  [{status.upper()}] {payload}', flush=True)
    print(f'[folds] DONE ok={n_ok}/{len(units)} '
          f'({(time.time() - t0) / 60:.1f}m)', flush=True)


def stage_data():
    from paper.src import data_build
    data_build.build()


def stage_universe_check():
    from paper.src import data_build, universe
    s = data_build.load_stocks(columns=['date', 'permno', 'me'])
    m = universe.top_n(s, C.UNIVERSE_N)
    cov = universe.cap_coverage(s, m)
    n = m.groupby('date').size()
    print(f'[universe] months={len(n)} | names/month min={n.min()} '
          f'max={n.max()} | cap coverage median={cov.median():.3f} '
          f'min={cov.min():.3f} | gate {C.COVERAGE_MIN}')
    assert cov.median() > C.COVERAGE_MIN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['data', 'universe_check', 'folds', 'walk',
                             'report'])
    ap.add_argument('--workers', type=int, default=7)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    os.makedirs(C.RESULTS, exist_ok=True)
    if a.stage == 'data':
        stage_data()
    elif a.stage == 'universe_check':
        stage_universe_check()
    elif a.stage == 'folds':
        stage_folds(a.workers, a.smoke)
    else:
        from paper import walk_report
        (walk_report.stage_walk if a.stage == 'walk'
         else walk_report.stage_report)(a.workers, a.smoke)


if __name__ == '__main__':
    main()
