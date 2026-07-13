"""Live-2026 appendix: out-of-sample continuation on the Compustat splice.

Primary results stay CRSP-pure (2011..2025-11). This runner trades the six
live formation months 2025-12..2026-05 with models trained ONLY on CRSP data
(< 2025-12-01: every training label is a CRSP return; Compustat enters
exclusively through the traded months' forward returns, universe caps, and
the 2026 panel feature rows). Splice validated in
experiments/2026-07-13-live2026-compustat.py --stage validate (top-1000
return corr 1.0000, 99.93% within 10bp).

Selection for trading-2026 = same pre-committed rules on folds with
validation end <= Jan 2026 (includes fold 115, CRSP-pure).

Outputs (separate from primary files):
  paper/results/live2026/selections_2026.csv
  paper/results/live2026/live_returns.csv
  paper/results/live2026/xsec_2026_<combo>.parquet
  paper/results/live2026/live_summary.md

Usage:
  .venv/bin/python -m paper.live2026 --stage walk --workers 7
  .venv/bin/python -m paper.live2026 --stage report
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

import config as repo_config                                    # noqa: E402
from paper import config as C                                   # noqa: E402
from paper.src import data_build, universe, model, portfolio    # noqa: E402
from paper.src import hmm as H                                  # noqa: E402
from paper.src import metrics as M                              # noqa: E402

LIVE_SRC = 'experiments/results/live2026'
OUT_DIR = os.path.join(C.RESULTS, 'live2026')
PANEL_LIVE = f'{LIVE_SRC}/panel_live.parquet'
STOCKS_LIVE = f'{LIVE_SRC}/stocks_live.parquet'
SEL_CSV = os.path.join(OUT_DIR, 'selections_2026.csv')
RET_CSV = os.path.join(OUT_DIR, 'live_returns.csv')
TRAIN_END = '2025-12-01'      # training data strictly CRSP (labels <= Dec-25)
SCORE_END = '2026-06-01'      # formation months 2025-12..2026-05


def _live_frames():
    panel = pd.read_parquet(PANEL_LIVE)
    panel['date'] = pd.to_datetime(panel['date'])
    raw = pd.read_parquet(STOCKS_LIVE,
                          columns=['permno', 'date', 'ret_adj', 'prc',
                                   'shrcd', 'exchcd', 'me'])
    raw['date'] = pd.to_datetime(raw['date'])
    stocks = data_build.build_stock_frame(raw, max_date=panel['date'].max())
    stocks = universe.top_n(stocks, C.UNIVERSE_N)
    return panel, stocks


def stage_walk(workers, smoke=False):
    from paper.src import selection as S
    os.makedirs(OUT_DIR, exist_ok=True)
    cells = pd.read_csv(C.FOLD_CELLS_CSV)
    n115 = int((cells['fold'] == 115).sum())
    assert n115 == 75, f'fold 115 incomplete: {n115}/75 — rerun S2 first'
    sel = S.select_years(cells, years=[2026])
    sel.to_csv(SEL_CSV, index=False)
    print(sel.to_string(index=False), flush=True)

    hmm_seeds, xgb_seeds = C.EVAL_HMM_SEEDS, C.EVAL_XGB_SEEDS
    n_iter, n_burnin = C.N_ITER, C.N_BURNIN
    if smoke:
        hmm_seeds, xgb_seeds = hmm_seeds[:3], xgb_seeds[:3]
        n_iter, n_burnin = 400, 100

    panel, stocks = _live_frames()
    done = set()
    if os.path.exists(RET_CSV):
        done = set(pd.read_csv(RET_CSV)['rule'])
    grouped = sel.groupby('combo')['rule'].apply(list).reset_index()
    for _, g in grouped.iterrows():
        combo = g['combo']
        rules = [ru for ru in g['rule'] if ru not in done]
        if not rules:
            print(f'[live] SKIP {combo} (done)', flush=True)
            continue
        print(f'[live] === 2026 {combo} (rules {rules}) ===', flush=True)
        feats_z = [f + '_z' for f in combo.split('+')]
        assert feats_z[0] == 'DD_z', combo
        p = panel[(panel['date'] >= C.PANEL_START)
                  & (panel['date'] < SCORE_END)]
        p = p.dropna(subset=feats_z).reset_index(drop=True)
        tr_p = p[p['date'] < TRAIN_END]
        Z_tr = tr_p[feats_z].values.astype(float)
        Z_full = p[feats_z].values.astype(float)
        signs = H.crisis_signs(Z_tr, pd.to_datetime(tr_p['date']).values,
                               repo_config.CRISIS_WINDOWS)
        pi_avg = H.fit_pi(Z_tr, Z_full, signs, hmm_seeds, workers,
                          n_iter, n_burnin)
        pi_df = pd.DataFrame({'date': pd.to_datetime(p['date']),
                              'pi': pi_avg})
        st = stocks.merge(pi_df, on='date', how='left').dropna(subset=['pi'])
        tr = st[st['date'] < TRAIN_END]
        te = st[(st['date'] >= TRAIN_END) & (st['date'] < SCORE_END)].copy()
        te['score_pi'] = model.ensemble_scores(
            tr[model.FEATURES_PI].values.astype(float),
            tr['ret_fwd'].values.astype(float),
            te[model.FEATURES_PI].values.astype(float), xgb_seeds,
            n_jobs=workers)
        te['score_nopi'] = model.ensemble_scores(
            tr[model.FEATURES_NOPI].values.astype(float),
            tr['ret_fwd'].values.astype(float),
            te[model.FEATURES_NOPI].values.astype(float), xgb_seeds,
            n_jobs=workers)
        cols = ['date', 'permno', 'me', 'pi', 'score_pi', 'score_nopi',
                'mom_12', 'ret_fwd']
        xp = os.path.join(OUT_DIR,
                          f"xsec_2026_{combo.replace('+', '_')}.parquet")
        te[cols].to_parquet(xp + '.tmp', index=False)
        os.replace(xp + '.tmp', xp)
        r, _ = portfolio.long_only_top(te, 'score_pi', C.DECILE_FRAC)
        b = portfolio.vw_benchmark(te)
        pi_m = (te[['date', 'pi']].drop_duplicates('date')
                .set_index('date')['pi'])
        block = pd.DataFrame({'date': r.index, 'strat_ret': r.values,
                              'bench_ret': b.reindex(r.index).values,
                              'pi': pi_m.reindex(r.index).values})
        block['combo'] = combo
        for ru in rules:
            out = block.copy()
            out['rule'] = ru
            out[['date', 'rule', 'combo', 'strat_ret', 'bench_ret',
                 'pi']].to_csv(RET_CSV, mode='a',
                               header=not os.path.exists(RET_CSV),
                               index=False)
        print(f'[live]   {len(r)} months  cum active='
              f'{((1 + r).prod() - (1 + b.reindex(r.index)).prod()):+.2%}',
              flush=True)
    print('[live] DONE', flush=True)


def stage_report(workers=1, smoke=False):
    rets = pd.read_csv(RET_CSV, parse_dates=['date'])
    sel = pd.read_csv(SEL_CSV)
    frames = []
    for combo in sel[sel['rule'] == 'argmax']['combo'].unique():
        xp = os.path.join(OUT_DIR,
                          f"xsec_2026_{combo.replace('+', '_')}.parquet")
        frames.append(pd.read_parquet(xp))
    x = pd.concat(frames, ignore_index=True).drop_duplicates(
        ['date', 'permno'])
    x['date'] = pd.to_datetime(x['date'])
    r_nopi, _ = portfolio.long_only_top(x, 'score_nopi', C.DECILE_FRAC)
    r_mom, _ = portfolio.long_only_top(x, 'mom_12', C.DECILE_FRAC)
    bench = portfolio.vw_benchmark(x)

    lines = ['# Live-2026 appendix: out-of-sample continuation '
             '(Compustat splice, GROSS)\n',
             'Models trained on CRSP data only (< 2025-12); traded months '
             '2025-12..2026-05. Six months — descriptive, not inferential. '
             'Splice gates: top-1000 return corr 1.0000, 99.93% within '
             '10bp; VW overlap corr 0.995.\n',
             '## Monthly active returns vs VW top-1000 benchmark\n']
    tbl = {}
    for ru, g in rets.groupby('rule'):
        s = g.set_index('date')['strat_ret'].sort_index()
        tbl[f'xgb_pi_{ru}'] = M.active(s, bench)
    tbl['xgb_nopi'] = M.active(r_nopi, bench)
    tbl['mom_12_1'] = M.active(r_mom, bench)
    tbl['benchmark_ret'] = bench
    pi_m = x[['date', 'pi']].drop_duplicates('date').set_index('date')['pi']
    tbl['pi'] = pi_m
    df = pd.DataFrame(tbl).sort_index()
    lines.append(df.round(4).to_string())
    lines.append('\n## Cumulative (6 months)\n')
    for k in ['xgb_pi_argmax', 'xgb_pi_rule_r', 'xgb_nopi', 'mom_12_1']:
        if k in df.columns:
            strat = df[k] + df['benchmark_ret']
            cum_s = float((1 + strat).prod() - 1)
            cum_b = float((1 + df['benchmark_ret']).prod() - 1)
            lines.append(f'{k}: total {cum_s:+.2%} vs benchmark {cum_b:+.2%} '
                         f'-> active {cum_s - cum_b:+.2%}')
    lines.append('\nSelections (fold evidence through validate-2025):')
    lines.append(sel.to_string(index=False))
    text = '\n'.join(lines)
    with open(os.path.join(OUT_DIR, 'live_summary.md'), 'w') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['walk', 'report'], required=True)
    ap.add_argument('--workers', type=int, default=7)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    (stage_walk if a.stage == 'walk' else stage_report)(a.workers, a.smoke)
