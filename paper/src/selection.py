"""Applied-objective selection CV: per-(combo, fold, hmm_seed) validation IR
of the long-only top-decile portfolio vs the VW top-1000 benchmark.

Worker pools parallelize over cells; run_cell is strictly single-process.
HMM uses the canonical posterior-mean machinery (paper.src.hmm), NOT the
thesis selection harness's last-draw shortcut — selection matches deployment
(disclosed deviation). rule_r tie-break: 1-SE eligibility, min feature count,
then higher mean, then combo name (ESS tie-break undefined for the IR
objective — disclosed).
"""
import os
import time

import numpy as np
import pandas as pd

import config as repo_config
from paper import config as C
from paper.src import data_build, universe, model, portfolio
from paper.src import hmm as H
from paper.src import metrics as M

_CACHE = {}


def _data():
    if 'panel' not in _CACHE:
        _CACHE['panel'] = data_build.load_panel()
        s = data_build.load_stocks(
            columns=['date', 'permno', 'me', 'ret_fwd']
                    + repo_config.MOM_FEATURES)
        _CACHE['stocks'] = universe.top_n(s, C.UNIVERSE_N)
    return _CACHE['panel'], _CACHE['stocks']


def run_cell(combo, fold, hmm_seed, n_iter=None, n_burnin=None,
             xgb_seeds=None, n_jobs=1):
    n_iter = n_iter or C.N_ITER
    n_burnin = n_burnin or C.N_BURNIN
    xgb_seeds = xgb_seeds or C.SEL_XGB_SEEDS
    fold_id, val_start, val_end = fold
    feats_z = [f + '_z' for f in combo.split('+')]
    assert feats_z[0] == 'DD_z', combo

    panel, stocks = _data()
    p = panel[(panel['date'] >= C.PANEL_START) & (panel['date'] < val_end)]
    p = p.dropna(subset=feats_z).reset_index(drop=True)
    tr_p = p[p['date'] < val_start]
    Z_tr = tr_p[feats_z].values.astype(float)
    Z_full = p[feats_z].values.astype(float)
    # Selection folds use the DD-anchored panic label (trainonly convention:
    # panic = deeper-drawdown state; dimension-safe for folds whose train
    # window predates the crisis windows). signs=[-1,0,..] implements it
    # exactly through fit_seed's sign-score rule.
    signs = np.zeros(len(feats_z))
    signs[0] = -1.0
    t0 = time.time()
    H._init_worker(Z_tr, Z_full, signs, n_iter, n_burnin)
    _, pi, _, _ = H.fit_seed(hmm_seed)
    hmm_sec = time.time() - t0

    pi_df = pd.DataFrame({'date': pd.to_datetime(p['date']), 'pi': pi})
    st = stocks[stocks['date'] < val_end].merge(pi_df, on='date', how='left')
    st = st.dropna(subset=['pi'])
    tr = st[st['date'] < val_start]
    te = st[st['date'] >= val_start].copy()
    if not len(te):
        return None
    t0 = time.time()
    te['score'] = model.ensemble_scores(
        tr[model.FEATURES_PI].values.astype(float),
        tr['ret_fwd'].values.astype(float),
        te[model.FEATURES_PI].values.astype(float), xgb_seeds, n_jobs)
    xgb_sec = time.time() - t0
    r, _ = portfolio.long_only_top(te, 'score', C.DECILE_FRAC)
    b = portfolio.vw_benchmark(te)
    return {'combo': combo, 'n_features': combo.count('+') + 1,
            'fold': fold_id, 'hmm_seed': hmm_seed,
            'n_xgb_seeds': len(xgb_seeds), 'val_ir': M.ir(r, b),
            'val_sharpe': M.sharpe(r), 'n_val_months': len(r),
            'hmm_sec': round(hmm_sec, 1), 'xgb_sec': round(xgb_sec, 1)}


def completed_cells(path):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(zip(df['combo'], df['fold'], df['hmm_seed']))


def append_row(path, row):
    pd.DataFrame([row]).to_csv(path, mode='a',
                               header=not os.path.exists(path), index=False)


def select_years(cells, years=None):
    per_fold = cells.groupby(['combo', 'fold'])['val_ir'].mean().unstack()
    D = pd.Series({c: c.count('+') + 1 for c in per_fold.index})
    rows = []
    for Y in (years or C.EVAL_YEARS):
        folds = [f for f in per_fold.columns if C.VAL_END.get(f, 9999) <= Y]
        sub = per_fold[folds]
        mean = sub.mean(axis=1)
        best = mean.idxmax()
        n = sub.loc[best].notna().sum()
        se = sub.loc[best].std() / np.sqrt(n)
        elig = mean[mean >= mean[best] - se]
        rows.append({'year': Y, 'rule': 'argmax', 'combo': best,
                     'cv_mean': mean[best], 'n_folds': len(folds),
                     'n_eligible': len(elig)})
        dmin = D[elig.index].min()
        cand = elig[D[elig.index] == dmin].sort_values(ascending=False)
        choice = sorted(cand[cand == cand.max()].index)[0]
        rows.append({'year': Y, 'rule': 'rule_r', 'combo': choice,
                     'cv_mean': mean[choice], 'n_folds': len(folds),
                     'n_eligible': len(elig)})
    return pd.DataFrame(rows)
