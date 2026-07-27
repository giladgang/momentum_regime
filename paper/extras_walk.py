"""Risk-aversion sigma-model + per-horizon SHAP, applied walk (rule_r, DD-only).

One pass over walk years 2011-2025, reusing a single per-year HMM refit
(panel DD_z, 50 seeds, deeper-drawdown = panic, identical to walk_report):

  (a) sigma model: XGB (10 seeds, frozen hyperparameters) trained on
      sigma_fwd_12m (rolling 12-m std of ret_adj shifted -12; thesis
      two_model_dual_util.py convention) with features mom_1..12 + pi +
      trail_sigma (12-m trailing std, shifted 1, min 6 obs). Training rows
      restricted to dates <= Dec Y-2 so no forward-vol label crosses into
      the trading year. Predicts sigma_hat for year-Y universe months.
  (b) SHAP: XGB r-model retrained at 3 seeds (walk-consistent budget, as
      shap_pi_share_by_year), TreeExplainer per seed on the year's eval
      months (<= 20k rows sampled); per-feature |SHAP| sums saved, plus
      long-leg (top-decile by the year's saved score_pi) sums.

Aggregation:
  results/tables/risk_aversion_sweep.csv  per gamma: sharpe/annret/vol/mdd/IR
  manuscript/plots/risk_aversion_applied.{pdf,png}
  results/tables/shap_horizon_applied.csv per-feature shares overall/calm/panic/long-leg
  manuscript/plots/shap_per_horizon_applied.{pdf,png}
Usage: python -m paper.extras_walk [--smoke]
"""
import argparse
import os

import numpy as np
import pandas as pd

import config as repo_config
from paper import config as C
from paper.src import data_build, universe, model
from paper.src import hmm as H
from paper.src import metrics as M

OUT_XTRA = os.path.join(C.RESULTS, 'extras')
MOMS = list(repo_config.MOM_FEATURES)
EPS, SIGMA_FLOOR = 0.01, 0.01
GAMMAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
          0.6, 0.7, 0.8, 0.9, 1.0]
SIGMA_SEEDS = list(range(1, 11))
SHAP_SEEDS = [1, 2, 3]
SHAP_SAMPLE = 20_000


def _stocks_with_sigma():
    s = data_build.load_stocks(
        columns=['date', 'permno', 'me', 'ret_adj', 'ret_fwd'] + MOMS)
    s = s.sort_values(['permno', 'date'])
    g = s.groupby('permno')['ret_adj']
    s['trail_sigma'] = g.transform(
        lambda x: x.shift(1).rolling(12, min_periods=6).std())
    roll = g.transform(lambda x: x.rolling(12, min_periods=6).std())
    s['sigma_fwd_12m'] = roll.groupby(s['permno']).shift(-12)
    return s


def year_pi(panel, year, hmm_seeds, workers, n_iter, n_burnin):
    y0, y1 = f'{year}-01-01', f'{year + 1}-01-01'
    p = panel[(panel['date'] >= C.PANEL_START) & (panel['date'] < y1)]
    p = p.dropna(subset=['DD_z']).reset_index(drop=True)
    Z = p[['DD_z']].values.astype(float)
    tr = (p['date'] < y0).values
    pi = H.fit_pi(Z[tr], Z, np.array([-1.0]), hmm_seeds, workers,
                  n_iter, n_burnin)
    return pd.DataFrame({'date': pd.to_datetime(p['date']), 'pi': pi})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--workers', type=int, default=7)
    a = ap.parse_args()
    os.makedirs(OUT_XTRA, exist_ok=True)
    hmm_seeds, n_iter, n_burnin = C.EVAL_HMM_SEEDS, C.N_ITER, C.N_BURNIN
    sigma_seeds, shap_seeds, years = SIGMA_SEEDS, SHAP_SEEDS, C.EVAL_YEARS
    shap_sample = SHAP_SAMPLE
    if a.smoke:
        hmm_seeds, n_iter, n_burnin = C.EVAL_HMM_SEEDS[:3], 400, 100
        sigma_seeds, shap_seeds, years, shap_sample = [1, 2], [1], [2020], 5000

    panel = data_build.load_panel()
    stocks = _stocks_with_sigma()
    med = stocks['trail_sigma'].median()
    stocks['trail_sigma'] = stocks['trail_sigma'].fillna(med).clip(lower=SIGMA_FLOOR)
    uni = universe.top_n(stocks, C.UNIVERSE_N)

    import shap as shap_lib
    from xgboost import XGBRegressor
    sig_blocks, shap_rows = [], []
    for year in years:
        y0, y1 = f'{year}-01-01', f'{year + 1}-01-01'
        pi_df = year_pi(panel, year, hmm_seeds, a.workers, n_iter, n_burnin)
        st = uni[uni['date'] < y1].merge(pi_df, on='date', how='left')
        st = st.dropna(subset=['pi'] + MOMS + ['ret_fwd'])
        tr, te = st[st['date'] < y0], st[st['date'] >= y0].copy()
        if not len(te):
            continue
        # (a) sigma model: labels only where the forward window stays < Jan Y
        cut = pd.Timestamp(f'{year - 1}-01-01')       # t <= Dec Y-2
        trs = tr[(tr['date'] < cut) & tr['sigma_fwd_12m'].notna()]
        feats = MOMS + ['pi', 'trail_sigma']
        Xtr, ytr = trs[feats].values.astype(float), trs['sigma_fwd_12m'].values
        Xte = te[feats].values.astype(float)
        pred = np.zeros(len(te))
        for sd in sigma_seeds:
            mm = XGBRegressor(n_estimators=repo_config.N_ESTIMATORS,
                              max_depth=repo_config.MAX_DEPTH,
                              learning_rate=repo_config.LEARNING_RATE,
                              subsample=repo_config.SUBSAMPLE,
                              colsample_bytree=repo_config.COLSAMPLE,
                              tree_method='hist', random_state=sd,
                              verbosity=0, n_jobs=a.workers)
            mm.fit(Xtr, ytr)
            pred += mm.predict(Xte)
        te['sigma_hat'] = np.clip(pred / len(sigma_seeds), SIGMA_FLOOR, None)
        sig_blocks.append(te[['date', 'permno', 'me', 'sigma_hat', 'pi',
                              'ret_fwd']].assign(year=year))
        # (b) SHAP on the r-model (3-seed retrain, walk-consistent)
        rfeats = MOMS + ['pi']
        Xr, yr = tr[rfeats].values.astype(float), tr['ret_fwd'].values.astype(float)
        Xe = te[rfeats].values.astype(float)
        idx = np.random.RandomState(42).choice(
            len(Xe), size=min(shap_sample, len(Xe)), replace=False)
        # long-leg flag from the saved production score
        xsec = pd.read_parquet(os.path.join(C.XSEC_DIR, f'xsec_{year}_DD.parquet'),
                               columns=['date', 'permno', 'score_pi'])
        te_key = te[['date', 'permno']].merge(
            xsec, on=['date', 'permno'], how='left')
        thresh = (te_key.groupby('date')['score_pi']
                  .transform(lambda x: x.quantile(0.9)))
        is_long = (te_key['score_pi'] >= thresh).values
        acc = np.zeros((len(idx), len(rfeats)))
        for sd in shap_seeds:
            mm = XGBRegressor(n_estimators=repo_config.N_ESTIMATORS,
                              max_depth=repo_config.MAX_DEPTH,
                              learning_rate=repo_config.LEARNING_RATE,
                              subsample=repo_config.SUBSAMPLE,
                              colsample_bytree=repo_config.COLSAMPLE,
                              tree_method='hist', random_state=sd,
                              verbosity=0, n_jobs=a.workers)
            mm.fit(Xr, yr)
            acc += np.abs(shap_lib.TreeExplainer(mm).shap_values(Xe[idx]))
        acc /= len(shap_seeds)
        sub_pi = te['pi'].values[idx]
        sub_long = is_long[idx]
        for scope, mask in [('overall', np.ones(len(idx), bool)),
                            ('calm', sub_pi < 0.5), ('panic', sub_pi >= 0.5),
                            ('long_leg', sub_long)]:
            if mask.sum() == 0:
                continue
            sums = acc[mask].sum(axis=0)
            shap_rows.append(dict(year=year, scope=scope, n=int(mask.sum()),
                                  **{f: float(v) for f, v in zip(rfeats, sums)}))
        print(f'[extras] {year}: sigma n_tr={len(trs)}, shap n={len(idx)}',
              flush=True)

    sig = pd.concat(sig_blocks, ignore_index=True)
    shp = pd.DataFrame(shap_rows)
    if a.smoke:
        print(sig.head(3).to_string(), '\n', shp.to_string(index=False))
        print('[extras] SMOKE OK'); return
    sig.to_parquet(os.path.join(OUT_XTRA, 'sigma_hat_walk.parquet'), index=False)
    shp.to_csv(os.path.join(OUT_XTRA, 'shap_sums_by_year.csv'), index=False)

    # ── risk-aversion sweep from saved r-hat (xsec) + sigma_hat ──────────────
    xs = []
    for year in C.EVAL_YEARS:
        p = os.path.join(C.XSEC_DIR, f'xsec_{year}_DD.parquet')
        if os.path.exists(p):
            xs.append(pd.read_parquet(p, columns=['date', 'permno', 'me',
                                                  'score_pi', 'ret_fwd']))
    xs = pd.concat(xs, ignore_index=True)
    xs = xs.merge(sig[['date', 'permno', 'sigma_hat']], on=['date', 'permno'],
                  how='inner')
    w = pd.read_csv(C.RETURNS_CSV, parse_dates=['date'])
    bench = (w[w['rule'] == 'rule_r'].set_index('date')['bench_ret'].sort_index())
    util_base = np.sign(xs['score_pi']) * np.log1p(np.abs(xs['score_pi']) / EPS)
    logsig = np.log(xs['sigma_hat'])
    rows = []
    for gam in GAMMAS:
        xs['_u'] = util_base - gam * logsig
        r = []
        for d, g in xs.groupby('date', sort=True):
            k = max(int(len(g) * C.DECILE_FRAC), 1)
            top = g.sort_values(['_u', 'permno'], ascending=[False, True]).head(k)
            wgt = top['me'] / top['me'].sum()
            r.append((d, float((wgt * top['ret_fwd']).sum())))
        r = pd.Series(dict(r)).sort_index()
        b = bench.reindex(r.index)
        rows.append(dict(gamma=gam, sharpe=round(M.sharpe(r), 3),
                         ann_ret=round(float((1 + r.mean()) ** 12 - 1), 4),
                         ann_vol=round(float(r.std() * np.sqrt(12)), 4),
                         max_dd=round(M.max_dd(r), 3), ir=round(M.ir(r, b), 3)))
        print(f'[sweep] gamma={gam}: {rows[-1]}', flush=True)
    sw = pd.DataFrame(rows)
    sw.to_csv(os.path.join(C.RESULTS, 'tables', 'risk_aversion_sweep.csv'),
              index=False)
    print(sw.to_string(index=False), flush=True)
    print('[extras] DONE', flush=True)


if __name__ == '__main__':
    main()
