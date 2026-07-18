"""The band itself as a LEARNED decision surface (XGBoost, expanding window).

Every no-trade band is a hold/sell rule for held-but-drifted stocks. Here that
rule IS the model: an XGBoost classifier trained on ex-post deferral value.

Decision points: stocks held by a static-band (E=20) book whose rank has
drifted outside the top decile. Ex-post label GOOD_HOLD = 1 if
  ret_fwd(stock) - ret_fwd(marginal replacement, the k-th ranked name)
  + spread_saved (round trip avoided now) > 0.
Features (PIT): rank_pct, lagged relative spread, mom_z, log size, pi, months
held. Trained expanding-window: for OOS year y, fit on decision points with
date < y-01-01; the learned band for year y is eband_it = 100 if
P(GOOD_HOLD)>0.5 else 10, fed to the stock_var_band engine policy.

Honesty guards: (i) OOS AUC of the classifier reported per strategy -- if ~0.5
the model has no signal and the "learned band" is noise; (ii) compared to the
static E=20 band AND to the uniform frontier at MATCHED turnover (cost); (iii)
net IR delta reported with the caveat that gross composition noise dominates.

Usage: .venv/bin/python -m paper.learned_band
Output: paper/results/banding_study/learned_band.{md,csv}
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)
warnings.filterwarnings('ignore')

from paper import config as C                                   # noqa: E402
from paper import execution as X                                # noqa: E402
from paper import banding_study as B                            # noqa: E402
from paper.src import signals                                   # noqa: E402

FEATS = ['rank_pct', 'rel_sp', 'mom_z', 'log_me', 'pi']
E_STATIC = 20
START_OOS = 2001


def _prep(s, sp, month_med):
    x = signals.build(s)
    x = x[(x['date'] >= C.SWEEP_START) & (x['date'] <= '2024-11-30')]
    x = x.reset_index(drop=True)
    x['ym'] = x['date'].dt.to_period('M')
    lag = sp[['permno', 'ym', 'hs']].copy()
    lag['ym'] = lag['ym'] + 1
    x = x.merge(lag.rename(columns={'hs': 'hs_lag'}), on=['permno', 'ym'],
                how='left')
    med = x.groupby('date')['hs_lag'].transform('median')
    x['rel_sp'] = (x['hs_lag'].astype('float64') / med).clip(0.25, 4).fillna(1.0)
    x['hs_now'] = x['hs_lag'].astype('float64').fillna(
        x['ym'].map(month_med)).fillna(float(month_med.median()))
    x['mom_z'] = x.groupby('date')['mom_12'].transform(
        lambda v: ((v - v.mean()) / v.std() if v.std() > 0 else 0.0)
    ).clip(-5, 5).fillna(0.0)
    x['log_me'] = np.log(x['me'].astype('float64'))
    g = x.groupby('date')['score']
    x['rank_pct'] = g.rank(ascending=False, pct=True)
    return x


def _decision_points(x):
    """Held-drifted stock-months of the static E=20 book, with labels."""
    _, led = X.simulate(x, ('nmv_band', (E_STATIC, E_STATIC)),
                        score_col='score')
    ev = led[led['cause'].isin(['entry', 'exit_rank', 'exit_univ'])]
    ev = ev.sort_values('date')
    hold, age, rows = set(), {}, []
    kth = (x.sort_values(['date', 'rank_pct'])
           .groupby('date').apply(lambda gg: gg[gg['rank_pct'] <= 0.10]
                                  ['ret_fwd'].mean()))     # replacement return
    by_m = {d: gg.set_index('permno') for d, gg in x.groupby('date')}
    dates = sorted(by_m)
    evg = {d: gg for d, gg in ev.groupby('date')}
    for d in dates:
        gg = by_m[d]
        for p in list(hold):
            if p not in gg.index:
                hold.discard(p); age.pop(p, None); continue
            age[p] = age.get(p, 0) + 1
            r = gg.loc[p]
            if float(r['rank_pct']) > 0.10:                 # drifted: decision
                lbl = float(r['ret_fwd']) - float(kth.get(d, 0.0)) \
                    + 2 * float(r['hs_now']) / 1e4
                rows.append({'date': d, 'permno': p, 'y': int(lbl > 0),
                             **{f: float(r[f]) for f in FEATS}})
        if d in evg:
            for _, e in evg[d].iterrows():
                if e['cause'] == 'entry':
                    hold.add(e['permno']); age[e['permno']] = 0
                else:
                    hold.discard(e['permno']); age.pop(e['permno'], None)
    return pd.DataFrame(rows)


def main():
    sp, month_med, _ = B.load_spreads()
    sp_pack = (sp, month_med, float(month_med.median()))
    rows = []
    for s in C.BS_STRATEGIES:
        x = _prep(s, sp, month_med)
        dp = _decision_points(x)
        if len(dp) < 500:
            print(f"[lb] {s}: only {len(dp)} decision points, skip"); continue
        # expanding-window predictions for every stock-month
        prob = pd.Series(np.nan, index=x.index)
        aucs = []
        for y in sorted({d.year for d in x['date']}):
            if y < START_OOS if s != 'xgb' else y < 2016:
                continue
            tr = dp[dp['date'] < f'{y}-01-01']
            if len(tr) < 300 or tr['y'].nunique() < 2:
                continue
            m = xgb.XGBClassifier(n_estimators=200, max_depth=3,
                                  learning_rate=0.05, subsample=0.8,
                                  reg_lambda=1.0, n_jobs=4, seed=0)
            m.fit(tr[FEATS], tr['y'])
            te = dp[(dp['date'] >= f'{y}-01-01') & (dp['date'] < f'{y+1}-01-01')]
            if len(te) > 20 and te['y'].nunique() > 1:
                from sklearn.metrics import roc_auc_score
                aucs.append(roc_auc_score(te['y'], m.predict_proba(
                    te[FEATS])[:, 1]))
            mask = x['date'].dt.year == y
            prob.loc[mask] = m.predict_proba(x.loc[mask, FEATS])[:, 1]
        oos = prob.notna()
        x_oos = x[oos].copy()
        x_oos['eband'] = np.where(prob[oos] > 0.5, 100.0, 10.0)
        mask_d = x_oos['date'].unique()

        def stats(pol_x, policy):
            r, led = X.simulate(pol_x, policy, score_col='score')
            cost = B.price_ledger(led, sp_pack)[0].reindex(r.index).fillna(0)
            mm = r.index.isin(mask_d)
            return (float(cost[mm].mean() * 12 * 1e4),
                    float(r['turnover'][mm].mean() * 12),
                    float((r['gross'] - cost)[mm].mean() * 12 * 100))
        c_l, to_l, n_l = stats(x_oos, ('stock_var_band', None))
        c_s, to_s, n_s = stats(x, ('nmv_band', (E_STATIC, E_STATIC)))
        # uniform frontier for matched-turnover cost comparison
        fr = []
        for E in [10, 15, 20, 30, 40, 60, 100]:
            cc, tt, _ = stats(x, ('nmv_band', (E, E)))
            fr.append((tt, cc))
        fr = np.array(sorted(fr))
        c_match = float(np.interp(to_l, fr[:, 0], fr[:, 1]))
        auc = float(np.mean(aucs)) if aucs else np.nan
        rows.append({'strategy': s, 'n_decisions': len(dp),
                     'oos_auc': round(auc, 3),
                     'cost_static': round(c_s, 2), 'cost_learned': round(c_l, 2),
                     'cost_unif_matched': round(c_match, 2),
                     'learned_minus_matched_pct':
                         round((c_match - c_l) / c_match * 100, 2)
                         if c_match else np.nan,
                     'to_learned': round(to_l, 2), 'to_static': round(to_s, 2),
                     'net_learned': round(n_l, 2), 'net_static': round(n_s, 2)})
        print(f"[lb] {s}: OOS AUC {auc:.3f} | cost learned {c_l:.2f} vs "
              f"uniform@same-turnover {c_match:.2f} "
              f"({(c_match-c_l)/c_match*100:+.1f}%) | net d "
              f"{n_l-n_s:+.2f}%", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(C.BANDING_DIR, 'learned_band.csv'), index=False)
    lines = [
        '# The band as a learned decision surface (XGBoost, expanding window)\n',
        'Hold/sell rule for drifted holdings learned from ex-post deferral'
        ' labels; eband=100 if P(GOOD_HOLD)>0.5 else 10. OOS AUC ~0.5 => no'
        ' signal. Cost compared to uniform band at MATCHED turnover.\n',
        R.to_string(index=False), '',
        'net deltas carry the 162x composition-noise caveat.']
    with open(os.path.join(C.BANDING_DIR, 'learned_band.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
