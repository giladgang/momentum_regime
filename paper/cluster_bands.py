"""Cluster-conditioned banding vs static/regime, honest walk-forward.

Arms (each fit with ONE global scale E, walk-forward-selected annually =
matched search vs static):
  monthly | static | regime (2-state GP) | cluster_month_econ |
  cluster_month_gp | cluster_stock_econ | cluster_stock_gp

Cluster labels are PIT-safe (annual expanding KMeans refit, paper/src/clusters).
Two per-cluster width rules, both parameter-free in the cluster dimension:
- econ: deep-loser clusters wider (hold), winners tighter -- tests Gilad's
  "trade less under cluster 3/4" hypothesis directly.
- gp: (spread_c/var_c)^(1/3) per cluster -- Constantinides/GP; DISAGREES with
  econ on deep-losers (high variance => GP says track tighter). Which the data
  favors is the point.

Usage: .venv/bin/python -m paper.cluster_bands
Output: paper/results/banding_study/cluster_bands.{md,csv}
"""
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from paper import config as C                                   # noqa: E402
from paper import execution as X                                # noqa: E402
from paper import banding_study as B                            # noqa: E402
from paper.src import metrics as M                              # noqa: E402
from paper.src import signals                                   # noqa: E402
from paper.src import clusters as CL                            # noqa: E402
from paper.gp_bands import _active, _regime_ratio               # noqa: E402

E_BASE = [10, 15, 20, 30, 40]
MONTHLY = ('nmv_band', (10, 10))
START_OOS = 2001
# econ direction: C0 winner (tighter) ... C3 deep-loser (wider). Hold losers.
ECON = np.array([0.7, 0.9, 1.2, 1.5])


def _pit_labels(frame, stock_z, book_z, first_year):
    """Annual expanding-refit PIT cluster labels for frame's months.
    Returns (month_lab: date->cluster, stock_lab: (row)->cluster aligned to
    frame). Months before first_year (need >=24m train) get cluster 0."""
    dates = pd.DatetimeIndex(sorted(frame['date'].unique()))
    years = sorted({d.year for d in dates})
    mlab = pd.Series(0, index=dates)
    slab = pd.Series(0, index=frame.index)
    fyi = frame.set_index('date')
    for y in years:
        cutoff = pd.Timestamp(f'{y}-01-01')
        if (book_z.index < cutoff).sum() < 24:
            continue
        tgt = dates[(dates.year == y)]
        try:
            ml = CL.assign_month_clusters(book_z, cutoff, tgt)
            mlab.loc[tgt] = ml.values
        except Exception:                                      # noqa: BLE001
            pass
        sz_y = stock_z[stock_z['date'].isin(set(tgt))]
        if len(sz_y):
            sl = CL.assign_stock_clusters(stock_z, cutoff, tgt)
            # map (date,permno)->cluster back onto frame rows of year y
            key = sz_y.assign(cl=sl.values).set_index(['date', 'permno'])['cl']
            fr_y = frame[frame['date'].isin(set(tgt))]
            idx = fr_y.set_index(['date', 'permno']).index
            slab.loc[fr_y.index] = key.reindex(idx).fillna(0).astype(int).values
    return mlab, slab


def _cluster_gp_ratios(book_z, sp_pack, panel_vw, mlab, cutoff):
    """(spread_c/spread_ref * var_ref/var_c)^(1/3) per month-cluster on prior
    data; ref = cluster 0 (winner). Falls back to 1.0."""
    _, month_med, _ = sp_pack
    m = pd.DataFrame({'date': mlab.index, 'cl': mlab.values})
    m = m[m['date'] < cutoff]
    m['ym'] = m['date'].dt.to_period('M')
    m['hs'] = m['ym'].map(month_med)
    m = m.merge(panel_vw, on='date', how='left')
    out = np.ones(CL.K)
    ref = m[m['cl'] == 0]
    if len(ref) < 6:
        return out
    s0, v0 = ref['hs'].mean(), ref['vwretd'].var()
    for c in range(CL.K):
        g = m[m['cl'] == c]
        if len(g) < 6 or g['hs'].mean() == 0 or g['vwretd'].var() == 0:
            continue
        out[c] = float(((g['hs'].mean() / s0) * (v0 / g['vwretd'].var()))
                       ** (1 / 3))
    return out


def _walk(arms, start_oos):
    """arms: {name: DataFrame(months x scale-cols)}. Select best scale per year
    on prior data, stitch OOS. Returns {name: stitched Series}."""
    any_df = next(iter(arms.values()))
    yrs = [y for y in sorted({d.year for d in any_df.index}) if y >= start_oos]
    rec = {a: [] for a in arms}
    for y in yrs:
        for a, df in arms.items():
            tr, te = df[df.index.year < y], df[df.index.year == y]
            if len(tr) < 60 or len(te) == 0:
                continue

            def ir(c):
                v = tr[c].dropna()
                return (v.mean() / v.std() * np.sqrt(12)
                        if len(v) > 12 and v.std() > 0 else -np.inf)
            rec[a].append(te[max(df.columns, key=ir)])
    return {a: pd.concat(v).sort_index() for a, v in rec.items() if v}


def _bh_fdr(pvals, q=0.05):
    s = pvals.sort_values()
    thr = (np.arange(1, len(s) + 1) / len(s)) * q
    return pd.Series(s.values <= thr, index=s.index)


def main():
    sp_pack = B.load_spreads()
    panel_vw = pd.read_parquet(C.PANEL_PARQUET, columns=['date', 'vwretd'])
    panel_vw['date'] = pd.to_datetime(panel_vw['date'])
    stocks = pd.read_parquet(C.STOCKS_PARQUET,
                             columns=['permno', 'date'] + CL.MOMS)
    stocks['date'] = pd.to_datetime(stocks['date'])
    stock_z = CL.build_stock_zcurves(stocks)

    rows, stitched_all = [], {}
    for s in ['momentum', 'xgb']:
        x = signals.build(s)
        x = x[x['date'] >= C.SWEEP_START].reset_index(drop=True)
        start = START_OOS if s != 'xgb' else 2013
        book_z = CL.book_zcurve(stock_z, x)
        first_year = int(pd.DatetimeIndex(book_z.index).year.min()) + 2
        mlab, slab = _pit_labels(x, stock_z, book_z, first_year)
        x = x.assign(cluster=x['date'].map(mlab), stock_cluster=slab.values)
        x['cluster'] = x['cluster'].fillna(0).astype(int)

        reg_ratio = _regime_ratio(x, sp_pack, panel_vw, f'{start}-01-01')
        gp = _cluster_gp_ratios(book_z, sp_pack, panel_vw, mlab,
                                f'{start}-01-01')
        b, bled = X.simulate(x, ('benchmark', None), score_col='score')
        bench_net = B._net(b['gross'], B.price_ledger(bled, sp_pack)[0])

        def act(pol):
            return _active(x, pol, sp_pack, bench_net)
        arms = {'monthly': {}, 'static': {}, 'regime': {},
                'cluster_month_econ': {}, 'cluster_month_gp': {},
                'cluster_stock_econ': {}, 'cluster_stock_gp': {}}
        for e in E_BASE:
            ep = float(np.clip(e * reg_ratio, 5, 60))
            arms['monthly'][e] = act(MONTHLY)
            arms['static'][e] = act(('nmv_band', (e, e)))
            arms['regime'][e] = act(('nmv_band', (e, ep)))
            ec = tuple(float(np.clip(e * ECON[c], 5, 60)) for c in range(CL.K))
            eg = tuple(float(np.clip(e * gp[c], 5, 60)) for c in range(CL.K))
            arms['cluster_month_econ'][e] = act(('cluster_band', ec))
            arms['cluster_month_gp'][e] = act(('cluster_band', eg))
            arms['cluster_stock_econ'][e] = act(
                ('stock_cluster_band', {c: ec[c] for c in range(CL.K)}))
            arms['cluster_stock_gp'][e] = act(
                ('stock_cluster_band', {c: eg[c] for c in range(CL.K)}))
        arms = {a: pd.DataFrame(d) for a, d in arms.items()}
        st = _walk(arms, start)
        stitched_all[s] = st
        z = pd.Series(0.0, index=st['static'].index)
        base = M.ir(st['static'], z)
        for a in st:
            d = (st[a] - st['static']).dropna()
            lo, hi = X.paired_block_bootstrap(
                st[a].reindex(d.index).astype(float),
                st['static'].reindex(d.index).astype(float))
            rows.append({'strategy': s, 'arm': a, 'oos_ir': M.ir(st[a], z),
                         'minus_static': M.ir(st[a], z) - base,
                         'ci_x12': f'[{lo*12:+.3f},{hi*12:+.3f}]',
                         'excl0': bool(lo > 0 or hi < 0)})
        print(f'[cb] {s}: ' + ' | '.join(
            f'{a} {M.ir(st[a], z):+.3f}' for a in st), flush=True)
    R = pd.DataFrame(rows)
    R.round(4).to_csv(os.path.join(C.BANDING_DIR, 'cluster_bands.csv'),
                      index=False)
    lines = ['# Cluster-conditioned banding vs static/regime (honest WF)\n',
             'One global scale per arm, walk-forward-selected annually '
             '(matched to static). PIT expanding-refit clusters. econ = hold '
             'deep-losers wider; gp = (spread/var)^(1/3) per cluster.\n',
             R.round(3).to_string(index=False), '']
    for s in stitched_all:
        sub = R[(R.strategy == s) & (R.arm != 'static')]
        best = sub.loc[sub['minus_static'].idxmax()]
        lines.append(f'{s}: best arm vs static = {best["arm"]} '
                     f'({best["minus_static"]:+.3f}, excl0 {best["excl0"]}).')
    with open(os.path.join(C.BANDING_DIR, 'cluster_bands.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
