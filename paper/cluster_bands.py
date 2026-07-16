"""Cluster-conditioned banding vs static/regime, honest walk-forward, ALL 7
strategies.

Arms (each fit with ONE global scale E, walk-forward-selected annually =
matched search vs static):
  monthly | static | regime (2-state GP) | cluster_month_econ |
  cluster_month_gp | cluster_stock_econ | cluster_stock_gp

Cluster labels are PIT-safe (annual expanding KMeans refit, paper/src/clusters).
Stock-level labels are strategy-independent, computed ONCE and reused.
econ = deep-losers wider (hold); gp = (spread/var)^(1/3) per cluster.
BH-FDR across the 5 conditioning arms x 7 strategies.

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
ECON = np.array([0.7, 0.9, 1.2, 1.5])       # C1 winner tighter ... C4 wider
COND_ARMS = ['regime', 'cluster_month_econ', 'cluster_month_gp',
             'cluster_stock_econ', 'cluster_stock_gp']


def _pit_stock_labels(stock_z):
    """Annual expanding-refit stock cluster per (date,permno), computed once
    (strategy-independent). Series indexed by (date, permno)."""
    dates = pd.DatetimeIndex(sorted(stock_z['date'].unique()))
    parts = []
    for y in sorted({d.year for d in dates}):
        cutoff = pd.Timestamp(f'{y}-01-01')
        if (stock_z['date'] < cutoff).sum() < 5000:
            continue
        tgt = dates[dates.year == y]
        sl = CL.assign_stock_clusters(stock_z, cutoff, tgt)
        parts.append(stock_z.loc[sl.index, ['date', 'permno']]
                     .assign(cl=sl.values))
    allm = pd.concat(parts) if parts else \
        pd.DataFrame(columns=['date', 'permno', 'cl'])
    return allm.set_index(['date', 'permno'])['cl']


def _pit_month_labels(frame, book_z):
    dates = pd.DatetimeIndex(sorted(frame['date'].unique()))
    mlab = pd.Series(0, index=dates)
    for y in sorted({d.year for d in dates}):
        cutoff = pd.Timestamp(f'{y}-01-01')
        if (book_z.index < cutoff).sum() < 24:
            continue
        tgt = dates[dates.year == y]
        try:
            mlab.loc[tgt] = CL.assign_month_clusters(book_z, cutoff, tgt).values
        except Exception:                                      # noqa: BLE001
            pass
    return mlab


def _cluster_gp_ratios(sp_pack, panel_vw, mlab, cutoff):
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


def _boot_p(a, b, n=2000, block=12, seed=0):
    d = (a - b).dropna().values
    rng = np.random.default_rng(seed)
    nn = len(d)
    means = np.empty(n)
    for i in range(n):
        idx = []
        while len(idx) < nn:
            s = int(rng.integers(0, nn))
            idx += list(range(s, s + block))
        means[i] = d[[j % nn for j in idx[:nn]]].mean()
    f = (means <= 0).mean()
    return 2 * min(f, 1 - f)


def main():
    sp_pack = B.load_spreads()
    panel_vw = pd.read_parquet(C.PANEL_PARQUET, columns=['date', 'vwretd'])
    panel_vw['date'] = pd.to_datetime(panel_vw['date'])
    stocks = pd.read_parquet(C.STOCKS_PARQUET,
                             columns=['permno', 'date'] + CL.MOMS)
    stocks['date'] = pd.to_datetime(stocks['date'])
    stock_z = CL.build_stock_zcurves(stocks)
    print('[cb] precomputing PIT stock labels (once) ...', flush=True)
    stock_lab = _pit_stock_labels(stock_z)

    rows = []
    for s in C.BS_STRATEGIES:
        x = signals.build(s)
        x = x[x['date'] >= C.SWEEP_START].reset_index(drop=True)
        start = START_OOS if s != 'xgb' else 2013
        book_z = CL.book_zcurve(stock_z, x)
        mlab = _pit_month_labels(x, book_z)
        sc = stock_lab.reindex(
            pd.MultiIndex.from_frame(x[['date', 'permno']]))
        x = x.assign(cluster=x['date'].map(mlab).fillna(0).astype(int),
                     stock_cluster=sc.fillna(0).astype(int).values)
        reg_ratio = _regime_ratio(x, sp_pack, panel_vw, f'{start}-01-01')
        gp = _cluster_gp_ratios(sp_pack, panel_vw, mlab, f'{start}-01-01')
        b, bled = X.simulate(x, ('benchmark', None), score_col='score')
        bench_net = B._net(b['gross'], B.price_ledger(bled, sp_pack)[0])

        def act(pol):
            return _active(x, pol, sp_pack, bench_net)
        arms = {a: {} for a in ['monthly', 'static'] + COND_ARMS}
        for e in E_BASE:
            ep = float(np.clip(e * reg_ratio, 5, 60))
            ec = tuple(float(np.clip(e * ECON[c], 5, 60)) for c in range(CL.K))
            eg = tuple(float(np.clip(e * gp[c], 5, 60)) for c in range(CL.K))
            arms['monthly'][e] = act(MONTHLY)
            arms['static'][e] = act(('nmv_band', (e, e)))
            arms['regime'][e] = act(('nmv_band', (e, ep)))
            arms['cluster_month_econ'][e] = act(('cluster_band', ec))
            arms['cluster_month_gp'][e] = act(('cluster_band', eg))
            arms['cluster_stock_econ'][e] = act(
                ('stock_cluster_band', {c: ec[c] for c in range(CL.K)}))
            arms['cluster_stock_gp'][e] = act(
                ('stock_cluster_band', {c: eg[c] for c in range(CL.K)}))
        st = _walk({a: pd.DataFrame(d) for a, d in arms.items()}, start)
        z = pd.Series(0.0, index=st['static'].index)
        base = M.ir(st['static'], z)
        for a in COND_ARMS:
            if a not in st:
                continue
            d = (st[a] - st['static']).dropna()
            lo, hi = X.paired_block_bootstrap(
                st[a].reindex(d.index).astype(float),
                st['static'].reindex(d.index).astype(float))
            rows.append({'strategy': s, 'arm': a, 'oos_ir': M.ir(st[a], z),
                         'minus_static': M.ir(st[a], z) - base,
                         'excl0': bool(lo > 0 or hi < 0),
                         'p_boot': _boot_p(st[a].reindex(d.index),
                                           st['static'].reindex(d.index))})
        print(f'[cb] {s}: static {base:+.3f} | ' + ' | '.join(
            f'{a.replace("cluster_","")} {M.ir(st[a], z)-base:+.3f}'
            for a in COND_ARMS if a in st), flush=True)
    R = pd.DataFrame(rows)
    # BH-FDR across all conditioning tests
    p = R['p_boot'].sort_values()
    thr = (np.arange(1, len(p) + 1) / len(p)) * 0.05
    R['bh_sig'] = False
    R.loc[p.index, 'bh_sig'] = (p.values <= thr)
    R.round(4).to_csv(os.path.join(C.BANDING_DIR, 'cluster_bands.csv'),
                      index=False)
    npos = int((R['minus_static'] > 0).sum())
    n_excl = int(R['excl0'].sum())
    n_bh = int(R['bh_sig'].sum())
    lines = ['# Cluster-conditioned banding, all 7 strategies (honest WF)\n',
             'One global scale per arm, walk-forward-selected annually. PIT '
             'expanding-refit clusters (stock labels computed once). econ = '
             'hold deep-losers wider; gp = (spread/var)^(1/3). BH-FDR q=0.05 '
             'across the 5 conditioning arms x 7 strategies.\n',
             R.round(3).to_string(index=False), '',
             f'Positive vs static: {npos}/{len(R)} arm-tests. '
             f'CI excludes 0: {n_excl}/{len(R)}. '
             f'Survive BH-FDR (q=0.05): {n_bh}/{len(R)}.']
    with open(os.path.join(C.BANDING_DIR, 'cluster_bands.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
