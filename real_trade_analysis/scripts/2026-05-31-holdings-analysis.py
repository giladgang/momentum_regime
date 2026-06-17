"""
2026-05-31-holdings-analysis.py
===============================
EXPLORATORY (real_trade_analysis). Permno-level holdings comparison:
  XGB long-only (train Top-1000, trade S&P 500; top-decile NYSE-P90, value-weighted, monthly)
  vs
  SPMO replica (Top-500, top-quintile risk-adj momentum, FMC x score, 9% cap, semi-annual)

Answers (a) which companies each holds (by permno; names join later via WRDS), and
(b) WHY the XGB has higher vol AND return -- via portfolio characteristics:
  # holdings, top-10 weight concentration, effective N (1/HHI), monthly turnover,
  value-weighted size percentile, value-weighted 12-mo momentum, and name-overlap
  between the two portfolios.

Names are NOT available (CRSP permno only; WRDS account locked). Join
results/holdings_*_top.csv on permno -> crsp.stocknames once WRDS is restored.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
_SC = os.path.join(_ROOT, 'real_trade_analysis', 'scripts')
_RES = os.path.join(_ROOT, 'real_trade_analysis', 'results')
def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(_SC, fn))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp = _load("spmo", "2026-05-31-spmo-replica-vs-xgb.py")
exp = sp.exp

from config import (TRAIN_END, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE,
                    SUBSAMPLE, COLSAMPLE, XGB_N_JOBS)
SEEDS = exp.SEEDS


def xgb_path(grps):
    """XGB long-only: monthly full rebalance, top-decile (NYSE P90), value-weighted."""
    out = {}
    for date, grp in grps:
        nyse = grp[grp['exchcd'] == 1]['score'].dropna()
        if len(nyse) < 10:
            continue
        hi = nyse.quantile(0.90)
        longs = grp[grp['score'] >= hi].dropna(subset=['me'])
        if longs['me'].sum() == 0:
            continue
        w = longs['me'] / longs['me'].sum()
        out[date] = dict(zip(longs['permno'], w.values))
    return out


def spmo_path(grps):
    """SPMO replica: semi-annual (Mar/Sep) top-quintile FMC x momentum-score, 9% cap, drift between."""
    out, prev = {}, {}
    for date, grp in grps:
        if date.month in (3, 9) or not prev:
            g = grp.dropna(subset=['me', 'spmo_signal', 'ret_fwd']).copy()
            if len(g) < 20:
                if prev:
                    out[date] = prev
                continue
            g['_ms'] = sp.momentum_score(g['spmo_signal']).values
            n = max(1, round(0.20 * len(g)))
            sel = g.nlargest(n, '_ms')
            w = sp.cap_weights(sel['me'] * sel['_ms'], sel['me'], 0.09, 3.0)
            wd = {p: float(v) for p, v in zip(sel['permno'], w.values)}
            out[date] = wd
            rmap = sel.set_index('permno')['ret_fwd']
            prev = {p: wd[p] * (1 + rmap.get(p, 0)) for p in wd}
            tot = sum(prev.values()); prev = {p: v / tot for p, v in prev.items()} if tot > 0 else {}
        else:
            held = grp[grp['permno'].isin(prev)].dropna(subset=['me', 'ret_fwd'])
            wp = {p: prev[p] for p in prev if p in set(held['permno'])}
            tot = sum(wp.values())
            if tot <= 0:
                continue
            wp = {p: v / tot for p, v in wp.items()}
            out[date] = wp
            rmap = held.set_index('permno')['ret_fwd']
            prev = {p: wp[p] * (1 + rmap.get(p, 0)) for p in wp}
            tot = sum(prev.values()); prev = {p: v / tot for p, v in prev.items()} if tot > 0 else {}
    return out


def summarize(path, lookups):
    """Portfolio characteristics + per-permno average weights from a {date:{permno:w}} path."""
    rows, permno_acc = [], {}
    prev = {}
    months = sorted(path.keys())
    for date in months:
        w = path[date]
        if not w:
            continue
        pr, m12 = lookups['pr'].get(date, {}), lookups['mom12'].get(date, {})
        ww = np.array(list(w.values()))
        top10 = np.sort(ww)[::-1][:10].sum()
        hhi = (ww ** 2).sum()
        size_pct = sum(v * pr.get(p, np.nan) for p, v in w.items())
        mom = sum(v * m12.get(p, np.nan) for p, v in w.items())
        turn = 0.5 * sum(abs(w.get(p, 0) - prev.get(p, 0)) for p in set(w) | set(prev))
        rows.append({'date': date, 'n': len(w), 'top10': top10, 'effN': 1 / hhi,
                     'size_pctile': size_pct, 'mom12': mom, 'turnover': turn})
        for p, v in w.items():
            permno_acc.setdefault(p, []).append(v)
        prev = w
    df = pd.DataFrame(rows).set_index('date')
    nM = len(df)
    char = {'avg_n': df['n'].mean(), 'top10_share': df['top10'].mean(), 'effN': df['effN'].mean(),
            'mo_turnover': df['turnover'].iloc[1:].mean(), 'size_pctile': df['size_pctile'].mean(),
            'mom12': df['mom12'].mean()}
    # per-permno: time-avg weight (over all months) + hold frequency
    pf = pd.DataFrame([{'permno': p, 'avg_wt': np.sum(v) / nM, 'pct_months': len(v) / nM,
                        'avg_wt_when_held': np.mean(v)} for p, v in permno_acc.items()])
    return char, pf, set(permno_acc.keys()), {d: set(path[d].keys()) for d in months if path[d]}


def main():
    base = sp.add_spmo_inputs(exp.build_features())
    df500 = exp.apply_size_screen(base, 500)
    train = exp.apply_size_screen(base, 1000); train = train[train['date'] < TRAIN_END]
    test = df500[df500['date'] >= TRAIN_END].copy()

    Xtr, ytr = train[exp.FEATURES].values.astype(float), train['ret_fwd'].values.astype(float)
    Xte = test[exp.FEATURES].values.astype(float)
    pred = np.zeros(len(Xte))
    for i, seed in enumerate(SEEDS, 1):
        m = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE,
                         subsample=SUBSAMPLE, colsample_bytree=COLSAMPLE, tree_method='hist',
                         random_state=seed, verbosity=0, n_jobs=XGB_N_JOBS)
        m.fit(Xtr, ytr); pred += m.predict(Xte)
        if i % 10 == 0:
            print(f"  xgb seed {i}/{len(SEEDS)}", flush=True)
    test['score'] = pred / len(SEEDS)
    test['spmo_signal'] = test['mom_value'] / test['sigma_m']

    # per-date lookups for size percentile and momentum
    lookups = {'pr': {}, 'mom12': {}}
    for date, grp in test.groupby('date'):
        g = grp.dropna(subset=['me'])
        lookups['pr'][date] = dict(zip(g['permno'], g['me'].rank(pct=True)))
        lookups['mom12'][date] = dict(zip(grp['permno'], grp['mom_12']))

    grps = list(test.groupby('date'))
    xgb_p = xgb_path(grps)
    spmo_p = spmo_path(grps)

    xc, xpf, xset, xmon = summarize(xgb_p, lookups)
    sc, spf, sset, smon = summarize(spmo_p, lookups)

    print("\n" + "=" * 78)
    print("  PORTFOLIO CHARACTERISTICS  (why XGB has higher vol & return)")
    print("=" * 78)
    print(f"  {'metric':<26} {'XGB long-only':>14} {'SPMO replica':>14}")
    print("  " + "-" * 56)
    labels = [('avg_n', 'avg # holdings', '{:.0f}'), ('top10_share', 'top-10 weight share', '{:.0%}'),
              ('effN', 'effective N (1/HHI)', '{:.0f}'), ('mo_turnover', 'monthly turnover', '{:.0%}'),
              ('size_pctile', 'VW size percentile', '{:.0%}'), ('mom12', 'VW 12-mo momentum', '{:.1%}')]
    for k, lbl, fmt in labels:
        print(f"  {lbl:<26} {fmt.format(xc[k]):>14} {fmt.format(sc[k]):>14}")
    print("=" * 78)

    # holdings overlap (per month: share of XGB names also in SPMO that month)
    common_months = sorted(set(xmon) & set(smon))
    ov = np.mean([len(xmon[d] & smon[d]) / len(xmon[d]) for d in common_months if xmon[d]])
    print(f"\n  Name overlap: avg {ov:.0%} of XGB holdings are also in SPMO that month")
    print(f"  permnos ever held: XGB {len(xset)}, SPMO {len(sset)}, common {len(xset & sset)}")

    print("\n  TOP-15 XGB HOLDINGS (by time-avg weight; names via WRDS later)")
    print(xpf.sort_values('avg_wt', ascending=False).head(15)
          .assign(avg_wt=lambda d: (d['avg_wt'] * 100).round(2), pct_months=lambda d: (d['pct_months'] * 100).round(0))
          [['permno', 'avg_wt', 'pct_months']].to_string(index=False))
    print("\n  TOP-15 SPMO HOLDINGS (by time-avg weight)")
    print(spf.sort_values('avg_wt', ascending=False).head(15)
          .assign(avg_wt=lambda d: (d['avg_wt'] * 100).round(2), pct_months=lambda d: (d['pct_months'] * 100).round(0))
          [['permno', 'avg_wt', 'pct_months']].to_string(index=False))

    xpf.sort_values('avg_wt', ascending=False).to_csv(os.path.join(_RES, 'holdings_xgb_top.csv'), index=False)
    spf.sort_values('avg_wt', ascending=False).to_csv(os.path.join(_RES, 'holdings_spmo_top.csv'), index=False)
    pd.DataFrame([{'strategy': 'XGB', **xc}, {'strategy': 'SPMO', **sc}]).to_csv(
        os.path.join(_RES, 'holdings_characteristics.csv'), index=False)
    print(f"\n  saved holdings_xgb_top.csv / holdings_spmo_top.csv / holdings_characteristics.csv")


if __name__ == '__main__':
    main()
