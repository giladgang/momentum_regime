"""
2026-05-31-selection-analysis-AF.py
===================================
EXPLORATORY (real_trade_analysis). Deep comparison of XGB (train Top-1000, trade
S&P 500, long-only top-decile VW, monthly) vs SPMO replica (Top-500, top-quintile
risk-adj momentum, FMC x score, semi-annual). One Top-1000 training; analyses A-F:

  A  Characteristic profile of picks (VW avg mom_1..12 term structure, pi_filter, size)
  B  Regime-split selection (calm vs panic) -- does XGB shift to short-horizon momentum in panic?
  C  Selection vs weighting attribution of the gross return gap
  D  XGB-only vs SPMO-only picks -- what each uniquely holds (features + realized fwd return)
  E  FF6 factor regressions (alpha, NW(6) t, loadings) for both net return streams
  F  Fee-stress: Sharpe vs cost level (one-way turnover bps; ~2x = per-side)

Names unavailable (permno only; WRDS locked). Reads only data inputs.
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
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(_SC, f)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ha = _load("holdings", "2026-05-31-holdings-analysis.py")   # xgb_path, spmo_path
sp = ha.sp; exp = ha.exp
lo = _load("lo", "2026-05-31-xgb-longonly-sp500.py")

from src.utils import metrics
from config import (TRAIN_END, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE,
                    SUBSAMPLE, COLSAMPLE, XGB_N_JOBS, FF_FACTORS_PATH)
SEEDS = exp.SEEDS
MOM = [f'mom_{i}' for i in range(1, 13)]


def vw_profile(path, feat_lk, months_filter=None):
    """Value-weighted average of each feature across the holdings, averaged over months."""
    acc = {f: [] for f in MOM + ['pi_filter', '_pr']}
    for date, w in path.items():
        if months_filter is not None and date not in months_filter:
            continue
        if not w:
            continue
        for f in acc:
            lk = feat_lk[f].get(date, {})
            acc[f].append(sum(v * lk.get(p, np.nan) for p, v in w.items()))
    return {f: np.nanmean(acc[f]) for f in acc}


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
        if i % 10 == 0: print(f"  xgb seed {i}/{len(SEEDS)}", flush=True)
    test['score'] = pred / len(SEEDS)
    test['spmo_signal'] = test['mom_value'] / test['sigma_m']

    # per-date feature lookups
    feat_lk = {f: {} for f in MOM + ['pi_filter', '_pr', 'ret_fwd']}
    pim = {}
    for date, grp in test.groupby('date'):
        g = grp.dropna(subset=['me'])
        feat_lk['_pr'][date] = dict(zip(g['permno'], g['me'].rank(pct=True)))
        for f in MOM + ['pi_filter', 'ret_fwd']:
            feat_lk[f][date] = dict(zip(grp['permno'], grp[f]))
        pim[date] = grp['pi_filter'].iloc[0]

    grps = list(test.groupby('date'))
    xgb_p, spmo_p = ha.xgb_path(grps), ha.spmo_path(grps)
    calm = {d for d in pim if pim[d] < 0.5}; panic = {d for d in pim if pim[d] >= 0.5}

    # ---- A: characteristic profile ----
    pa_x, pa_s = vw_profile(xgb_p, feat_lk), vw_profile(spmo_p, feat_lk)
    print("\n" + "=" * 78 + "\n  A. CHARACTERISTIC PROFILE OF PICKS (value-weighted avg)\n" + "=" * 78)
    print(f"  {'feature':<12} {'XGB':>9} {'SPMO':>9}")
    for f in MOM + ['pi_filter', '_pr']:
        lbl = 'size_pctile' if f == '_pr' else f
        print(f"  {lbl:<12} {pa_x[f]:>9.3f} {pa_s[f]:>9.3f}")

    # ---- B: regime-split momentum term structure ----
    bx_c, bx_p = vw_profile(xgb_p, feat_lk, calm), vw_profile(xgb_p, feat_lk, panic)
    bs_c, bs_p = vw_profile(spmo_p, feat_lk, calm), vw_profile(spmo_p, feat_lk, panic)
    print("\n" + "=" * 78 + "\n  B. REGIME-SPLIT MOMENTUM (VW avg of mom_1, mom_6, mom_12; short vs long horizon)\n" + "=" * 78)
    print(f"  {'':<22}{'mom_1':>8}{'mom_6':>8}{'mom_12':>8}   short/long (mom_1/mom_12)")
    for nm, prof in [('XGB  calm', bx_c), ('XGB  panic', bx_p), ('SPMO calm', bs_c), ('SPMO panic', bs_p)]:
        sl = prof['mom_1'] / prof['mom_12'] if prof['mom_12'] else np.nan
        print(f"  {nm:<22}{prof['mom_1']:>8.3f}{prof['mom_6']:>8.3f}{prof['mom_12']:>8.3f}   {sl:>8.2f}")

    # ---- C: selection vs weighting attribution (gross monthly return gap) ----
    common_c, xonly_c, sonly_c, gap_c = [], [], [], []
    for date in sorted(set(xgb_p) & set(spmo_p)):
        wx, ws = xgb_p[date], spmo_p[date]; r = feat_lk['ret_fwd'][date]
        if not wx or not ws: continue
        allp = set(wx) | set(ws)
        common = sum((wx.get(p, 0) - ws.get(p, 0)) * r.get(p, 0) for p in (set(wx) & set(ws)))
        xonly = sum(wx[p] * r.get(p, 0) for p in (set(wx) - set(ws)))
        sonly = -sum(ws[p] * r.get(p, 0) for p in (set(ws) - set(wx)))
        common_c.append(common); xonly_c.append(xonly); sonly_c.append(sonly)
        gap_c.append(common + xonly + sonly)
    print("\n" + "=" * 78 + "\n  C. SELECTION vs WEIGHTING ATTRIBUTION of gross return gap (annualized)\n" + "=" * 78)
    print(f"  total gross gap (XGB - SPMO):     {np.mean(gap_c)*12:>+7.1%}/yr")
    print(f"    from different weights on shared names: {np.mean(common_c)*12:>+7.1%}/yr")
    print(f"    from XGB-only names:                    {np.mean(xonly_c)*12:>+7.1%}/yr")
    print(f"    from NOT holding SPMO-only names:       {np.mean(sonly_c)*12:>+7.1%}/yr")

    # ---- D: XGB-only vs SPMO-only picks ----
    def only_profile(get_only):
        acc = {f: [] for f in ['mom_1', 'mom_12', 'pi_filter', '_pr', 'ret_fwd']}
        for date in sorted(set(xgb_p) & set(spmo_p)):
            names = get_only(set(xgb_p[date]), set(spmo_p[date]))
            for f in acc:
                lk = feat_lk[f].get(date, {})
                vals = [lk.get(p, np.nan) for p in names if p in lk]
                if vals: acc[f].append(np.nanmean(vals))
        return {f: np.nanmean(acc[f]) for f in acc}
    xo = only_profile(lambda x, s: x - s); so = only_profile(lambda x, s: s - x)
    print("\n" + "=" * 78 + "\n  D. UNIQUE PICKS: XGB-only vs SPMO-only names (equal-wt avg of the unique set)\n" + "=" * 78)
    print(f"  {'metric':<14}{'XGB-only':>10}{'SPMO-only':>11}")
    for f, lbl in [('mom_1', 'mom_1'), ('mom_12', 'mom_12'), ('pi_filter', 'pi_filter'),
                   ('_pr', 'size_pctile'), ('ret_fwd', 'realized fwd ret')]:
        print(f"  {lbl:<14}{xo[f]:>10.3f}{so[f]:>11.3f}")

    # ---- E: FF6 regressions ----
    ff = pd.read_parquet(os.path.join(_ROOT, FF_FACTORS_PATH)).copy()
    ff.index = pd.to_datetime(ff.index).to_period('M')
    facs = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD']
    xgb_ret = lo.long_only_port(test, 'score'); spmo_ret = sp.build_longonly(test, 'spmo_signal', 'score', 0.20, 0.09)
    try:
        import statsmodels.api as sm
        print("\n" + "=" * 78 + "\n  E. FF6 FACTOR REGRESSIONS (NW(6); alpha annualized)\n" + "=" * 78)
        print(f"  {'strategy':<12}{'alpha':>8}{'t(a)':>6}  " + "".join(f"{f:>7}" for f in facs))
        for nm, r in [('XGB LO', xgb_ret), ('SPMO', spmo_ret)]:
            # ret at date t is the t->t+1 (ret_fwd) return -> align to the month it is REALIZED (t+1)
            r = r.dropna(); r.index = r.index.to_period('M') + 1
            d = pd.concat([r.rename('r'), ff], axis=1, join='inner').dropna()
            y = d['r'] - d['RF']; X = sm.add_constant(d[facs])
            res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
            row = f"  {nm:<12}{res.params['const']*12:>7.1%}{res.tvalues['const']:>6.2f}  " + "".join(f"{res.params[f]:>+7.2f}" for f in facs)
            print(row)
    except Exception as e:
        print("  (statsmodels unavailable:", repr(e)[:80], ")")

    # ---- F: fee stress ----
    print("\n" + "=" * 78 + "\n  F. FEE STRESS (Sharpe, full 2011-2024; bps = one-way turnover rate; ~2x=per-side)\n" + "=" * 78)
    print(f"  {'fee (bps)':<10}{'XGB LO Sharpe':>15}{'SPMO Sharpe':>14}")
    for bps in [0, 10, 20, 30, 50, 100]:
        fee = bps / 1e4
        xr = lo.long_only_port(test, 'score', fee=fee)
        sp.FEE = fee; sr = sp.build_longonly(test, 'spmo_signal', 'score', 0.20, 0.09)
        note = '  <- prod default' if bps == 10 else ('  <- ~10bps/side' if bps == 20 else '')
        print(f"  {bps:<10}{metrics(xr)[2]:>15.2f}{metrics(sr)[2]:>14.2f}{note}")
    sp.FEE = 0.001

    print("\n  done.")


if __name__ == '__main__':
    main()
