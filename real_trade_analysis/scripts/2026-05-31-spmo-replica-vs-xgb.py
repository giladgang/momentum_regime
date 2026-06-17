"""
2026-05-31-spmo-replica-vs-xgb.py
=================================
EXPLORATORY (experiments/ -- throwaway).

Replicate the S&P 500 Momentum index rules (the index SPMO tracks) on our CRSP
data, then run the thesis XGB regime-momentum signal through the IDENTICAL
construction. Holding construction fixed and swapping only the signal isolates:
"does the regime-aware XGB signal beat S&P's risk-adjusted momentum signal when
both are built into the same S&P 500 Momentum machinery?"

Faithful to "S&P Momentum Indices Methodology" (Mar 2026):
  * Universe        : Top-500 by market cap each month (S&P 500 size proxy).
  * Momentum value  : 12-month return excluding the most recent month,
                      price_{M-2}/price_{M-14} - 1  (Appendix A). Implemented as
                      the cumulative return over months t-13..t-2.
  * Risk adjustment : momentum value / sigma  (Appendix A.2).
  * Z / score       : winsorized z (+/-3) -> 1+z if z>0 else 1/(1-z)  (Appendix B).
  * Selection       : top quintile (~100 of 500)  (Constituent Selection).
  * Weighting       : FMC x momentum score, capped at min(9%, 3x cap weight)
                      (Constituent Weightings). No sector/country caps for S&P 500
                      Momentum.
  * Rebalance       : semi-annual, March & September  (Index Maintenance).
  * Long-only.

Fidelity gaps (forced by monthly CRSP; documented for the writeup):
  1. sigma from MONTHLY returns over the 12-mo window, not daily (no WRDS daily
     access from this network -- pull script staged at _pull_daily_top500.py).
  2. Full market cap `me`, not float-adjusted FMC (no IWF in repo).
  3. 20% turnover buffer rule skipped (smooths turnover, not the signal).
  4. Month-granular Mar/Sep timing, not the exact 3rd-Friday mechanics.
  5. Eligibility ~ "has 12-mo momentum history" instead of the 150-trading-day rule.

Reuses build_features / apply_size_screen / constants from the russell1000
module (its main() is __main__-guarded). Reads only data inputs; writes only its
own CSV next to this file.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

_spec = importlib.util.spec_from_file_location(
    "xgb_russell1000", os.path.join(_ROOT, 'real_trade_analysis', 'scripts', '2026-05-30-xgb-russell1000.py'))
exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp)

from src.utils import metrics
from config import (TRAIN_END, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE,
                    SUBSAMPLE, COLSAMPLE, XGB_N_JOBS)

TOP_N        = 500            # S&P 500 size proxy
TOP_FRAC     = 0.20           # top quintile selected
REBAL_MONTHS = {3, 9}        # March & September
WCAP         = 0.09           # 9% single-name cap
WCAP_MULT    = 3.0            # ... or 3x the stock's cap weight, whichever lower
FEE          = 0.001          # 10 bps one-way on turnover
SEEDS        = exp.SEEDS      # 50, from config
_CRSP        = os.path.join(_ROOT, 'data', 'crsp_msf_raw.parquet')


# ── S&P momentum-score transform (Appendix B), applied within each month ──────
def momentum_score(signal):
    """Winsorized z (+/-3) -> 1+z (z>0) / 1/(1-z) (z<0) / 1 (z=0). Always positive."""
    s = signal.astype(float)
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(1.0, index=s.index)
    z = ((s - s.mean()) / sd).clip(-3, 3)
    sc = np.where(z > 0, 1 + z, 1.0 / (1.0 - z))
    sc = np.where(np.isclose(z, 0.0), 1.0, sc)
    return pd.Series(sc, index=s.index)


def cap_weights(raw, me, cap, cap_mult):
    """Normalize raw weights, then iteratively cap at min(cap, cap_mult*capweight)."""
    w = (raw / raw.sum())
    if cap is None:
        return w
    capwt = me / me.sum()
    capvec = np.minimum(cap, cap_mult * capwt)
    for _ in range(200):
        over = w > capvec + 1e-12
        if not over.any():
            break
        excess = (w[over] - capvec[over]).sum()
        w[over] = capvec[over]
        under = ~over
        pool = w[under].sum()
        if pool <= 0:
            break
        w[under] = w[under] + excess * w[under] / pool
    return w


def _drift(weights, retmap):
    """Grow weights by realized returns; keep only names with a return; renormalize."""
    nw = {p: weights[p] * (1 + retmap[p]) for p in weights if p in retmap.index}
    tot = sum(nw.values())
    return {p: v / tot for p, v in nw.items()} if tot > 0 else {}


def build_longonly(test, signal_col, weight_mode, top_frac, cap):
    """Long-only, semi-annual (Mar/Sep) rebalanced, hold-and-drift between.

    weight_mode='score' -> FMC x momentum_score(signal); 'cap' -> FMC only.
    Returns a monthly return series (uses ret_fwd, t->t+1).
    """
    monthly, prev_w = [], {}
    for date, grp in test.groupby('date'):
        is_rebal = (date.month in REBAL_MONTHS) or (not prev_w)
        if is_rebal:
            g = grp.dropna(subset=['me', signal_col, 'ret_fwd']).copy()
            if len(g) < 20:
                continue
            if weight_mode == 'score':
                g['_ms'] = momentum_score(g[signal_col])
                n_sel = max(1, round(top_frac * len(g)))
                sel = g.nlargest(n_sel, '_ms')
                raw = sel['me'] * sel['_ms']
            else:  # cap-weighted benchmark over the whole (or top) universe
                n_sel = max(1, round(top_frac * len(g)))
                sel = g.nlargest(n_sel, 'me') if top_frac < 1.0 else g
                raw = sel['me']
            w = cap_weights(raw, sel['me'], cap, WCAP_MULT)
            w = {p: float(v) for p, v in zip(sel['permno'], w)}
            retmap = sel.set_index('permno')['ret_fwd']
            turn = 0.5 * sum(abs(w.get(p, 0) - prev_w.get(p, 0)) for p in set(w) | set(prev_w))
            r = sum(w[p] * retmap[p] for p in w) - FEE * turn
            prev_w = _drift(w, retmap)
            monthly.append({'date': date, 'ret': r})
        else:
            held = grp[grp['permno'].isin(prev_w)].dropna(subset=['me', 'ret_fwd'])
            if held.empty:
                continue
            retmap = held.set_index('permno')['ret_fwd']
            wp = {p: prev_w[p] for p in prev_w if p in retmap.index}
            tot = sum(wp.values())
            if tot <= 0:
                continue
            wp = {p: v / tot for p, v in wp.items()}
            r = sum(wp[p] * retmap[p] for p in wp)
            prev_w = _drift(wp, retmap)
            monthly.append({'date': date, 'ret': r})
    return pd.DataFrame(monthly).set_index('date')['ret']


def regime_row(r, pim, label):
    r = r.dropna()
    p = pim.reindex(r.index)
    calm, panic = r[p < 0.5], r[p >= 0.5]
    sh = lambda x: x.mean() / x.std() * np.sqrt(12) if len(x) > 1 and x.std() > 0 else np.nan
    ann, vol, sharpe, mdd = metrics(r)
    return {'label': label, 'months': len(r), 'ann_ret': ann, 'vol': vol,
            'sharpe': sharpe, 'mdd': mdd, 'calm': sh(calm), 'panic': sh(panic)}


def add_spmo_inputs(df):
    """Merge S&P momentum value (12-mo skip-most-recent) + monthly sigma onto df."""
    raw = pd.read_parquet(_CRSP, columns=['permno', 'date', 'ret_adj', 'shrcd', 'exchcd', 'prc'])
    raw['date'] = pd.to_datetime(raw['date'])
    raw = raw.sort_values(['permno', 'date'])
    raw = raw[raw['shrcd'].isin([10, 11]) & raw['exchcd'].isin([1, 2, 3]) & (raw['prc'].abs() > 1.0)]
    lr = np.log1p(raw['ret_adj'].clip(lower=-0.999))
    raw['_lr_s2'] = lr.groupby(raw['permno']).shift(2)        # skip most recent month
    raw['_r_s2']  = raw.groupby('permno')['ret_adj'].shift(2)
    g = raw.groupby('permno', sort=False)
    # 12-month window t-13..t-2
    raw['mom_value'] = np.expm1(g['_lr_s2'].rolling(12, min_periods=12).sum()
                                .reset_index(level=0, drop=True).sort_index())
    raw['sigma_m']   = (g['_r_s2'].rolling(12, min_periods=12).std()
                        .reset_index(level=0, drop=True).sort_index())
    out = df.merge(raw[['permno', 'date', 'mom_value', 'sigma_m']], on=['permno', 'date'], how='left')
    return out


def main():
    df = exp.build_features()
    df = add_spmo_inputs(df)
    df = exp.apply_size_screen(df, TOP_N)          # Top-500 each month
    print(f"Top-{TOP_N} panel: {len(df):,} rows  "
          f"({df['date'].min().date()} -> {df['date'].max().date()})", flush=True)

    # XGB ensemble: train+trade on Top-500, predict every test row.
    train = df[df['date'] < TRAIN_END]
    test  = df[df['date'] >= TRAIN_END].copy()
    Xtr = train[exp.FEATURES].values.astype(float)
    ytr = train['ret_fwd'].values.astype(float)
    Xte = test[exp.FEATURES].values.astype(float)
    preds = np.zeros(len(Xte))
    for i, seed in enumerate(SEEDS, 1):
        m = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                         learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                         colsample_bytree=COLSAMPLE, tree_method='hist',
                         random_state=seed, verbosity=0, n_jobs=XGB_N_JOBS)
        m.fit(Xtr, ytr)
        preds += m.predict(Xte)
        if i % 10 == 0:
            print(f"  xgb seed {i}/{len(SEEDS)}", flush=True)
    preds /= len(SEEDS)
    test['xgb_signal']  = preds
    test['spmo_signal'] = test['mom_value'] / test['sigma_m']   # risk-adjusted momentum

    pim = test.drop_duplicates('date').set_index('date')['pi_filter']

    strategies = [
        ('S&P500 proxy (cap-wt)',  build_longonly(test, 'me',          'cap',   1.00, None)),
        ('SPMO replica (risk-adj)', build_longonly(test, 'spmo_signal', 'score', TOP_FRAC, WCAP)),
        ('Your algo (SPMO-style)',  build_longonly(test, 'xgb_signal',  'score', TOP_FRAC, WCAP)),
    ]
    rows = [regime_row(r, pim, lbl) for lbl, r in strategies]

    print("\n" + "=" * 82)
    print("  S&P 500 MOMENTUM (SPMO) RULES REPLICA vs THESIS XGB  --  long-only, semi-annual")
    print("=" * 82)
    print(f"  {'Strategy':<26} {'Mo':>4} {'AnnRet':>8} {'Vol':>7} {'Sharpe':>7} "
          f"{'MDD':>8} {'CalmSh':>7} {'PanicSh':>8}")
    print("  " + "-" * 78)
    for r in rows:
        print(f"  {r['label']:<26} {r['months']:>4} {r['ann_ret']:>7.1%} {r['vol']:>6.1%} "
              f"{r['sharpe']:>7.2f} {r['mdd']:>7.1%} {r['calm']:>7.2f} {r['panic']:>8.2f}")
    print("=" * 82)

    out = os.path.join(_ROOT, 'real_trade_analysis', 'results', '2026-05-31-spmo-replica-vs-xgb_returns.csv')
    pd.DataFrame({lbl: r for (lbl, r) in strategies}).to_csv(out)
    print(f"  saved {out}")
    summ = os.path.join(_ROOT, 'real_trade_analysis', 'results', '2026-05-31-spmo-replica-vs-xgb_summary.csv')
    pd.DataFrame(rows).to_csv(summ, index=False)
    print(f"  saved {summ}")


if __name__ == '__main__':
    main()
