"""Clean-room term-structure time-varying band — Gate 0 (ceiling + trust gate).
Design: docs/superpowers/specs/2026-07-20-tv-band-design.md
Plan:   docs/superpowers/plans/2026-07-20-tv-band-gate0.md

Imports NOTHING from paper.execution / paper.banding_study by design: this is an
independent re-implementation whose `monthly` policy must reproduce walk_returns.csv.
Usage: .venv/bin/python -m paper.tv_band --stage gate0
"""
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from paper import config as C                                   # noqa: E402
from paper.src.clusters import MOMS                             # noqa: E402  (col names only)

XSEC = os.path.join(C.RESULTS, 'xsec')
HS_PARQUET = os.path.join(C.RESULTS, 's5', 'half_spreads.parquet')
STOCKS = C.STOCKS_PARQUET
WALK = os.path.join(C.RESULTS, 'walk_returns.csv')
OUT_DIR = os.path.join(C.RESULTS, 'banding_study')
YEARS = list(range(2011, 2026))


def load_panel():
    """Monthly main-model panel: xsec _DD frames joined to the 12 momentum horizons."""
    frames = []
    for y in YEARS:
        fp = os.path.join(XSEC, f'xsec_{y}_DD.parquet')
        frames.append(pd.read_parquet(
            fp, columns=['date', 'permno', 'me', 'pi', 'score_pi', 'ret_fwd']))
    x = pd.concat(frames, ignore_index=True)
    st = pd.read_parquet(STOCKS, columns=['date', 'permno'] + MOMS)
    p = x.merge(st, on=['date', 'permno'], how='left')
    return p.sort_values(['date', 'permno']).reset_index(drop=True)


def load_spreads():
    """(sp[permno,ym,hs_bp], per-month median ffilled, earliest-month median)."""
    sp = pd.read_parquet(HS_PARQUET, columns=['permno', 'ym', 'hs'])
    sp = sp.dropna(subset=['hs']).copy()
    sp['hs'] = sp['hs'] * 1e4                        # decimal -> bp
    month_med = sp.groupby('ym')['hs'].median()
    full = pd.period_range(month_med.index.min(), month_med.index.max(), freq='M')
    month_med = month_med.reindex(full).ffill()
    return sp, month_med, float(month_med.iloc[0])


def _vw_weights(g, members):
    me = g.set_index('permno')['me'].reindex(sorted(members))
    w = me / me.sum()
    return dict(zip(w.index, w.values))


def policy_benchmark(g, held, pi_t):
    return set(g['permno'])


def policy_monthly(g, held, pi_t):
    k = max(int(len(g) * 0.10), 1)
    return set(g.sort_values(['score_pi', 'permno'], ascending=[False, True])
               ['permno'].head(k))


def simulate(panel, policy):
    """Long-only VW active-return backtest with drift-adjusted turnover.
    `policy(g, held, pi_t) -> set(permno)`; bench return uses the VW universe.
    """
    rows, ledger = [], []
    hold, prev_ret = {}, {}
    for t, g in panel.groupby('date', sort=True):
        g = g.dropna(subset=['score_pi', 'me', 'ret_fwd'])
        pi_t = float(g['pi'].iloc[0])
        # drift last month's book by realized return, renormalize
        drift = {p: w * (1 + prev_ret.get(p, 0.0)) for p, w in hold.items()}
        tot = sum(drift.values()) or 1.0
        drift = {p: w / tot for p, w in drift.items()}

        members = policy(g, set(hold), pi_t)
        tgt = _vw_weights(g, members)

        traded = 0.0
        for p in set(drift) | set(tgt):
            dw = tgt.get(p, 0.0) - drift.get(p, 0.0)
            if abs(dw) < 1e-12:
                continue
            traded += abs(dw)
            ledger.append({'date': t, 'permno': p, 'dw': dw})
        ret = g.set_index('permno')['ret_fwd']
        book = float(sum(w * ret[p] for p, w in tgt.items()))
        bench = float((g['me'] / g['me'].sum() * g['ret_fwd']).sum())
        rows.append({'date': t, 'book': book, 'bench': bench,
                     'active': book - bench, 'turnover': traded / 2,
                     'n_names': len(tgt), 'pi': pi_t})
        hold = tgt
        prev_ret = {p: float(ret[p]) for p in tgt}
    m = pd.DataFrame(rows).set_index('date')
    return m, pd.DataFrame(ledger)


def policy_band(E_enter_pct, E_exit_pct):
    ee, ex = E_enter_pct / 100.0, E_exit_pct / 100.0

    def _policy(g, held, pi_t):
        n = len(g)
        ranked = g.sort_values(['score_pi', 'permno'],
                               ascending=[False, True])['permno'].tolist()
        rank_pct = {p: (i + 1) / n for i, p in enumerate(ranked)}
        univ = set(ranked)
        keep = {p for p in held if p in univ and rank_pct[p] <= ex}
        add = {p for p in ranked if rank_pct[p] <= ee}
        return keep | add
    return _policy


def price(ledger, sp_pack, flat_bp=None):
    sp, month_med, full_med = sp_pack
    if ledger is None or len(ledger) == 0:
        return pd.Series(dtype=float)
    m = ledger.copy()
    m['ym'] = m['date'].dt.to_period('M')
    if flat_bp is not None:
        m['hs'] = float(flat_bp)
    else:
        m = m.merge(sp, on=['permno', 'ym'], how='left')
        m['hs'] = m['hs'].fillna(m['ym'].map(month_med)).fillna(full_med)
    cost = (m['dw'].abs() * m['hs'] / 1e4).groupby(m['date']).sum()
    return cost


def net(active, cost):
    return active - cost.reindex(active.index).fillna(0.0)
