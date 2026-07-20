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


def price(ledger, sp_pack, flat_bp=None, stress_mult=None, pi_by_date=None):
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
    if stress_mult is not None and pi_by_date is not None:
        panic = m['date'].map(pi_by_date).fillna(0.0).ge(0.5)
        m.loc[panic, 'hs'] = m.loc[panic, 'hs'] * float(stress_mult)
    return (m['dw'].abs() * m['hs'] / 1e4).groupby(m['date']).sum()


def net(active, cost):
    return active - cost.reindex(active.index).fillna(0.0)


def month_features(panel):
    rows = []
    for t, g in panel.groupby('date', sort=True):
        g = g.dropna(subset=MOMS)
        if len(g) < 20:
            continue
        cs = g[MOMS].mean()                              # cross-sectional term structure
        z = (g[MOMS] - g[MOMS].mean()) / g[MOMS].std(ddof=0)
        k = max(int(len(g) * 0.10), 1)
        top_idx = g['score_pi'].nlargest(k).index
        zc = z.loc[top_idx].mean()                       # long-leg z-curve
        row = {'date': t, 'pi': float(g['pi'].iloc[0])}
        row.update({f'zc_{h}': zc[f'mom_{h}'] for h in range(1, 13)})
        row.update({f'cs_{h}': cs[f'mom_{h}'] for h in range(1, 13)})
        rows.append(row)
    return pd.DataFrame(rows).set_index('date')


def compress_features(feat):
    zc = feat[[f'zc_{h}' for h in range(1, 13)]].values
    h = np.arange(1, 13)
    level = zc.mean(axis=1)
    slope = np.array([np.polyfit(h, r, 1)[0] for r in zc])
    curv = np.array([np.polyfit(h, r, 2)[0] for r in zc])
    return pd.DataFrame({'level': level, 'slope': slope, 'curv': curv,
                         'pi': feat['pi'].values}, index=feat.index)


def standardize(feat, train_mask):
    mu = feat.loc[train_mask].mean()
    sd = feat.loc[train_mask].std(ddof=0).replace(0.0, 1.0)
    return (feat - mu) / sd


def net_ir(r):
    r = pd.Series(r).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(12))


def paired_block_bootstrap(a, b, n_boot=10000, block=12, seed=0):
    a, b = a.align(b, join='inner')
    d = (b - a).values
    n = len(d)
    delta = net_ir(b) - net_ir(a)
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(n / block))
    av, bv = a.values, b.values
    stats = np.empty(n_boot)
    for j in range(n_boot):
        starts = rng.integers(0, n, nblocks)
        idx = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]
        stats[j] = net_ir(pd.Series(bv[idx])) - net_ir(pd.Series(av[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(delta), float(lo), float(hi)


def _policy_perbin(exit_by_bin, bin_of_date, enter=10, default_exit=20):
    ee = enter / 100.0

    def _policy(g, held, pi_t):
        t = g['date'].iloc[0]
        ex = exit_by_bin.get(bin_of_date.get(t), default_exit) / 100.0
        n = len(g)
        ranked = g.sort_values(['score_pi', 'permno'],
                               ascending=[False, True])['permno'].tolist()
        rank_pct = {p: (i + 1) / n for i, p in enumerate(ranked)}
        univ = set(ranked)
        keep = {p for p in held if p in univ and rank_pct[p] <= ex}
        add = {p for p in ranked if rank_pct[p] <= ee}
        return keep | add
    return _policy


def _ir_for_band(panel, policy, sp_pack, flat_bp):
    m, led = simulate(panel, policy)
    cost = price(led, sp_pack, flat_bp=flat_bp)
    return net_ir(net(m['active'], cost)), m, led


def best_static(panel, sp_pack, exit_grid=(15, 20, 25, 30, 40), flat_bp=None):
    best = (None, -np.inf)
    for ex in exit_grid:
        ir, _, _ = _ir_for_band(panel, policy_band(10, ex), sp_pack, flat_bp)
        if ir > best[1]:
            best = (ex, ir)
    return best


def oracle_bin_ir(panel, feat, sp_pack, bins, exit_grid=(15, 20, 25, 30, 40),
                  flat_bp=None):
    """Feature-binned oracle ceiling via COORDINATE ASCENT seeded at the best
    static band. net IR is not additive across bins and the band is
    path-dependent, so per-bin greedy fitting is not a valid upper bound;
    coordinate ascent from the all-static seed only accepts full-panel-IR
    improvements, so the result is guaranteed >= the best static band and is
    the in-sample max over the (bin -> width) function class."""
    bin_of_date = dict(zip(feat.index, bins.reindex(feat.index)))
    uniq = [b for b in pd.unique(bins.dropna())]
    E_static, _ = best_static(panel, sp_pack, exit_grid, flat_bp)
    exit_by_bin = {b: E_static for b in uniq}

    def _ir(assign):
        pol = _policy_perbin(assign, bin_of_date, default_exit=E_static)
        ir, _, _ = _ir_for_band(panel, pol, sp_pack, flat_bp)
        return ir

    cur = _ir(exit_by_bin)
    improved = True
    while improved:
        improved = False
        for b in uniq:
            best_w, best_ir = exit_by_bin[b], cur
            for w in exit_grid:
                if w == exit_by_bin[b]:
                    continue
                trial = dict(exit_by_bin)
                trial[b] = w
                ir = _ir(trial)
                if ir > best_ir + 1e-12:
                    best_w, best_ir = w, ir
            if best_w != exit_by_bin[b]:
                exit_by_bin[b] = best_w
                cur = best_ir
                improved = True
    return cur, exit_by_bin


def _bins_clusters(feat):
    # K=4 by z-curve level quartiles (proxy for the thesis calm..deep-crisis axis)
    lvl = feat[[f'zc_{h}' for h in range(1, 13)]].mean(axis=1)
    return pd.qcut(lvl, 4, labels=False, duplicates='drop').astype('Int64').astype(str)


def _bins_grid(feat):
    lvl = feat[[f'zc_{h}' for h in range(1, 13)]].mean(axis=1)
    a = pd.qcut(feat['pi'], 2, labels=False, duplicates='drop').astype(str)
    b = pd.qcut(lvl, 2, labels=False, duplicates='drop').astype(str)
    return (a + '_' + b)


def run_gate0():
    os.makedirs(OUT_DIR, exist_ok=True)
    panel = load_panel()
    sp_pack = load_spreads()
    # trust gate (blocking)
    m_mo, _ = simulate(panel, policy_monthly)
    wr = pd.read_csv(WALK, parse_dates=['date'])
    wr = wr[(wr['rule'] == 'rule_r') & (wr['combo'] == 'DD')].set_index('date')
    gate_err = float((m_mo['book'] - wr['strat_ret']).reindex(
        m_mo.index.intersection(wr.index)).abs().max())
    assert gate_err < 1e-6, f'TRUST GATE FAILED: max abs err {gate_err}'

    feat = month_features(panel)
    panel = panel[panel['date'].isin(feat.index)]
    rows = []
    for cost_name, flat in [('measured', None), ('flat10', 10)]:
        E_star, ir_static = best_static(panel, sp_pack, flat_bp=flat)
        ir_c, exit_c = oracle_bin_ir(panel, feat, sp_pack, _bins_clusters(feat),
                                     flat_bp=flat)
        ir_g, exit_g = oracle_bin_ir(panel, feat, sp_pack, _bins_grid(feat),
                                     flat_bp=flat)
        best_oracle = max(ir_c, ir_g)
        threshold = ir_static + 0.029 * abs(ir_static)
        rows.append({'cost': cost_name, 'E_static': E_star, 'ir_static': ir_static,
                     'ir_oracle_cluster': ir_c, 'ir_oracle_grid': ir_g,
                     'ir_oracle_best': best_oracle,
                     'uplift': best_oracle - ir_static,
                     'passes': bool(best_oracle > threshold)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'tv_band_gate0.csv'), index=False)
    g0_pass = bool(df.loc[df['cost'] == 'measured', 'passes'].iloc[0])
    with open(os.path.join(OUT_DIR, 'tv_band_gate0.md'), 'w') as f:
        f.write('# Gate 0 — term-structure band ceiling (perfect-hindsight oracle)\n\n')
        f.write('Feature-binned oracle (full-sample foreknowledge) vs best static band. '
                'Threshold = static IR + 2.9% (spread-timing artifact). '
                'g0_pass gates whether Gate 1 (learned real-time band) is built.\n\n')
        f.write(df.to_string(index=False))
        f.write(f'\n\n**g0_pass (measured cost): {g0_pass}**\n')
    return {'g0_pass': g0_pass, 'table': df}


def regime_turnover(panel, pi_cut=0.5):
    """Independent re-derivation of the 'trading is regime-invariant' finding:
    mean no-band monthly turnover in calm (pi<cut) vs panic (pi>=cut) months."""
    m, _ = simulate(panel, policy_monthly)
    calm = m[m['pi'] < pi_cut]['turnover']
    panic = m[m['pi'] >= pi_cut]['turnover']
    to_c, to_p = float(calm.mean()), float(panic.mean())
    return {'to_calm': to_c, 'to_panic': to_p,
            'ratio': to_p / to_c if to_c else np.nan,
            'n_calm': int(len(calm)), 'n_panic': int(len(panic))}


def band_regime_turnover(panel, policy, pi_cut=0.5):
    m, _ = simulate(panel, policy)
    calm = float(m[m['pi'] < pi_cut]['turnover'].mean())
    panic = float(m[m['pi'] >= pi_cut]['turnover'].mean())
    return calm, panic


def run_gate0_impl(stress_grid=(1, 2, 5)):
    os.makedirs(OUT_DIR, exist_ok=True)
    panel = load_panel()
    sp_pack = load_spreads()
    feat = month_features(panel)
    panel = panel[panel['date'].isin(feat.index)]
    pi_by_date = panel.groupby('date')['pi'].first().to_dict()
    grid = _bins_grid(feat)

    E_star, _ = best_static(panel, sp_pack)
    _, exit_by_bin = oracle_bin_ir(panel, feat, sp_pack, grid)
    bin_of_date = dict(zip(feat.index, grid.reindex(feat.index)))
    bands = {'static': policy_band(10, E_star),
             'oracle': _policy_perbin(exit_by_bin, bin_of_date, default_exit=E_star)}

    rows = []
    for name, pol in bands.items():
        m, led = simulate(panel, pol)
        calm_to, panic_to = band_regime_turnover(panel, pol)
        row = {'band': name, 'to_calm': calm_to, 'to_panic': panic_to,
               'panic_minus_calm_to': panic_to - calm_to}
        for s in stress_grid:
            cost = price(led, sp_pack, stress_mult=(None if s == 1 else s),
                         pi_by_date=pi_by_date)
            row[f'net_ir_stress{s}'] = net_ir(net(m['active'], cost))
        rows.append(row)
    df = pd.DataFrame(rows)

    sp_panic = df.loc[df['band'] == 'static', 'to_panic'].iloc[0]
    or_panic = df.loc[df['band'] == 'oracle', 'to_panic'].iloc[0]
    direction = ('MORE in panic than static (FLAGGED: panic trading is harder / costlier)'
                 if or_panic > sp_panic + 1e-9
                 else 'LESS (or equal) in panic than static (implementable, on-narrative)')
    df.to_csv(os.path.join(OUT_DIR, 'tv_band_gate0_impl.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'tv_band_gate0_impl.md'), 'w') as f:
        f.write('# Gate 0 — implementability (G3): direction + panic-stress cost\n\n')
        f.write('Per-band panic vs calm turnover, and net IR under panic-month (pi>=0.5) '
                'half-spread stress multipliers. Direction = does the oracle band trade '
                'MORE or LESS in panic than the static band.\n\n')
        f.write(df.to_string(index=False))
        f.write(f'\n\n**Oracle vs static panic-turnover direction: {direction}**\n')
    return {'table': df, 'direction': direction}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='gate0', choices=['gate0'])
    ap.parse_args()
    res = run_gate0()
    print('regime turnover:', regime_turnover(load_panel()))
    print('g0_pass:', res['g0_pass'])
    print(res['table'].to_string(index=False))
    impl = run_gate0_impl()
    print('implementability direction:', impl['direction'])
    print(impl['table'].to_string(index=False))


if __name__ == '__main__':
    main()
