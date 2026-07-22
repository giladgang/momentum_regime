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
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

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
    improvements, so the result is guaranteed >= the best static band. It is
    a coordinate-wise LOCAL optimum of the (bin -> width) function class, not
    a verified global max, and is therefore a conservative (lower-bound)
    estimate of the true bin-oracle ceiling. It also varies E_exit only
    (entry fixed at 10%), a conservative subclass of the full (E_enter,
    E_exit) design, so the measured ceiling is a lower bound on the true
    (E_enter, E_exit) ceiling too."""
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
    cluster_bins = _bins_clusters(feat)
    grid_bins = _bins_grid(feat)

    E_star, _ = best_static(panel, sp_pack)
    _, exit_c = oracle_bin_ir(panel, feat, sp_pack, cluster_bins)
    _, exit_g = oracle_bin_ir(panel, feat, sp_pack, grid_bins)
    cluster_bin_of_date = dict(zip(feat.index, cluster_bins.reindex(feat.index)))
    grid_bin_of_date = dict(zip(feat.index, grid_bins.reindex(feat.index)))
    bands = {
        'static': policy_band(10, E_star),
        # headline ceiling (best oracle reported by run_gate0)
        'oracle_cluster': _policy_perbin(exit_c, cluster_bin_of_date, default_exit=E_star),
        'oracle_grid': _policy_perbin(exit_g, grid_bin_of_date, default_exit=E_star),
    }

    rows = []
    for name in ['static', 'oracle_cluster', 'oracle_grid']:
        pol = bands[name]
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
    cl_panic = df.loc[df['band'] == 'oracle_cluster', 'to_panic'].iloc[0]
    # verdict is based on the CLUSTER oracle: run_gate0 reports the best of
    # {cluster, grid} oracles as the headline ceiling, and the cluster oracle
    # is that headline band (net IR ~0.464 vs the grid oracle's ~0.440), so
    # the implementability verdict must describe the band actually claimed
    # as the ceiling, not the (also-reported) grid oracle.
    direction = ('MORE in panic than static (FLAGGED: panic trading is harder / costlier)'
                 if cl_panic > sp_panic + 1e-9
                 else 'LESS (or equal) in panic than static (implementable, on-narrative)')
    df.to_csv(os.path.join(OUT_DIR, 'tv_band_gate0_impl.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'tv_band_gate0_impl.md'), 'w') as f:
        f.write('# Gate 0 — implementability (G3): direction + panic-stress cost\n\n')
        f.write('Per-band panic vs calm turnover, and net IR under panic-month (pi>=0.5) '
                'half-spread stress multipliers, for the static band and BOTH feature-binned '
                'oracles (cluster and grid). run_gate0 reports the headline ceiling as the '
                'best of {oracle_cluster, oracle_grid} net IR, which is the oracle_cluster '
                'band; direction/stress below are shown for both oracles, but the verdict is '
                'based on oracle_cluster (the headline-ceiling band), i.e. does it trade MORE '
                'or LESS in panic than the static band.\n\n')
        f.write(df.to_string(index=False))
        f.write(f'\n\n**Oracle (cluster, headline ceiling) vs static panic-turnover '
                f'direction: {direction}**\n')
    return {'table': df, 'direction': direction}


def make_feature_policy(ee_ex_fn):
    """simulate-compatible policy from a per-date (E_enter_pct, E_exit_pct) function."""
    def _policy(g, held, pi_t):
        t = g['date'].iloc[0]
        ee_pct, ex_pct = ee_ex_fn(t)
        ee, ex = ee_pct / 100.0, ex_pct / 100.0
        n = len(g)
        ranked = g.sort_values(['score_pi', 'permno'],
                               ascending=[False, True])['permno'].tolist()
        rank_pct = {p: (i + 1) / n for i, p in enumerate(ranked)}
        univ = set(ranked)
        keep = {p for p in held if p in univ and rank_pct[p] <= ex}
        add = {p for p in ranked if rank_pct[p] <= ee}
        return keep | add
    return _policy


def _train_subpanel(panel, feat, train_mask):
    train_dates = set(feat.index[train_mask])
    return panel[panel['date'].isin(train_dates)]


def walkforward(panel, feat, sp_pack, fit_fn, start_oos=2013, flat_bp=None):
    years = sorted({d.year for d in feat.index if d.year >= start_oos})
    stitched = []
    for Y in years:
        train_mask = feat.index.year < Y
        if train_mask.sum() < 24:
            continue
        policy = fit_fn(panel, feat, sp_pack, train_mask)
        m, led = simulate(panel, policy)
        cost = price(led, sp_pack, flat_bp=flat_bp)
        r = net(m['active'], cost)
        stitched.append(r[r.index.year == Y])
    return pd.concat(stitched).sort_index()


def fit_monthly(panel, feat, sp_pack, train_mask):
    return policy_monthly


def fit_static(panel, feat, sp_pack, train_mask):
    sub = _train_subpanel(panel, feat, train_mask)
    E, _ = best_static(sub, sp_pack)
    return policy_band(10, E)


def fit_cluster_band(panel, feat, sp_pack, train_mask, exit_grid=(15, 20, 25, 30, 40)):
    # K=4 z-curve-level bins with edges fit on TRAIN only; per-bin exit width by
    # coordinate ascent (seeded at best static) on the TRAIN sub-panel.
    ftr = feat.loc[train_mask]
    lvl_tr = ftr[[f'zc_{h}' for h in range(1, 13)]].mean(axis=1)
    _, edges = pd.qcut(lvl_tr, 4, labels=False, retbins=True, duplicates='drop')
    edges = edges.copy(); edges[0] = -np.inf; edges[-1] = np.inf
    lvl_all = feat[[f'zc_{h}' for h in range(1, 13)]].mean(axis=1)
    bin_all = pd.cut(lvl_all, bins=edges, labels=False, include_lowest=True)
    bin_of_date = {d: (int(b) if pd.notna(b) else -1) for d, b in bin_all.items()}
    uniq = sorted({b for d, b in bin_of_date.items() if train_mask[feat.index.get_loc(d)]})

    sub = _train_subpanel(panel, feat, train_mask)
    E0, _ = best_static(sub, sp_pack)
    width = {b: E0 for b in uniq}

    def _ir(assign):
        pol = make_feature_policy(lambda t: (10, assign.get(bin_of_date.get(t, -1), E0)))
        m, led = simulate(sub, pol)
        return net_ir(net(m['active'], price(led, sp_pack)))

    cur = _ir(width)
    improved = True
    while improved:
        improved = False
        for b in uniq:
            best_w, best_ir = width[b], cur
            for w in exit_grid:
                if w == width[b]:
                    continue
                trial = dict(width); trial[b] = w
                v = _ir(trial)
                if v > best_ir + 1e-12:
                    best_w, best_ir = w, v
            if best_w != width[b]:
                width[b] = best_w; cur = best_ir; improved = True

    return make_feature_policy(lambda t: (10, width.get(bin_of_date.get(t, -1), E0)))


def trailing_optimal_targets(panel, feat, sp_pack, window=36,
                             enter_grid=(5, 10, 15), exit_grid=(15, 20, 25, 30, 40)):
    """Per month t: the (E_enter, E_exit) maximizing net IR over the trailing
    `window` months ending at t. Supervised target for Trainers B/C."""
    dates = list(feat.index)
    combos = [(ee, ex) for ee in enter_grid for ex in exit_grid if ee <= ex]
    series = {}
    for ee, ex in combos:
        m, led = simulate(panel, policy_band(ee, ex))
        series[(ee, ex)] = net(m['active'], price(led, sp_pack))
    rows = []
    for i, t in enumerate(dates):
        win = dates[max(0, i - window + 1):i + 1]
        best, best_ir = (10, 20), -np.inf
        for c in combos:
            ir = net_ir(series[c].reindex(win).dropna())
            if ir > best_ir:
                best, best_ir = c, ir
        rows.append({'date': t, 'tgt_enter': best[0], 'tgt_exit': best[1]})
    return pd.DataFrame(rows).set_index('date')


_COMP_COLS = ['level', 'slope', 'curv', 'pi']


def _feature_matrix(feat, cols, train_mask):
    comp = compress_features(feat)[cols]
    mu = comp.loc[train_mask].mean()
    sd = comp.loc[train_mask].std(ddof=0).replace(0.0, 1.0)
    return (comp - mu) / sd, mu, sd


def fit_trainer_A(panel, feat, sp_pack, train_mask, feat_cols=None):
    """Linear-exp band policy E=clip(base*exp(w.z),lo,hi), fit by Nelder-Mead
    maximizing TRAIN net IR. Compressed 4-dim features keep the search low-dim."""
    cols = feat_cols or _COMP_COLS
    Z, _, _ = _feature_matrix(feat, cols, train_mask)
    sub = _train_subpanel(panel, feat, train_mask)
    E0, _ = best_static(sub, sp_pack)
    k = len(cols)

    def policy_from(theta):
        be, bx = theta[0], theta[1]
        we, wx = theta[2:2 + k], theta[2 + k:2 + 2 * k]

        def ee_ex(t):
            z = Z.loc[t].values
            ee = float(np.clip(be * np.exp(z @ we), 5, 60))
            ex = float(np.clip(bx * np.exp(z @ wx), ee, 60))
            return ee, ex
        return make_feature_policy(ee_ex)

    def neg_ir(theta):
        m, led = simulate(sub, policy_from(theta))
        return -net_ir(net(m['active'], price(led, sp_pack)))

    theta0 = np.concatenate([[10.0, float(E0)], np.zeros(2 * k)])
    res = minimize(neg_ir, theta0, method='Nelder-Mead',
                   options={'maxiter': 200, 'xatol': 1e-2, 'fatol': 1e-4})
    theta = res.x if np.isfinite(res.fun) else theta0
    if -neg_ir(theta) < -neg_ir(theta0):     # never worse than the static seed
        theta = theta0
    return policy_from(theta)


_FULL_COLS = ([f'zc_{h}' for h in range(1, 13)] +
              [f'cs_{h}' for h in range(1, 13)] + ['pi'])


def _fit_supervised(panel, feat, sp_pack, train_mask, window, make_model):
    tgt = trailing_optimal_targets(panel, feat, sp_pack, window=window)
    mu = feat.loc[train_mask, _FULL_COLS].mean()
    sd = feat.loc[train_mask, _FULL_COLS].std(ddof=0).replace(0.0, 1.0)
    Z = (feat[_FULL_COLS] - mu) / sd
    tr_idx = feat.index[train_mask]
    Xtr = Z.loc[tr_idx].values
    me = make_model().fit(Xtr, tgt.loc[tr_idx, 'tgt_enter'].values)
    mx = make_model().fit(Xtr, tgt.loc[tr_idx, 'tgt_exit'].values)

    def ee_ex(t):
        z = Z.loc[t].values.reshape(1, -1)
        ee = float(np.clip(me.predict(z)[0], 5, 15))
        ex = float(np.clip(mx.predict(z)[0], ee, 40))
        return ee, ex
    return make_feature_policy(ee_ex)


def fit_trainer_B(panel, feat, sp_pack, train_mask, window=36):
    return _fit_supervised(panel, feat, sp_pack, train_mask, window,
                           lambda: GradientBoostingRegressor(
                               n_estimators=200, max_depth=2, learning_rate=0.05,
                               subsample=0.8, random_state=0))


def fit_trainer_C(panel, feat, sp_pack, train_mask, window=36):
    return _fit_supervised(panel, feat, sp_pack, train_mask, window,
                           lambda: Ridge(alpha=10.0))


def rebound_band_spec(panel, feat, sp_pack, train_mask, widen_bins,
                      widen_grid=(30, 40, 50, 60)):
    """Rebound-theory band: WIDEN (trade less) in the below-average z-curve bins
    (bin 0 ~ thesis cluster 4, bin 1 ~ cluster 3), never tighter than static.
    Widen width tuned on TRAIN net IR. Returns (policy, bin_of_date, width_by_bin)."""
    zc = [f'zc_{h}' for h in range(1, 13)]
    ftr = feat.loc[train_mask]
    _, edges = pd.qcut(ftr[zc].mean(axis=1), 4, labels=False,
                       retbins=True, duplicates='drop')
    edges = edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_all = pd.cut(feat[zc].mean(axis=1), bins=edges, labels=False,
                     include_lowest=True)
    bin_of_date = {d: (int(b) if pd.notna(b) else -1) for d, b in bin_all.items()}
    sub = _train_subpanel(panel, feat, train_mask)
    E0, _ = best_static(sub, sp_pack)

    def width_map(w):
        return {b: (w if b in widen_bins else E0) for b in range(4)}

    def ir_for(wbb):
        pol = make_feature_policy(
            lambda t, m=wbb: (10, m.get(bin_of_date.get(t, -1), E0)))
        mo, led = simulate(sub, pol)
        return net_ir(net(mo['active'], price(led, sp_pack)))

    best_wbb, best_ir = width_map(E0), ir_for(width_map(E0))
    for w in widen_grid:
        if w <= E0:
            continue
        wbb = width_map(w)
        v = ir_for(wbb)
        if v > best_ir + 1e-12:
            best_wbb, best_ir = wbb, v
    policy = make_feature_policy(
        lambda t, m=best_wbb: (10, m.get(bin_of_date.get(t, -1), E0)))
    return policy, bin_of_date, best_wbb


def fit_rebound_c4(panel, feat, sp_pack, train_mask):
    return rebound_band_spec(panel, feat, sp_pack, train_mask, (0,))[0]


def fit_rebound_c3(panel, feat, sp_pack, train_mask):
    return rebound_band_spec(panel, feat, sp_pack, train_mask, (1,))[0]


def fit_rebound_both(panel, feat, sp_pack, train_mask):
    return rebound_band_spec(panel, feat, sp_pack, train_mask, (0, 1))[0]


def band_width_attributes(feat, bin_of_date, width_by_bin, pi_cut=0.5):
    zc = [f'zc_{h}' for h in range(1, 13)]
    cs = [f'cs_{h}' for h in range(1, 13)]
    lvl, csl = feat[zc].mean(axis=1), feat[cs].mean(axis=1)
    rows = []
    for b in sorted(set(bin_of_date.values())):
        idx = feat.index.isin([x for x, bb in bin_of_date.items() if bb == b])
        s = feat[idx]
        rows.append({'bin': b, 'exit_width': width_by_bin.get(b, np.nan),
                     'n': int(idx.sum()), 'avg_pi': float(s['pi'].mean()),
                     'frac_panic': float((s['pi'] >= pi_cut).mean()),
                     'avg_zc_level': float(lvl[idx].mean()),
                     'avg_cs_mom': float(csl[idx].mean())})
    return pd.DataFrame(rows)


def _delta_ci(arm_series, static_series, n_boot=10000, block=12, seed=0):
    """(delta, lo, hi, se, p) for net_ir(arm) - net_ir(static), circular block
    resample. p is a two-sided bootstrap p-value (fraction of resamples on the
    opposite side of 0, doubled)."""
    a, b = static_series.align(arm_series, join='inner')
    av, bv, n = a.values, b.values, len(a)
    delta = net_ir(b) - net_ir(a)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    stats = np.empty(n_boot)
    for j in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) % n
                              for s in rng.integers(0, n, nb)])[:n]
        stats[j] = net_ir(pd.Series(bv[idx])) - net_ir(pd.Series(av[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    p = min(1.0, 2.0 * min((stats <= 0).mean(), (stats >= 0).mean()))
    return float(delta), float(lo), float(hi), float(stats.std(ddof=1)), float(p)


def _bh_reject(pvals, q=0.05):
    """Benjamini-Hochberg: boolean array, True where H0 rejected at FDR q."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        reject[order[:kmax + 1]] = True
    return reject


def run_gate1(start_oos=2013, flat_bp=None, arms=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    panel = load_panel()
    feat = month_features(panel)
    panel = panel[panel['date'].isin(feat.index)]
    sp = load_spreads()
    if arms is None:
        arms = {'monthly': fit_monthly, 'static': fit_static,
                'cluster_band': fit_cluster_band, 'trainer_A': fit_trainer_A,
                'trainer_B': fit_trainer_B, 'trainer_C': fit_trainer_C,
                'rebound_c4': fit_rebound_c4, 'rebound_c3': fit_rebound_c3,
                'rebound_both': fit_rebound_both}
    assert 'static' in arms, "run_gate1 requires the 'static' arm as the baseline"
    series = {name: walkforward(panel, feat, sp, fn, start_oos, flat_bp)
              for name, fn in arms.items()}
    static_s = series['static']
    rows = []
    for name, s in series.items():
        delta, lo, hi, se, p = _delta_ci(s, static_s)
        rows.append({'arm': name, 'oos_ir': net_ir(s), 'minus_static': delta,
                     'ci_lo': lo, 'ci_hi': hi, 'se': se, 'p': p,
                     'excl0': bool(lo > 0 or hi < 0),
                     'passes_1se': bool(delta - se > 0)})
    df = pd.DataFrame(rows)
    learned_names = ['cluster_band', 'trainer_A', 'trainer_B', 'trainer_C',
                     'rebound_c4', 'rebound_c3', 'rebound_both']
    lmask = df['arm'].isin(learned_names)
    # BH-FDR across the learned arms (multiple-testing correction, q=0.05)
    df['bh_sig'] = False
    lp = df.loc[lmask, 'p'].values
    if len(lp):
        df.loc[lmask, 'bh_sig'] = _bh_reject(lp, q=0.05)
    learned = df[lmask]
    # honest win: positive AND survives BH-FDR (CI/1-SE reported but not the gate)
    g1_win = bool(((learned['minus_static'] > 0) & learned['bh_sig']).any())
    df.to_csv(os.path.join(OUT_DIR, 'tv_band_gate1.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'tv_band_gate1.md'), 'w') as f:
        f.write('# Gate 1 — learned real-time term-structure band (walk-forward OOS)\n\n')
        f.write(f'OOS {start_oos}-2025, annual re-fit on prior months only, stitched. '
                'g1_win requires a learned arm with minus_static>0 that survives BH-FDR '
                '(q=0.05) across the 7 learned arms (the multiple-testing correction the '
                'prior campaign used). excl0 (bootstrap CI) and passes_1se are reported '
                'alongside but are NOT the gate, since testing 7 arms inflates single-arm '
                'significance.\n\n')
        f.write(df.to_string(index=False))
        f.write(f'\n\n**g1_win: {g1_win}**\n')
    if 'rebound_both' in arms:
        full_mask = np.ones(len(feat), dtype=bool)
        _, bod, wbb = rebound_band_spec(panel, feat, sp, full_mask, widen_bins=(0, 1))
        band_width_attributes(feat, bod, wbb).to_csv(
            os.path.join(OUT_DIR, 'tv_band_gate1_attributes.csv'), index=False)
    return {'table': df, 'g1_win': g1_win}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='gate0', choices=['gate0', 'gate1'])
    args = ap.parse_args()
    if args.stage == 'gate1':
        res = run_gate1()
        print('g1_win:', res['g1_win'])
        print(res['table'].to_string(index=False))
        return
    res = run_gate0()
    print('regime turnover:', regime_turnover(load_panel()))
    print('g0_pass:', res['g0_pass'])
    print(res['table'].to_string(index=False))
    impl = run_gate0_impl()
    print('implementability direction:', impl['direction'])
    print(impl['table'].to_string(index=False))


if __name__ == '__main__':
    main()
