"""Regime-conditional banding sweep: 7 strategies x band policies x costs.

Spec: docs/superpowers/specs/2026-07-15-regime-conditional-banding-study-design.md
Gates G1-G3 registered in paper/results/PAPER_NOTES.md (2026-07-15, pre-run).

Usage:
  .venv/bin/python -m paper.banding_study --stage s6gate
  .venv/bin/python -m paper.banding_study --stage sweep  --workers 7
  .venv/bin/python -m paper.banding_study --stage report
  (--smoke: 2 strategies x tiny grid)
Outputs in paper/results/banding_study/: cells.csv, best_cells.csv,
delta_bootstrap.csv, frontier.csv, report.md

Pricing conventions:
- half_spreads.parquet stores DECIMAL half-spreads (winsor [1e-4, 0.02] =
  [1, 200] bp); converted to bp on load, so cost_t = sum_i |dw|*hs_bp/1e4.
- Missing (permno, month) hs -> that month's cross-sectional median of
  available stocks; month with NO spread data (pre-2011 until the extended
  panel lands) -> full-panel median. Fallback share exposed per cell as
  `hs_fallback_frac` (no silent fills; the sweep reruns on the extended
  panel).
- Stress columns multiply hs by m in STRESS_MULT ONLY in months where the
  strategy frame's pi >= 0.5. Benchmark = simulate(x, ('benchmark', None)),
  priced identically (benchmark pays reconstitution, S6 convention).
"""
import argparse
import ast
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from paper import config as C                                   # noqa: E402
from paper import execution as X                                # noqa: E402
from paper.src import metrics as M                              # noqa: E402
from paper.src import signals                                   # noqa: E402

HS_PARQUET = os.path.join(C.RESULTS, 's5', 'half_spreads.parquet')
S6_CELLS = os.path.join(C.RESULTS, 's6_banding', 'cells.csv')
FF_FACTORS = os.path.join(_REPO, 'data', 'ff_factors.parquet')  # READ ONLY
SMOKE_STRATEGIES = ['momentum', 'xgb']
TIERS = ['monthly', 'static_best', 'regime_best']
LO5 = ['Mkt-RF', 'size', 'value', 'robust', 'conservative']
LO6 = LO5 + ['winner']
HAC_LAGS = 6


# ── grid ──────────────────────────────────────────────────────────────────

def cell_grid():
    diag = [('nmv_band', (e, e)) for e in C.E_GRID]
    two = [('nmv_band', (ec, ep))
           for ec in C.E_GRID for ep in C.E_GRID if ec != ep]
    three = [('regime3_band', (ec, ecr, erec))
             for ec, ecr, erec in itertools.product(C.E_GRID, repeat=3)
             if ecr != erec]
    return diag + two + three


def smoke_grid():
    return [('nmv_band', (10, 10)), ('nmv_band', (20, 20)),
            ('nmv_band', (20, 10)), ('regime3_band', (20, 20, 10))]


def family(policy):
    name, arg = policy
    if name == 'nmv_band':
        return 'diag' if arg[0] == arg[1] else 'regime2'
    return 'regime3'


def _policy_from_params(params):
    arg = ast.literal_eval(params)
    return ('nmv_band', arg) if len(arg) == 2 else ('regime3_band', arg)


# ── selection + gate helpers ──────────────────────────────────────────────

def select_best(cells):
    rows = []
    for s, g in cells.groupby('strategy'):
        d = g[g['family'] == 'diag']
        r = g[g['family'] != 'diag']
        if len(d) == 0:
            raise ValueError(f'{s}: no diagonal cells in results — cannot '
                             'pick static_best')
        if len(r) == 0:
            raise ValueError(f'{s}: no off-diagonal cells in results — '
                             'cannot pick regime_best')
        bd = d.loc[d['net_ir_meas'].idxmax()]
        br = r.loc[r['net_ir_meas'].idxmax()]
        # cost_intensity = monthly-tier (10, 10) to_mo x its own mean_hs_bp;
        # both must come from the SAME (10, 10) row -- g's row order is
        # cells.csv append order (nondeterministic imap_unordered), so
        # g['mean_hs_bp'].iloc[0] is NOT necessarily the (10, 10) row.
        mono = g[g['params'] == '(10, 10)']
        if len(mono):
            ci = (float(mono['to_mo'].iloc[0])
                  * float(mono['mean_hs_bp'].iloc[0]))
        else:
            ci = float('nan')
        rows.append({'strategy': s,
                     'static_params': bd['params'],
                     'static_net_ir': bd['net_ir_meas'],
                     'regime_params': br['params'],
                     'regime_family': br['family'],
                     'regime_net_ir': br['net_ir_meas'],
                     'delta_net_ir': br['net_ir_meas'] - bd['net_ir_meas'],
                     'cost_intensity': ci})
    return pd.DataFrame(rows)


def gate2_spearman(best):
    from scipy.stats import spearmanr
    return float(spearmanr(best['cost_intensity'],
                           best['delta_net_ir']).statistic)


# ── spread pricing ────────────────────────────────────────────────────────

def load_spreads():
    """(panel[permno, ym, hs in bp], per-month median, full-panel median)."""
    sp = pd.read_parquet(HS_PARQUET, columns=['permno', 'ym', 'hs'])
    sp = sp.dropna(subset=['hs']).copy()
    sp['hs'] = sp['hs'] * 1e4                    # decimal -> basis points
    month_med = sp.groupby('ym')['hs'].median()
    return sp, month_med, float(sp['hs'].median())


def price_ledger(led, sp_pack, stress_mult=None):
    """Monthly spread cost from a trade ledger: cost_t = sum |dw|*hs/1e4.

    Returns (cost Series indexed by date, fallback fraction of trades,
    trade-weighted mean hs in bp). stress_mult multiplies hs ONLY in months
    where the formation pi >= 0.5.
    """
    sp, month_med, full_med = sp_pack
    if led is None or len(led) == 0:
        return pd.Series(dtype=float), 0.0, np.nan
    m = led[['date', 'permno', 'dw', 'pi']].copy()
    m['ym'] = m['date'].dt.to_period('M')
    m = m.merge(sp, on=['permno', 'ym'], how='left')
    fallback = float(m['hs'].isna().mean())
    m['hs'] = m['hs'].fillna(m['ym'].map(month_med)).fillna(full_med)
    if stress_mult is not None:
        m['hs'] = m['hs'] * np.where(m['pi'] >= 0.5, stress_mult, 1.0)
    vol = m['dw'].abs()
    cost = (vol * m['hs'] / 1e4).groupby(m['date']).sum()
    mean_hs = float((vol * m['hs']).sum() / vol.sum()) if vol.sum() > 0 \
        else np.nan
    return cost, fallback, mean_hs


def _net(r_gross, cost):
    return r_gross - cost.reindex(r_gross.index).fillna(0.0)


# ── long-only factor alphas ───────────────────────────────────────────────

def load_lo_design():
    """Mkt-RF + excess long legs (leg - RF), month-end indexed."""
    ff = pd.read_parquet(FF_FACTORS)
    key = {c.lower().replace('-', '').replace('_', ''): c for c in ff.columns}
    ff.index = pd.DatetimeIndex(ff.index) + pd.offsets.MonthEnd(0)
    legs = pd.read_parquet(C.FF_LONG_LEGS)[['size', 'value', 'robust',
                                            'conservative', 'winner']]
    legs.index = pd.DatetimeIndex(legs.index) + pd.offsets.MonthEnd(0)
    rf = ff[key['rf']]
    ex = legs.sub(rf.reindex(legs.index), axis=0)
    mkt = ff[key['mktrf']].rename('Mkt-RF')
    return pd.concat([mkt, ex], axis=1).dropna()


def alpha_lo(act, design, cols):
    """HAC(6) alpha of a formation-indexed active return on the LO factor
    set; factors aligned to the realization month t+1. Returns (alpha*12, t).
    """
    import statsmodels.api as sm
    y = act.dropna()
    if len(y) < 24:
        return np.nan, np.nan
    idx = (pd.DatetimeIndex(y.index) + pd.offsets.MonthEnd(0)
           + pd.offsets.MonthEnd(1))
    m = pd.DataFrame({'ret': y.values}, index=idx).join(
        design[cols], how='inner').dropna()
    if len(m) < 24:
        return np.nan, np.nan
    res = sm.OLS(m['ret'].values,
                 sm.add_constant(m[cols].values)).fit(
        cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})
    return float(res.params[0]) * 12, float(res.tvalues[0])


# ── per-cell metrics ──────────────────────────────────────────────────────

def _cell_metrics(strategy, policy, x, bench, sp_pack, design):
    """One row of cells.csv. bench: monthly df with gross/turnover/net_*."""
    r, led = X.simulate(x, policy, score_col='score')
    act_g = M.active(r['gross'], bench['gross'])
    cost, fb_frac, mean_hs = price_ledger(led, sp_pack)
    net = _net(r['gross'], cost)
    net_b = bench['net_meas']
    act_net = M.active(net, net_b)
    st = x.groupby('date')['state3'].first().reindex(r.index)
    split = pd.Timestamp(C.SPLIT_DATE)
    vol = r['traded'].mean() - 2 * bench['turnover'].mean()
    a5, t5 = alpha_lo(act_net, design, LO5)
    a6, t6 = alpha_lo(act_net, design, LO6)
    row = {'strategy': strategy,
           'family': family(policy),
           'params': str(policy[1]),
           'gross_ir': M.ir(r['gross'], bench['gross']),
           'net_ir_0': M.ir(r['gross'], bench['gross']),
           'net_ir_meas': M.ir(net, net_b),
           'to_mo': float(r['turnover'].mean()),
           'to_calm': float(r['turnover'][st == 0].mean()),
           'to_crash': float(r['turnover'][st == 1].mean()),
           'to_recovery': float(r['turnover'][st == 2].mean()),
           'n_names': float(r['n_names'].mean()),
           'breakeven_bp': (float(act_g.mean() / vol * 1e4) if vol > 0
                            else np.inf),
           'alpha_ff5lo': a5, 't_ff5lo': t5,
           'alpha_ff6lo': a6, 't_ff6lo': t6,
           'mean_hs_bp': mean_hs,
           'hs_fallback_frac': fb_frac,
           'net_ir_meas_pre': M.ir(net[net.index < split],
                                   net_b[net_b.index < split]),
           'net_ir_meas_post': M.ir(net[net.index >= split],
                                    net_b[net_b.index >= split])}
    for mult in C.STRESS_MULT:
        cst, _, _ = price_ledger(led, sp_pack, stress_mult=mult)
        row[f'net_ir_stress{mult}'] = M.ir(_net(r['gross'], cst),
                                           bench[f'net_stress{mult}'])
    return row


# ── spawn-pool sweep (mirrors pipeline.stage_folds) ───────────────────────

_WCACHE = {}


def _wget(key, fn):
    if key not in _WCACHE:
        _WCACHE[key] = fn()
    return _WCACHE[key]


def _run_cell_unit(args):
    strategy, policy, frame_path, bench_path = args
    try:
        x = _wget(('frame', frame_path),
                  lambda: pd.read_parquet(frame_path))
        bench = _wget(('bench', bench_path),
                      lambda: pd.read_parquet(bench_path))
        sp_pack = _wget('spreads', load_spreads)
        design = _wget('design', load_lo_design)
        row = _cell_metrics(strategy, policy, x, bench, sp_pack, design)
        return ('ok', row)
    except Exception as e:                                      # noqa: BLE001
        return ('failed', f'{strategy}/{policy}: {e!r}')


def _completed(out):
    if not os.path.exists(out):
        return set()
    done = pd.read_csv(out, usecols=['strategy', 'params'])
    return set(zip(done['strategy'], done['params']))


def _append_row(out, row):
    pd.DataFrame([row]).to_csv(out, mode='a', index=False,
                               header=not os.path.exists(out))


def price_cells(x, bench_gross, bench_ledger, sp_pack):
    """Benchmark-side pricing shared by every cell of one strategy: monthly
    df with gross, turnover and the measured/stressed net series."""
    bench = bench_gross[['gross', 'turnover']].copy()
    cost, _, _ = price_ledger(bench_ledger, sp_pack)
    bench['net_meas'] = _net(bench['gross'], cost)
    for mult in C.STRESS_MULT:
        cst, _, _ = price_ledger(bench_ledger, sp_pack, stress_mult=mult)
        bench[f'net_stress{mult}'] = _net(bench['gross'], cst)
    return bench


def _prep_strategy(strategy, tmp_dir):
    """Build the signal frame + priced benchmark once; persist for workers."""
    x = signals.build(strategy)
    x = x[x['date'] >= C.SWEEP_START].reset_index(drop=True)
    b, bled = X.simulate(x, ('benchmark', None), score_col='score')
    bench = price_cells(x, b, bled, load_spreads())
    frame_path = os.path.join(tmp_dir, f'frame_{strategy}.parquet')
    bench_path = os.path.join(tmp_dir, f'bench_{strategy}.parquet')
    x.to_parquet(frame_path, index=False)
    bench.to_parquet(bench_path)
    return frame_path, bench_path


def run_strategy(strategy, workers, smoke=False, out=None):
    """Sweep one strategy's grid (resume-safe on (strategy, params));
    returns that strategy's cells as a DataFrame."""
    out = out or os.path.join(C.BANDING_DIR, 'cells.csv')
    grid = smoke_grid() if smoke else cell_grid()
    done = _completed(out)
    todo = [p for p in grid if (strategy, str(p[1])) not in done]
    print(f'[sweep] {strategy}: {len(todo)} cells to run '
          f'({len(grid)} grid, done={len(grid) - len(todo)})', flush=True)
    if todo:
        tmp_dir = os.path.join(C.BANDING_DIR, 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        frame_path, bench_path = _prep_strategy(strategy, tmp_dir)
        units = [(strategy, p, frame_path, bench_path) for p in todo]
        from multiprocessing import get_context
        ctx = get_context('spawn')
        t0, n_ok = time.time(), 0
        with ctx.Pool(workers) as pool:
            for status, payload in pool.imap_unordered(_run_cell_unit, units,
                                                       chunksize=1):
                if status == 'ok':
                    _append_row(out, payload)
                    n_ok += 1
                    if n_ok % 10 == 0:
                        rate = (time.time() - t0) / n_ok
                        eta = rate * (len(units) - n_ok) / 60
                        print(f'  [{strategy} {n_ok}/{len(units)}] '
                              f'{rate:.0f}s/cell ETA {eta:.0f}m', flush=True)
                else:
                    print(f'  [FAILED] {payload}', flush=True)
        print(f'[sweep] {strategy} DONE ok={n_ok}/{len(units)} '
              f'({(time.time() - t0) / 60:.1f}m)', flush=True)
    cells = pd.read_csv(out)
    return cells[cells['strategy'] == strategy].reset_index(drop=True)


def stage_sweep(workers, smoke=False):
    os.makedirs(C.BANDING_DIR, exist_ok=True)
    out = os.path.join(C.BANDING_DIR, 'cells.csv')
    if smoke:
        out = out.replace('.csv', '_smoke.csv')
    strategies = SMOKE_STRATEGIES if smoke else C.BS_STRATEGIES
    for s in strategies:
        try:
            run_strategy(s, workers, smoke=smoke, out=out)
        except Exception as e:                                  # noqa: BLE001
            print(f'[skip] {s}: {e}', flush=True)
    print(f'[sweep] cells -> {out}', flush=True)


# ── S6 reproduction gate ──────────────────────────────────────────────────

def stage_s6gate(tol=1e-3):
    """xgb x nmv_band 5x5 at FLAT 10bp must reproduce S6 cells.csv.

    S6 convention (paper/banding.py): net = gross - traded*10bp/1e4; the
    benchmark pays reconstitution 2*TO*10bp/1e4 off walk_returns bench_ret.
    NOT the measured-spread pricing.
    """
    x = signals.build('xgb')
    br = pd.read_csv(C.RETURNS_CSV, parse_dates=['date'])
    bench = br[br.rule == 'rule_r'].set_index('date')['bench_ret']
    benchmark, _ = X.simulate(x, ('benchmark', None), score_col='score')
    net_b = bench - benchmark['turnover'].reindex(bench.index) * 2 * 10 / 1e4
    rows = []
    for ec in C.E_GRID:
        for ep in C.E_GRID:
            r, _ = X.simulate(x, ('nmv_band', (ec, ep)), score_col='score')
            net = r['gross'] - r['traded'] * 10 / 1e4
            rows.append({'e_calm': ec, 'e_panic': ep,
                         'net_ir_10bp_new': M.ir(net, net_b)})
    s6 = pd.read_csv(S6_CELLS)
    cmp_ = pd.DataFrame(rows).merge(
        s6[['e_calm', 'e_panic', 'net_ir_10bp']], on=['e_calm', 'e_panic'])
    cmp_['diff'] = (cmp_['net_ir_10bp_new'] - cmp_['net_ir_10bp']).abs()
    print(cmp_.round(4).to_string(index=False))
    ok = bool(cmp_['diff'].max() < tol)
    print(f"[s6gate] max |diff| = {cmp_['diff'].max():.2e} "
          f"(tol {tol:g}) -> {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit('[s6gate] BLOCKED: xgb nmv_band grid does not '
                         'reproduce S6 at flat 10bp')
    return cmp_


# ── gates + report (run by Task 11) ───────────────────────────────────────

def evaluate_gates(cells, boot=None, frontier=None):
    """G1-G3 verdicts. boot: delta_bootstrap df (ci_lo/ci_hi per strategy);
    frontier: frontier df (tier/method/net_sharpe). cells-only fields are
    filled without them."""
    best = select_best(cells)
    n = len(best)
    n_pos = int((best['delta_net_ir'] > 0).sum())
    g1 = {'n_strategies': n, 'n_delta_pos': n_pos,
          'majority_delta_pos': bool(n_pos > n / 2)}
    if boot is not None:
        w = boot.merge(best[['strategy', 'delta_net_ir']], on='strategy')
        winners = w[w['delta_net_ir'] > 0]
        n_excl = int(((winners['ci_lo'] > 0) | (winners['ci_hi'] < 0)).sum())
        g1['n_winner_ci_excl0'] = n_excl
        g1['pass'] = bool(g1['majority_delta_pos']
                          and n_excl > len(winners) / 2)
    rho = gate2_spearman(best)
    g2 = {'spearman': rho, 'pass': bool(rho > 0)}
    g3 = {}
    if frontier is not None:
        f = (frontier[frontier['method'] == 'ir_weighted']
             .set_index('tier')['net_sharpe'])
        g3 = {t: float(f[t]) for t in TIERS if t in f.index}
        if {'regime_best', 'static_best'} <= set(g3):
            g3['pass'] = bool(g3['regime_best'] > g3['static_best'])
    return {'G1': g1, 'G2': g2, 'G3': g3}


def _tier_net_ir(cells, best_row, tier):
    if tier == 'monthly':
        g = cells[(cells['strategy'] == best_row['strategy'])
                  & (cells['params'] == '(10, 10)')]
        if len(g) == 0:
            raise ValueError(f"{best_row['strategy']}: no (10, 10) "
                             'monthly-tier cell in results — cannot '
                             'compute monthly tier net IR')
        return float(g['net_ir_meas'].iloc[0])
    return float(best_row['static_net_ir' if tier == 'static_best'
                          else 'regime_net_ir'])


def _frontier(cells, best, acts):
    """IR-weighted combined book per tier (+ scipy MVE robustness row).
    Weights prop. to max(net_ir_meas, 0) per strategy under the tier,
    renormalized monthly over the strategies with data that month."""
    rows = []
    for tier in TIERS:
        w = {}
        for _, br in best.iterrows():
            s = br['strategy']
            if (tier, s) in acts:
                w[s] = max(_tier_net_ir(cells, br, tier), 0.0)
        tot = sum(w.values())
        if not w or tot == 0:
            continue
        w = {s: v / tot for s, v in w.items()}
        A = pd.DataFrame({s: acts[(tier, s)] for s in w}).sort_index()
        wmat = A.notna().astype(float).mul(pd.Series(w), axis=1)
        wmat = wmat.div(wmat.sum(axis=1), axis=0)
        comb = (A * wmat).sum(axis=1, min_count=1).dropna()
        rows.append({'tier': tier, 'method': 'ir_weighted',
                     'net_sharpe': M.sharpe(comb), 'n_strategies': len(w),
                     'n_months': len(comb),
                     'weights': str({s: round(v, 3) for s, v in w.items()})})
        common = A.dropna()
        if len(common) >= 24 and len(w) >= 2:
            from scipy.optimize import minimize

            def neg_sharpe(v, R=common.values):
                p = R @ v
                sd = p.std()
                return -(p.mean() / sd) if sd > 0 else 0.0
            n = A.shape[1]
            res = minimize(neg_sharpe, np.ones(n) / n, method='SLSQP',
                           bounds=[(0.0, 1.0)] * n,
                           constraints=({'type': 'eq',
                                         'fun': lambda v: v.sum() - 1.0},))
            p = pd.Series(common.values @ res.x, index=common.index)
            rows.append({'tier': tier, 'method': 'mve',
                         'net_sharpe': M.sharpe(p), 'n_strategies': n,
                         'n_months': len(common),
                         'weights': str({s: round(float(v), 3) for s, v
                                         in zip(common.columns, res.x)})})
    return pd.DataFrame(rows)


def _honesty(cells):
    """Re-select best cells on pre-SPLIT_DATE only, evaluate post; XGB
    exempt (its sample starts at SPLIT_DATE)."""
    rows = []
    for s, g in cells.groupby('strategy'):
        if s == 'xgb' or g['net_ir_meas_pre'].notna().sum() == 0:
            continue
        d = g[g['family'] == 'diag']
        r = g[g['family'] != 'diag']
        pd_ = d.loc[d['net_ir_meas_pre'].idxmax()]
        pr = r.loc[r['net_ir_meas_pre'].idxmax()]
        fd = d.loc[d['net_ir_meas'].idxmax()]
        fr = r.loc[r['net_ir_meas'].idxmax()]
        rows.append({'strategy': s,
                     'pre_static_params': pd_['params'],
                     'pre_regime_params': pr['params'],
                     'pre_delta': pr['net_ir_meas_pre']
                     - pd_['net_ir_meas_pre'],
                     'post_delta_preselected': pr['net_ir_meas_post']
                     - pd_['net_ir_meas_post'],
                     'full_delta': fr['net_ir_meas'] - fd['net_ir_meas'],
                     'post_delta_fullselected': fr['net_ir_meas_post']
                     - fd['net_ir_meas_post']})
    return pd.DataFrame(rows)


def stage_report(smoke=False):
    cells_path = os.path.join(C.BANDING_DIR,
                              'cells_smoke.csv' if smoke else 'cells.csv')
    cells = pd.read_csv(cells_path)
    best = select_best(cells)
    best.round(4).to_csv(os.path.join(C.BANDING_DIR, 'best_cells.csv'),
                         index=False)
    sp_pack = load_spreads()

    # net-of-measured-cost active series for the 3 tiers of each strategy
    acts, boot_rows = {}, []
    for _, br in best.iterrows():
        s = br['strategy']
        try:
            x = signals.build(s)
        except Exception as e:                                  # noqa: BLE001
            print(f'[skip] {s}: {e}', flush=True)
            continue
        x = x[x['date'] >= C.SWEEP_START].reset_index(drop=True)
        b, bled = X.simulate(x, ('benchmark', None), score_col='score')
        bench = price_cells(x, b, bled, sp_pack)
        tier_params = {'monthly': '(10, 10)',
                       'static_best': br['static_params'],
                       'regime_best': br['regime_params']}
        for tier, params in tier_params.items():
            r, led = X.simulate(x, _policy_from_params(params),
                                score_col='score')
            cost, _, _ = price_ledger(led, sp_pack)
            acts[(tier, s)] = M.active(_net(r['gross'], cost),
                                       bench['net_meas'])
        lo, hi = X.paired_block_bootstrap(acts[('regime_best', s)],
                                          acts[('static_best', s)])
        boot_rows.append({'strategy': s,
                          'delta_net_ir': br['delta_net_ir'],
                          'ci_lo': lo, 'ci_hi': hi,
                          'ci_lo_x12': lo * 12, 'ci_hi_x12': hi * 12,
                          'excludes_zero': bool(lo > 0 or hi < 0)})
    boot = pd.DataFrame(boot_rows)
    boot.round(4).to_csv(os.path.join(C.BANDING_DIR, 'delta_bootstrap.csv'),
                         index=False)
    frontier = _frontier(cells, best, acts)
    frontier.round(4).to_csv(os.path.join(C.BANDING_DIR, 'frontier.csv'),
                             index=False)
    gates = evaluate_gates(cells, boot=boot, frontier=frontier)
    honesty = _honesty(cells)

    def verdict(g):
        return ('HIT' if g.get('pass') else 'MISS') if 'pass' in g else 'n/a'
    lines = [
        '# Banding study: results\n',
        f'Cells: {len(cells)} ({cells_path}). Gates registered pre-run in '
        'paper/results/PAPER_NOTES.md (2026-07-15).\n',
        '## Best cells (selected on net IR at measured spreads)\n',
        best.round(3).to_string(index=False), '',
        '## Paired block-bootstrap, regime_best - static_best '
        '(net-of-measured-cost active, CI x12)\n',
        boot.round(4).to_string(index=False), '',
        '## Frontier (combined book net Sharpe per tier)\n',
        frontier.round(3).to_string(index=False), '',
        '## Gates\n',
        f"- G1 (majority delta>0, winner CIs exclude 0): "
        f"{verdict(gates['G1'])} {gates['G1']}",
        f"- G2 (Spearman cost intensity vs delta > 0): "
        f"{verdict(gates['G2'])} {gates['G2']}",
        f"- G3 (IR-weighted book Sharpe, regime > static): "
        f"{verdict(gates['G3'])} {gates['G3']}", '',
        '## Selection honesty (select pre-SPLIT, evaluate post; XGB exempt)'
        '\n',
        (honesty.round(3).to_string(index=False) if len(honesty)
         else '(no pre-split data yet)'), '',
        '## Fallback pricing share per strategy (mean hs_fallback_frac)\n',
        cells.groupby('strategy')['hs_fallback_frac'].mean().round(3)
        .to_string(), '',
    ]
    text = '\n'.join(lines)
    with open(os.path.join(C.BANDING_DIR, 'report.md'), 'w') as f:
        f.write(text + '\n')
    print(text)
    print('Saved ->', C.BANDING_DIR)
    return gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['sweep', 's6gate', 'report'])
    ap.add_argument('--workers', type=int, default=7)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    os.makedirs(C.BANDING_DIR, exist_ok=True)
    if a.stage == 'sweep':
        stage_sweep(a.workers, a.smoke)
    elif a.stage == 's6gate':
        stage_s6gate()
    else:
        stage_report(smoke=a.smoke)


if __name__ == '__main__':
    main()
