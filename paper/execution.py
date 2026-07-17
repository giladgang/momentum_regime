"""S5 execution/cost engine (plan: docs/superpowers/plans/2026-07-13-s5-execution.md).

Simulates execution policies over the persisted monthly cross-sections with
drift-adjusted turnover accounting and a per-trade ledger. GATE: the
`monthly` policy must reproduce walk_returns.csv gross returns exactly.

Usage: .venv/bin/python -m paper.execution   (runs all layers -> paper/results/s5/)
"""
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from paper import config as C                                   # noqa: E402
from paper.src import metrics as M                              # noqa: E402

S5 = os.path.join(C.RESULTS, 's5')
FLAT_BPS = [0, 5, 10, 20, 50]
BANDS = [15, 20, 30, 40]
PI_BANDS = [(40, 15), (40, 20), (30, 15)]
BLOCK = 12
N_BOOT = 10000


def load_xsec(score_col='score_pi'):
    sel = pd.read_csv(C.SELECTIONS_CSV)
    rr = sel[sel['rule'] == 'rule_r'].set_index('year')['combo']
    frames = []
    for y in C.EVAL_YEARS:
        xp = os.path.join(C.XSEC_DIR,
                          f"xsec_{y}_{rr.loc[y].replace('+', '_')}.parquet")
        frames.append(pd.read_parquet(xp))
    x = pd.concat(frames, ignore_index=True)
    x['date'] = pd.to_datetime(x['date'])
    return x.sort_values(['date', 'permno']).reset_index(drop=True)


def _target_weights(g, members, cap=None):
    me = g.set_index('permno')['me'].reindex(sorted(members))
    w = (me / me.sum()).values
    if cap:
        for _ in range(50):
            over = w > cap
            if not over.any():
                break
            excess = (w[over] - cap).sum()
            w[over] = cap
            free = ~over
            if not free.any() or w[free].sum() == 0:
                break
            w[free] += excess * w[free] / w[free].sum()
    return dict(zip(sorted(members), w))


def _members(g, score_col, policy, prev, pi_t, pi_prev, mi=0):
    """Target membership per policy. g indexed rows of month t; mi = month index."""
    n = len(g)
    k = max(int(n * C.DECILE_FRAC), 1)
    ranked = g.sort_values([score_col, 'permno'],
                           ascending=[False, True])['permno'].tolist()
    rank_pct = {p: (i + 1) / n for i, p in enumerate(ranked)}
    univ = set(g['permno'])
    name, arg = policy

    if name == 'benchmark':
        return univ
    if name == 'monthly':
        return set(ranked[:k])
    if name == 'calm_bench':
        return univ if pi_t < 0.5 else set(ranked[:k])
    if name == 'band' or (name == 'pi_band'):
        if name == 'band':
            b = arg / 100.0
        else:
            bc, bp = arg
            b = (bp if pi_t >= 0.5 else bc) / 100.0
        keep = {p for p in prev if p in univ and rank_pct[p] <= b}
        fill = [p for p in ranked if p not in keep][:max(k - len(keep), 0)]
        return keep | set(fill)
    if name == 'panic_no_sell':
        if pi_t < 0.5 or not prev:
            return set(ranked[:k])
        keep = {p for p in prev if p in univ}
        fill = [p for p in ranked if p not in keep][:max(k - len(keep), 0)]
        return keep | set(fill)
    if name == 'panic_hold_losers':
        # in panic, DON'T sell the beaten-down held names (below-median
        # momentum among current holdings); trade the rest to the decile.
        # Parameter-free; targets "hold the fallen names through recovery" as
        # a turnover reducer. thr = momentum-percentile cutoff (default 0.5).
        thr = arg if arg is not None else 0.5
        if pi_t < 0.5 or not prev:
            return set(ranked[:k])
        held = [p for p in prev if p in univ]
        mom = g.set_index('permno')['mom_12']
        mh = mom.reindex(held).dropna()
        cut = mh.quantile(thr) if len(mh) else np.nan
        losers = {p for p in held if p in mom.index and mom[p] <= cut}
        fill = [p for p in ranked if p not in losers][:max(k - len(losers), 0)]
        return losers | set(fill)
    if name == 'flip_freeze':
        flip = (pi_prev is None) or ((pi_t >= 0.5) != (pi_prev >= 0.5))
        if flip or not prev:
            return set(ranked[:k])
        keep = {p for p in prev if p in univ}
        fill = [p for p in ranked if p not in keep][:max(k - len(keep), 0)]
        return keep | set(fill)
    if name == 'freq':
        # unconditional reduced-frequency rebalance every `arg` months
        if mi % arg == 0 or not prev:
            return set(ranked[:k])
        keep = {p for p in prev if p in univ}
        fill = [p for p in ranked if p not in keep][:max(k - len(keep), 0)]
        return keep | set(fill)
    if name == 'nmv_band':
        # S6: literature banding (NMV 2016 / DNMV 2023 fn.11) — enter at the
        # decile, HOLD until rank falls out of E; count floats (no fill-to-k).
        # arg = (E_calm, E_panic) in percent; E switches on pi at formation.
        ec, ep = arg
        e = (ep if pi_t >= 0.5 else ec) / 100.0
        keep = {p for p in prev if p in univ and rank_pct[p] <= e}
        return set(ranked[:k]) | keep
    if name == 'regime3_band':
        # 3-state NMV banding: keep-band E switches on state3
        # (0 calm, 1 crash, 2 recovery). Same buy/hold semantics as
        # nmv_band: enter at the decile, hold until rank falls out of E.
        ec, ecr, erec = arg
        s3 = int(g['state3'].iloc[0]) if 'state3' in g else 0
        e = {0: ec, 1: ecr, 2: erec}[s3] / 100.0
        keep = {p for p in prev if p in univ and rank_pct[p] <= e}
        return set(ranked[:k]) | keep
    if name == 'cluster_band':
        # K-state NMV banding: keep-band E switches on the MONTH's momentum-
        # shape cluster label (constant within the month, like state3).
        # arg = (E_0, ..., E_{K-1}); reduces to nmv_band(E,E) when all equal.
        s = int(g['cluster'].iloc[0]) if 'cluster' in g else 0
        e = arg[s] / 100.0
        keep = {p for p in prev if p in univ and rank_pct[p] <= e}
        return set(ranked[:k]) | keep
    if name == 'stock_cluster_band':
        # per-STOCK NMV band: keep stock p if its rank_pct <= E[cluster_of_p].
        # arg = {cluster -> E}. Reduces to nmv_band(E,E) when all E equal.
        cl = (g.set_index('permno')['stock_cluster']
              if 'stock_cluster' in g else None)
        keep = set()
        for p in prev:
            if p not in univ:
                continue
            c = int(cl[p]) if cl is not None and p in cl.index else 0
            if rank_pct[p] <= arg[c] / 100.0:
                keep.add(p)
        return set(ranked[:k]) | keep
    if name == 'var_band':
        # continuous per-month band width from the frame column 'eband'
        # (E_t = smooth function of market factors, computed upstream).
        # Reduces to nmv_band(E,E) when eband is constant = E.
        e = (float(g['eband'].iloc[0]) if 'eband' in g else arg) / 100.0
        keep = {p for p in prev if p in univ and rank_pct[p] <= e}
        return set(ranked[:k]) | keep
    if name == 'stock_var_band':
        # continuous PER-STOCK band width from the per-stock-month column
        # 'eband' (E_it = smooth function of the stock's momentum / spread and
        # market pi, computed upstream). keep stock p if rank_pct[p] <=
        # eband_p/100. Reduces to nmv_band(E,E) when eband is constant = E.
        eb = (g.set_index('permno')['eband'] if 'eband' in g else None)
        keep = set()
        for p in prev:
            if p not in univ:
                continue
            e_p = (float(eb[p]) if eb is not None and p in eb.index
                   else float(arg)) / 100.0
            if rank_pct[p] <= e_p:
                keep.add(p)
        return set(ranked[:k]) | keep
    if name == 'rm_band':
        # band width as a function of (regime x past-momentum bucket):
        # keep-band E depends on the month regime (pi>=0.5) and whether the
        # held stock's mom_12 is below the month median. arg =
        # (E_calm_hi, E_calm_lo, E_panic_hi, E_panic_lo). Reduces to
        # nmv_band(E,E) when all four equal.
        ech, ecl, eph, epl = arg
        panic = pi_t >= 0.5
        mom = g.set_index('permno')['mom_12']
        med = float(mom.median())
        keep = set()
        for p in prev:
            if p not in univ:
                continue
            low = (p in mom.index) and (mom[p] <= med)
            e = (epl if low else eph) if panic else (ecl if low else ech)
            if rank_pct[p] <= e / 100.0:
                keep.add(p)
        return set(ranked[:k]) | keep
    if name == 'regime_patient':
        # THE PRODUCT: benchmark (or wide-band tilt) in calm; one decisive
        # reorganization into the model basket when pi crosses enter;
        # no-sell freeze inside panic; one reorganization home when pi
        # falls below exit (hysteresis).
        enter, exit_, calm_mode = arg['enter'], arg['exit'], arg['calm']
        in_panic = arg['state']
        if in_panic['on']:
            if pi_t < exit_:
                in_panic['on'] = False          # reorganize home
            else:                               # freeze (forced exits only)
                keep = {p for p in prev if p in univ}
                fill = [p for p in ranked
                        if p not in keep][:max(k - len(keep), 0)]
                return keep | set(fill)
        else:
            if pi_t >= enter:
                in_panic['on'] = True           # reorganize into the basket
                return set(ranked[:k])
        if calm_mode == 'bench':
            return univ
        # calm momentum tilt with wide band (40%)
        keepable = {p for p in prev if p in univ and rank_pct[p] <= 0.40}
        fill = [p for p in ranked
                if p not in keepable][:max(k - len(keepable), 0)]
        return keepable | set(fill)
    raise ValueError(name)


def simulate(x, policy, score_col='score_pi', cap=None):
    """Returns (monthly df, ledger df). policy = (name, arg)."""
    rows, ledger = [], []
    hold, prev_ret, pi_prev = {}, {}, None
    for mi, (t, g) in enumerate(x.groupby('date', sort=True)):
        pi_t = float(g['pi'].iloc[0])
        univ = set(g['permno'])
        med_mom = float(g[g['permno'].isin(hold)]['mom_12'].median()) \
            if hold else np.nan

        drift = {p: w * (1 + prev_ret.get(p, 0.0)) for p, w in hold.items()}
        tot = sum(drift.values()) or 1.0
        drift = {p: w / tot for p, w in drift.items()}

        members = _members(g, score_col, policy, set(hold), pi_t, pi_prev, mi)
        tgt = _target_weights(g, members, cap)

        traded = 0.0
        mom_map = g.set_index('permno')['mom_12']
        for p in set(drift) | set(tgt):
            dw = tgt.get(p, 0.0) - drift.get(p, 0.0)
            if abs(dw) < 1e-12:
                continue
            traded += abs(dw)
            if p not in tgt:
                cause = 'exit_univ' if p not in univ else 'exit_rank'
                ttype = ('loser' if p in mom_map.index
                         and mom_map[p] < med_mom else 'winner')
            elif p not in drift:
                cause, ttype = 'entry', 'entry'
            else:
                cause, ttype = 'weight', 'weight'
            ledger.append({'date': t, 'permno': p, 'dw': dw, 'cause': cause,
                           'ttype': ttype, 'pi': pi_t})
        ret_map = g.set_index('permno')['ret_fwd']
        gross = float(sum(w * ret_map[p] for p, w in tgt.items()))
        rows.append({'date': t, 'gross': gross, 'turnover': traded / 2,
                     'traded': traded, 'pi': pi_t, 'n_names': len(tgt)})
        hold = tgt
        prev_ret = {p: float(ret_map[p]) for p in tgt}
        pi_prev = pi_t
    return (pd.DataFrame(rows).set_index('date'),
            pd.DataFrame(ledger))


def paired_block_bootstrap(a, b, n_boot=N_BOOT, block=BLOCK, seed=0):
    """CI of mean(a-b) via paired moving-block bootstrap (same blocks)."""
    d = (a - b).dropna().values
    n = len(d)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n // block + 1))
    means = np.array([
        np.concatenate([d[s:s + block] for s in row])[:n].mean()
        for row in starts])
    return (float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def main():
    os.makedirs(S5, exist_ok=True)
    x = load_xsec()
    bench_ret = pd.read_csv(C.RETURNS_CSV, parse_dates=['date'])
    bench = (bench_ret[bench_ret.rule == 'rule_r']
             .set_index('date')['bench_ret'])

    # ── GATE: monthly policy reproduces the walk ──
    walk = (bench_ret[bench_ret.rule == 'rule_r']
            .set_index('date')['strat_ret'])
    mon, led_mon = simulate(x, ('monthly', None))
    err = (mon['gross'] - walk.reindex(mon.index)).abs().max()
    assert err < 1e-10, f'GATE FAIL: monthly vs walk max err {err}'
    print(f'[gate] monthly reproduces walk exactly (max err {err:.1e})')
    benchmark, _ = simulate(x, ('benchmark', None))
    print(f"[gate] benchmark reconstitution turnover "
          f"{benchmark['turnover'].mean():.3%}/mo | "
          f"monthly baseline {mon['turnover'].mean():.3%}/mo")

    # ── policy ladder ──
    policies = [('monthly', None), ('panic_no_sell', None),
                ('flip_freeze', None), ('calm_bench', None)]
    policies += [('band', b) for b in BANDS]
    policies += [('pi_band', p) for p in PI_BANDS]

    results, ledgers = {}, {}
    for pol in policies:
        key = pol[0] if pol[1] is None else f'{pol[0]}{pol[1]}'
        results[key], ledgers[key] = simulate(x, pol)
        print(f'[sim] {key}: TO {results[key].turnover.mean():.2%}/mo',
              flush=True)

    net_bench_to = benchmark['turnover']
    rows = []
    base_net = {}
    for key, r in results.items():
        act_g = r['gross'] - bench.reindex(r.index)
        row = {'policy': key, 'to_mo': r['turnover'].mean(),
               'gross_ir': M.ir(r['gross'], bench)}
        for c in FLAT_BPS:
            net = r['gross'] - r['traded'] * c / 1e4
            net_b = bench - net_bench_to.reindex(bench.index) * 2 * c / 1e4
            row[f'net_ir_{c}bp'] = M.ir(net, net_b)
            if c == 10:
                base_net[key] = M.active(net, net_b)
        row['breakeven_bp'] = (act_g.mean() /
                               (r['traded'].mean()
                                - 2 * net_bench_to.mean()) * 1e4
                               if r['traded'].mean() > 2 * net_bench_to.mean()
                               else np.inf)
        rows.append(row)
    ladder = pd.DataFrame(rows)
    # paired bootstrap of net-active difference vs monthly baseline (10bp)
    cis = []
    for key in ladder['policy']:
        lo, hi = paired_block_bootstrap(base_net[key], base_net['monthly'])
        cis.append(f'[{lo * 12:+.3f},{hi * 12:+.3f}]')
    ladder['d_vs_monthly_10bp_CI'] = cis
    ladder = ladder.round(3)
    ladder.to_csv(os.path.join(S5, 'policy_ladder.csv'), index=False)
    print(ladder.to_string(index=False))

    # ── Layer 1: turnover by regime x cause + alpha earned ──
    led = ledgers['monthly']
    led['regime'] = np.where(led['pi'] >= 0.5, 'panic', 'calm')
    dec = (led.assign(vol=led['dw'].abs())
           .groupby(['regime', 'cause'])['vol'].sum()
           .unstack(fill_value=0.0))
    n_cal = (~(mon['pi'] >= 0.5)).sum()
    n_pan = (mon['pi'] >= 0.5).sum()
    dec['months'] = [n_cal, n_pan]
    act = mon['gross'] - bench.reindex(mon.index)
    dec['active_earned_mo'] = [float(act[mon['pi'] < 0.5].mean()),
                               float(act[mon['pi'] >= 0.5].mean())]
    dec['traded_mo'] = dec[[c for c in ['entry', 'exit_rank', 'exit_univ',
                                        'weight'] if c in dec]].sum(axis=1) \
        / dec['months']
    dec.round(4).to_csv(os.path.join(S5, 'turnover_decomposition.csv'))
    print(dec.round(4).to_string())

    # ── Layer 4b: counterfactual panic loser-sells ──
    fwd = x.pivot_table(index='date', columns='permno', values='ret_fwd')
    dates = list(fwd.index)
    didx = {d: i for i, d in enumerate(dates)}
    def fwd_ret(p, d, h):
        i = didx[d]
        rs = [fwd.iloc[j][p] for j in range(i, min(i + h, len(dates)))
              if p in fwd.columns]
        rs = [r for r in rs if pd.notna(r)]
        return float(np.prod([1 + r for r in rs]) - 1) if rs else np.nan
    sells = led[(led.regime == 'panic') & (led.cause == 'exit_rank')
                & (led.ttype == 'loser')]
    buys = led[(led.regime == 'panic') & (led.cause == 'entry')]
    cf = []
    for h in (3, 6):
        s = np.array([fwd_ret(r.permno, r.date, h)
                      for r in sells.itertuples()])
        b = np.array([fwd_ret(r.permno, r.date, h)
                      for r in buys.itertuples()])
        cf.append({'horizon_mo': h, 'n_sells': len(s),
                   'sold_fwd': float(np.nanmean(s)),
                   'bought_fwd': float(np.nanmean(b)),
                   'sold_minus_bought': float(np.nanmean(s)
                                              - np.nanmean(b))})
    cfd = pd.DataFrame(cf).round(4)
    cfd.to_csv(os.path.join(S5, 'counterfactual_panic_sells.csv'),
               index=False)
    print(cfd.to_string(index=False))
    print('\nSaved ->', S5)


def simulate_blend(x, mode, base_w, rf=None, pi_map=None, lam_map=None):
    """Holdings-level blend comparators (tier 3).
    base_w: dict date -> {permno: weight} of the risky book (capped).
    pi_scale: w = pi*book + (1-pi)*benchmark. bsc: w = lam*book + (1-lam)*cash(RF).
    Same drift-adjusted accounting; traded volume counts stock legs."""
    rows = []
    hold, cash, prev_ret, prev_rf = {}, 0.0, {}, 0.0
    for t, g in x.groupby('date', sort=True):
        ret_map = g.set_index('permno')['ret_fwd']
        drift = {p: w * (1 + prev_ret.get(p, 0.0)) for p, w in hold.items()}
        dc = cash * (1 + prev_rf)
        tot = (sum(drift.values()) + dc) or 1.0
        drift = {p: w / tot for p, w in drift.items()}
        dc /= tot
        book = base_w[t]
        if mode == 'pi_scale':
            pi_t = float(pi_map[t])
            bench_w = _target_weights(g, set(g['permno']), None)
            tgt = {p: pi_t * book.get(p, 0.0) + (1 - pi_t) * bench_w.get(p, 0.0)
                   for p in set(book) | set(bench_w)}
            tgt_cash = 0.0
        else:                                   # bsc
            lam = float(lam_map[t])
            tgt = {p: lam * w for p, w in book.items()}
            tgt_cash = 1 - lam
        traded = float(sum(abs(tgt.get(p, 0.0) - drift.get(p, 0.0))
                           for p in set(tgt) | set(drift)))
        rf_t = float(rf[t]) if rf is not None else 0.0
        gross = float(sum(w * ret_map.get(p, 0.0) for p, w in tgt.items())
                      + tgt_cash * rf_t)
        rows.append({'date': t, 'gross': gross, 'turnover': traded / 2,
                     'traded': traded, 'pi': float(g['pi'].iloc[0])})
        hold, cash = tgt, tgt_cash
        prev_ret = {p: float(ret_map.get(p, 0.0)) for p in tgt}
        prev_rf = rf_t
    return pd.DataFrame(rows).set_index('date')


def product_main():
    """Regime-Patient Momentum vs the three comparator tiers (flat engine).
    All runs 5%-capped (the product recipe); registered comparator grid."""
    os.makedirs(S5, exist_ok=True)
    x = load_xsec()
    br = pd.read_csv(C.RETURNS_CSV, parse_dates=['date'])
    bench = br[br.rule == 'rule_r'].set_index('date')['bench_ret']
    walk = br[br.rule == 'rule_r'].set_index('date')['strat_ret']
    mon_g, _ = simulate(x, ('monthly', None))
    assert (mon_g['gross'] - walk.reindex(mon_g.index)).abs().max() < 1e-10
    bench_sim, _ = simulate(x, ('benchmark', None))
    nb_to = bench_sim['turnover']

    ff = pd.read_parquet('data/ff_factors.parquet')['RF']
    ff.index = ff.index + pd.offsets.MonthEnd(0)
    months = sorted(x['date'].unique())
    rf = {t: float(ff.reindex([pd.Timestamp(t) + pd.offsets.MonthEnd(1)]
                              ).iloc[0] or 0.0) for t in months}

    runs = {}
    def patient(exit_, calm):
        return ('regime_patient', {'enter': 0.5, 'exit': exit_,
                                   'calm': calm, 'state': {'on': False}})
    for ex in (0.4, 0.3):
        for calm in ('bench', 'tilt'):
            key = f'PRODUCT_patient(x{ex},{calm})'
            runs[key], _ = simulate(x, patient(ex, calm), cap=0.05)
    for key, pol, sc in [('mom_monthly', ('monthly', None), 'mom_12')] + \
            [(f'mom_band{b}', ('band', b), 'mom_12') for b in (20, 30, 40)] + \
            [(f'mom_freq{k}', ('freq', k), 'mom_12') for k in (3, 6, 12)]:
        runs[key], _ = simulate(x, pol, score_col=sc, cap=0.05)
    runs['pi_monthly_capped'], _ = simulate(x, ('monthly', None), cap=0.05)

    # tier 3 blends
    pi_map = x.groupby('date')['pi'].first()
    base_pi = {}
    for t, g in x.groupby('date', sort=True):
        k = max(int(len(g) * C.DECILE_FRAC), 1)
        top = g.sort_values(['score_pi', 'permno'],
                            ascending=[False, True]).head(k)
        base_pi[t] = _target_weights(g, set(top['permno']), 0.05)
    base_mom = {}
    for t, g in x.groupby('date', sort=True):
        k = max(int(len(g) * C.DECILE_FRAC), 1)
        top = g.sort_values(['mom_12', 'permno'],
                            ascending=[False, True]).head(k)
        base_mom[t] = _target_weights(g, set(top['permno']), 0.05)
    runs['ALT_pi_scale'] = simulate_blend(x, 'pi_scale', base_pi,
                                          rf=rf, pi_map=pi_map)
    mom_series = runs['mom_monthly']['gross']
    sig = mom_series.expanding(12).std().shift(1) * np.sqrt(12)
    lam = (0.15 / sig).clip(upper=1.0).fillna(1.0)
    runs['ALT_bsc_mom'] = simulate_blend(x, 'bsc', base_mom, rf=rf,
                                         lam_map=lam.to_dict())

    rows, net10 = [], {}
    for key, r in runs.items():
        act_g = r['gross'] - bench.reindex(r.index)
        row = {'strategy': key, 'to_mo': r['turnover'].mean(),
               'gross_ir': M.ir(r['gross'], bench)}
        for c in (5, 10, 20):
            net = r['gross'] - r['traded'] * c / 1e4
            net_b = bench - nb_to.reindex(bench.index) * 2 * c / 1e4
            row[f'net_ir_{c}bp'] = M.ir(net, net_b)
            if c == 10:
                net10[key] = M.active(net, net_b)
        vol = r['traded'].mean() - 2 * nb_to.mean()
        row['breakeven_bp'] = act_g.mean() / vol * 1e4 if vol > 0 else np.inf
        rows.append(row)
    tab = pd.DataFrame(rows)
    best_t2 = tab[tab.strategy.str.startswith('mom_')] \
        .set_index('strategy')['net_ir_10bp'].idxmax()
    cis = []
    for key in tab['strategy']:
        lo, hi = paired_block_bootstrap(net10[key], net10[best_t2])
        cis.append(f'[{lo * 12:+.3f},{hi * 12:+.3f}]')
    tab[f'd_vs_{best_t2}_CI'] = cis
    tab['bench_ir_ref'] = 0.0
    tab = tab.round(3)
    tab.to_csv(os.path.join(S5, 'product_comparison.csv'), index=False)
    print(f'(paired CIs vs best tier-2 at 10bp = {best_t2})')
    print(tab.to_string(index=False))
    print('\nSaved ->', os.path.join(S5, 'product_comparison.csv'))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--product', action='store_true')
    if ap.parse_args().product:
        product_main()
    else:
        main()
