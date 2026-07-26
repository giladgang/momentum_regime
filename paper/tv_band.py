"""Clean-room term-structure time-varying band study (Gate 0 ceiling + Gate 1 OOS).

WHAT THIS FILE IS
-----------------
A self-contained backtest that asks: can we beat a *static* rebalancing band by
making the band width a *function of the market state* (the 12-horizon momentum
term structure + the HMM regime probability pi)? The underlying strategy is the
applied regime-aware XGB selector: long-only, value-weighted, top-decile of a
~1000-stock large-cap US universe. Performance = net-of-cost ACTIVE return vs the
value-weighted universe, summarized as the annualized information ratio (net IR).

WHY IT IS A SEPARATE ENGINE (the "clean room")
----------------------------------------------
An earlier multi-strategy campaign concluded regime-conditional banding is null.
To rule out that this was a bug in the production engine (paper/execution.py),
this module re-implements the whole backtest from scratch and imports NOTHING
from paper.execution / paper.banding_study. The check that it is faithful is the
TRUST GATE: its no-band `monthly` policy must reproduce the production
`walk_returns.csv` to <1e-6 per month (it does, to ~1.7e-16). Only then is any
band result trusted.

TWO STAGES
----------
Gate 0 (run_gate0): the CEILING. With perfect hindsight, how much better is a
  state-varying band than static? Answers "does the signal even exist".
Gate 1 (run_gate1): the REALIZABLE test. Can a band learned only from PAST data
  (walk-forward, annual re-fit) beat static out of sample, after an honest
  multiple-testing correction? Answers "can you actually trade it".

Design:  docs/superpowers/specs/2026-07-20-tv-band-design.md
Plans:   docs/superpowers/plans/2026-07-20-tv-band-gate0.md, ...-gate1.md
Usage:   .venv/bin/python -m paper.tv_band --stage gate0   (ceiling + diagnostics)
         .venv/bin/python -m paper.tv_band --stage gate1   (walk-forward sweep)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize                 # Trainer A: black-box net-IR fit
from sklearn.ensemble import GradientBoostingRegressor   # Trainer B (the "XGB" learner)
from sklearn.linear_model import Ridge                   # Trainer C (linear baseline)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from paper import config as C                       # noqa: E402  (paths only)
from paper.src.clusters import MOMS                 # noqa: E402  (the 12 mom_* col names)

# ── data locations (all local artifacts; nothing is pulled) ────────────────────
XSEC = os.path.join(C.RESULTS, 'xsec')              # per-month cross-sections (XGB score, pi, ...)
HS_PARQUET = os.path.join(C.RESULTS, 's5', 'half_spreads.parquet')   # measured cost panel
STOCKS = C.STOCKS_PARQUET                           # source of the 12 momentum horizons
WALK = os.path.join(C.RESULTS, 'walk_returns.csv')  # production returns (trust-gate target)
OUT_DIR = os.path.join(C.RESULTS, 'banding_study')
YEARS = list(range(2011, 2026))                     # applied test window


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_panel():
    """Monthly main-model panel. One row per (date, permno) with the XGB score
    (`score_pi`), the regime probability (`pi`), market cap (`me`), next-month
    return (`ret_fwd`), and the 12 momentum horizons (`mom_1..mom_12`).

    The `_DD` cross-sections are the main model (DD-only HMM regime). The 12
    momentum horizons live in `stocks.parquet` and are joined on (date, permno).
    """
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
    """Measured trading-cost panel. Returns (per-(permno,ym) half-spread in bp,
    per-month cross-sectional median half-spread [forward-filled over gap
    months], earliest-month median). The medians are the PIT fallback for names
    with no quoted spread that month — never a whole-panel or future value.
    """
    sp = pd.read_parquet(HS_PARQUET, columns=['permno', 'ym', 'hs'])
    sp = sp.dropna(subset=['hs']).copy()
    sp['hs'] = sp['hs'] * 1e4                        # stored as a decimal fraction -> basis points
    month_med = sp.groupby('ym')['hs'].median()
    full = pd.period_range(month_med.index.min(), month_med.index.max(), freq='M')
    month_med = month_med.reindex(full).ffill()      # gap months inherit the last PAST median
    return sp, month_med, float(month_med.iloc[0])


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# A "policy" is a callable (g, held, pi_t) -> set(permno): given this month's
# cross-section `g`, the currently-held names, and pi, it returns the target
# holdings. Every band variant is just a different policy.
# ══════════════════════════════════════════════════════════════════════════════
def _vw_weights(g, members):
    """Value-weight a set of names by market cap (weights sum to 1)."""
    me = g.set_index('permno')['me'].reindex(sorted(members))
    w = me / me.sum()
    return dict(zip(w.index, w.values))


def policy_benchmark(g, held, pi_t):
    """The benchmark: hold the entire universe (value-weighted)."""
    return set(g['permno'])


def policy_monthly(g, held, pi_t):
    """No band: every month, hold the top decile by XGB score (ties by permno)."""
    k = max(int(len(g) * 0.10), 1)
    return set(g.sort_values(['score_pi', 'permno'], ascending=[False, True])
               ['permno'].head(k))


def simulate(panel, policy):
    """Run a policy through the panel. Returns (monthly frame, trade ledger).

    Long-only value-weighted book; the reported return is the ACTIVE return
    (book minus the value-weighted universe benchmark). Turnover is
    DRIFT-ADJUSTED: last month's weights first grow by their realized return and
    are renormalized, so the reported turnover counts only *active* rebalancing,
    not the passive drift of a buy-and-hold book. `turnover` is one-way (traded
    volume / 2); the `ledger` has one row per name traded (its weight change),
    which is what `price()` costs.
    """
    rows, ledger = [], []
    hold, prev_ret = {}, {}
    for t, g in panel.groupby('date', sort=True):
        g = g.dropna(subset=['score_pi', 'me', 'ret_fwd'])
        pi_t = float(g['pi'].iloc[0])
        # passively drift last month's book, then renormalize to weights summing to 1
        drift = {p: w * (1 + prev_ret.get(p, 0.0)) for p, w in hold.items()}
        tot = sum(drift.values()) or 1.0
        drift = {p: w / tot for p, w in drift.items()}

        members = policy(g, set(hold), pi_t)         # the band decides who to hold
        tgt = _vw_weights(g, members)

        traded = 0.0
        for p in set(drift) | set(tgt):
            dw = tgt.get(p, 0.0) - drift.get(p, 0.0)  # trade = target minus drifted weight
            if abs(dw) < 1e-12:
                continue
            traded += abs(dw)
            ledger.append({'date': t, 'permno': p, 'dw': dw})
        ret = g.set_index('permno')['ret_fwd']
        book = float(sum(w * ret[p] for p, w in tgt.items()))
        bench = float((g['me'] / g['me'].sum() * g['ret_fwd']).sum())   # VW universe
        rows.append({'date': t, 'book': book, 'bench': bench,
                     'active': book - bench, 'turnover': traded / 2,   # /2 = one-way
                     'n_names': len(tgt), 'pi': pi_t})
        hold = tgt
        prev_ret = {p: float(ret[p]) for p in tgt}
    m = pd.DataFrame(rows).set_index('date')
    return m, pd.DataFrame(ledger)


# ── the band mechanic ──────────────────────────────────────────────────────────
def policy_band(E_enter_pct, E_exit_pct):
    """A percentile no-trade band (hysteresis). Returns a policy that: ADDS a
    non-held name while its score rank% <= E_enter, and HOLDS a held name until
    its rank% falls past the wider E_exit. Wider E_exit => hold longer => trade
    less. The static baseline is one fixed (E_enter=10, E_exit=E*) pair.
    """
    ee, ex = E_enter_pct / 100.0, E_exit_pct / 100.0

    def _policy(g, held, pi_t):
        n = len(g)
        ranked = g.sort_values(['score_pi', 'permno'],
                               ascending=[False, True])['permno'].tolist()
        rank_pct = {p: (i + 1) / n for i, p in enumerate(ranked)}   # 1/n = best-ranked
        univ = set(ranked)
        keep = {p for p in held if p in univ and rank_pct[p] <= ex}  # hold band
        add = {p for p in ranked if rank_pct[p] <= ee}               # entry band
        return keep | add
    return _policy


# ── costs ──────────────────────────────────────────────────────────────────────
def price(ledger, sp_pack, flat_bp=None, stress_mult=None, pi_by_date=None):
    """Trading cost per month = sum over trades of |weight change| * half-spread.

    Default: measured half-spreads (PIT median fallback for missing names).
    `flat_bp`: a constant spread instead (robustness). `stress_mult`+`pi_by_date`:
    multiply the spread only in panic months (pi>=0.5) to stress-test whether an
    edge depends on cheap panic execution.
    """
    sp, month_med, full_med = sp_pack
    if ledger is None or len(ledger) == 0:
        return pd.Series(dtype=float)
    m = ledger.copy()
    m['ym'] = m['date'].dt.to_period('M')
    if flat_bp is not None:
        m['hs'] = float(flat_bp)
    else:
        m = m.merge(sp, on=['permno', 'ym'], how='left')
        m['hs'] = m['hs'].fillna(m['ym'].map(month_med)).fillna(full_med)   # PIT fallback
    if stress_mult is not None and pi_by_date is not None:
        panic = m['date'].map(pi_by_date).fillna(0.0).ge(0.5)
        m.loc[panic, 'hs'] = m.loc[panic, 'hs'] * float(stress_mult)
    return (m['dw'].abs() * m['hs'] / 1e4).groupby(m['date']).sum()   # bp -> decimal


def net(active, cost):
    """Net-of-cost active return (missing cost months treated as zero cost)."""
    return active - cost.reindex(active.index).fillna(0.0)


# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONING FEATURES (the market-state a band could key on)
# ══════════════════════════════════════════════════════════════════════════════
def month_features(panel):
    """Per-month state features (all point-in-time — month-t cross-section only):
      zc_1..zc_12 : the LONG-LEG Z-CURVE — the top-decile picks' momentum at each
                    horizon, cross-sectionally z-scored (the picks' "shape", the
                    signal the thesis mechanism turns on).
      cs_1..cs_12 : the CROSS-SECTIONAL term structure — universe-mean momentum
                    at each horizon (the market's own momentum state).
      pi          : the HMM regime (stress) probability.
    """
    rows = []
    for t, g in panel.groupby('date', sort=True):
        g = g.dropna(subset=MOMS)
        if len(g) < 20:
            continue
        cs = g[MOMS].mean()                              # cross-sectional term structure
        z = (g[MOMS] - g[MOMS].mean()) / g[MOMS].std(ddof=0)   # per-horizon cross-sec z-score
        k = max(int(len(g) * 0.10), 1)
        top_idx = g['score_pi'].nlargest(k).index        # this month's top-decile picks
        zc = z.loc[top_idx].mean()                       # long-leg z-curve = mean z of the picks
        row = {'date': t, 'pi': float(g['pi'].iloc[0])}
        row.update({f'zc_{h}': zc[f'mom_{h}'] for h in range(1, 13)})
        row.update({f'cs_{h}': cs[f'mom_{h}'] for h in range(1, 13)})
        rows.append(row)
    return pd.DataFrame(rows).set_index('date')


def compress_features(feat):
    """Reduce the 12-horizon z-curve to its shape: level (mean), slope (deg-1
    polyfit), curvature (deg-2 leading coeff), plus pi. Keeps Trainer A's search
    low-dimensional (overfit mitigation)."""
    zc = feat[[f'zc_{h}' for h in range(1, 13)]].values
    h = np.arange(1, 13)
    level = zc.mean(axis=1)
    slope = np.array([np.polyfit(h, r, 1)[0] for r in zc])
    curv = np.array([np.polyfit(h, r, 2)[0] for r in zc])
    return pd.DataFrame({'level': level, 'slope': slope, 'curv': curv,
                         'pi': feat['pi'].values}, index=feat.index)


def standardize(feat, train_mask):
    """Z-score each column using ONLY train-window rows' mean/std (no leakage)."""
    mu = feat.loc[train_mask].mean()
    sd = feat.loc[train_mask].std(ddof=0).replace(0.0, 1.0)
    return (feat - mu) / sd


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS (objective + significance)
# ══════════════════════════════════════════════════════════════════════════════
def net_ir(r):
    """Annualized information ratio = mean/std * sqrt(12). This is the objective
    and the selection metric everywhere: banding is a small-mean/high-variance,
    benchmark-relative problem, so the standardized (leverage-invariant) IR is
    the right target, not raw return. 0 if degenerate."""
    r = pd.Series(r).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(12))


def paired_block_bootstrap(a, b, n_boot=10000, block=12, seed=0):
    """95% CI for net_ir(b) - net_ir(a) via a CIRCULAR BLOCK bootstrap (block=12
    months preserves autocorrelation that an iid bootstrap would destroy).
    Returns (point delta, lo, hi)."""
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


# ══════════════════════════════════════════════════════════════════════════════
# GATE 0 — the perfect-hindsight ceiling
# ══════════════════════════════════════════════════════════════════════════════
def _policy_perbin(exit_by_bin, bin_of_date, enter=10, default_exit=20):
    """A band whose exit width depends on the month's feature BIN (entry fixed at
    `enter`). Used by the oracle and the implementability report."""
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
    """Convenience: net IR of a policy over a panel (also returns the raw frame/ledger)."""
    m, led = simulate(panel, policy)
    cost = price(led, sp_pack, flat_bp=flat_bp)
    return net_ir(net(m['active'], cost)), m, led


def best_static(panel, sp_pack, exit_grid=(15, 20, 25, 30, 40), flat_bp=None):
    """The static baseline: the single exit width (entry fixed at 10%) maximizing
    net IR over the panel. Returns (E*, its net IR)."""
    best = (None, -np.inf)
    for ex in exit_grid:
        ir, _, _ = _ir_for_band(panel, policy_band(10, ex), sp_pack, flat_bp)
        if ir > best[1]:
            best = (ex, ir)
    return best


def oracle_bin_ir(panel, feat, sp_pack, bins, exit_grid=(15, 20, 25, 30, 40),
                  flat_bp=None):
    """The Gate-0 CEILING: with full-sample foreknowledge, the best exit width per
    feature bin. Found by COORDINATE ASCENT seeded at the best static band — net
    IR is not additive across bins and the band is path-dependent, so per-bin
    greedy fitting is NOT a valid upper bound, whereas ascent from the all-static
    seed accepts only full-panel-IR improvements and is therefore guaranteed
    >= static. (It is a coordinate-wise LOCAL optimum, and varies E_exit only, so
    it is a *conservative* estimate of the true feature ceiling.)
    Returns (ceiling net IR, chosen width per bin).
    """
    bin_of_date = dict(zip(feat.index, bins.reindex(feat.index)))
    uniq = [b for b in pd.unique(bins.dropna())]
    E_static, _ = best_static(panel, sp_pack, exit_grid, flat_bp)
    exit_by_bin = {b: E_static for b in uniq}          # seed: every bin = static

    def _ir(assign):
        pol = _policy_perbin(assign, bin_of_date, default_exit=E_static)
        ir, _, _ = _ir_for_band(panel, pol, sp_pack, flat_bp)
        return ir

    cur = _ir(exit_by_bin)
    improved = True
    while improved:                                    # sweep bins until nothing improves
        improved = False
        for b in uniq:
            best_w, best_ir = exit_by_bin[b], cur
            for w in exit_grid:
                if w == exit_by_bin[b]:
                    continue
                trial = dict(exit_by_bin)
                trial[b] = w
                ir = _ir(trial)
                if ir > best_ir + 1e-12:               # accept only genuine improvements
                    best_w, best_ir = w, ir
            if best_w != exit_by_bin[b]:
                exit_by_bin[b] = best_w
                cur = best_ir
                improved = True
    return cur, exit_by_bin


def _bins_clusters(feat):
    """Coarse state bins: K=4 quartiles of the z-curve LEVEL (a proxy for the
    thesis calm..deep-crisis axis; bin 0 = most below-average = crisis)."""
    lvl = feat[[f'zc_{h}' for h in range(1, 13)]].mean(axis=1)
    return pd.qcut(lvl, 4, labels=False, duplicates='drop').astype('Int64').astype(str)


def _bins_grid(feat):
    """Alternative binning: a 2x2 grid of (pi tercile) x (z-curve level)."""
    lvl = feat[[f'zc_{h}' for h in range(1, 13)]].mean(axis=1)
    a = pd.qcut(feat['pi'], 2, labels=False, duplicates='drop').astype(str)
    b = pd.qcut(lvl, 2, labels=False, duplicates='drop').astype(str)
    return (a + '_' + b)


def run_gate0():
    """Gate 0 driver: trust gate -> best static -> feature-binned oracle ceiling,
    for measured and flat-10bp costs. Writes tv_band_gate0.{md,csv}. `g0_pass` is
    True if the oracle beats static by more than the ~2.9% spread-timing artifact,
    which is the necessary condition for building Gate 1."""
    os.makedirs(OUT_DIR, exist_ok=True)
    panel = load_panel()
    sp_pack = load_spreads()
    # TRUST GATE (blocking): the no-band book must reproduce production returns.
    m_mo, _ = simulate(panel, policy_monthly)
    wr = pd.read_csv(WALK, parse_dates=['date'])
    wr = wr[(wr['rule'] == 'rule_r') & (wr['combo'] == 'DD')].set_index('date')
    gate_err = float((m_mo['book'] - wr['strat_ret']).reindex(
        m_mo.index.intersection(wr.index)).abs().max())
    assert gate_err < 1e-6, f'TRUST GATE FAILED: max abs err {gate_err}'

    feat = month_features(panel)
    panel = panel[panel['date'].isin(feat.index)]      # keep months with features
    rows = []
    for cost_name, flat in [('measured', None), ('flat10', 10)]:
        E_star, ir_static = best_static(panel, sp_pack, flat_bp=flat)
        ir_c, exit_c = oracle_bin_ir(panel, feat, sp_pack, _bins_clusters(feat),
                                     flat_bp=flat)
        ir_g, exit_g = oracle_bin_ir(panel, feat, sp_pack, _bins_grid(feat),
                                     flat_bp=flat)
        best_oracle = max(ir_c, ir_g)                  # headline ceiling = better binning
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


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS (mechanism + implementability)
# ══════════════════════════════════════════════════════════════════════════════
def regime_turnover(panel, pi_cut=0.5):
    """Independent re-derivation of the 'trading is regime-invariant' finding:
    mean no-band monthly turnover in calm (pi<cut) vs panic (pi>=cut). If trading
    barely varies by regime, a regime-conditioned band has little to exploit."""
    m, _ = simulate(panel, policy_monthly)
    calm = m[m['pi'] < pi_cut]['turnover']
    panic = m[m['pi'] >= pi_cut]['turnover']
    to_c, to_p = float(calm.mean()), float(panic.mean())
    return {'to_calm': to_c, 'to_panic': to_p,
            'ratio': to_p / to_c if to_c else np.nan,
            'n_calm': int(len(calm)), 'n_panic': int(len(panic))}


def band_regime_turnover(panel, policy, pi_cut=0.5):
    """Calm vs panic mean turnover for a GIVEN band policy (the implementability
    direction: does this band trade more or less in panic than static)."""
    m, _ = simulate(panel, policy)
    calm = float(m[m['pi'] < pi_cut]['turnover'].mean())
    panic = float(m[m['pi'] >= pi_cut]['turnover'].mean())
    return calm, panic


def run_gate0_impl(stress_grid=(1, 2, 5)):
    """Gate-0 implementability (G3): for static and BOTH oracle bands, report
    panic-vs-calm turnover and net IR under panic-cost stress (x1/x2/x5). The
    DIRECTION verdict is based on the cluster oracle (the headline ceiling): does
    it trade MORE in panic (flagged: harder/costlier) or LESS (implementable)?
    Writes tv_band_gate0_impl.{md,csv}."""
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
    # Verdict describes the CLUSTER oracle (the headline ceiling, ~0.464 vs the
    # grid oracle's ~0.440) — i.e. the band actually claimed as the ceiling.
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


# ══════════════════════════════════════════════════════════════════════════════
# GATE 1 — the realizable, walk-forward test
# Each "arm" is a fit_fn(panel, feat, sp_pack, train_mask) -> policy, fit on
# PRIOR months only. walkforward() re-fits it annually and stitches the OOS years.
# ══════════════════════════════════════════════════════════════════════════════
def make_feature_policy(ee_ex_fn):
    """Turn a per-date (E_enter, E_exit) function into a simulate-compatible band
    policy. This is how a learned model's monthly width predictions become a band."""
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
    """The panel restricted to the training months (used for fitting each arm)."""
    train_dates = set(feat.index[train_mask])
    return panel[panel['date'].isin(train_dates)]


def walkforward(panel, feat, sp_pack, fit_fn, start_oos=2013, flat_bp=None):
    """Stitched out-of-sample series. For each OOS year Y, re-fit the arm on months
    with year<Y ONLY, apply it over the panel, and keep just year-Y net returns.
    No leakage: every fitted parameter uses only pre-Y data; features are
    contemporaneous. (start_oos=2013 = when the XGB score first exists.)"""
    years = sorted({d.year for d in feat.index if d.year >= start_oos})
    stitched = []
    for Y in years:
        train_mask = feat.index.year < Y
        if train_mask.sum() < 24:                      # need >=2y of history to fit
            continue
        policy = fit_fn(panel, feat, sp_pack, train_mask)
        m, led = simulate(panel, policy)
        cost = price(led, sp_pack, flat_bp=flat_bp)
        r = net(m['active'], cost)
        stitched.append(r[r.index.year == Y])          # keep ONLY the OOS year
    return pd.concat(stitched).sort_index()


# ── the arms (each returns a policy fit on train months only) ───────────────────
def fit_monthly(panel, feat, sp_pack, train_mask):
    """Arm: no band (rebalance fully every month)."""
    return policy_monthly


def fit_static(panel, feat, sp_pack, train_mask):
    """Arm: the best single static band width, re-selected each year on prior data."""
    sub = _train_subpanel(panel, feat, train_mask)
    E, _ = best_static(sub, sp_pack)
    return policy_band(10, E)


def fit_cluster_band(panel, feat, sp_pack, train_mask, exit_grid=(15, 20, 25, 30, 40)):
    """Arm: a free per-cluster width. K=4 z-curve-level bins with edges fit on TRAIN
    only, then a per-bin exit width by coordinate ascent (seeded at best static) on
    the TRAIN sub-panel. Can narrow OR widen any bin — the general realizable oracle."""
    ftr = feat.loc[train_mask]
    lvl_tr = ftr[[f'zc_{h}' for h in range(1, 13)]].mean(axis=1)
    _, edges = pd.qcut(lvl_tr, 4, labels=False, retbins=True, duplicates='drop')
    edges = edges.copy(); edges[0] = -np.inf; edges[-1] = np.inf   # open-ended outer bins
    lvl_all = feat[[f'zc_{h}' for h in range(1, 13)]].mean(axis=1)
    bin_all = pd.cut(lvl_all, bins=edges, labels=False, include_lowest=True)  # apply train edges
    bin_of_date = {d: (int(b) if pd.notna(b) else -1) for d, b in bin_all.items()}
    uniq = sorted({b for d, b in bin_of_date.items() if train_mask[feat.index.get_loc(d)]})

    sub = _train_subpanel(panel, feat, train_mask)
    E0, _ = best_static(sub, sp_pack)
    width = {b: E0 for b in uniq}                       # seed all bins at static

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
    """The SUPERVISED TARGET for Trainers B/C: for each month t, the (E_enter,
    E_exit) that maximized net IR over the trailing `window` months ending at t.
    Note this target is intrinsically noisy month-to-month (a few realizations
    drive it) — which is the root reason the learned bands can't beat static."""
    dates = list(feat.index)
    combos = [(ee, ex) for ee in enter_grid for ex in exit_grid if ee <= ex]
    series = {}
    for ee, ex in combos:                              # precompute each fixed band's net series
        m, led = simulate(panel, policy_band(ee, ex))
        series[(ee, ex)] = net(m['active'], price(led, sp_pack))
    rows = []
    for i, t in enumerate(dates):
        win = dates[max(0, i - window + 1):i + 1]       # trailing window ENDING at t (causal)
        best, best_ir = (10, 20), -np.inf
        for c in combos:
            ir = net_ir(series[c].reindex(win).dropna())
            if ir > best_ir:
                best, best_ir = c, ir
        rows.append({'date': t, 'tgt_enter': best[0], 'tgt_exit': best[1]})
    return pd.DataFrame(rows).set_index('date')


_COMP_COLS = ['level', 'slope', 'curv', 'pi']          # Trainer A features (compressed)


def _feature_matrix(feat, cols, train_mask):
    """Standardized feature matrix (train-only mean/std -> no leakage)."""
    comp = compress_features(feat)[cols]
    mu = comp.loc[train_mask].mean()
    sd = comp.loc[train_mask].std(ddof=0).replace(0.0, 1.0)
    return (comp - mu) / sd, mu, sd


def fit_trainer_A(panel, feat, sp_pack, train_mask, feat_cols=None):
    """Arm A: a linear-exponential band policy E = clip(base * exp(w.z), lo, hi),
    with (base_enter, base_exit, w_enter, w_exit) fit by Nelder-Mead maximizing
    TRAIN net IR directly (the literal 'maximize net IR' objective). Uses the
    compressed 4-dim features to keep the black-box search low-dimensional. Falls
    back to the static seed if the optimized policy is worse in-sample."""
    cols = feat_cols or _COMP_COLS
    Z, _, _ = _feature_matrix(feat, cols, train_mask)
    sub = _train_subpanel(panel, feat, train_mask)
    E0, _ = best_static(sub, sp_pack)
    k = len(cols)

    def policy_from(theta):
        be, bx = theta[0], theta[1]                     # base enter/exit widths
        we, wx = theta[2:2 + k], theta[2 + k:2 + 2 * k]  # feature weights per output

        def ee_ex(t):
            z = Z.loc[t].values
            ee = float(np.clip(be * np.exp(z @ we), 5, 60))
            ex = float(np.clip(bx * np.exp(z @ wx), ee, 60))   # enforce exit >= enter
            return ee, ex
        return make_feature_policy(ee_ex)

    def neg_ir(theta):
        m, led = simulate(sub, policy_from(theta))
        return -net_ir(net(m['active'], price(led, sp_pack)))

    theta0 = np.concatenate([[10.0, float(E0)], np.zeros(2 * k)])   # seed = static band
    res = minimize(neg_ir, theta0, method='Nelder-Mead',
                   options={'maxiter': 200, 'xatol': 1e-2, 'fatol': 1e-4})
    theta = res.x if np.isfinite(res.fun) else theta0
    if -neg_ir(theta) < -neg_ir(theta0):               # never return worse than the static seed
        theta = theta0
    return policy_from(theta)


_FULL_COLS = ([f'zc_{h}' for h in range(1, 13)] +      # Trainers B/C: the FULL 25-feature set
              [f'cs_{h}' for h in range(1, 13)] + ['pi'])


def _fit_supervised(panel, feat, sp_pack, train_mask, window, make_model):
    """Fit a supervised regressor (B: GBM, C: ridge) on the full features to
    predict the trailing-window optimal band. Trees can't ingest net IR directly,
    so they learn the (features -> ex-post-optimal-band) map on TRAIN months, then
    predict a width each OOS month. Standardization uses train-only moments."""
    tgt = trailing_optimal_targets(panel, feat, sp_pack, window=window)
    mu = feat.loc[train_mask, _FULL_COLS].mean()
    sd = feat.loc[train_mask, _FULL_COLS].std(ddof=0).replace(0.0, 1.0)
    Z = (feat[_FULL_COLS] - mu) / sd
    tr_idx = feat.index[train_mask]
    Xtr = Z.loc[tr_idx].values
    me = make_model().fit(Xtr, tgt.loc[tr_idx, 'tgt_enter'].values)   # one model per output
    mx = make_model().fit(Xtr, tgt.loc[tr_idx, 'tgt_exit'].values)

    def ee_ex(t):
        z = Z.loc[t].values.reshape(1, -1)
        ee = float(np.clip(me.predict(z)[0], 5, 15))
        ex = float(np.clip(mx.predict(z)[0], ee, 40))
        return ee, ex
    return make_feature_policy(ee_ex)


def fit_trainer_B(panel, feat, sp_pack, train_mask, window=36):
    """Arm B: gradient-boosted trees (the 'XGB' band learner) on the 25 features."""
    return _fit_supervised(panel, feat, sp_pack, train_mask, window,
                           lambda: GradientBoostingRegressor(
                               n_estimators=200, max_depth=2, learning_rate=0.05,
                               subsample=0.8, random_state=0))


def fit_trainer_C(panel, feat, sp_pack, train_mask, window=36):
    """Arm C: ridge (the linear supervised baseline; isolates how much of B is
    nonlinearity vs the target definition)."""
    return _fit_supervised(panel, feat, sp_pack, train_mask, window,
                           lambda: Ridge(alpha=10.0))


def rebound_band_spec(panel, feat, sp_pack, train_mask, widen_bins,
                      widen_grid=(30, 40, 50, 60)):
    """Rebound-theory arm: WIDEN (trade less) in the below-average z-curve bins
    (bin 0 ~ thesis cluster 4 / deep crisis, bin 1 ~ cluster 3 / recovery), never
    tighter than static — the implementable 'hold through' direction. The widen
    width is tuned on TRAIN net IR. Returns (policy, bin_of_date, width_by_bin)."""
    zc = [f'zc_{h}' for h in range(1, 13)]
    ftr = feat.loc[train_mask]
    _, edges = pd.qcut(ftr[zc].mean(axis=1), 4, labels=False,
                       retbins=True, duplicates='drop')      # bin edges on TRAIN only
    edges = edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_all = pd.cut(feat[zc].mean(axis=1), bins=edges, labels=False,
                     include_lowest=True)
    bin_of_date = {d: (int(b) if pd.notna(b) else -1) for d, b in bin_all.items()}
    sub = _train_subpanel(panel, feat, train_mask)
    E0, _ = best_static(sub, sp_pack)

    def width_map(w):                                  # widen the target bins to w, rest static
        return {b: (w if b in widen_bins else E0) for b in range(4)}

    def ir_for(wbb):
        pol = make_feature_policy(
            lambda t, m=wbb: (10, m.get(bin_of_date.get(t, -1), E0)))
        mo, led = simulate(sub, pol)
        return net_ir(net(mo['active'], price(led, sp_pack)))

    best_wbb, best_ir = width_map(E0), ir_for(width_map(E0))   # seed = static
    for w in widen_grid:
        if w <= E0:                                    # rebound theory: WIDEN only (never tighter)
            continue
        wbb = width_map(w)
        v = ir_for(wbb)
        if v > best_ir + 1e-12:
            best_wbb, best_ir = wbb, v
    policy = make_feature_policy(
        lambda t, m=best_wbb: (10, m.get(bin_of_date.get(t, -1), E0)))
    return policy, bin_of_date, best_wbb


def fit_rebound_c4(panel, feat, sp_pack, train_mask):
    """Arm: widen only in the deepest-below bin (~ thesis cluster 4, deep crisis)."""
    return rebound_band_spec(panel, feat, sp_pack, train_mask, (0,))[0]


def fit_rebound_c3(panel, feat, sp_pack, train_mask):
    """Arm: widen only in the below-average bin (~ thesis cluster 3, recovery)."""
    return rebound_band_spec(panel, feat, sp_pack, train_mask, (1,))[0]


def fit_rebound_both(panel, feat, sp_pack, train_mask):
    """Arm: widen in both below-average bins (clusters 3 and 4)."""
    return rebound_band_spec(panel, feat, sp_pack, train_mask, (0, 1))[0]


def band_width_attributes(feat, bin_of_date, width_by_bin, pi_cut=0.5):
    """Interpretability: per bin, the chosen width and the market it represents
    (avg pi, fraction panic, avg z-curve level, avg cross-sectional momentum).
    Answers 'what market states does the band widen on?'."""
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


# ── honesty gate: significance + multiple-testing correction ───────────────────
def _delta_ci(arm_series, static_series, n_boot=10000, block=12, seed=0):
    """(delta, lo, hi, se, p) for net_ir(arm) - net_ir(static) via circular block
    resample. `p` is a two-sided bootstrap p-value (fraction of resamples on the
    opposite side of 0, doubled) — fed to BH-FDR in run_gate1."""
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
    """Benjamini-Hochberg step-up: boolean array, True where H0 (arm=static) is
    rejected at false-discovery rate q. This is the multiple-testing correction —
    testing many arms inflates the chance one clears an uncorrected CI."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        kmax = np.max(np.where(passed)[0])             # step-up: reject up to the largest passing rank
        reject[order[:kmax + 1]] = True
    return reject


def run_gate1(start_oos=2013, flat_bp=None, arms=None):
    """Gate 1 driver: walk-forward each arm, compare to the static arm, and gate
    on BH-FDR. `g1_win` is True only if a learned arm has minus_static>0 AND
    survives the multiple-testing correction. Writes tv_band_gate1.{md,csv} and a
    band-width interpretability CSV. (excl0 / passes_1se are reported but are NOT
    the gate — single-arm significance is inflated across 7 arms.)"""
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
    # BH-FDR across the learned arms that are candidates for "beats static"
    # (positive-delta only): a one-sided question, so a significantly-WORSE arm
    # must not enter the family and inflate the step-up threshold.
    df['bh_sig'] = False
    pos = lmask & (df['minus_static'] > 0)
    pp = df.loc[pos, 'p'].values
    if len(pp):
        df.loc[pos, 'bh_sig'] = _bh_reject(pp, q=0.05)
    learned = df[lmask]
    g1_win = bool(learned['bh_sig'].any())             # honest win: survives BH-FDR
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
        # interpretability: which market states get which width (full-sample, descriptive)
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
    if args.stage == 'gate1':                          # walk-forward learned-band sweep
        res = run_gate1()
        print('g1_win:', res['g1_win'])
        print(res['table'].to_string(index=False))
        return
    # gate0: ceiling + regime-turnover diagnostic + implementability
    res = run_gate0()
    print('regime turnover:', regime_turnover(load_panel()))
    print('g0_pass:', res['g0_pass'])
    print(res['table'].to_string(index=False))
    impl = run_gate0_impl()
    print('implementability direction:', impl['direction'])
    print(impl['table'].to_string(index=False))


if __name__ == '__main__':
    main()
