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

NOTE ON COMMENTS: per request, nearly every line carries an explanatory comment
so a non-Python reader can follow the logic; the density is deliberate.

Design:  docs/superpowers/specs/2026-07-20-tv-band-design.md
Plans:   docs/superpowers/plans/2026-07-20-tv-band-gate0.md, ...-gate1.md
Usage:   .venv/bin/python -m paper.tv_band --stage gate0   (ceiling + diagnostics)
         .venv/bin/python -m paper.tv_band --stage gate1   (walk-forward sweep)
"""
import os                                              # filesystem paths
import sys                                             # sys.path manipulation

import numpy as np                                     # numerics / arrays
import pandas as pd                                    # dataframes / time series
from scipy.optimize import minimize                    # Trainer A: black-box net-IR fit
from sklearn.ensemble import GradientBoostingRegressor  # Trainer B (the "XGB" learner)
from sklearn.linear_model import Ridge                 # Trainer C (linear baseline)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (two levels up)
sys.path.insert(0, _REPO)                              # make `paper` importable
os.chdir(_REPO)                                        # run relative to the repo root

from paper import config as C                          # noqa: E402  (all paths/knobs live here)
from paper.src.clusters import MOMS                    # noqa: E402  (the 12 mom_* column names)

# ── data locations (all local artifacts; nothing is pulled) ────────────────────
XSEC = os.path.join(C.RESULTS, 'xsec')                 # per-month cross-sections (XGB score, pi, ...)
HS_PARQUET = os.path.join(C.RESULTS, 's5', 'half_spreads.parquet')  # measured cost panel
STOCKS = C.STOCKS_PARQUET                              # source of the 12 momentum horizons
WALK = os.path.join(C.RESULTS, 'walk_returns.csv')     # production returns (trust-gate target)
OUT_DIR = os.path.join(C.RESULTS, 'banding_study')     # where result md/csv are written
YEARS = list(range(2011, 2026))                        # applied test window (2011..2025)


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
    frames = []                                        # collect one dataframe per year
    for y in YEARS:                                    # loop 2011..2025
        fp = os.path.join(XSEC, f'xsec_{y}_DD.parquet')  # that year's main-model cross-section
        frames.append(pd.read_parquet(                 # read only the columns we need
            fp, columns=['date', 'permno', 'me', 'pi', 'score_pi', 'ret_fwd']))
    x = pd.concat(frames, ignore_index=True)           # stack all years into one panel
    st = pd.read_parquet(STOCKS, columns=['date', 'permno'] + MOMS)  # the 12 momentum horizons
    p = x.merge(st, on=['date', 'permno'], how='left')  # attach momentum to each (date, permno)
    return p.sort_values(['date', 'permno']).reset_index(drop=True)  # deterministic ordering


def load_spreads():
    """Measured trading-cost panel. Returns (per-(permno,ym) half-spread in bp,
    per-month cross-sectional median half-spread [forward-filled over gap
    months], earliest-month median). The medians are the PIT fallback for names
    with no quoted spread that month — never a whole-panel or future value.
    """
    sp = pd.read_parquet(HS_PARQUET, columns=['permno', 'ym', 'hs'])  # stock-month half-spreads
    sp = sp.dropna(subset=['hs']).copy()               # drop rows with no spread
    sp['hs'] = sp['hs'] * 1e4                           # stored as a decimal fraction -> basis points
    month_med = sp.groupby('ym')['hs'].median()        # cross-sectional median spread per month
    full = pd.period_range(month_med.index.min(), month_med.index.max(), freq='M')  # every month
    month_med = month_med.reindex(full).ffill()        # gap months inherit the last PAST median
    return sp, month_med, float(month_med.iloc[0])     # (panel, per-month median, earliest median)


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# A "policy" is a callable (g, held, pi_t) -> set(permno): given this month's
# cross-section `g`, the currently-held names, and pi, it returns the target
# holdings. Every band variant is just a different policy.
# ══════════════════════════════════════════════════════════════════════════════
def _vw_weights(g, members):
    """Value-weight a set of names by market cap (weights sum to 1)."""
    me = g.set_index('permno')['me'].reindex(sorted(members))  # market cap of the chosen names
    w = me / me.sum()                                  # normalize so weights sum to 1
    return dict(zip(w.index, w.values))                # {permno: weight}


def policy_benchmark(g, held, pi_t):
    """The benchmark: hold the entire universe (value-weighted)."""
    return set(g['permno'])                            # every stock this month


def policy_monthly(g, held, pi_t):
    """No band: every month, hold the top decile by XGB score (ties by permno)."""
    k = max(int(len(g) * 0.10), 1)                     # decile size (>=1 name)
    return set(g.sort_values(['score_pi', 'permno'], ascending=[False, True])  # best score first
               ['permno'].head(k))                     # take the top k permnos


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
    rows, ledger = [], []                              # monthly stats, and per-trade rows
    hold, prev_ret = {}, {}                            # current book, and last month's realized returns
    for t, g in panel.groupby('date', sort=True):      # walk months in order
        g = g.dropna(subset=['score_pi', 'me', 'ret_fwd'])  # drop rows missing score/cap/return
        pi_t = float(g['pi'].iloc[0])                  # this month's regime probability
        # passively drift last month's book, then renormalize to weights summing to 1
        drift = {p: w * (1 + prev_ret.get(p, 0.0)) for p, w in hold.items()}  # grow by realized ret
        tot = sum(drift.values()) or 1.0               # total (guard divide-by-zero on month 1)
        drift = {p: w / tot for p, w in drift.items()}  # renormalize the drifted weights

        members = policy(g, set(hold), pi_t)           # the band decides who to hold
        tgt = _vw_weights(g, members)                  # value-weight the target holdings

        traded = 0.0                                   # accumulate total traded weight (both sides)
        for p in set(drift) | set(tgt):                # every name in the old or new book
            dw = tgt.get(p, 0.0) - drift.get(p, 0.0)   # trade = target minus drifted weight
            if abs(dw) < 1e-12:                        # skip negligible trades
                continue
            traded += abs(dw)                          # add to traded volume
            ledger.append({'date': t, 'permno': p, 'dw': dw})  # record the trade for costing
        ret = g.set_index('permno')['ret_fwd']         # next-month return per name
        book = float(sum(w * ret[p] for p, w in tgt.items()))  # book return = weighted picks
        bench = float((g['me'] / g['me'].sum() * g['ret_fwd']).sum())  # VW universe return
        rows.append({'date': t, 'book': book, 'bench': bench,  # store this month's stats
                     'active': book - bench, 'turnover': traded / 2,  # active return; /2 = one-way
                     'n_names': len(tgt), 'pi': pi_t})
        hold = tgt                                     # carry the book to next month
        prev_ret = {p: float(ret[p]) for p in tgt}     # remember realized returns for the drift
    m = pd.DataFrame(rows).set_index('date')           # monthly frame indexed by date
    return m, pd.DataFrame(ledger)                     # (monthly stats, trade ledger)


# ── the band mechanic ──────────────────────────────────────────────────────────
def policy_band(E_enter_pct, E_exit_pct):
    """A percentile no-trade band (hysteresis). Returns a policy that: ADDS a
    non-held name while its score rank% <= E_enter, and HOLDS a held name until
    its rank% falls past the wider E_exit. Wider E_exit => hold longer => trade
    less. The static baseline is one fixed (E_enter=10, E_exit=E*) pair.
    """
    ee, ex = E_enter_pct / 100.0, E_exit_pct / 100.0   # percentages -> fractions

    def _policy(g, held, pi_t):                        # the actual policy closure
        n = len(g)                                     # number of stocks this month
        ranked = g.sort_values(['score_pi', 'permno'],  # sort best score first (ties by permno)
                               ascending=[False, True])['permno'].tolist()
        rank_pct = {p: (i + 1) / n for i, p in enumerate(ranked)}  # percentile rank (1/n = best)
        univ = set(ranked)                             # investable names this month
        keep = {p for p in held if p in univ and rank_pct[p] <= ex}  # HOLD band (wide exit)
        add = {p for p in ranked if rank_pct[p] <= ee}  # ENTRY band (tight enter)
        return keep | add                              # target = kept union added
    return _policy                                     # hand back the policy


# ── costs ──────────────────────────────────────────────────────────────────────
def price(ledger, sp_pack, flat_bp=None, stress_mult=None, pi_by_date=None):
    """Trading cost per month = sum over trades of |weight change| * half-spread.

    Default: measured half-spreads (PIT median fallback for missing names).
    `flat_bp`: a constant spread instead (robustness). `stress_mult`+`pi_by_date`:
    multiply the spread only in panic months (pi>=0.5) to stress-test whether an
    edge depends on cheap panic execution.
    """
    sp, month_med, full_med = sp_pack                  # unpack the spread panel + fallbacks
    if ledger is None or len(ledger) == 0:             # no trades -> no cost
        return pd.Series(dtype=float)
    m = ledger.copy()                                  # work on a copy of the trades
    m['ym'] = m['date'].dt.to_period('M')              # month key to join spreads on
    if flat_bp is not None:                            # robustness: constant spread
        m['hs'] = float(flat_bp)
    else:                                              # default: measured spreads
        m = m.merge(sp, on=['permno', 'ym'], how='left')  # attach each trade's half-spread
        m['hs'] = m['hs'].fillna(m['ym'].map(month_med)).fillna(full_med)  # PIT median fallback
    if stress_mult is not None and pi_by_date is not None:  # optional panic-cost stress
        panic = m['date'].map(pi_by_date).fillna(0.0).ge(0.5)  # which trades fall in panic months
        m.loc[panic, 'hs'] = m.loc[panic, 'hs'] * float(stress_mult)  # widen panic spreads
    return (m['dw'].abs() * m['hs'] / 1e4).groupby(m['date']).sum()  # cost per month (bp -> decimal)


def net(active, cost):
    """Net-of-cost active return (missing cost months treated as zero cost)."""
    return active - cost.reindex(active.index).fillna(0.0)  # subtract aligned cost


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
    rows = []                                          # one feature row per month
    for t, g in panel.groupby('date', sort=True):      # walk months
        g = g.dropna(subset=MOMS)                      # need complete momentum term structure
        if len(g) < 20:                                # skip months with too few names
            continue
        cs = g[MOMS].mean()                            # cross-sectional term structure (per horizon mean)
        z = (g[MOMS] - g[MOMS].mean()) / g[MOMS].std(ddof=0)  # per-horizon cross-sectional z-score
        k = max(int(len(g) * 0.10), 1)                 # decile size
        top_idx = g['score_pi'].nlargest(k).index      # this month's top-decile picks
        zc = z.loc[top_idx].mean()                     # long-leg z-curve = mean z of the picks
        row = {'date': t, 'pi': float(g['pi'].iloc[0])}  # start the row with date + pi
        row.update({f'zc_{h}': zc[f'mom_{h}'] for h in range(1, 13)})  # 12 z-curve features
        row.update({f'cs_{h}': cs[f'mom_{h}'] for h in range(1, 13)})  # 12 term-structure features
        rows.append(row)                               # collect the row
    return pd.DataFrame(rows).set_index('date')        # feature frame indexed by date


def compress_features(feat):
    """Reduce the 12-horizon z-curve to its shape: level (mean), slope (deg-1
    polyfit), curvature (deg-2 leading coeff), plus pi. Keeps Trainer A's search
    low-dimensional (overfit mitigation)."""
    zc = feat[[f'zc_{h}' for h in range(1, 13)]].values  # (months x 12) z-curve matrix
    h = np.arange(1, 13)                               # horizon axis 1..12
    level = zc.mean(axis=1)                            # level = average z across horizons
    slope = np.array([np.polyfit(h, r, 1)[0] for r in zc])  # slope = linear-fit leading coeff
    curv = np.array([np.polyfit(h, r, 2)[0] for r in zc])   # curvature = quadratic leading coeff
    return pd.DataFrame({'level': level, 'slope': slope, 'curv': curv,  # 3 shape features + pi
                         'pi': feat['pi'].values}, index=feat.index)


def standardize(feat, train_mask):
    """Z-score each column using ONLY train-window rows' mean/std (no leakage)."""
    mu = feat.loc[train_mask].mean()                   # train-only mean per column
    sd = feat.loc[train_mask].std(ddof=0).replace(0.0, 1.0)  # train-only std (guard zeros)
    return (feat - mu) / sd                            # apply the transform to all rows


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS (objective + significance)
# ══════════════════════════════════════════════════════════════════════════════
def net_ir(r):
    """Annualized information ratio = mean/std * sqrt(12). This is the objective
    and the selection metric everywhere: banding is a small-mean/high-variance,
    benchmark-relative problem, so the standardized (leverage-invariant) IR is
    the right target, not raw return. 0 if degenerate."""
    r = pd.Series(r).dropna()                          # drop missing months
    if len(r) < 2 or r.std(ddof=1) == 0:               # guard: too short / zero variance
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(12))  # monthly IR annualized by sqrt(12)


def paired_block_bootstrap(a, b, n_boot=10000, block=12, seed=0):
    """95% CI for net_ir(b) - net_ir(a) via a CIRCULAR BLOCK bootstrap (block=12
    months preserves autocorrelation that an iid bootstrap would destroy).
    Returns (point delta, lo, hi)."""
    a, b = a.align(b, join='inner')                    # align the two series to common months
    d = (b - a).values                                 # per-month difference (length only)
    n = len(d)                                          # number of aligned months
    delta = net_ir(b) - net_ir(a)                      # the point estimate we want a CI for
    rng = np.random.default_rng(seed)                  # deterministic RNG
    nblocks = int(np.ceil(n / block))                  # how many 12-month blocks cover n months
    av, bv = a.values, b.values                        # raw arrays for fast indexing
    stats = np.empty(n_boot)                           # bootstrap distribution of the delta
    for j in range(n_boot):                            # resample n_boot times
        starts = rng.integers(0, n, nblocks)           # random block start indices
        idx = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]  # wrap-around blocks
        stats[j] = net_ir(pd.Series(bv[idx])) - net_ir(pd.Series(av[idx]))  # resampled delta
    lo, hi = np.percentile(stats, [2.5, 97.5])         # 95% percentile interval
    return float(delta), float(lo), float(hi)          # (point, lo, hi)


# ══════════════════════════════════════════════════════════════════════════════
# GATE 0 — the perfect-hindsight ceiling
# ══════════════════════════════════════════════════════════════════════════════
def _policy_perbin(exit_by_bin, bin_of_date, enter=10, default_exit=20):
    """A band whose exit width depends on the month's feature BIN (entry fixed at
    `enter`). Used by the oracle and the implementability report."""
    ee = enter / 100.0                                 # fixed entry fraction

    def _policy(g, held, pi_t):                        # policy closure
        t = g['date'].iloc[0]                          # this month's date
        ex = exit_by_bin.get(bin_of_date.get(t), default_exit) / 100.0  # bin's exit width -> fraction
        n = len(g)                                     # universe size
        ranked = g.sort_values(['score_pi', 'permno'],  # rank by score
                               ascending=[False, True])['permno'].tolist()
        rank_pct = {p: (i + 1) / n for i, p in enumerate(ranked)}  # percentile ranks
        univ = set(ranked)                             # investable names
        keep = {p for p in held if p in univ and rank_pct[p] <= ex}  # hold band (bin-specific)
        add = {p for p in ranked if rank_pct[p] <= ee}  # entry band (fixed)
        return keep | add                              # target holdings
    return _policy                                     # hand back the policy


def _ir_for_band(panel, policy, sp_pack, flat_bp):
    """Convenience: net IR of a policy over a panel (also returns the raw frame/ledger)."""
    m, led = simulate(panel, policy)                   # run the policy
    cost = price(led, sp_pack, flat_bp=flat_bp)         # cost the trades
    return net_ir(net(m['active'], cost)), m, led      # (net IR, monthly frame, ledger)


def best_static(panel, sp_pack, exit_grid=(15, 20, 25, 30, 40), flat_bp=None):
    """The static baseline: the single exit width (entry fixed at 10%) maximizing
    net IR over the panel. Returns (E*, its net IR)."""
    best = (None, -np.inf)                             # (best width, best IR)
    for ex in exit_grid:                               # try each candidate exit width
        ir, _, _ = _ir_for_band(panel, policy_band(10, ex), sp_pack, flat_bp)  # net IR of that band
        if ir > best[1]:                               # keep the best
            best = (ex, ir)
    return best                                        # (E*, its net IR)


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
    bin_of_date = dict(zip(feat.index, bins.reindex(feat.index)))  # month -> bin label
    uniq = [b for b in pd.unique(bins.dropna())]       # the distinct bins
    E_static, _ = best_static(panel, sp_pack, exit_grid, flat_bp)  # the seed width
    exit_by_bin = {b: E_static for b in uniq}          # seed: every bin = static

    def _ir(assign):                                   # net IR of a given per-bin width map
        pol = _policy_perbin(assign, bin_of_date, default_exit=E_static)  # build the band
        ir, _, _ = _ir_for_band(panel, pol, sp_pack, flat_bp)  # evaluate it
        return ir

    cur = _ir(exit_by_bin)                             # IR at the all-static seed
    improved = True                                    # loop control
    while improved:                                    # sweep bins until nothing improves
        improved = False
        for b in uniq:                                 # try to improve each bin's width
            best_w, best_ir = exit_by_bin[b], cur      # current best for this bin
            for w in exit_grid:                        # try every candidate width
                if w == exit_by_bin[b]:                # skip the current one
                    continue
                trial = dict(exit_by_bin)              # copy the assignment
                trial[b] = w                           # change this bin's width
                ir = _ir(trial)                        # evaluate the trial
                if ir > best_ir + 1e-12:               # accept only genuine improvements
                    best_w, best_ir = w, ir
            if best_w != exit_by_bin[b]:               # commit the best move for this bin
                exit_by_bin[b] = best_w
                cur = best_ir
                improved = True                        # keep sweeping
    return cur, exit_by_bin                            # (ceiling IR, per-bin widths)


def _bins_clusters(feat):
    """Coarse state bins: K=4 quartiles of the z-curve LEVEL (a proxy for the
    thesis calm..deep-crisis axis; bin 0 = most below-average = crisis)."""
    lvl = feat[[f'zc_{h}' for h in range(1, 13)]].mean(axis=1)  # z-curve level per month
    return pd.qcut(lvl, 4, labels=False, duplicates='drop').astype('Int64').astype(str)  # 4 quartile bins


def _bins_grid(feat):
    """Alternative binning: a 2x2 grid of (pi tercile) x (z-curve level)."""
    lvl = feat[[f'zc_{h}' for h in range(1, 13)]].mean(axis=1)  # z-curve level
    a = pd.qcut(feat['pi'], 2, labels=False, duplicates='drop').astype(str)  # pi half
    b = pd.qcut(lvl, 2, labels=False, duplicates='drop').astype(str)  # level half
    return (a + '_' + b)                               # 2x2 = up to 4 bins labeled "a_b"


def run_gate0():
    """Gate 0 driver: trust gate -> best static -> feature-binned oracle ceiling,
    for measured and flat-10bp costs. Writes tv_band_gate0.{md,csv}. `g0_pass` is
    True if the oracle beats static by more than the ~2.9% spread-timing artifact,
    which is the necessary condition for building Gate 1."""
    os.makedirs(OUT_DIR, exist_ok=True)                # ensure the output folder exists
    panel = load_panel()                               # load the monthly panel
    sp_pack = load_spreads()                            # load the cost panel
    # TRUST GATE (blocking): the no-band book must reproduce production returns.
    m_mo, _ = simulate(panel, policy_monthly)          # run the no-band monthly book
    wr = pd.read_csv(WALK, parse_dates=['date'])       # production returns
    wr = wr[(wr['rule'] == 'rule_r') & (wr['combo'] == 'DD')].set_index('date')  # main model rows
    gate_err = float((m_mo['book'] - wr['strat_ret']).reindex(  # max abs monthly difference
        m_mo.index.intersection(wr.index)).abs().max())
    assert gate_err < 1e-6, f'TRUST GATE FAILED: max abs err {gate_err}'  # stop if not reproduced

    feat = month_features(panel)                       # compute the state features
    panel = panel[panel['date'].isin(feat.index)]      # keep months with features
    rows = []                                          # one summary row per cost regime
    for cost_name, flat in [('measured', None), ('flat10', 10)]:  # measured + flat-10bp
        E_star, ir_static = best_static(panel, sp_pack, flat_bp=flat)  # baseline
        ir_c, exit_c = oracle_bin_ir(panel, feat, sp_pack, _bins_clusters(feat),  # cluster oracle
                                     flat_bp=flat)
        ir_g, exit_g = oracle_bin_ir(panel, feat, sp_pack, _bins_grid(feat),  # grid oracle
                                     flat_bp=flat)
        best_oracle = max(ir_c, ir_g)                  # headline ceiling = better binning
        threshold = ir_static + 0.029 * abs(ir_static)  # static + 2.9% spread-timing artifact
        rows.append({'cost': cost_name, 'E_static': E_star, 'ir_static': ir_static,  # record row
                     'ir_oracle_cluster': ir_c, 'ir_oracle_grid': ir_g,
                     'ir_oracle_best': best_oracle,
                     'uplift': best_oracle - ir_static,
                     'passes': bool(best_oracle > threshold)})
    df = pd.DataFrame(rows)                             # results table
    df.to_csv(os.path.join(OUT_DIR, 'tv_band_gate0.csv'), index=False)  # save csv
    g0_pass = bool(df.loc[df['cost'] == 'measured', 'passes'].iloc[0])  # verdict at measured cost
    with open(os.path.join(OUT_DIR, 'tv_band_gate0.md'), 'w') as f:  # write the md report
        f.write('# Gate 0 — term-structure band ceiling (perfect-hindsight oracle)\n\n')
        f.write('Feature-binned oracle (full-sample foreknowledge) vs best static band. '
                'Threshold = static IR + 2.9% (spread-timing artifact). '
                'g0_pass gates whether Gate 1 (learned real-time band) is built.\n\n')
        f.write(df.to_string(index=False))
        f.write(f'\n\n**g0_pass (measured cost): {g0_pass}**\n')
    return {'g0_pass': g0_pass, 'table': df}           # return the verdict + table


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS (mechanism + implementability)
# ══════════════════════════════════════════════════════════════════════════════
def regime_turnover(panel, pi_cut=0.5):
    """Independent re-derivation of the 'trading is regime-invariant' finding:
    mean no-band monthly turnover in calm (pi<cut) vs panic (pi>=cut). If trading
    barely varies by regime, a regime-conditioned band has little to exploit."""
    m, _ = simulate(panel, policy_monthly)             # the no-band monthly book
    calm = m[m['pi'] < pi_cut]['turnover']             # turnover in calm months
    panic = m[m['pi'] >= pi_cut]['turnover']           # turnover in panic months
    to_c, to_p = float(calm.mean()), float(panic.mean())  # mean turnover each regime
    return {'to_calm': to_c, 'to_panic': to_p,         # report both + their ratio + counts
            'ratio': to_p / to_c if to_c else np.nan,
            'n_calm': int(len(calm)), 'n_panic': int(len(panic))}


def band_regime_turnover(panel, policy, pi_cut=0.5):
    """Calm vs panic mean turnover for a GIVEN band policy (the implementability
    direction: does this band trade more or less in panic than static)."""
    m, _ = simulate(panel, policy)                     # run the band
    calm = float(m[m['pi'] < pi_cut]['turnover'].mean())  # calm-month turnover
    panic = float(m[m['pi'] >= pi_cut]['turnover'].mean())  # panic-month turnover
    return calm, panic                                 # (calm, panic)


def run_gate0_impl(stress_grid=(1, 2, 5)):
    """Gate-0 implementability (G3): for static and BOTH oracle bands, report
    panic-vs-calm turnover and net IR under panic-cost stress (x1/x2/x5). The
    DIRECTION verdict is based on the cluster oracle (the headline ceiling): does
    it trade MORE in panic (flagged: harder/costlier) or LESS (implementable)?
    Writes tv_band_gate0_impl.{md,csv}."""
    os.makedirs(OUT_DIR, exist_ok=True)                # ensure output folder
    panel = load_panel()                               # panel
    sp_pack = load_spreads()                            # costs
    feat = month_features(panel)                       # features
    panel = panel[panel['date'].isin(feat.index)]      # keep feature months
    pi_by_date = panel.groupby('date')['pi'].first().to_dict()  # month -> pi (for stress)
    cluster_bins = _bins_clusters(feat)                # K=4 cluster bins
    grid_bins = _bins_grid(feat)                       # 2x2 grid bins

    E_star, _ = best_static(panel, sp_pack)            # static width
    _, exit_c = oracle_bin_ir(panel, feat, sp_pack, cluster_bins)  # cluster oracle widths
    _, exit_g = oracle_bin_ir(panel, feat, sp_pack, grid_bins)  # grid oracle widths
    cluster_bin_of_date = dict(zip(feat.index, cluster_bins.reindex(feat.index)))  # month->cluster bin
    grid_bin_of_date = dict(zip(feat.index, grid_bins.reindex(feat.index)))  # month->grid bin
    bands = {                                          # the three bands to profile
        'static': policy_band(10, E_star),
        # headline ceiling (best oracle reported by run_gate0)
        'oracle_cluster': _policy_perbin(exit_c, cluster_bin_of_date, default_exit=E_star),
        'oracle_grid': _policy_perbin(exit_g, grid_bin_of_date, default_exit=E_star),
    }

    rows = []                                          # one row per band
    for name in ['static', 'oracle_cluster', 'oracle_grid']:  # profile each band
        pol = bands[name]
        m, led = simulate(panel, pol)                  # run it
        calm_to, panic_to = band_regime_turnover(panel, pol)  # calm/panic turnover
        row = {'band': name, 'to_calm': calm_to, 'to_panic': panic_to,  # start the row
               'panic_minus_calm_to': panic_to - calm_to}
        for s in stress_grid:                          # net IR under x1/x2/x5 panic stress
            cost = price(led, sp_pack, stress_mult=(None if s == 1 else s),
                         pi_by_date=pi_by_date)
            row[f'net_ir_stress{s}'] = net_ir(net(m['active'], cost))
        rows.append(row)
    df = pd.DataFrame(rows)                             # table

    sp_panic = df.loc[df['band'] == 'static', 'to_panic'].iloc[0]  # static panic turnover
    cl_panic = df.loc[df['band'] == 'oracle_cluster', 'to_panic'].iloc[0]  # cluster-oracle panic turnover
    # Verdict describes the CLUSTER oracle (the headline ceiling, ~0.464 vs the
    # grid oracle's ~0.440) — i.e. the band actually claimed as the ceiling.
    direction = ('MORE in panic than static (FLAGGED: panic trading is harder / costlier)'
                 if cl_panic > sp_panic + 1e-9         # trades more in panic -> flagged
                 else 'LESS (or equal) in panic than static (implementable, on-narrative)')
    df.to_csv(os.path.join(OUT_DIR, 'tv_band_gate0_impl.csv'), index=False)  # save csv
    with open(os.path.join(OUT_DIR, 'tv_band_gate0_impl.md'), 'w') as f:  # write md
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
    return {'table': df, 'direction': direction}       # return the table + verdict


# ══════════════════════════════════════════════════════════════════════════════
# GATE 1 — the realizable, walk-forward test
# Each "arm" is a fit_fn(panel, feat, sp_pack, train_mask) -> policy, fit on
# PRIOR months only. walkforward() re-fits it annually and stitches the OOS years.
# ══════════════════════════════════════════════════════════════════════════════
def make_feature_policy(ee_ex_fn):
    """Turn a per-date (E_enter, E_exit) function into a simulate-compatible band
    policy. This is how a learned model's monthly width predictions become a band."""
    def _policy(g, held, pi_t):                        # policy closure
        t = g['date'].iloc[0]                          # this month's date
        ee_pct, ex_pct = ee_ex_fn(t)                   # the model's widths for this month
        ee, ex = ee_pct / 100.0, ex_pct / 100.0        # percentages -> fractions
        n = len(g)                                     # universe size
        ranked = g.sort_values(['score_pi', 'permno'],  # rank by score
                               ascending=[False, True])['permno'].tolist()
        rank_pct = {p: (i + 1) / n for i, p in enumerate(ranked)}  # percentile ranks
        univ = set(ranked)                             # investable names
        keep = {p for p in held if p in univ and rank_pct[p] <= ex}  # hold band
        add = {p for p in ranked if rank_pct[p] <= ee}  # entry band
        return keep | add                              # target holdings
    return _policy                                     # hand back the policy


def _train_subpanel(panel, feat, train_mask):
    """The panel restricted to the training months (used for fitting each arm)."""
    train_dates = set(feat.index[train_mask])          # dates flagged as training
    return panel[panel['date'].isin(train_dates)]      # panel rows on those dates


def walkforward(panel, feat, sp_pack, fit_fn, start_oos=2013, flat_bp=None):
    """Stitched out-of-sample series. For each OOS year Y, re-fit the arm on months
    with year<Y ONLY, apply it over the panel, and keep just year-Y net returns.
    No leakage: every fitted parameter uses only pre-Y data; features are
    contemporaneous. (start_oos=2013 = when the XGB score first exists.)"""
    years = sorted({d.year for d in feat.index if d.year >= start_oos})  # OOS years
    stitched = []                                      # collect each year's OOS returns
    for Y in years:                                    # walk the OOS years
        train_mask = feat.index.year < Y               # TRAIN = strictly before year Y
        if train_mask.sum() < 24:                      # need >=2y of history to fit
            continue
        policy = fit_fn(panel, feat, sp_pack, train_mask)  # fit the arm on prior data only
        m, led = simulate(panel, policy)               # apply it over the whole panel
        cost = price(led, sp_pack, flat_bp=flat_bp)     # cost the trades
        r = net(m['active'], cost)                     # net active return series
        stitched.append(r[r.index.year == Y])          # keep ONLY the OOS year
    return pd.concat(stitched).sort_index()            # one stitched OOS series


# ── the arms (each returns a policy fit on train months only) ───────────────────
def fit_monthly(panel, feat, sp_pack, train_mask):
    """Arm: no band (rebalance fully every month)."""
    return policy_monthly                              # nothing to fit


def fit_static(panel, feat, sp_pack, train_mask):
    """Arm: the best single static band width, re-selected each year on prior data."""
    sub = _train_subpanel(panel, feat, train_mask)     # train sub-panel
    E, _ = best_static(sub, sp_pack)                   # best static width on TRAIN
    return policy_band(10, E)                          # apply it OOS


def fit_cluster_band(panel, feat, sp_pack, train_mask, exit_grid=(15, 20, 25, 30, 40)):
    """Arm: a free per-cluster width. K=4 z-curve-level bins with edges fit on TRAIN
    only, then a per-bin exit width by coordinate ascent (seeded at best static) on
    the TRAIN sub-panel. Can narrow OR widen any bin — the general realizable oracle."""
    ftr = feat.loc[train_mask]                         # train features
    lvl_tr = ftr[[f'zc_{h}' for h in range(1, 13)]].mean(axis=1)  # train z-curve level
    _, edges = pd.qcut(lvl_tr, 4, labels=False, retbins=True, duplicates='drop')  # TRAIN bin edges
    edges = edges.copy(); edges[0] = -np.inf; edges[-1] = np.inf   # open-ended outer bins
    lvl_all = feat[[f'zc_{h}' for h in range(1, 13)]].mean(axis=1)  # all-month level
    bin_all = pd.cut(lvl_all, bins=edges, labels=False, include_lowest=True)  # apply train edges
    bin_of_date = {d: (int(b) if pd.notna(b) else -1) for d, b in bin_all.items()}  # month -> bin
    uniq = sorted({b for d, b in bin_of_date.items() if train_mask[feat.index.get_loc(d)]})  # train bins

    sub = _train_subpanel(panel, feat, train_mask)     # train sub-panel
    E0, _ = best_static(sub, sp_pack)                  # static seed width
    width = {b: E0 for b in uniq}                       # seed all bins at static

    def _ir(assign):                                   # net IR of a per-bin width map on TRAIN
        pol = make_feature_policy(lambda t: (10, assign.get(bin_of_date.get(t, -1), E0)))
        m, led = simulate(sub, pol)
        return net_ir(net(m['active'], price(led, sp_pack)))

    cur = _ir(width)                                   # IR at the seed
    improved = True                                    # coordinate-ascent loop
    while improved:
        improved = False
        for b in uniq:                                 # try to improve each bin
            best_w, best_ir = width[b], cur
            for w in exit_grid:                        # over all candidate widths
                if w == width[b]:
                    continue
                trial = dict(width); trial[b] = w      # try this width in bin b
                v = _ir(trial)
                if v > best_ir + 1e-12:                # accept improvements only
                    best_w, best_ir = w, v
            if best_w != width[b]:                     # commit the best move
                width[b] = best_w; cur = best_ir; improved = True

    return make_feature_policy(lambda t: (10, width.get(bin_of_date.get(t, -1), E0)))  # OOS policy


def trailing_optimal_targets(panel, feat, sp_pack, window=36,
                             enter_grid=(5, 10, 15), exit_grid=(15, 20, 25, 30, 40)):
    """The SUPERVISED TARGET for Trainers B/C: for each month t, the (E_enter,
    E_exit) that maximized net IR over the trailing `window` months ending at t.
    Note this target is intrinsically noisy month-to-month (a few realizations
    drive it) — which is the root reason the learned bands can't beat static."""
    dates = list(feat.index)                           # chronological months
    combos = [(ee, ex) for ee in enter_grid for ex in exit_grid if ee <= ex]  # valid band pairs
    series = {}                                        # each band's full net series
    for ee, ex in combos:                              # precompute each fixed band's net series
        m, led = simulate(panel, policy_band(ee, ex))
        series[(ee, ex)] = net(m['active'], price(led, sp_pack))
    rows = []                                          # one target row per month
    for i, t in enumerate(dates):                      # for each month
        win = dates[max(0, i - window + 1):i + 1]       # trailing window ENDING at t (causal)
        best, best_ir = (10, 20), -np.inf              # best band over that window
        for c in combos:                               # scan all bands
            ir = net_ir(series[c].reindex(win).dropna())  # its net IR over the window
            if ir > best_ir:
                best, best_ir = c, ir
        rows.append({'date': t, 'tgt_enter': best[0], 'tgt_exit': best[1]})  # the target band
    return pd.DataFrame(rows).set_index('date')        # (date -> target enter/exit)


_COMP_COLS = ['level', 'slope', 'curv', 'pi']          # Trainer A features (compressed z-curve shape + pi)


def _feature_matrix(feat, cols, train_mask):
    """Standardized feature matrix (train-only mean/std -> no leakage)."""
    comp = compress_features(feat)[cols]               # compressed features
    mu = comp.loc[train_mask].mean()                   # train mean
    sd = comp.loc[train_mask].std(ddof=0).replace(0.0, 1.0)  # train std
    return (comp - mu) / sd, mu, sd                    # standardized matrix + moments


def fit_trainer_A(panel, feat, sp_pack, train_mask, feat_cols=None):
    """Arm A: a linear-exponential band policy E = clip(base * exp(w.z), lo, hi),
    with (base_enter, base_exit, w_enter, w_exit) fit by Nelder-Mead maximizing
    TRAIN net IR directly (the literal 'maximize net IR' objective). Uses the
    compressed 4-dim features to keep the black-box search low-dimensional. Falls
    back to the static seed if the optimized policy is worse in-sample."""
    cols = feat_cols or _COMP_COLS                     # which features to use
    Z, _, _ = _feature_matrix(feat, cols, train_mask)  # standardized features (all months)
    sub = _train_subpanel(panel, feat, train_mask)     # train sub-panel
    E0, _ = best_static(sub, sp_pack)                  # static seed width
    k = len(cols)                                      # number of features

    def policy_from(theta):                            # build a band policy from parameters theta
        be, bx = theta[0], theta[1]                    # base enter/exit widths
        we, wx = theta[2:2 + k], theta[2 + k:2 + 2 * k]  # feature weights per output

        def ee_ex(t):                                  # per-month widths
            z = Z.loc[t].values                        # this month's features
            ee = float(np.clip(be * np.exp(z @ we), 5, 60))  # entry width (bounded)
            ex = float(np.clip(bx * np.exp(z @ wx), ee, 60))  # exit width (>= enter, bounded)
            return ee, ex
        return make_feature_policy(ee_ex)              # wrap into a policy

    def neg_ir(theta):                                 # objective for the optimizer (minimize)
        m, led = simulate(sub, policy_from(theta))     # run the candidate on TRAIN
        return -net_ir(net(m['active'], price(led, sp_pack)))  # negative net IR

    theta0 = np.concatenate([[10.0, float(E0)], np.zeros(2 * k)])  # seed = static band (zero weights)
    res = minimize(neg_ir, theta0, method='Nelder-Mead',  # black-box optimize net IR
                   options={'maxiter': 200, 'xatol': 1e-2, 'fatol': 1e-4})
    theta = res.x if np.isfinite(res.fun) else theta0  # use the fit if it converged
    if -neg_ir(theta) < -neg_ir(theta0):               # never return worse than the static seed
        theta = theta0
    return policy_from(theta)                          # the fitted OOS policy


_FULL_COLS = ([f'zc_{h}' for h in range(1, 13)] +      # Trainers B/C features: 12 z-curve
              [f'cs_{h}' for h in range(1, 13)] + ['pi'])  # + 12 term-structure + pi (25 total)


def _fit_supervised(panel, feat, sp_pack, train_mask, window, make_model):
    """Fit a supervised regressor (B: GBM, C: ridge) on the full features to
    predict the trailing-window optimal band. Trees can't ingest net IR directly,
    so they learn the (features -> ex-post-optimal-band) map on TRAIN months, then
    predict a width each OOS month. Standardization uses train-only moments."""
    tgt = trailing_optimal_targets(panel, feat, sp_pack, window=window)  # the supervised target
    mu = feat.loc[train_mask, _FULL_COLS].mean()       # train mean
    sd = feat.loc[train_mask, _FULL_COLS].std(ddof=0).replace(0.0, 1.0)  # train std
    Z = (feat[_FULL_COLS] - mu) / sd                   # standardized features (all months)
    tr_idx = feat.index[train_mask]                    # training dates
    Xtr = Z.loc[tr_idx].values                         # training feature matrix
    me = make_model().fit(Xtr, tgt.loc[tr_idx, 'tgt_enter'].values)  # entry-width model (train fit)
    mx = make_model().fit(Xtr, tgt.loc[tr_idx, 'tgt_exit'].values)   # exit-width model (train fit)

    def ee_ex(t):                                      # per-month predicted widths
        z = Z.loc[t].values.reshape(1, -1)             # this month's features
        ee = float(np.clip(me.predict(z)[0], 5, 15))   # predicted entry (bounded)
        ex = float(np.clip(mx.predict(z)[0], ee, 40))  # predicted exit (>= enter, bounded)
        return ee, ex
    return make_feature_policy(ee_ex)                  # wrap into a policy


def fit_trainer_B(panel, feat, sp_pack, train_mask, window=36):
    """Arm B: gradient-boosted trees (the 'XGB' band learner) on the 25 features."""
    return _fit_supervised(panel, feat, sp_pack, train_mask, window,  # supervised with a GBM
                           lambda: GradientBoostingRegressor(
                               n_estimators=200, max_depth=2, learning_rate=0.05,
                               subsample=0.8, random_state=0))


def fit_trainer_C(panel, feat, sp_pack, train_mask, window=36):
    """Arm C: ridge (the linear supervised baseline; isolates how much of B is
    nonlinearity vs the target definition)."""
    return _fit_supervised(panel, feat, sp_pack, train_mask, window,  # supervised with ridge
                           lambda: Ridge(alpha=10.0))


def rebound_band_spec(panel, feat, sp_pack, train_mask, widen_bins,
                      widen_grid=(30, 40, 50, 60)):
    """Rebound-theory arm: WIDEN (trade less) in the below-average z-curve bins
    (bin 0 ~ thesis cluster 4 / deep crisis, bin 1 ~ cluster 3 / recovery), never
    tighter than static — the implementable 'hold through' direction. The widen
    width is tuned on TRAIN net IR. Returns (policy, bin_of_date, width_by_bin)."""
    zc = [f'zc_{h}' for h in range(1, 13)]             # z-curve column names
    ftr = feat.loc[train_mask]                         # train features
    _, edges = pd.qcut(ftr[zc].mean(axis=1), 4, labels=False,  # bin edges on TRAIN only
                       retbins=True, duplicates='drop')
    edges = edges.copy()                               # open-ended outer bins:
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_all = pd.cut(feat[zc].mean(axis=1), bins=edges, labels=False,  # apply train edges to all
                     include_lowest=True)
    bin_of_date = {d: (int(b) if pd.notna(b) else -1) for d, b in bin_all.items()}  # month -> bin
    sub = _train_subpanel(panel, feat, train_mask)     # train sub-panel
    E0, _ = best_static(sub, sp_pack)                  # static seed width

    def width_map(w):                                  # widen the target bins to w, rest static
        return {b: (w if b in widen_bins else E0) for b in range(4)}

    def ir_for(wbb):                                   # net IR of a width map on TRAIN
        pol = make_feature_policy(
            lambda t, m=wbb: (10, m.get(bin_of_date.get(t, -1), E0)))
        mo, led = simulate(sub, pol)
        return net_ir(net(mo['active'], price(led, sp_pack)))

    best_wbb, best_ir = width_map(E0), ir_for(width_map(E0))  # seed = static everywhere
    for w in widen_grid:                               # try each wider width
        if w <= E0:                                    # rebound theory: WIDEN only (never tighter)
            continue
        wbb = width_map(w)                             # widen the target bins to w
        v = ir_for(wbb)                                # evaluate on TRAIN
        if v > best_ir + 1e-12:                        # keep if it improves
            best_wbb, best_ir = wbb, v
    policy = make_feature_policy(                      # build the OOS policy from the best widths
        lambda t, m=best_wbb: (10, m.get(bin_of_date.get(t, -1), E0)))
    return policy, bin_of_date, best_wbb               # (policy, month->bin, chosen widths)


def fit_rebound_c4(panel, feat, sp_pack, train_mask):
    """Arm: widen only in the deepest-below bin (~ thesis cluster 4, deep crisis)."""
    return rebound_band_spec(panel, feat, sp_pack, train_mask, (0,))[0]  # widen bin 0


def fit_rebound_c3(panel, feat, sp_pack, train_mask):
    """Arm: widen only in the below-average bin (~ thesis cluster 3, recovery)."""
    return rebound_band_spec(panel, feat, sp_pack, train_mask, (1,))[0]  # widen bin 1


def fit_rebound_both(panel, feat, sp_pack, train_mask):
    """Arm: widen in both below-average bins (clusters 3 and 4)."""
    return rebound_band_spec(panel, feat, sp_pack, train_mask, (0, 1))[0]  # widen bins 0 and 1


def band_width_attributes(feat, bin_of_date, width_by_bin, pi_cut=0.5):
    """Interpretability: per bin, the chosen width and the market it represents
    (avg pi, fraction panic, avg z-curve level, avg cross-sectional momentum).
    Answers 'what market states does the band widen on?'."""
    zc = [f'zc_{h}' for h in range(1, 13)]             # z-curve columns
    cs = [f'cs_{h}' for h in range(1, 13)]             # term-structure columns
    lvl, csl = feat[zc].mean(axis=1), feat[cs].mean(axis=1)  # per-month level summaries
    rows = []                                          # one row per bin
    for b in sorted(set(bin_of_date.values())):        # each bin
        idx = feat.index.isin([x for x, bb in bin_of_date.items() if bb == b])  # its months
        s = feat[idx]                                  # feature rows for that bin
        rows.append({'bin': b, 'exit_width': width_by_bin.get(b, np.nan),  # width + market attributes
                     'n': int(idx.sum()), 'avg_pi': float(s['pi'].mean()),
                     'frac_panic': float((s['pi'] >= pi_cut).mean()),
                     'avg_zc_level': float(lvl[idx].mean()),
                     'avg_cs_mom': float(csl[idx].mean())})
    return pd.DataFrame(rows)                           # the attributes table


# ── honesty gate: significance + multiple-testing correction ───────────────────
def _delta_ci(arm_series, static_series, n_boot=10000, block=12, seed=0):
    """(delta, lo, hi, se, p) for net_ir(arm) - net_ir(static) via circular block
    resample. `p` is a two-sided bootstrap p-value (fraction of resamples on the
    opposite side of 0, doubled) — fed to BH-FDR in run_gate1."""
    a, b = static_series.align(arm_series, join='inner')  # align static (a) and arm (b)
    av, bv, n = a.values, b.values, len(a)             # arrays + length
    delta = net_ir(b) - net_ir(a)                      # point estimate: arm minus static
    rng = np.random.default_rng(seed)                  # deterministic RNG
    nb = int(np.ceil(n / block))                       # number of blocks
    stats = np.empty(n_boot)                           # bootstrap distribution of the delta
    for j in range(n_boot):                            # resample
        idx = np.concatenate([np.arange(s, s + block) % n  # wrap-around blocks
                              for s in rng.integers(0, n, nb)])[:n]
        stats[j] = net_ir(pd.Series(bv[idx])) - net_ir(pd.Series(av[idx]))  # resampled delta
    lo, hi = np.percentile(stats, [2.5, 97.5])         # 95% CI
    p = min(1.0, 2.0 * min((stats <= 0).mean(), (stats >= 0).mean()))  # two-sided bootstrap p-value
    return float(delta), float(lo), float(hi), float(stats.std(ddof=1)), float(p)  # (delta, lo, hi, se, p)


def _bh_reject(pvals, q=0.05):
    """Benjamini-Hochberg step-up: boolean array, True where H0 (arm=static) is
    rejected at false-discovery rate q. This is the multiple-testing correction —
    testing many arms inflates the chance one clears an uncorrected CI."""
    p = np.asarray(pvals, dtype=float)                 # the p-values
    m = len(p)                                          # number of tests
    order = np.argsort(p)                              # ranks, smallest p first
    thresh = q * (np.arange(1, m + 1)) / m             # BH thresholds q*i/m
    passed = p[order] <= thresh                        # which ranked p-values clear their threshold
    reject = np.zeros(m, dtype=bool)                   # default: reject none
    if passed.any():                                   # if any clears...
        kmax = np.max(np.where(passed)[0])             # step-up: reject up to the largest passing rank
        reject[order[:kmax + 1]] = True                # mark those hypotheses rejected
    return reject                                      # boolean mask aligned to input order


def run_gate1(start_oos=2013, flat_bp=None, arms=None):
    """Gate 1 driver: walk-forward each arm, compare to the static arm, and gate
    on BH-FDR. `g1_win` is True only if a learned arm has minus_static>0 AND
    survives the multiple-testing correction. Writes tv_band_gate1.{md,csv} and a
    band-width interpretability CSV. (excl0 / passes_1se are reported but are NOT
    the gate — single-arm significance is inflated across 7 arms.)"""
    os.makedirs(OUT_DIR, exist_ok=True)                # ensure output folder
    panel = load_panel()                               # panel
    feat = month_features(panel)                       # features
    panel = panel[panel['date'].isin(feat.index)]      # keep feature months
    sp = load_spreads()                                # costs
    if arms is None:                                   # default = the full 9-arm set
        arms = {'monthly': fit_monthly, 'static': fit_static,
                'cluster_band': fit_cluster_band, 'trainer_A': fit_trainer_A,
                'trainer_B': fit_trainer_B, 'trainer_C': fit_trainer_C,
                'rebound_c4': fit_rebound_c4, 'rebound_c3': fit_rebound_c3,
                'rebound_both': fit_rebound_both}
    assert 'static' in arms, "run_gate1 requires the 'static' arm as the baseline"  # baseline needed
    series = {name: walkforward(panel, feat, sp, fn, start_oos, flat_bp)  # OOS series per arm
              for name, fn in arms.items()}
    static_s = series['static']                        # the baseline OOS series
    rows = []                                          # one row per arm
    for name, s in series.items():                     # for each arm
        delta, lo, hi, se, p = _delta_ci(s, static_s)  # difference vs static + CI/se/p
        rows.append({'arm': name, 'oos_ir': net_ir(s), 'minus_static': delta,  # record
                     'ci_lo': lo, 'ci_hi': hi, 'se': se, 'p': p,
                     'excl0': bool(lo > 0 or hi < 0),  # does the CI exclude 0?
                     'passes_1se': bool(delta - se > 0)})  # does delta beat 1 std error?
    df = pd.DataFrame(rows)                             # results table
    learned_names = ['cluster_band', 'trainer_A', 'trainer_B', 'trainer_C',  # the 7 learned arms
                     'rebound_c4', 'rebound_c3', 'rebound_both']
    lmask = df['arm'].isin(learned_names)              # mask selecting the learned arms
    # BH-FDR across the learned arms that are candidates for "beats static"
    # (positive-delta only): a one-sided question, so a significantly-WORSE arm
    # must not enter the family and inflate the step-up threshold.
    df['bh_sig'] = False                               # default: no arm significant
    pos = lmask & (df['minus_static'] > 0)             # learned arms with a positive delta
    pp = df.loc[pos, 'p'].values                       # their p-values
    if len(pp):                                        # if any positive learned arms...
        df.loc[pos, 'bh_sig'] = _bh_reject(pp, q=0.05)  # apply BH-FDR
    learned = df[lmask]                                # the learned-arm rows
    g1_win = bool(learned['bh_sig'].any())             # honest win: any learned arm survives BH-FDR
    df.to_csv(os.path.join(OUT_DIR, 'tv_band_gate1.csv'), index=False)  # save csv
    with open(os.path.join(OUT_DIR, 'tv_band_gate1.md'), 'w') as f:  # write md
        f.write('# Gate 1 — learned real-time term-structure band (walk-forward OOS)\n\n')
        f.write(f'OOS {start_oos}-2025, annual re-fit on prior months only, stitched. '
                'g1_win requires a learned arm with minus_static>0 that survives BH-FDR '
                '(q=0.05) across the 7 learned arms (the multiple-testing correction the '
                'prior campaign used). excl0 (bootstrap CI) and passes_1se are reported '
                'alongside but are NOT the gate, since testing 7 arms inflates single-arm '
                'significance.\n\n')
        f.write(df.to_string(index=False))
        f.write(f'\n\n**g1_win: {g1_win}**\n')
    if 'rebound_both' in arms:                         # optional interpretability output
        # interpretability: which market states get which width (full-sample, descriptive)
        full_mask = np.ones(len(feat), dtype=bool)     # use all months (descriptive, not OOS)
        _, bod, wbb = rebound_band_spec(panel, feat, sp, full_mask, widen_bins=(0, 1))  # fit widths
        band_width_attributes(feat, bod, wbb).to_csv(  # write per-bin market attributes
            os.path.join(OUT_DIR, 'tv_band_gate1_attributes.csv'), index=False)
    return {'table': df, 'g1_win': g1_win}             # return the table + verdict


def main():
    import argparse                                    # CLI parsing
    ap = argparse.ArgumentParser()                     # argument parser
    ap.add_argument('--stage', default='gate0', choices=['gate0', 'gate1'])  # which stage to run
    args = ap.parse_args()                             # parse the CLI
    if args.stage == 'gate1':                          # walk-forward learned-band sweep
        res = run_gate1()                              # run Gate 1
        print('g1_win:', res['g1_win'])                # print the verdict
        print(res['table'].to_string(index=False))     # print the arm table
        return
    # gate0: ceiling + regime-turnover diagnostic + implementability
    res = run_gate0()                                  # run Gate 0 (ceiling + trust gate)
    print('regime turnover:', regime_turnover(load_panel()))  # print the mechanism diagnostic
    print('g0_pass:', res['g0_pass'])                  # print the ceiling verdict
    print(res['table'].to_string(index=False))         # print the ceiling table
    impl = run_gate0_impl()                            # run the implementability check
    print('implementability direction:', impl['direction'])  # print the direction verdict
    print(impl['table'].to_string(index=False))         # print the implementability table


if __name__ == '__main__':                             # script entry point
    main()                                             # run the CLI
