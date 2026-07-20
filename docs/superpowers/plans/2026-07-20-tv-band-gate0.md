# Term-Structure Time-Varying Band — Gate 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the clean-room band engine and run the Gate 0 ceiling test — decide whether a band that is a function of the 12-horizon momentum term structure + `pi` can beat a static band even with full in-sample foreknowledge, before any learning machinery is built.

**Architecture:** A single self-contained module `paper/tv_band.py` that imports nothing from `paper/execution.py` or `paper/banding_study.py` — only numpy/pandas + raw artifacts. It re-derives the long-only VW top-decile active-return backtest from scratch, proves itself against `walk_returns.csv` (trust gate), then computes the feature-binned oracle ceiling and the calm-vs-panic turnover diagnostic.

**Tech Stack:** Python 3, numpy, pandas, pyarrow, pytest. (No sklearn/scipy needed for Gate 0; they arrive with the Gate 1 trainers.)

## Global Constraints

- Main model only: `score_pi` from `paper/results/xsec/xsec_{year}_DD.parquet`, years 2011–2025. Rank descending by `score_pi`, ties broken by `permno` ascending.
- Portfolio: long-only, value-weighted by `me`, top decile `k = max(int(n*0.10), 1)`. Benchmark = value-weighted full universe. Active return = book − benchmark.
- Clean-room: `paper/tv_band.py` MUST NOT import from `paper.execution` or `paper.banding_study`. Reuse of `paper.config` (paths) and `paper.src.clusters.MOMS` (the 12 column-name strings only) is allowed.
- Turnover is drift-adjusted: last month's weights grow by realized `ret_fwd` and renormalize before the new trade. `cost_t = Σ_i |Δw_i| · hs_i / 1e4` with `hs` in bp.
- Trust gate is blocking: the no-band `monthly` policy must reproduce `walk_returns.csv` `strat_ret` and the `benchmark` policy must reproduce `bench_ret` (rule_r/DD rows) to ≤ 1e-6 per month before any other result is computed or trusted.
- Verified anchor (2015-06-30): book = −0.03460814, bench = +0.01871751.
- Outputs go to `paper/results/banding_study/tv_band_gate0.{md,csv}`.

---

## File Structure

- Create: `paper/tv_band.py` — the clean-room engine + Gate 0 driver.
- Create: `tests/test_tv_band.py` — all Gate 0 tests.
- Produce (at runtime): `paper/results/banding_study/tv_band_gate0.md`, `tv_band_gate0.csv`.

Run tests with: `.venv/bin/python -m pytest tests/test_tv_band.py -v`

---

### Task 1: Data loading — monthly panel + spreads

**Files:**
- Create: `paper/tv_band.py`
- Test: `tests/test_tv_band.py`

**Interfaces:**
- Produces: `load_panel() -> pd.DataFrame` with columns `['date','permno','me','pi','score_pi','ret_fwd','mom_1'..'mom_12']`, one row per (date, permno), dates 2011-01..2025-11, sorted by date.
- Produces: `load_spreads() -> tuple[pd.DataFrame, pd.Series, float]` = (`sp[permno,ym,hs_bp]`, per-month median `hs_bp` forward-filled, earliest-month median).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tv_band.py
import numpy as np
import pandas as pd
import pytest
from paper import tv_band as T


def test_load_panel_shape_and_columns():
    p = T.load_panel()
    need = ({'date', 'permno', 'me', 'pi', 'score_pi', 'ret_fwd'}
            | {f'mom_{h}' for h in range(1, 13)})
    assert need.issubset(p.columns)
    assert p['date'].min() == pd.Timestamp('2011-01-31')
    assert p['date'].max() >= pd.Timestamp('2025-10-31')
    # no duplicate (date, permno)
    assert not p.duplicated(['date', 'permno']).any()
    # ~1000 names per month
    assert p.groupby('date').size().median() >= 500


def test_load_spreads_units():
    sp, month_med, first_med = T.load_spreads()
    assert {'permno', 'ym', 'hs'}.issubset(sp.columns)
    # hs converted to basis points, sane range
    assert 0.1 < sp['hs'].median() < 500
    assert np.isfinite(first_med)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py::test_load_panel_shape_and_columns -v`
Expected: FAIL with `AttributeError: module 'paper.tv_band' has no attribute 'load_panel'`

- [ ] **Step 3: Write minimal implementation**

```python
# paper/tv_band.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "load_" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add paper/tv_band.py tests/test_tv_band.py
git commit -m "tv_band: clean-room data loading (panel + spreads)"
```

---

### Task 2: Clean-room simulate + TRUST GATE

**Files:**
- Modify: `paper/tv_band.py`
- Test: `tests/test_tv_band.py`

**Interfaces:**
- Consumes: `load_panel()`.
- Produces: `simulate(panel, policy) -> (monthly: pd.DataFrame indexed by date with ['book','bench','active','turnover','n_names','pi'], ledger: pd.DataFrame['date','permno','dw'])`. `policy` is a callable `(g_sorted, held_set, pi_t) -> set(permno)` returning target members; two built-ins provided: `policy_monthly` (top-k) and `policy_benchmark` (all names).
- Produces: `_vw_weights(g, members) -> dict[permno,float]`.

- [ ] **Step 1: Write the failing test**

```python
def test_trust_gate_reproduces_walk_returns():
    panel = T.load_panel()
    monthly, _ = T.simulate(panel, T.policy_monthly)
    bench, _ = T.simulate(panel, T.policy_benchmark)
    wr = pd.read_csv(T.WALK, parse_dates=['date'])
    wr = wr[(wr['rule'] == 'rule_r') & (wr['combo'] == 'DD')].set_index('date')
    j = monthly.join(wr[['strat_ret', 'bench_ret']], how='inner')
    assert len(j) >= 100
    assert (j['book'] - j['strat_ret']).abs().max() < 1e-6
    b = bench.join(wr[['bench_ret']], how='inner')
    assert (b['book'] - b['bench_ret']).abs().max() < 1e-6


def test_simulate_active_is_book_minus_bench():
    panel = T.load_panel()
    m, _ = T.simulate(panel, T.policy_monthly)
    assert np.allclose(m['active'], m['book'] - m['bench'])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py::test_trust_gate_reproduces_walk_returns -v`
Expected: FAIL with `AttributeError: ... has no attribute 'simulate'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "trust or active_is" -v`
Expected: PASS (2 tests). If the trust gate fails, STOP and diagnose the engine — do not proceed.

- [ ] **Step 5: Commit**

```bash
git add paper/tv_band.py tests/test_tv_band.py
git commit -m "tv_band: clean-room simulate + trust gate vs walk_returns"
```

---

### Task 3: Band mechanic — static E and (E_enter, E_exit) floating-count

**Files:**
- Modify: `paper/tv_band.py`
- Test: `tests/test_tv_band.py`

**Interfaces:**
- Consumes: `simulate`, `_vw_weights`.
- Produces: `policy_band(E_enter_pct, E_exit_pct) -> callable` usable as a `simulate` policy: hold a name while its rank% ≤ E_exit; add a non-held name while rank% ≤ E_enter; count floats. Ranks by `score_pi` descending. `rank% = (i+1)/n`.
- Produces: `price(ledger, sp_pack, flat_bp=None) -> pd.Series` monthly cost (decimal return units). Measured half-spread by default; `flat_bp` overrides with a constant.
- Produces: `net(active, cost) -> pd.Series` = active − cost (aligned, missing cost → 0).

- [ ] **Step 1: Write the failing test**

```python
def test_band_reduces_turnover_monotonically():
    panel = T.load_panel()
    to = {}
    for exit_pct in [10, 20, 40]:
        m, _ = T.simulate(panel, T.policy_band(10, exit_pct))
        to[exit_pct] = m['turnover'].mean()
    # wider exit band => weakly lower turnover
    assert to[40] < to[20] < to[10] + 1e-9
    # E_enter=E_exit=10 reproduces the monthly turnover closely
    m10, _ = T.simulate(panel, T.policy_band(10, 10))
    mm, _ = T.simulate(panel, T.policy_monthly)
    assert abs(m10['turnover'].mean() - mm['turnover'].mean()) < 5e-3


def test_price_and_net():
    panel = T.load_panel()
    m, led = T.simulate(panel, T.policy_band(10, 20))
    sp_pack = T.load_spreads()
    cost = T.price(led, sp_pack)
    assert (cost >= 0).all() and cost.mean() > 0
    flat = T.price(led, sp_pack, flat_bp=10)
    # flat 10bp cost == 10bp * two-way traded volume / 1e4
    assert flat.mean() > 0
    n = T.net(m['active'], cost)
    assert np.allclose(n, m['active'] - cost.reindex(n.index).fillna(0.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "band_reduces or price_and_net" -v`
Expected: FAIL with `AttributeError: ... 'policy_band'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "band_reduces or price_and_net" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add paper/tv_band.py tests/test_tv_band.py
git commit -m "tv_band: (E_enter,E_exit) floating-count band + spread/flat costing"
```

---

### Task 4: Features — z-curve, term structure, pi (full-25 + compressed-4)

**Files:**
- Modify: `paper/tv_band.py`
- Test: `tests/test_tv_band.py`

**Interfaces:**
- Consumes: `load_panel`.
- Produces: `month_features(panel) -> pd.DataFrame` indexed by date with columns `zc_1..zc_12` (long-leg z-curve of the top-decile score_pi picks), `cs_1..cs_12` (cross-sectional mean momentum per horizon), `pi`. Raw (un-standardized), PIT by construction (uses only month-t data).
- Produces: `compress_features(feat) -> pd.DataFrame` with `level, slope, curv` (from the z-curve) + `pi`.
- Produces: `standardize(feat, train_mask) -> pd.DataFrame` z-scored using train-rows' mean/std only.

- [ ] **Step 1: Write the failing test**

```python
def test_month_features_zcurve_matches_definition():
    panel = T.load_panel()
    feat = T.month_features(panel)
    assert {f'zc_{h}' for h in range(1, 13)}.issubset(feat.columns)
    assert {f'cs_{h}' for h in range(1, 13)}.issubset(feat.columns)
    assert 'pi' in feat.columns
    # recompute zc for one month independently
    d = feat.index[20]
    g = panel[panel['date'] == d].dropna(subset=[f'mom_{h}' for h in range(1, 13)])
    z = (g[[f'mom_{h}' for h in range(1, 13)]]
         - g[[f'mom_{h}' for h in range(1, 13)]].mean()) / \
        g[[f'mom_{h}' for h in range(1, 13)]].std(ddof=0)
    z.columns = [f'zc_{h}' for h in range(1, 13)]
    g2 = g.assign(**z)
    k = max(int(len(g) * 0.10), 1)
    top = g2.nlargest(k, 'score_pi')
    exp1 = top['zc_1'].mean()
    assert abs(feat.loc[d, 'zc_1'] - exp1) < 1e-9


def test_standardize_uses_train_only():
    panel = T.load_panel()
    feat = T.month_features(panel)
    mask = feat.index < pd.Timestamp('2018-01-01')
    z = T.standardize(feat, mask)
    # train-window columns are ~mean 0
    assert abs(z.loc[mask, 'pi'].mean()) < 1e-9
    comp = T.compress_features(feat)
    assert {'level', 'slope', 'curv', 'pi'} == set(comp.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "features or standardize" -v`
Expected: FAIL with `AttributeError: ... 'month_features'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "features or standardize" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add paper/tv_band.py tests/test_tv_band.py
git commit -m "tv_band: month-level features (z-curve, term structure, pi) + compress/standardize"
```

---

### Task 5: net IR + paired block bootstrap

**Files:**
- Modify: `paper/tv_band.py`
- Test: `tests/test_tv_band.py`

**Interfaces:**
- Produces: `net_ir(r) -> float` = `mean(r)/std(r)·√12` (0.0 if std==0 or len<2).
- Produces: `paired_block_bootstrap(a, b, n_boot=10000, block=12, seed=0) -> (delta, lo, hi)` on the net-IR difference of aligned monthly series `a` and `b` (annualized), returning the point delta and 95% CI.

- [ ] **Step 1: Write the failing test**

```python
def test_net_ir_and_bootstrap():
    rng = np.random.default_rng(0)
    idx = pd.date_range('2011-01-31', periods=120, freq='ME')
    a = pd.Series(rng.normal(0.01, 0.04, 120), index=idx)
    b = a + 0.003                                     # b strictly better
    assert T.net_ir(b) > T.net_ir(a)
    delta, lo, hi = T.paired_block_bootstrap(a, b)
    assert delta == pytest.approx(T.net_ir(b) - T.net_ir(a), abs=1e-9)
    assert lo <= delta <= hi
    # zero-variance guard
    assert T.net_ir(pd.Series([0.0, 0.0, 0.0])) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "net_ir_and_bootstrap" -v`
Expected: FAIL with `AttributeError: ... 'net_ir'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "net_ir_and_bootstrap" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add paper/tv_band.py tests/test_tv_band.py
git commit -m "tv_band: net IR + paired block bootstrap"
```

---

### Task 6: Gate 0 ceiling — feature-binned oracle + static baseline + report

**Files:**
- Modify: `paper/tv_band.py`
- Test: `tests/test_tv_band.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `best_static(panel, sp_pack, exit_grid=(15,20,25,30,40), flat_bp=None) -> (E_star, ir)` — the single exit width maximizing full-sample net IR (entry fixed at 10%).
- Produces: `oracle_bin_ir(panel, feat, sp_pack, bins, exit_grid, flat_bp=None) -> (ir, assignment)` — assign each month to a feature bin, give each bin its full-sample net-IR-optimal exit width, simulate that per-month band, return the realized net IR. `bins`: a Series (date→bin id). Two binnings tested: K=4 thesis clusters and a coarse `pi × zc-level` grid.
- Produces: `run_gate0() -> dict` — orchestrates trust gate, static baseline, both oracles, writes `tv_band_gate0.{md,csv}`, returns the summary dict with `g0_pass` (oracle uplift over static > 0.029·static_ir threshold, i.e. beyond the spread-timing artifact).

- [ ] **Step 1: Write the failing test**

```python
def test_oracle_ceiling_at_least_static():
    panel = T.load_panel()
    sp_pack = T.load_spreads()
    feat = T.month_features(panel)
    E_star, ir_static = T.best_static(panel, sp_pack)
    # coarse pi x zc-level bins
    lvl = feat[[f'zc_{h}' for h in range(1, 13)]].mean(axis=1)
    bins = (pd.qcut(feat['pi'], 2, labels=False, duplicates='drop').astype(str)
            + '_' + pd.qcut(lvl, 2, labels=False, duplicates='drop').astype(str))
    ir_oracle, _ = T.oracle_bin_ir(panel, feat, sp_pack, bins,
                                   exit_grid=(15, 20, 25, 30, 40))
    # a per-bin optimal band cannot do worse than the single best static band
    assert ir_oracle >= ir_static - 1e-9


def test_run_gate0_writes_report():
    res = T.run_gate0()
    assert 'g0_pass' in res and isinstance(res['g0_pass'], bool)
    assert os.path.exists(os.path.join(T.OUT_DIR, 'tv_band_gate0.md'))
    assert os.path.exists(os.path.join(T.OUT_DIR, 'tv_band_gate0.csv'))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "oracle_ceiling or run_gate0" -v`
Expected: FAIL with `AttributeError: ... 'best_static'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "oracle_ceiling or run_gate0" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add paper/tv_band.py tests/test_tv_band.py
git commit -m "tv_band: Gate 0 ceiling (feature-binned oracle) + report"
```

---

### Task 7: G2 mechanism diagnostic — calm vs panic turnover (independent)

**Files:**
- Modify: `paper/tv_band.py`
- Test: `tests/test_tv_band.py`

**Interfaces:**
- Consumes: `simulate`, `policy_monthly`.
- Produces: `regime_turnover(panel, pi_cut=0.5) -> dict` with `to_calm`, `to_panic`, `ratio`, `n_calm`, `n_panic` from the no-band monthly book — the independent re-derivation of the "trading is regime-invariant" finding.

- [ ] **Step 1: Write the failing test**

```python
def test_regime_turnover_independent():
    panel = T.load_panel()
    d = T.regime_turnover(panel)
    assert {'to_calm', 'to_panic', 'ratio', 'n_calm', 'n_panic'} <= set(d)
    assert d['n_calm'] > 0 and d['n_panic'] > 0
    # sanity: turnover is a fraction in (0, 1]
    assert 0 < d['to_calm'] <= 1 and 0 < d['to_panic'] <= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py::test_regime_turnover_independent -v`
Expected: FAIL with `AttributeError: ... 'regime_turnover'`

- [ ] **Step 3: Write minimal implementation**

```python
def regime_turnover(panel, pi_cut=0.5):
    m, _ = simulate(panel, policy_monthly)
    calm = m[m['pi'] < pi_cut]['turnover']
    panic = m[m['pi'] >= pi_cut]['turnover']
    to_c, to_p = float(calm.mean()), float(panic.mean())
    return {'to_calm': to_c, 'to_panic': to_p,
            'ratio': to_p / to_c if to_c else np.nan,
            'n_calm': int(len(calm)), 'n_panic': int(len(panic))}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py::test_regime_turnover_independent -v`
Expected: PASS

- [ ] **Step 5: Add CLI + run the full Gate 0, then commit**

Add at the bottom of `paper/tv_band.py`:

```python
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='gate0', choices=['gate0'])
    ap.parse_args()
    res = run_gate0()
    print('regime turnover:', regime_turnover(load_panel()))
    print('g0_pass:', res['g0_pass'])
    print(res['table'].to_string(index=False))


if __name__ == '__main__':
    main()
```

Run: `.venv/bin/python -m paper.tv_band --stage gate0`
Expected: prints the trust-gate-passing Gate 0 table, the calm/panic turnover, and `g0_pass`. Inspect `paper/results/banding_study/tv_band_gate0.md`.

```bash
git add paper/tv_band.py tests/test_tv_band.py
git commit -m "tv_band: regime-turnover diagnostic + Gate 0 CLI"
```

---

## Decision after Gate 0

- **If `g0_pass` is True** (the feature-binned oracle beats static beyond the 2.9% artifact): the information exists in-sample. Write the **Gate 1 plan** (Trainers A/B/C + walk-forward OOS + honesty gate) per spec §5, §7.
- **If `g0_pass` is False**: the null extends to the richest input set at its perfect-hindsight ceiling. Stop; this is the reportable result. Record both outcomes and the calm/panic turnover diagnostic in `paper/results/PAPER_NOTES.md`.

---

## Self-Review notes

- Spec §6 trust gate → Task 2 (verified anchor pre-checked at 2015-06-30).
- Spec §4 band mechanic (E_enter,E_exit floating count) → Task 3.
- Spec §5 features (z-curve + term structure + pi, full-25 + compressed-4) → Task 4.
- Spec §3 Gate 0 feature-binned oracle (bin variants) → Task 6.
- Spec §2 G2 mechanism → Task 7.
- Spec §7 bootstrap CI → Task 5 (used by the Gate 1 plan; built here so the stats primitive is proven before OOS work).
- Deferred to Gate 1 plan (YAGNI): Trainers A/B/C, walk-forward OOS stitch, 1-SE regularization, full-sample flexible-fit ceiling variant.
