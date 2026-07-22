# Term-Structure Time-Varying Band — Gate 1 (Learned OOS Band) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Test out-of-sample whether a band that learns `(E_enter, E_exit)` from the 12-horizon momentum term structure + `pi` beats a static band — the real (walk-forward) test that Gate 0's in-sample ceiling (`g0_pass=True`, +18.5%) only established as *possible*.

**Architecture:** Extend the existing clean-room module `paper/tv_band.py` (all Gate-0 primitives already built + tested) with a walk-forward OOS harness and four learned arms, staged behind `--stage gate1`. Every arm is fit on prior months only, re-fit annually, and its year-Y realized net active returns are stitched into one OOS series (2013→2025). All arms are compared to the walk-forward best-static band with a paired block-bootstrap CI and a 1-SE honesty gate.

**Tech Stack:** Python 3, numpy, pandas, scipy.optimize (Nelder-Mead), sklearn (GradientBoostingRegressor, Ridge), pytest. All confirmed available.

## Global Constraints

- Extend `paper/tv_band.py`; import NOTHING from `paper/execution.py` or `paper/banding_study.py`. Reuse the module's own tested primitives: `load_panel`, `load_spreads`, `simulate`, `policy_band`, `_policy_perbin`, `_bins_clusters`, `month_features`, `standardize`, `net_ir`, `paired_block_bootstrap`, `best_static`, `price`, `net`, `net_ir`, `MOMS`, `OUT_DIR`.
- Band mechanic is unchanged: long-only VW top-decile floating-count hold-band; hold while rank% ≤ E_exit, add while rank% ≤ E_enter; rank by `score_pi` desc, ties by `permno` asc; `rank% = (i+1)/n`.
- **No leakage:** for OOS year Y, every fitted parameter (static `E`, cluster bin edges + widths, Trainer A weights, Trainer B/C models + their trailing-window targets, feature standardization mean/std) uses ONLY months with `year < Y`. Features themselves are contemporaneous/PIT (month-t cross-section) and may be read for OOS months; only *parameters* are prior-only.
- **Walk-forward stitch:** OOS from 2013 (matches the main model's XGB score availability). For each year Y ≥ 2013, fit on `year < Y`, apply to year Y, keep only year-Y net returns, concatenate.
- **Objective / selection:** annualized net IR = `mean/std·√12` (ddof=1). Costs: measured half-spreads primary, flat 10bp robustness.
- **Honesty gate (the win condition):** a learned arm beats static only if OOS `net_ir(arm) − net_ir(static) > 0`, the paired block-bootstrap CI (block 12, 10000 reps) excludes 0, AND the arm still beats static under 1-SE regularization (defined in Task 5). Report all arms regardless.
- Outputs: `paper/results/banding_study/tv_band_gate1.{md,csv}`.
- Reference OOS numbers to beat (walk-forward, from prior campaign, main model xgb): static band OOS net IR ≈ 0.585; the prior regularized regime bands did NOT beat it. Gate-0 in-sample ceiling was 0.464 on the 2011-2025 measured-cost active series (different window/metric base — do not expect equality; the Gate-1 number is its own baseline).

---

## File Structure

- Modify: `paper/tv_band.py` — add the walk-forward harness, four arms, Gate-1 report, `--stage gate1`.
- Modify: `tests/test_tv_band.py` — add Gate-1 tests.
- Produce: `paper/results/banding_study/tv_band_gate1.{md,csv}`.

Run tests: `.venv/bin/python -m pytest tests/test_tv_band.py -v`

---

### Task 1: Walk-forward harness + feature-policy + baselines + the decisive cluster-band arm

This is the decisive cheap OOS test: the walk-forward cluster band is the realizable analog of the Gate-0 oracle (same K=4 z-curve binning, but per-cluster widths and bin edges fit on prior data only). If it does not beat static OOS, the discrete term-structure band is null out of sample regardless of the fancier trainers.

**Files:**
- Modify: `paper/tv_band.py`
- Test: `tests/test_tv_band.py`

**Interfaces:**
- Produces: `make_feature_policy(ee_ex_fn) -> policy` — a `simulate`-compatible policy where `ee_ex_fn(date) -> (E_enter_pct, E_exit_pct)`; applies the standard floating-count band with those per-month percentiles.
- Produces: `walkforward(panel, feat, sp_pack, fit_fn, start_oos=2013, flat_bp=None) -> pd.Series` — stitched OOS monthly net active returns. `fit_fn(panel, feat, sp_pack, train_mask) -> policy` is fit on `train_mask` months only; the returned policy is simulated over the full panel and only year-Y months are kept.
- Produces: `fit_monthly`, `fit_static`, `fit_cluster_band` — three `fit_fn`s.
- Produces: `_train_subpanel(panel, feat, train_mask) -> panel` helper (rows whose date is a train month).

- [ ] **Step 1: Write the failing test**

```python
def test_walkforward_static_and_cluster_band():
    panel = T.load_panel()
    feat = T.month_features(panel)
    panel = panel[panel['date'].isin(feat.index)]
    sp = T.load_spreads()
    wf_static = T.walkforward(panel, feat, sp, T.fit_static, start_oos=2013)
    wf_cluster = T.walkforward(panel, feat, sp, T.fit_cluster_band, start_oos=2013)
    # OOS series cover 2013..2025, aligned, non-empty
    assert wf_static.index.min().year == 2013
    assert len(wf_static) == len(wf_cluster) > 100
    # both are finite net-return series
    assert wf_static.notna().all() and wf_cluster.notna().all()
    # sanity: static WF net IR is in a plausible band for the main model
    assert 0.0 < T.net_ir(wf_static) < 1.5


def test_make_feature_policy_matches_static_band():
    panel = T.load_panel()
    # a constant feature policy (E_enter=10,E_exit=25 for all dates) must equal policy_band(10,25)
    pol_feat = T.make_feature_policy(lambda t: (10, 25))
    a, _ = T.simulate(panel, pol_feat)
    b, _ = T.simulate(panel, T.policy_band(10, 25))
    assert np.allclose(a['turnover'].values, b['turnover'].values)
    assert np.allclose(a['book'].values, b['book'].values)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "walkforward_static or feature_policy_matches" -v`
Expected: FAIL with `AttributeError: ... 'make_feature_policy'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "walkforward_static or feature_policy_matches" -v`
Expected: PASS (2 tests). `test_walkforward_static_and_cluster_band` runs several fits × sims and may take 1–3 minutes — let it finish.

- [ ] **Step 5: Commit**

```bash
git add paper/tv_band.py tests/test_tv_band.py
git commit -m "tv_band: Gate 1 walk-forward harness + feature-policy + static/cluster-band arms"
```

---

### Task 2: Trailing-window ex-post-optimal band target (for supervised trainers)

**Files:**
- Modify: `paper/tv_band.py`
- Test: `tests/test_tv_band.py`

**Interfaces:**
- Produces: `trailing_optimal_targets(panel, feat, sp_pack, window=36, enter_grid=(5,10,15), exit_grid=(15,20,25,30,40)) -> pd.DataFrame` indexed by date with columns `tgt_enter, tgt_exit` — for each month t, the `(E_enter, E_exit)` that maximizes net IR over the trailing `window` months ending at t (inclusive). Months with fewer than `window` prior months use all available. This is the supervised target for Trainers B/C. Enter grid restricted to ≤ exit.

- [ ] **Step 1: Write the failing test**

```python
def test_trailing_optimal_targets():
    panel = T.load_panel()
    feat = T.month_features(panel)
    panel = panel[panel['date'].isin(feat.index)]
    sp = T.load_spreads()
    tgt = T.trailing_optimal_targets(panel, feat, sp, window=36)
    assert {'tgt_enter', 'tgt_exit'}.issubset(tgt.columns)
    # targets respect enter <= exit and live in the grids
    assert (tgt['tgt_enter'] <= tgt['tgt_exit']).all()
    assert tgt['tgt_exit'].isin([15, 20, 25, 30, 40]).all()
    assert len(tgt) > 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "trailing_optimal" -v`
Expected: FAIL with `AttributeError: ... 'trailing_optimal_targets'`

- [ ] **Step 3: Write minimal implementation**

```python
def trailing_optimal_targets(panel, feat, sp_pack, window=36,
                             enter_grid=(5, 10, 15), exit_grid=(15, 20, 25, 30, 40)):
    dates = list(feat.index)
    # precompute each (enter,exit) band's monthly net series once over the full panel
    combos = [(ee, ex) for ee in enter_grid for ex in exit_grid if ee <= ex]
    series = {}
    for ee, ex in combos:
        m, led = simulate(panel, policy_band(ee, ex))
        series[(ee, ex)] = net(m['active'], price(led, sp_pack))
    rows = []
    for i, t in enumerate(dates):
        lo = max(0, i - window + 1)
        win = dates[lo:i + 1]
        best, best_ir = (10, 20), -np.inf
        for c in combos:
            r = series[c].reindex(win).dropna()
            ir = net_ir(r)
            if ir > best_ir:
                best, best_ir = c, ir
        rows.append({'date': t, 'tgt_enter': best[0], 'tgt_exit': best[1]})
    return pd.DataFrame(rows).set_index('date')
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "trailing_optimal" -v`
Expected: PASS. (Precomputes ~13 band series once, then windows — a couple minutes.)

- [ ] **Step 5: Commit**

```bash
git add paper/tv_band.py tests/test_tv_band.py
git commit -m "tv_band: Gate 1 trailing-window optimal-band target builder"
```

---

### Task 3: Trainer A — linear-exp policy fit by black-box net-IR maximization

**Files:**
- Modify: `paper/tv_band.py`
- Test: `tests/test_tv_band.py`

**Interfaces:**
- Produces: `fit_trainer_A(panel, feat, sp_pack, train_mask, feat_cols=None) -> policy` — fits `(base_enter, base_exit, w_enter·x, w_exit·x)` by Nelder-Mead (`scipy.optimize.minimize`) maximizing TRAIN-window net IR of the resulting feature band; returns a `make_feature_policy`. `E = clip(base·exp(w·z), 5, 60)`, `z` = standardized features (train mean/std). Uses the compressed 4-dim features (`level, slope, curv, pi`) by default to keep the search low-dimensional (spec §10 overfit mitigation).
- Produces: `_feature_matrix(feat, cols, train_mask) -> (Z_all, mu, sd)` — standardized feature matrix using train-only moments.

- [ ] **Step 1: Write the failing test**

```python
def test_trainer_A_beats_static_in_sample():
    panel = T.load_panel()
    feat = T.month_features(panel)
    panel = panel[panel['date'].isin(feat.index)]
    sp = T.load_spreads()
    mask = feat.index.year < 2020            # train on <2020
    polA = T.fit_trainer_A(panel, feat, sp, mask)
    sub = T._train_subpanel(panel, feat, mask)
    # in-sample, the fitted policy should not be worse than static (optimizer seeded/bounded)
    mA, ledA = T.simulate(sub, polA)
    irA = T.net_ir(T.net(mA['active'], T.price(ledA, sp)))
    E, irS = T.best_static(sub, sp)
    assert irA >= irS - 0.02             # allows tiny optimizer slack; expect >= static
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "trainer_A" -v`
Expected: FAIL with `AttributeError: ... 'fit_trainer_A'`

- [ ] **Step 3: Write minimal implementation**

```python
from scipy.optimize import minimize                     # add near top imports

_COMP_COLS = ['level', 'slope', 'curv', 'pi']


def _feature_matrix(feat, cols, train_mask):
    comp = compress_features(feat)[cols]
    mu = comp.loc[train_mask].mean()
    sd = comp.loc[train_mask].std(ddof=0).replace(0.0, 1.0)
    Z = (comp - mu) / sd
    return Z, mu, sd


def fit_trainer_A(panel, feat, sp_pack, train_mask, feat_cols=None):
    cols = feat_cols or _COMP_COLS
    Z, _, _ = _feature_matrix(feat, cols, train_mask)
    sub = _train_subpanel(panel, feat, train_mask)
    E0, _ = best_static(sub, sp_pack)
    k = len(cols)

    def unpack(theta):
        be, bx = theta[0], theta[1]
        we, wx = theta[2:2 + k], theta[2 + k:2 + 2 * k]
        return be, bx, we, wx

    def policy_from(theta):
        be, bx, we, wx = unpack(theta)
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
    theta = res.x if (res.success or np.isfinite(res.fun)) else theta0
    # never return something worse than the static seed
    if -neg_ir(theta) < -neg_ir(theta0):
        theta = theta0
    return policy_from(theta)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "trainer_A" -v`
Expected: PASS. (Nelder-Mead runs many simulate() evals — 1–3 minutes.)

- [ ] **Step 5: Commit**

```bash
git add paper/tv_band.py tests/test_tv_band.py
git commit -m "tv_band: Gate 1 Trainer A (linear-exp policy, black-box net-IR fit)"
```

---

### Task 4: Trainers B (GBM) + C (ridge) — supervised on the trailing-window target

**Files:**
- Modify: `paper/tv_band.py`
- Test: `tests/test_tv_band.py`

**Interfaces:**
- Produces: `fit_trainer_B(panel, feat, sp_pack, train_mask, window=36) -> policy` — GBM (`GradientBoostingRegressor`) per output (enter, exit) fit on TRAIN-window `(features → trailing_optimal_targets)`; predicts per-month `(E_enter, E_exit)`, clipped to grids and `enter ≤ exit`.
- Produces: `fit_trainer_C(...) -> policy` — same but `Ridge`.
- Both use the full 25-dim features (z-curve + term structure + pi) standardized on train.

- [ ] **Step 1: Write the failing test**

```python
def test_trainers_B_C_produce_valid_bands():
    panel = T.load_panel()
    feat = T.month_features(panel)
    panel = panel[panel['date'].isin(feat.index)]
    sp = T.load_spreads()
    mask = feat.index.year < 2020
    for fit in (T.fit_trainer_B, T.fit_trainer_C):
        pol = fit(panel, feat, sp, mask)
        m, led = T.simulate(panel, pol)          # applies to full panel incl OOS
        # produces a valid book every month (floating count between ~5% and ~60%)
        assert (m['n_names'] > 0).all()
        assert T.net(m['active'], T.price(led, sp)).notna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "trainers_B_C" -v`
Expected: FAIL with `AttributeError: ... 'fit_trainer_B'`

- [ ] **Step 3: Write minimal implementation**

```python
from sklearn.ensemble import GradientBoostingRegressor   # add near top imports
from sklearn.linear_model import Ridge

_FULL_COLS = ([f'zc_{h}' for h in range(1, 13)] +
              [f'cs_{h}' for h in range(1, 13)] + ['pi'])


def _fit_supervised(panel, feat, sp_pack, train_mask, window, make_model):
    tgt = trailing_optimal_targets(panel, feat, sp_pack, window=window)
    mu = feat.loc[train_mask, _FULL_COLS].mean()
    sd = feat.loc[train_mask, _FULL_COLS].std(ddof=0).replace(0.0, 1.0)
    Z = (feat[_FULL_COLS] - mu) / sd
    tr = train_mask
    Xtr = Z.loc[feat.index[tr]].values
    me = make_model().fit(Xtr, tgt.loc[feat.index[tr], 'tgt_enter'].values)
    mx = make_model().fit(Xtr, tgt.loc[feat.index[tr], 'tgt_exit'].values)

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "trainers_B_C" -v`
Expected: PASS. (Builds the trailing target once per fit — a couple minutes.)

- [ ] **Step 5: Commit**

```bash
git add paper/tv_band.py tests/test_tv_band.py
git commit -m "tv_band: Gate 1 Trainers B (GBM) + C (ridge), supervised on trailing target"
```

---

### Task 5: Gate 1 walk-forward evaluation + honesty gate + report + CLI

**Files:**
- Modify: `paper/tv_band.py`
- Test: `tests/test_tv_band.py`

**Interfaces:**
- Produces: `run_gate1(start_oos=2013, flat_bp=None) -> dict` — runs `walkforward` for arms `monthly, static, cluster_band, trainer_A, trainer_B, trainer_C`; computes each arm's stitched-OOS net IR, the (arm − static) delta, its paired block-bootstrap CI, and the 1-SE-regularized verdict (arm adopted only if `delta − 1·SE_delta > 0`, where `SE_delta` is the bootstrap std of the delta). Writes `tv_band_gate1.{md,csv}` and returns the summary. `g1_win` = any arm passes the full honesty gate.
- Modifies: `main()` — add `--stage gate1` running `run_gate1()` and printing the arm table + `g1_win`.

- [ ] **Step 1: Write the failing test**

```python
def test_run_gate1_report_structure_and_gate_logic():
    # Fast subset (monthly + static only) exercises assembly, static-delta=0, gate,
    # and file write without the ~40-min full learned sweep (that runs via CLI, Step 5).
    res = T.run_gate1(start_oos=2013,
                      arms={'monthly': T.fit_monthly, 'static': T.fit_static})
    df = res['table']
    assert {'monthly', 'static'} == set(df['arm'])
    assert {'oos_ir', 'minus_static', 'ci_lo', 'ci_hi', 'se', 'passes_1se'}.issubset(df.columns)
    # static's delta vs itself is 0
    assert abs(df.loc[df['arm'] == 'static', 'minus_static'].iloc[0]) < 1e-9
    # no learned arm in this subset => g1_win False
    assert res['g1_win'] is False
    assert os.path.exists(os.path.join(T.OUT_DIR, 'tv_band_gate1.md'))


def test_delta_ci_zero_for_identical_series():
    rng = np.random.default_rng(0)
    idx = pd.date_range('2013-01-31', periods=120, freq='ME')
    s = pd.Series(rng.normal(0.01, 0.04, 120), index=idx)
    delta, lo, hi, se = T._delta_ci(s, s)          # arm == static
    assert abs(delta) < 1e-9 and se >= 0 and lo <= 0 <= hi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "run_gate1 or delta_ci" -v`
Expected: FAIL with `AttributeError: ... 'run_gate1'` / `'_delta_ci'`

- [ ] **Step 3: Write minimal implementation**

```python
def _delta_ci(arm_series, static_series, n_boot=10000, block=12, seed=0):
    """(delta, lo, hi, se) for net_ir(arm) - net_ir(static). Single circular-block
    resample loop; extends paired_block_bootstrap's method to also return the SE."""
    a, b = static_series.align(arm_series, join='inner')     # a=static, b=arm
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
    return float(delta), float(lo), float(hi), float(stats.std(ddof=1))


def run_gate1(start_oos=2013, flat_bp=None, arms=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    panel = load_panel()
    feat = month_features(panel)
    panel = panel[panel['date'].isin(feat.index)]
    sp = load_spreads()
    if arms is None:
        arms = {'monthly': fit_monthly, 'static': fit_static,
                'cluster_band': fit_cluster_band, 'trainer_A': fit_trainer_A,
                'trainer_B': fit_trainer_B, 'trainer_C': fit_trainer_C}
    assert 'static' in arms, "run_gate1 requires the 'static' arm as the baseline"
    series = {name: walkforward(panel, feat, sp, fn, start_oos, flat_bp)
              for name, fn in arms.items()}
    static_s = series['static']
    rows = []
    for name, s in series.items():
        delta, lo, hi, se = _delta_ci(s, static_s)
        rows.append({'arm': name, 'oos_ir': net_ir(s), 'minus_static': delta,
                     'ci_lo': lo, 'ci_hi': hi, 'se': se,
                     'excl0': bool(lo > 0 or hi < 0),
                     'passes_1se': bool(delta - se > 0)})
    df = pd.DataFrame(rows)
    learned = df[df['arm'].isin(['cluster_band', 'trainer_A', 'trainer_B', 'trainer_C'])]
    g1_win = bool(((learned['minus_static'] > 0) & learned['excl0'] &
                   learned['passes_1se']).any())
    df.to_csv(os.path.join(OUT_DIR, 'tv_band_gate1.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'tv_band_gate1.md'), 'w') as f:
        f.write('# Gate 1 — learned real-time term-structure band (walk-forward OOS)\n\n')
        f.write(f'OOS {start_oos}-2025, annual re-fit on prior months only, stitched. '
                'A learned arm wins only if minus_static>0 AND its bootstrap CI excludes 0 '
                'AND it survives 1-SE regularization (minus_static - SE > 0).\n\n')
        f.write(df.to_string(index=False))
        f.write(f'\n\n**g1_win: {g1_win}**\n')
    return {'table': df, 'g1_win': g1_win}
```

Extend `main()`: add `'gate1'` to the `--stage` choices and, when selected, run and print:

```python
    if args.stage == 'gate1':
        res = run_gate1()
        print('g1_win:', res['g1_win'])
        print(res['table'].to_string(index=False))
        return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "run_gate1 or delta_ci" -v`
Expected: PASS (2 tests). The structure test runs only the `monthly`+`static` subset (static = 13 annual `best_static` fits), ~1–2 minutes; if it exceeds the harness 120s timeout it will background — read the output file to confirm PASS. `_delta_ci` test is instant.

- [ ] **Step 5: Run the full Gate 1 end-to-end via CLI, then commit**

Run the full 6-arm sweep **detached** (Trainer A's Nelder-Mead makes this ~30–45 minutes):
`nohup .venv/bin/python -m paper.tv_band --stage gate1 > paper/results/banding_study/gate1_run.log 2>&1 &`
Wait for it to finish (poll the log / the written `tv_band_gate1.md`), confirm the arm table has all 6 arms and a `g1_win` verdict, and that the run is clean. Then:

```bash
git add paper/tv_band.py tests/test_tv_band.py paper/results/banding_study/tv_band_gate1.csv paper/results/banding_study/tv_band_gate1.md
git commit -m "tv_band: Gate 1 walk-forward evaluation + honesty gate + report + CLI"
```

---

### Task 6: Rebound-theory band arm (Test 3) + band-width interpretability

Encodes the thesis rebound mechanism as a *constrained* band: WIDEN (trade less) during the
below-average / reversal regimes (thesis clusters 3 and 4), never narrower than static. This is
the *implementable* direction (trade less when it is hardest) and it is expected to **cost**
return in cluster 4 (the establishment phase, where the Gate-0 oracle showed the profit comes
from trading *more*) while being closer to free in cluster 3 (the hold-through-recovery phase).
The decomposition (c4-only / c3-only / both) is the finding. Also adds the interpretability
output: what market attributes correspond to each band width.

Mapping: `_bins_clusters` sorts months into K=4 z-curve-level quartiles; the two lowest-level
bins are the below-average baskets — **bin 0 ≈ thesis cluster 4** (deepest below average, active
crisis), **bin 1 ≈ thesis cluster 3** (below average, recovery). Bins 2–3 are the above-average
(continuation) baskets and are left at the static width.

**Files:**
- Modify: `paper/tv_band.py`
- Test: `tests/test_tv_band.py`

**Interfaces:**
- Produces: `rebound_band_spec(panel, feat, sp_pack, train_mask, widen_bins, widen_grid=(30,40,50,60)) -> (policy, bin_of_date, width_by_bin)` — PIT K=4 z-curve-level bins (edges fit on train); for bins in `widen_bins`, tune ONE wider-than-static exit width on train net IR; other bins use best static.
- Produces: `fit_rebound_c4`, `fit_rebound_c3`, `fit_rebound_both` — `fit_fn` wrappers (`widen_bins` = `(0,)`, `(1,)`, `(0,1)`).
- Produces: `band_width_attributes(feat, bin_of_date, width_by_bin, pi_cut=0.5) -> pd.DataFrame` — per bin: chosen exit width + avg pi, frac panic, avg z-curve level, avg cross-sectional momentum, month count.
- Modifies: `run_gate1` — adds the three rebound arms to the default arm set and to the learned-arm honesty gate; writes `tv_band_gate1_attributes.csv` from a full-sample `rebound_band_spec((0,1))`.

- [ ] **Step 1: Write the failing test**

```python
def test_rebound_band_widens_only_in_rebound_bins():
    panel = T.load_panel()
    feat = T.month_features(panel)
    panel = panel[panel['date'].isin(feat.index)]
    sp = T.load_spreads()
    mask = feat.index.year < 2020
    pol, bod, wbb = T.rebound_band_spec(panel, feat, sp, mask, widen_bins=(0, 1))
    E, _ = T.best_static(T._train_subpanel(panel, feat, mask), sp)
    for b, w in wbb.items():
        if b in (0, 1):
            assert w >= E                 # rebound bins widen (or hold) — never tighter
        else:
            assert w == E                 # continuation bins stay at static
    attrs = T.band_width_attributes(feat, bod, wbb)
    # the two widened bins are the lowest z-curve-level (below-average) baskets
    lo = set(attrs.sort_values('avg_zc_level').head(2)['bin'].tolist())
    assert lo == {0, 1}
    # panic concentrates in the deepest bin (bin 0 ≈ cluster 4)
    a = attrs.set_index('bin')
    assert a.loc[0, 'frac_panic'] >= a.loc[3, 'frac_panic']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "rebound_band_widens" -v`
Expected: FAIL with `AttributeError: ... 'rebound_band_spec'`

- [ ] **Step 3: Write minimal implementation**

```python
def rebound_band_spec(panel, feat, sp_pack, train_mask, widen_bins,
                      widen_grid=(30, 40, 50, 60)):
    zc = [f'zc_{h}' for h in range(1, 13)]
    ftr = feat.loc[train_mask]
    _, edges = pd.qcut(ftr[zc].mean(axis=1), 4, labels=False,
                       retbins=True, duplicates='drop')
    edges = edges.copy(); edges[0] = -np.inf; edges[-1] = np.inf
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
        wbb = width_map(w); v = ir_for(wbb)
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
```

Then MODIFY `run_gate1` (from Task 5): add the three rebound arms to the default `arms` dict —

```python
        arms = {'monthly': fit_monthly, 'static': fit_static,
                'cluster_band': fit_cluster_band, 'trainer_A': fit_trainer_A,
                'trainer_B': fit_trainer_B, 'trainer_C': fit_trainer_C,
                'rebound_c4': fit_rebound_c4, 'rebound_c3': fit_rebound_c3,
                'rebound_both': fit_rebound_both}
```

— extend the learned-arm honesty filter to include them —

```python
    learned = df[df['arm'].isin(['cluster_band', 'trainer_A', 'trainer_B',
                                 'trainer_C', 'rebound_c4', 'rebound_c3',
                                 'rebound_both'])]
```

— and, after writing the main table, emit the interpretability CSV (full-sample spec; descriptive, not a performance claim) —

```python
    full_mask = np.ones(len(feat), dtype=bool)
    _, bod, wbb = rebound_band_spec(panel, feat, sp, full_mask, widen_bins=(0, 1))
    band_width_attributes(feat, bod, wbb).to_csv(
        os.path.join(OUT_DIR, 'tv_band_gate1_attributes.csv'), index=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tv_band.py -k "rebound_band_widens" -v`
Expected: PASS. (rebound_band_spec on the train sub-panel is cheap — `best_static` + a 4-point widen grid — a few seconds.)

- [ ] **Step 5: Commit**

```bash
git add paper/tv_band.py tests/test_tv_band.py
git commit -m "tv_band: Gate 1 rebound-theory band arm (c3/c4/both) + band-width interpretability"
```

Note: the three rebound arms are cheap (small train-only grid), so adding them to the full `run_gate1` sweep costs little beyond their three walk-forwards. The expected result is that `rebound_c3` is near-free vs static while `rebound_c4` costs return — the return-vs-implementability decomposition that is Test 3's finding.

---

## Decision after Gate 1

- **If `g1_win` is True** (a learned arm beats static OOS, CI excludes 0, survives 1-SE): the term-structure band is a real, realizable cost-mitigation result — the headline positive finding. Then run the implementability direction (Task 8 `run_gate0_impl` analog) on the winning OOS band, and write it into the paper.
- **If `g1_win` is False** (expected, given the prior campaign): the null extends to the richest input set with a proper learned function — the ceiling exists in-sample (Gate 0) but is NOT realizable out of sample. Report the ceiling-vs-realized gap as the honest finding: a clean negative result that strengthens "a well-chosen static band is enough."

## Self-Review notes

- Spec §5 Trainer A (linear-exp, black-box) → Task 3; Trainers B/C (supervised, trailing target) → Task 4 (+ Task 2 target).
- Spec §7 walk-forward + bootstrap CI + 1-SE honesty gate → Task 1 harness + Task 5 gate.
- Spec §10 overfit mitigations: Trainer A uses compressed 4-dim features; 1-SE gate; Gate 0 already gated this build. Trailing-window length is a knob (default 36) — Task 2 exposes it for a sensitivity pass in the paper.
- Decisive cheap pre-check (cluster_band, Task 1) mirrors the Gate-0 oracle and can falsify before Trainers A/B/C matter.
- Reuses only tested Gate-0 primitives; no import of execution.py/banding_study.py.
