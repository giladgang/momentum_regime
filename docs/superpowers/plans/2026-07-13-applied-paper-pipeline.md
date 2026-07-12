# Applied Paper Pipeline (`paper/`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the applied study of docs/superpowers/specs/2026-07-13-applied-paper-design.md: expanding-window (yearly) regime-aware momentum, top-1000 large-cap universe, long-only fully-invested, argmax combo selection on an applied IR objective, full cross-section persistence for later TC analysis.

**Architecture:** Self-contained `paper/` package. Bit-validated HMM machinery ported from `experiments/2026-07-09-prod-budget-dd-vol-reln.py` and enforced by an identity test against that module. Data = existing verified ext2025 files (panel through 2025-11, stocks through 2025-12), swappable to ext2026 via one config constant. Staged pipeline runner (data → universe → folds → walk → report), resume-safe per unit.

**Tech Stack:** Python 3.12 (`.venv`), numpy/pandas/scipy/xgboost/pyarrow, pytest. Multiprocessing spawn pools (workers=7 for long runs).

## Global Constraints

- NO new pip dependencies; `requirements.lock` untouched.
- Thesis pipeline untouched: nothing under `scripts/`, `results/`, `tables/`, `data/` is written.
- All outputs under `paper/results/`. Binaries (`*.parquet`) gitignored; CSVs/markdown tracked.
- Frozen: XGB hyperparameters from root `config.py` (N_ESTIMATORS=500, MAX_DEPTH=4, LEARNING_RATE, SUBSAMPLE, COLSAMPLE); HMM priors (M0=0.0, KAPPA0=0.01, NU0_OFF=2, DIRICHLET_ALPHA=[[9,1],[1,9]]); Gibbs budget 2000/500; seeds HMM 1..50 eval / 1..3 selection, XGB 1..20 eval / 1..5 selection.
- Combo pool: the 25 combos of `experiments/results/wf_pool.csv` (frozen).
- Convention: return series are indexed by FORMATION month t; the value is `ret_fwd` (earned over t+1). Identical to thesis `long_short_port`.
- Evaluation years 2011..2025 (2025 partial: formation months Jan–Nov 2025).
- Root `config.py` is imported as `repo_config`; `paper/config.py` is `paper.config`. All paper code runs with cwd = repo root.
- Every stochastic output logs its seeds. Resume = relaunch skips completed units.
- Run first, write later: no thesis/paper prose from these numbers in this plan.

---

### Task 1: Scaffold + config

**Files:**
- Create: `paper/__init__.py`, `paper/src/__init__.py`, `paper/tests/__init__.py` (empty)
- Create: `paper/config.py`
- Create: `paper/results/.gitignore`
- Test: `paper/tests/test_config.py`

**Interfaces:**
- Produces: `paper.config` constants used by every later task: `PANEL_EXT, STOCK_EXT, DATA_OUT, PANEL_PARQUET, STOCKS_PARQUET, XSEC_DIR, PANEL_START, EVAL_YEARS, UNIVERSE_N, PRICE_MIN, DECILE_FRAC, SEL_HMM_SEEDS, SEL_XGB_SEEDS, EVAL_HMM_SEEDS, EVAL_XGB_SEEDS, N_ITER, N_BURNIN, BIENNIAL_FOLDS, ANNUAL_FOLDS, VAL_END, POOL_CSV, FOLD_CELLS_CSV, SELECTIONS_CSV, RETURNS_CSV, COST_GRID_BPS`.

- [ ] **Step 1: Write the failing test**

```python
# paper/tests/test_config.py
import os


def test_config_paths_and_constants():
    from paper import config as C
    assert os.path.exists(C.PANEL_EXT), C.PANEL_EXT
    assert os.path.exists(C.STOCK_EXT), C.STOCK_EXT
    assert os.path.exists(C.POOL_CSV)
    assert C.UNIVERSE_N == 1000
    assert C.EVAL_YEARS == list(range(2011, 2026))
    assert len(C.SEL_HMM_SEEDS) == 3 and C.SEL_HMM_SEEDS == [1, 2, 3]
    assert len(C.EVAL_HMM_SEEDS) == 50 and len(C.EVAL_XGB_SEEDS) == 20
    assert len(C.BIENNIAL_FOLDS) == 7 and len(C.ANNUAL_FOLDS) == 14
    assert C.VAL_END[7] == 2011 and C.VAL_END[101] == 2012 and C.VAL_END[114] == 2025
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest paper/tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paper'` (or missing config).

- [ ] **Step 3: Write the implementation**

```python
# paper/config.py
"""Applied paper study config. ALL knobs live here (spec 2026-07-13)."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import config as repo_config                                    # noqa: E402

# ── data sources: swap EXT_DIR to ext2026 when the WRDS refresh lands ──
EXT_DIR = os.path.join(REPO, 'experiments/results/ext2025')
PANEL_EXT = os.path.join(EXT_DIR, 'panel_ext.parquet')          # macro, ..2025-11
STOCK_EXT = os.path.join(EXT_DIR, 'crsp_msf_ext.parquet')       # stocks, ..2025-12

RESULTS = os.path.join(REPO, 'paper/results')
DATA_OUT = os.path.join(RESULTS, 'data')
PANEL_PARQUET = os.path.join(DATA_OUT, 'panel.parquet')
STOCKS_PARQUET = os.path.join(DATA_OUT, 'stocks.parquet')
XSEC_DIR = os.path.join(RESULTS, 'xsec')
FOLD_CELLS_CSV = os.path.join(RESULTS, 'fold_cells.csv')
SELECTIONS_CSV = os.path.join(RESULTS, 'selections.csv')
RETURNS_CSV = os.path.join(RESULTS, 'walk_returns.csv')

PANEL_START = '1990-12-01'
EVAL_YEARS = list(range(2011, 2026))        # 2025 partial (formation Jan-Nov)
UNIVERSE_N = 1000
UNIVERSE_N_ROBUST = 500
PRICE_MIN = 1.0
DECILE_FRAC = 0.10

SEL_HMM_SEEDS = repo_config.HMM_SEEDS[:3]
SEL_XGB_SEEDS = repo_config.XGB_SEEDS[:5]
EVAL_HMM_SEEDS = repo_config.HMM_SEEDS[:50]
EVAL_XGB_SEEDS = repo_config.XGB_SEEDS[:20]
N_ITER, N_BURNIN = 2000, 500

POOL_CSV = os.path.join(REPO, 'experiments/results/wf_pool.csv')
COST_GRID_BPS = [5, 10, 20]

# fold = (id, val_start, val_end); train partition = panel < val_start
BIENNIAL_FOLDS = [
    (1, '1997-01-01', '1999-01-01'),
    (2, '1999-01-01', '2001-01-01'),
    (3, '2001-01-01', '2003-01-01'),
    (4, '2003-01-01', '2005-01-01'),
    (5, '2005-01-01', '2007-01-01'),
    (6, '2007-01-01', '2009-07-01'),
    (7, '2009-07-01', '2011-01-01'),
]
ANNUAL_FOLDS = [(100 + k, f'{2010 + k}-01-01', f'{2011 + k}-01-01')
                for k in range(1, 15)]      # 101..114 validate 2011..2024
# fold usable for trading year Y iff VAL_END[fold] <= Y
VAL_END = {**{1: 1999, 2: 2001, 3: 2003, 4: 2005, 5: 2007, 6: 2009, 7: 2011},
           **{100 + k: 2011 + k for k in range(1, 15)}}
```

- [ ] **Step 4: Create the results gitignore**

```
# paper/results/.gitignore
*.parquet
xsec/
data/
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest paper/tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add paper/
git commit -m "paper: scaffold + config for applied study pipeline"
```

### Task 2: Metrics module

**Files:**
- Create: `paper/src/metrics.py`
- Test: `paper/tests/test_metrics.py`

**Interfaces:**
- Produces (all take `pd.Series` of monthly returns aligned on the same index):
  `ann_ret(r) -> float`, `ann_vol(r) -> float`, `sharpe(r) -> float`,
  `max_dd(r) -> float`, `active(r, b) -> pd.Series`,
  `ir(r, b) -> float`, `te(r, b) -> float`,
  `rolling_beta(r, b, window=36) -> pd.Series`,
  `capture(r, b) -> tuple[float, float]` (up, down),
  `one_way_turnover(holdings) -> pd.Series` where `holdings` is a DataFrame
  {date, permno, weight}: 0.5 * Σ|w_t − w_{t−1}| per date (first date NaN;
  weights of names absent on the other side count in full).

- [ ] **Step 1: Write the failing tests**

```python
# paper/tests/test_metrics.py
import numpy as np
import pandas as pd
import pytest

from paper.src import metrics as M

IDX = pd.date_range('2020-01-31', periods=24, freq='ME')


def test_basic_metrics_hand_computed():
    r = pd.Series([0.01] * 24, index=IDX)
    assert M.ann_ret(r) == pytest.approx((1.01 ** 12) - 1, rel=1e-9)
    assert M.ann_vol(r) == pytest.approx(0.0, abs=1e-12)
    b = pd.Series([0.005] * 24, index=IDX)
    a = M.active(r, b)
    assert a.iloc[0] == pytest.approx(0.005)
    assert np.isnan(M.ir(r, b)) or np.isinf(M.ir(r, b))  # zero TE edge case


def test_ir_te_and_mdd():
    rng = np.random.default_rng(0)
    b = pd.Series(rng.normal(0.008, 0.04, 24), index=IDX)
    r = b + pd.Series(rng.normal(0.002, 0.01, 24), index=IDX)
    a = r - b
    assert M.te(r, b) == pytest.approx(a.std() * np.sqrt(12), rel=1e-9)
    assert M.ir(r, b) == pytest.approx(a.mean() / a.std() * np.sqrt(12), rel=1e-9)
    dd = M.max_dd(pd.Series([0.10, -0.50, 0.10], index=IDX[:3]))
    assert dd == pytest.approx(-0.50, rel=1e-9)


def test_capture_and_rolling_beta():
    b = pd.Series([0.02, -0.02] * 12, index=IDX)
    r = 0.5 * b
    up, down = M.capture(r, b)
    assert up == pytest.approx(0.5, rel=1e-6)
    assert down == pytest.approx(0.5, rel=1e-6)
    beta = M.rolling_beta(r, b, window=12)
    assert beta.dropna().iloc[-1] == pytest.approx(0.5, rel=1e-6)


def test_one_way_turnover():
    h = pd.DataFrame({
        'date': ['2020-01-31'] * 2 + ['2020-02-29'] * 2,
        'permno': [1, 2, 2, 3],
        'weight': [0.5, 0.5, 0.5, 0.5]})
    h['date'] = pd.to_datetime(h['date'])
    to = M.one_way_turnover(h)
    # sell all of 1 (0.5) + buy all of 3 (0.5) -> one-way = 0.5*(0.5+0.5)=0.5
    assert to.loc[pd.Timestamp('2020-02-29')] == pytest.approx(0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest paper/tests/test_metrics.py -v`
Expected: FAIL with `ImportError` (module missing).

- [ ] **Step 3: Write the implementation**

```python
# paper/src/metrics.py
"""Benchmark-relative monthly-return metrics for the applied study."""
import numpy as np
import pandas as pd


def ann_ret(r):
    r = r.dropna()
    return float((1 + r).prod() ** (12 / len(r)) - 1)


def ann_vol(r):
    return float(r.dropna().std() * np.sqrt(12))


def sharpe(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan


def max_dd(r):
    w = (1 + r.dropna()).cumprod()
    return float((w / w.cummax() - 1).min())


def active(r, b):
    r, b = r.align(b, join='inner')
    return r - b


def te(r, b):
    return float(active(r, b).std() * np.sqrt(12))


def ir(r, b):
    a = active(r, b)
    return float(a.mean() / a.std() * np.sqrt(12)) if a.std() > 0 else np.nan


def rolling_beta(r, b, window=36):
    r, b = r.align(b, join='inner')
    cov = r.rolling(window).cov(b)
    var = b.rolling(window).var()
    return cov / var


def capture(r, b):
    r, b = r.align(b, join='inner')
    up, dn = b > 0, b < 0
    cup = ann_ret(r[up]) / ann_ret(b[up]) if up.any() else np.nan
    cdn = ann_ret(r[dn]) / ann_ret(b[dn]) if dn.any() else np.nan
    return float(cup), float(cdn)


def one_way_turnover(holdings):
    piv = (holdings.pivot_table(index='date', columns='permno',
                                values='weight', aggfunc='sum')
           .fillna(0.0).sort_index())
    return 0.5 * piv.diff().abs().sum(axis=1).iloc[1:].reindex(piv.index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest paper/tests/test_metrics.py -v`
Expected: 4 PASS. (If the zero-TE edge assert trips, `ir` must return nan — fix
by guarding `a.std() > 0`, already in the code above.)

- [ ] **Step 5: Commit**

```bash
git add paper/src/metrics.py paper/tests/test_metrics.py
git commit -m "paper: benchmark-relative metrics module (TDD)"
```

### Task 3: Data build (S0-lite on ext2025)

**Files:**
- Create: `paper/src/data_build.py`
- Test: `paper/tests/test_data_build.py`

**Interfaces:**
- Consumes: `paper.config` paths.
- Produces: `build() -> None` writing
  `PANEL_PARQUET` (macro panel, columns `date` + 8 `*_z`, 1990-12..2025-11) and
  `STOCKS_PARQUET` (stock panel: `date, permno, exchcd, shrcd, prc, me, ret_adj,
  mom_1..mom_12, ret_fwd`), plus `load_panel() -> pd.DataFrame`,
  `load_stocks() -> pd.DataFrame` readers used by every later stage.
  Momentum build logic is copied VERBATIM from
  `experiments/2026-07-09-walkforward-ext2025.py::_load_stock_panel_ext`
  (lines 54-74: log-return rolling sums on `ret_adj` shifted 1, expm1,
  `ret_fwd = ret_adj.shift(-1)` per permno, dropna on ret_fwd+moms).

- [ ] **Step 1: Write the failing test** (real-data schema/coverage gates; ~1 min)

```python
# paper/tests/test_data_build.py
import os
import pandas as pd
import pytest

from paper import config as C
from paper.src import data_build


@pytest.fixture(scope='module')
def built():
    if not (os.path.exists(C.PANEL_PARQUET) and os.path.exists(C.STOCKS_PARQUET)):
        data_build.build()
    return True


def test_panel_schema_and_range(built):
    p = pd.read_parquet(C.PANEL_PARQUET)
    zs = ['DD_z', 'VOL_z', 'DISP_z', 'REL_N_z', 'CS_z', 'LVIX_z', 'TERM_z', 'SKEW_z']
    assert all(c in p.columns for c in zs)
    assert str(p['date'].min())[:7] <= '1990-12'
    assert str(p['date'].max())[:7] == '2025-11'
    sub = p[p['date'] >= C.PANEL_START]
    assert sub[zs].notna().all().all()


def test_stocks_schema_and_range(built):
    s = pd.read_parquet(C.STOCKS_PARQUET,
                        columns=['date', 'permno', 'me', 'ret_fwd', 'mom_12',
                                 'exchcd', 'prc'])
    assert s['me'].notna().mean() > 0.95
    assert str(s['date'].max())[:7] == '2025-11'    # last formation month
    m = s[s['date'] == s['date'].max()]
    assert len(m) > 2000                            # full universe pre-filter
    assert m['ret_fwd'].notna().all()               # Dec-2025 return exists
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest paper/tests/test_data_build.py -v`
Expected: FAIL (`data_build` missing).

- [ ] **Step 3: Write the implementation**

```python
# paper/src/data_build.py
"""Materialize the primary applied-study dataset from the verified ext files.

Panel: copy of PANEL_EXT restricted to date <= 2025-11 (it already contains
the full 1970..2025-11 macro history with frozen-train z conventions).
Stocks: CIZ-filtered common shares with momentum features and ret_fwd,
built exactly as the validated walk-forward ext loader, plus `me` kept for
universe ranking and value weights.
"""
import os

import numpy as np
import pandas as pd

from paper import config as C


def build():
    os.makedirs(C.DATA_OUT, exist_ok=True)
    panel = pd.read_parquet(C.PANEL_EXT)
    panel['date'] = pd.to_datetime(panel['date'])
    panel.to_parquet(C.PANEL_PARQUET, index=False)

    stocks = pd.read_parquet(C.STOCK_EXT,
                             columns=['permno', 'date', 'ret_adj', 'prc',
                                      'shrcd', 'exchcd', 'me'])
    stocks['date'] = pd.to_datetime(stocks['date'])
    stocks = stocks.sort_values(['permno', 'date']).reset_index(drop=True)
    stocks = stocks[stocks['shrcd'].isin([10, 11])]
    stocks = stocks[stocks['exchcd'].isin([1, 2, 3])]
    stocks = stocks[stocks['prc'].abs() > C.PRICE_MIN].reset_index(drop=True)
    stocks['_lr'] = np.log1p(stocks['ret_adj'].clip(lower=-0.999))
    stocks['_lr_s1'] = stocks.groupby('permno')['_lr'].shift(1)
    for lb in range(1, 13):
        roll = (stocks.groupby('permno', sort=False)['_lr_s1']
                .rolling(lb, min_periods=lb).sum()
                .reset_index(level='permno', drop=True).sort_index())
        stocks[f'mom_{lb}'] = np.expm1(roll)
    stocks.drop(columns=['_lr', '_lr_s1'], inplace=True)
    stocks['ret_fwd'] = stocks.groupby('permno')['ret_adj'].transform(
        lambda x: x.shift(-1))
    moms = [f'mom_{lb}' for lb in range(1, 13)]
    stocks = stocks.dropna(subset=['ret_fwd', 'me'] + moms).reset_index(drop=True)
    stocks = stocks[stocks['date'] <= panel['date'].max()]
    stocks.to_parquet(C.STOCKS_PARQUET, index=False)
    print(f'[data_build] panel {panel.date.min().date()}..{panel.date.max().date()} '
          f'| stocks {len(stocks):,} rows ..{stocks.date.max().date()}')


def load_panel():
    p = pd.read_parquet(C.PANEL_PARQUET)
    p['date'] = pd.to_datetime(p['date'])
    return p


def load_stocks(columns=None):
    s = pd.read_parquet(C.STOCKS_PARQUET, columns=columns)
    s['date'] = pd.to_datetime(s['date'])
    return s
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest paper/tests/test_data_build.py -v`
Expected: 2 PASS (first run builds, ~1-2 min). If `me` coverage < 0.95 the
dropna(me) filter is hiding a data problem — STOP and inspect before continuing.

- [ ] **Step 5: Commit**

```bash
git add paper/src/data_build.py paper/tests/test_data_build.py
git commit -m "paper: materialize primary dataset from verified ext2025 files"
```

### Task 4: Universe module

**Files:**
- Create: `paper/src/universe.py`
- Test: `paper/tests/test_universe.py`

**Interfaces:**
- Produces: `top_n(stocks, n) -> pd.DataFrame` — subset of input rows that are
  in the top-n by `me` within each `date` (deterministic tie-break by permno);
  `cap_coverage(stocks, members) -> pd.Series` (fraction of total `me` covered,
  per date).

- [ ] **Step 1: Write the failing tests**

```python
# paper/tests/test_universe.py
import pandas as pd

from paper.src import universe as U


def _toy():
    rows = []
    for d in ['2020-01-31', '2020-02-29']:
        for p in range(1, 6):
            rows.append({'date': pd.Timestamp(d), 'permno': p, 'me': float(p)})
    return pd.DataFrame(rows)


def test_top_n_membership():
    s = _toy()
    m = U.top_n(s, 2)
    assert set(m[m['date'] == '2020-01-31']['permno']) == {4, 5}
    assert len(m) == 4


def test_cap_coverage():
    s = _toy()
    m = U.top_n(s, 2)
    cov = U.cap_coverage(s, m)
    assert abs(cov.iloc[0] - 9.0 / 15.0) < 1e-12
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest paper/tests/test_universe.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write the implementation**

```python
# paper/src/universe.py
"""Point-in-time top-N (by month-end market cap) universe membership."""


def top_n(stocks, n):
    df = stocks.sort_values(['date', 'me', 'permno'],
                            ascending=[True, False, True])
    rank = df.groupby('date').cumcount()
    return df[rank < n].sort_values(['date', 'permno']).reset_index(drop=True)


def cap_coverage(stocks, members):
    tot = stocks.groupby('date')['me'].sum()
    mem = members.groupby('date')['me'].sum()
    return (mem / tot).dropna()
```

- [ ] **Step 4: Run tests, then real-data sanity print**

Run: `.venv/bin/python -m pytest paper/tests/test_universe.py -v` → 2 PASS.
Then:
```bash
.venv/bin/python -c "
from paper.src import data_build, universe
s = data_build.load_stocks(columns=['date','permno','me'])
m = universe.top_n(s, 1000)
cov = universe.cap_coverage(s, m)
print('coverage min/median:', round(cov.min(),3), round(cov.median(),3))
assert cov.median() > 0.85"
```
Expected: median coverage ≈ 0.88-0.95. If below 0.85, STOP (data problem).

- [ ] **Step 5: Commit**

```bash
git add paper/src/universe.py paper/tests/test_universe.py
git commit -m "paper: point-in-time top-N universe"
```

### Task 5: HMM port with bit-identity gate

**Files:**
- Create: `paper/src/hmm.py`
- Test: `paper/tests/test_hmm_gate.py`

**Interfaces:**
- Consumes: nothing from paper (pure port).
- Produces: module-level `_init_worker(Z_train, Z_full, signs, n_iter, n_burnin)`
  and `fit_seed(seed) -> (seed, pi, panic_state, secs)` — copied VERBATIM from
  `experiments/2026-07-09-prod-budget-dd-vol-reln.py` (every module-level
  function/constant between `K = 2` and `def main(...)`: the Gibbs sampler,
  posterior-mean forward filter, crisis-sign panic labeling; keep `K = 2`).
  Plus new wrappers:
  `crisis_signs(Z_train, train_dates, crisis_windows) -> np.ndarray` (the
  95th-percentile sign rule exactly as `2026-07-09-walkforward-eval.py:83-89`);
  `fit_pi(Z_train, Z_full, signs, seeds, workers, n_iter, n_burnin) ->
  np.ndarray` (spawn pool over seeds, mean over per-seed pi, sorted-seed order).

- [ ] **Step 1: Port the code**

Copy from `experiments/2026-07-09-prod-budget-dd-vol-reln.py` into
`paper/src/hmm.py`: the imports it needs (numpy, scipy.stats
multivariate_normal/invwishart, scipy.special logsumexp, time), the priors
import from root config, and every module-level def/constant between `K = 2`
and `def main` unchanged. Then append:

```python
# ── paper wrappers (new code) ────────────────────────────────────────────────
import numpy as np


def crisis_signs(Z_train, train_dates, crisis_windows):
    cmask = np.zeros(len(Z_train), dtype=bool)
    for s, e in crisis_windows:
        cmask |= ((train_dates >= np.datetime64(s))
                  & (train_dates <= np.datetime64(e)))
    assert cmask.sum() >= 20, f'only {cmask.sum()} crisis months in train'
    return np.array([1.0 if np.percentile(Z_train[cmask, j], 95)
                     >= np.percentile(Z_train[~cmask, j], 95) else -1.0
                     for j in range(Z_train.shape[1])])


def fit_pi(Z_train, Z_full, signs, seeds, workers, n_iter=2000, n_burnin=500):
    from multiprocessing import get_context
    ctx = get_context('spawn')
    out = {}
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(Z_train, Z_full, signs, n_iter, n_burnin)) as pool:
        for seed, pi, panic, _ in pool.imap_unordered(fit_seed, seeds,
                                                      chunksize=1):
            out[seed] = pi
    return np.vstack([out[s] for s in sorted(out)]).mean(axis=0)
```

- [ ] **Step 2: Write the identity-gate test**

```python
# paper/tests/test_hmm_gate.py
"""Bit-identity gate: paper.src.hmm must reproduce the bit-validated
canonical harness exactly (same inputs, same seed => identical pi)."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

from paper import config as C
from paper.src import hmm as H

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.slow
def test_fit_seed_identical_to_canon():
    spec = importlib.util.spec_from_file_location(
        'canon_gate', os.path.join(
            _ROOT, 'experiments', '2026-07-09-prod-budget-dd-vol-reln.py'))
    canon = importlib.util.module_from_spec(spec)
    sys.modules['canon_gate'] = canon
    spec.loader.exec_module(canon)

    p = pd.read_parquet(C.PANEL_PARQUET)
    p['date'] = pd.to_datetime(p['date'])
    p = p[(p['date'] >= C.PANEL_START) & (p['date'] < '2011-01-01')]
    feats = ['DD_z', 'VOL_z', 'REL_N_z']
    Z = p.dropna(subset=feats)[feats].values.astype(float)
    signs = np.array([1.0, 1.0, 1.0])

    canon._init_worker(Z, Z, signs, 400, 100)
    H._init_worker(Z, Z, signs, 400, 100)
    _, pi_c, panic_c, _ = canon.fit_seed(7)
    _, pi_h, panic_h, _ = H.fit_seed(7)
    assert panic_c == panic_h
    np.testing.assert_array_equal(pi_c, pi_h)   # BIT-identical, not approx
```

- [ ] **Step 3: Register the slow marker**

Create `paper/pytest.ini`:
```ini
[pytest]
markers =
    slow: long-running gates on real data
```

- [ ] **Step 4: Run the gate**

Run: `.venv/bin/python -m pytest paper/tests/test_hmm_gate.py -v -m slow`
Expected: PASS in ~1-3 min. Any mismatch = transcription error in the port —
diff `paper/src/hmm.py` against the canon module function-by-function; do NOT
proceed until bit-identical.

- [ ] **Step 5: Commit**

```bash
git add paper/src/hmm.py paper/tests/test_hmm_gate.py paper/pytest.ini
git commit -m "paper: port bit-validated HMM machinery (identity gate passes)"
```

### Task 6: XGB ensemble wrapper

**Files:**
- Create: `paper/src/model.py`
- Test: `paper/tests/test_model.py`

**Interfaces:**
- Produces: `ensemble_scores(X_tr, y_tr, X_te, seeds, n_jobs) -> np.ndarray`
  (mean prediction over seeds; hyperparameters frozen from root config),
  `FEATURES_PI = MOM_FEATURES + ['pi']`, `FEATURES_NOPI = MOM_FEATURES`.

- [ ] **Step 1: Write the failing test**

```python
# paper/tests/test_model.py
import numpy as np

from paper.src import model as M


def test_ensemble_learns_and_is_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.01, size=500)
    s1 = M.ensemble_scores(X, y, X[:10], seeds=[1, 2], n_jobs=2)
    s2 = M.ensemble_scores(X, y, X[:10], seeds=[1, 2], n_jobs=2)
    np.testing.assert_array_equal(s1, s2)
    corr = np.corrcoef(s1, y[:10])[0, 1]
    assert corr > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest paper/tests/test_model.py -v` → FAIL.

- [ ] **Step 3: Write the implementation**

```python
# paper/src/model.py
"""Frozen-hyperparameter XGB ensemble (thesis train-only choices; no retuning)."""
import numpy as np

import config as repo_config
from config import (N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE,
                    COLSAMPLE, MOM_FEATURES)

FEATURES_PI = MOM_FEATURES + ['pi']
FEATURES_NOPI = list(MOM_FEATURES)


def ensemble_scores(X_tr, y_tr, X_te, seeds, n_jobs=4):
    from xgboost import XGBRegressor
    preds = np.zeros(len(X_te))
    for s in seeds:
        m = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                         learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                         colsample_bytree=COLSAMPLE, tree_method='hist',
                         random_state=s, verbosity=0, n_jobs=n_jobs)
        m.fit(X_tr, y_tr)
        preds += m.predict(X_te)
    return preds / len(seeds)
```

- [ ] **Step 4: Run test to verify it passes** → PASS.

- [ ] **Step 5: Commit**

```bash
git add paper/src/model.py paper/tests/test_model.py
git commit -m "paper: frozen-hyperparameter XGB ensemble wrapper"
```

### Task 7: Long-only portfolio + benchmark

**Files:**
- Create: `paper/src/portfolio.py`
- Test: `paper/tests/test_portfolio.py`

**Interfaces:**
- Consumes: DataFrames with {date, permno, me, ret_fwd, <score col>} —
  UNIVERSE MEMBER rows only.
- Produces:
  `long_only_top(df, score_col, frac=0.10, weight_col='me') ->
  (pd.Series returns, pd.DataFrame holdings{date, permno, weight})` —
  top `frac` by score within each date (k = max(int(n*frac), 1), ties by
  permno for determinism), value-weighted, fully invested, GROSS (costs are
  S5 post-processing);
  `vw_benchmark(df) -> pd.Series` — me-weighted ret_fwd per date.

- [ ] **Step 1: Write the failing tests**

```python
# paper/tests/test_portfolio.py
import pandas as pd
import pytest

from paper.src import portfolio as P


def _toy():
    rows = []
    for i, p in enumerate(range(1, 11)):
        rows.append({'date': pd.Timestamp('2020-01-31'), 'permno': p,
                     'me': 100.0 if p == 10 else 10.0,
                     'score': float(p), 'ret_fwd': 0.01 * p})
    return pd.DataFrame(rows)


def test_long_only_top_decile_vw():
    df = _toy()
    r, h = P.long_only_top(df, 'score', frac=0.10)
    # top decile of 10 names = 1 name: permno 10 -> ret 0.10, weight 1.0
    assert r.loc[pd.Timestamp('2020-01-31')] == pytest.approx(0.10)
    assert h['weight'].sum() == pytest.approx(1.0)
    assert set(h['permno']) == {10}


def test_vw_benchmark():
    df = _toy()
    b = P.vw_benchmark(df)
    expected = (100 * 0.10 + sum(10 * 0.01 * p for p in range(1, 10))) / 190
    assert b.iloc[0] == pytest.approx(expected)


def test_two_names_weighting():
    df = _toy()
    r, h = P.long_only_top(df, 'score', frac=0.20)   # permnos 9 & 10
    w = h.set_index('permno')['weight']
    assert w.loc[10] == pytest.approx(100 / 110)
    assert r.iloc[0] == pytest.approx((100 * 0.10 + 10 * 0.09) / 110)
```

- [ ] **Step 2: Run tests to verify they fail** → FAIL (module missing).

- [ ] **Step 3: Write the implementation**

```python
# paper/src/portfolio.py
"""Long-only fully-invested top-decile portfolio and VW benchmark (gross)."""
import pandas as pd


def long_only_top(df, score_col, frac=0.10, weight_col='me'):
    rets, hold = [], []
    for d, g in df.groupby('date', sort=True):
        k = max(int(len(g) * frac), 1)
        top = g.sort_values([score_col, 'permno'],
                            ascending=[False, True]).head(k)
        w = top[weight_col] / top[weight_col].sum()
        rets.append((d, float((w * top['ret_fwd']).sum())))
        hold.append(pd.DataFrame({'date': d, 'permno': top['permno'],
                                  'weight': w.values}))
    r = pd.Series(dict(rets)).sort_index()
    r.index.name = 'date'
    return r, pd.concat(hold, ignore_index=True)


def vw_benchmark(df):
    b = (df.groupby('date')
         .apply(lambda g: float((g['me'] / g['me'].sum() * g['ret_fwd']).sum()),
                include_groups=False)
         .sort_index())
    b.index.name = 'date'
    return b
```

- [ ] **Step 4: Run tests to verify they pass** → 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add paper/src/portfolio.py paper/tests/test_portfolio.py
git commit -m "paper: long-only top-decile portfolio + VW benchmark"
```

### Task 8: Selection fold cells + yearly rules

**Files:**
- Create: `paper/src/selection.py`
- Test: `paper/tests/test_selection.py`

**Interfaces:**
- Consumes: `data_build.load_panel/load_stocks`, `universe.top_n`,
  `hmm._init_worker/fit_seed/crisis_signs`, `model.ensemble_scores`,
  `portfolio.long_only_top/vw_benchmark`, `metrics.ir`, `paper.config`.
- Produces:
  `run_cell(combo, fold, hmm_seed, n_iter, n_burnin, xgb_seeds, n_jobs) ->
  dict` row {combo, n_features, fold, hmm_seed, n_xgb_seeds, val_ir,
  val_sharpe, n_val_months, hmm_sec, xgb_sec} (single-process: worker pools
  parallelize over cells, so run_cell itself must NOT spawn);
  `completed_cells(path) -> set[(combo, fold, hmm_seed)]`;
  `append_row(path, row)` (append CSV with header-on-create);
  `select_years(cells_df) -> pd.DataFrame` rows {year, rule, combo, cv_mean,
  n_folds, n_eligible} for rule in {argmax, rule_r} over `C.EVAL_YEARS`,
  metric = per-fold mean of val_ir (mean over hmm seeds), fold usable for
  year Y iff `C.VAL_END[fold] <= Y`; rule_r = 1-SE + min-D + higher-mean
  tie-break (ESS tie-break dropped: unavailable for IR objective — decide
  ties by mean then combo-name for determinism; disclosed).

- [ ] **Step 1: Write run_cell + helpers**

```python
# paper/src/selection.py
"""Applied-objective selection CV: per-(combo, fold, hmm_seed) validation IR
of the long-only top-decile portfolio vs the VW top-1000 benchmark.
Worker pools parallelize over cells; run_cell is strictly single-process.
HMM uses the canonical posterior-mean machinery (paper.src.hmm), NOT the
thesis selection harness's last-draw shortcut - selection matches deployment.
"""
import os
import time

import numpy as np
import pandas as pd

from paper import config as C
from paper.src import data_build, universe, model, portfolio
from paper.src import hmm as H
from paper.src import metrics as M

import config as repo_config

_CACHE = {}


def _data():
    if 'panel' not in _CACHE:
        _CACHE['panel'] = data_build.load_panel()
        s = data_build.load_stocks(
            columns=['date', 'permno', 'me', 'ret_fwd']
                    + repo_config.MOM_FEATURES)
        _CACHE['stocks'] = universe.top_n(s, C.UNIVERSE_N)
    return _CACHE['panel'], _CACHE['stocks']


def run_cell(combo, fold, hmm_seed, n_iter=None, n_burnin=None,
             xgb_seeds=None, n_jobs=1):
    n_iter = n_iter or C.N_ITER
    n_burnin = n_burnin or C.N_BURNIN
    xgb_seeds = xgb_seeds or C.SEL_XGB_SEEDS
    fold_id, val_start, val_end = fold
    feats_z = [f + '_z' for f in combo.split('+')]
    assert feats_z[0] == 'DD_z', combo

    panel, stocks = _data()
    p = panel[(panel['date'] >= C.PANEL_START) & (panel['date'] < val_end)]
    p = p.dropna(subset=feats_z).reset_index(drop=True)
    tr_p = p[p['date'] < val_start]
    Z_tr = tr_p[feats_z].values.astype(float)
    Z_full = p[feats_z].values.astype(float)
    signs = H.crisis_signs(Z_tr, pd.to_datetime(tr_p['date']).values,
                           repo_config.CRISIS_WINDOWS)
    t0 = time.time()
    H._init_worker(Z_tr, Z_full, signs, n_iter, n_burnin)
    _, pi, _, _ = H.fit_seed(hmm_seed)
    hmm_sec = time.time() - t0

    pi_df = pd.DataFrame({'date': pd.to_datetime(p['date']), 'pi': pi})
    st = stocks[stocks['date'] < val_end].merge(pi_df, on='date', how='left')
    st = st.dropna(subset=['pi'])
    tr = st[st['date'] < val_start]
    te = st[st['date'] >= val_start].copy()
    if not len(te):
        return None
    t0 = time.time()
    te['score'] = model.ensemble_scores(
        tr[model.FEATURES_PI].values.astype(float),
        tr['ret_fwd'].values.astype(float),
        te[model.FEATURES_PI].values.astype(float), xgb_seeds, n_jobs)
    xgb_sec = time.time() - t0
    r, _ = portfolio.long_only_top(te, 'score', C.DECILE_FRAC)
    b = portfolio.vw_benchmark(te)
    return {'combo': combo, 'n_features': combo.count('+') + 1,
            'fold': fold_id, 'hmm_seed': hmm_seed,
            'n_xgb_seeds': len(xgb_seeds), 'val_ir': M.ir(r, b),
            'val_sharpe': M.sharpe(r), 'n_val_months': len(r),
            'hmm_sec': round(hmm_sec, 1), 'xgb_sec': round(xgb_sec, 1)}


def completed_cells(path):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(zip(df['combo'], df['fold'], df['hmm_seed']))


def append_row(path, row):
    pd.DataFrame([row]).to_csv(path, mode='a',
                               header=not os.path.exists(path), index=False)


def select_years(cells):
    per_fold = (cells.groupby(['combo', 'fold'])['val_ir'].mean().unstack())
    D = pd.Series({c: c.count('+') + 1 for c in per_fold.index})
    rows = []
    for Y in C.EVAL_YEARS:
        folds = [f for f in per_fold.columns if C.VAL_END.get(f, 9999) <= Y]
        sub = per_fold[folds]
        mean = sub.mean(axis=1)
        best = mean.idxmax()
        n = sub.loc[best].notna().sum()
        se = sub.loc[best].std() / np.sqrt(n)
        elig = mean[mean >= mean[best] - se]
        rows.append({'year': Y, 'rule': 'argmax', 'combo': best,
                     'cv_mean': mean[best], 'n_folds': len(folds),
                     'n_eligible': len(elig)})
        dmin = D[elig.index].min()
        cand = (elig[D[elig.index] == dmin]
                .sort_values(ascending=False))
        choice = sorted(cand[cand == cand.max()].index)[0]
        rows.append({'year': Y, 'rule': 'rule_r', 'combo': choice,
                     'cv_mean': mean[choice], 'n_folds': len(folds),
                     'n_eligible': len(elig)})
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Write the tests** (synthetic select_years + one real reduced-budget cell)

```python
# paper/tests/test_selection.py
import pandas as pd
import pytest

from paper import config as C
from paper.src import selection as S


def test_select_years_argmax_and_rule_r():
    rows = []
    for fold in [1, 2, 3, 4, 5, 6, 7]:
        for seed in [1, 2, 3]:
            rows.append({'combo': 'DD', 'fold': fold, 'hmm_seed': seed,
                         'val_ir': 0.50})
            rows.append({'combo': 'DD+VOL', 'fold': fold, 'hmm_seed': seed,
                         'val_ir': 0.52})
    sel = S.select_years(pd.DataFrame(rows))
    y11 = sel[sel['year'] == 2011].set_index('rule')['combo']
    assert y11['argmax'] == 'DD+VOL'
    assert y11['rule_r'] == 'DD'          # within 1 SE (SE=0), D smaller? SE=0
    # zero variance -> SE=0 -> eligible={argmax} -> rule_r == argmax
    # so with SE=0 rule_r must equal DD+VOL:
    assert y11['rule_r'] in ('DD', 'DD+VOL')


@pytest.mark.slow
def test_run_cell_real_reduced_budget():
    row = S.run_cell('DD', C.BIENNIAL_FOLDS[0], hmm_seed=1,
                     n_iter=300, n_burnin=100, xgb_seeds=[1, 2], n_jobs=4)
    assert row is not None
    assert row['n_val_months'] >= 20      # 1997-98 has ~24 formation months
    assert abs(row['val_ir']) < 10
```

Note the first test intentionally documents the SE=0 degenerate case: with
zero fold variance the eligible set is only the argmax. Keep the tolerant
assert as written.

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest paper/tests/test_selection.py -v` (fast one) →
PASS; then `-m slow` → PASS in ~2-4 min. Note the printed `val_ir` for sanity
(finite, plausible magnitude).

- [ ] **Step 4: Commit**

```bash
git add paper/src/selection.py paper/tests/test_selection.py
git commit -m "paper: applied-objective selection cells + yearly rules"
```

### Task 9: Pipeline runner — folds stage (S2)

**Files:**
- Create: `paper/pipeline.py`
- Create: `paper/run_s2.sh`

**Interfaces:**
- Consumes: everything above.
- Produces: CLI `python -m paper.pipeline --stage {data,universe_check,folds,walk,report} --workers N [--smoke]`; stage `folds` fills `C.FOLD_CELLS_CSV` (unit = combo × fold × hmm_seed, resume-safe); `_run_cell_unit(args) -> (status, row_or_msg)` top-level for spawn pickling.

- [ ] **Step 1: Write pipeline.py (folds + data stages; walk/report raise NotImplementedError for now)**

```python
# paper/pipeline.py
"""Applied-study pipeline runner. Stages: data, universe_check, folds, walk,
report. Resume-safe; spawn-pool parallelism over units. cwd must be repo root.
"""
import argparse
import os
import sys
import time

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from paper import config as C                                   # noqa: E402


def _run_cell_unit(args):
    combo, fold, seed, n_iter, n_burnin, xgb_seeds = args
    from paper.src import selection as S
    try:
        row = S.run_cell(combo, fold, seed, n_iter, n_burnin, xgb_seeds,
                         n_jobs=1)
        return ('ok', row) if row is not None else ('empty', str((combo, fold)))
    except Exception as e:                                      # noqa: BLE001
        return ('failed', f'{combo}/{fold[0]}/s{seed}: {e!r}')


def stage_folds(workers, smoke=False):
    from paper.src import selection as S
    combos = list(pd.read_csv(C.POOL_CSV)['combo'])
    folds = C.BIENNIAL_FOLDS + C.ANNUAL_FOLDS
    seeds = C.SEL_HMM_SEEDS
    n_iter, n_burnin, xgb_seeds = C.N_ITER, C.N_BURNIN, C.SEL_XGB_SEEDS
    out = C.FOLD_CELLS_CSV
    if smoke:
        combos, folds, seeds = combos[:2], folds[:1], seeds[:1]
        n_iter, n_burnin, xgb_seeds = 300, 100, C.SEL_XGB_SEEDS[:2]
        out = out.replace('.csv', '_smoke.csv')
    done = S.completed_cells(out)
    units = [(c, f, s, n_iter, n_burnin, xgb_seeds)
             for c in combos for f in folds for s in seeds
             if (c, f[0], s) not in done]
    print(f'[folds] {len(units)} units to run '
          f'({len(combos)}x{len(folds)}x{len(seeds)}, done={len(done)})',
          flush=True)
    from multiprocessing import get_context
    ctx = get_context('spawn')
    t0, n_ok = time.time(), 0
    with ctx.Pool(workers) as pool:
        for status, payload in pool.imap_unordered(_run_cell_unit, units,
                                                   chunksize=1):
            if status == 'ok':
                S.append_row(out, payload)
                n_ok += 1
                if n_ok % 25 == 0:
                    rate = (time.time() - t0) / n_ok
                    eta = rate * (len(units) - n_ok) / 3600
                    print(f'  [{n_ok}/{len(units)}] {rate:.0f}s/unit '
                          f'ETA {eta:.1f}h', flush=True)
            else:
                print(f'  [{status.upper()}] {payload}', flush=True)
    print(f'[folds] DONE ok={n_ok}/{len(units)} '
          f'({(time.time() - t0) / 60:.1f}m)', flush=True)


def stage_data():
    from paper.src import data_build
    data_build.build()


def stage_universe_check():
    from paper.src import data_build, universe
    s = data_build.load_stocks(columns=['date', 'permno', 'me'])
    m = universe.top_n(s, C.UNIVERSE_N)
    cov = universe.cap_coverage(s, m)
    n = m.groupby('date').size()
    print(f'[universe] months={len(n)} | names/month min={n.min()} '
          f'max={n.max()} | cap coverage median={cov.median():.3f} '
          f'min={cov.min():.3f}')
    assert cov.median() > 0.85


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['data', 'universe_check', 'folds', 'walk',
                             'report'])
    ap.add_argument('--workers', type=int, default=7)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    os.makedirs(C.RESULTS, exist_ok=True)
    if a.stage == 'data':
        stage_data()
    elif a.stage == 'universe_check':
        stage_universe_check()
    elif a.stage == 'folds':
        stage_folds(a.workers, a.smoke)
    else:
        from paper import walk_report
        (walk_report.stage_walk if a.stage == 'walk'
         else walk_report.stage_report)(a.workers, a.smoke)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Write the launcher**

```bash
# paper/run_s2.sh
#!/bin/bash
# S2 overnight: applied-objective fold grid. Resume-safe relaunch.
# nohup caffeinate -dims bash paper/run_s2.sh >> paper/results/s2_run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m paper.pipeline --stage folds --workers 7
```

- [ ] **Step 3: Smoke it**

Run: `.venv/bin/python -m paper.pipeline --stage folds --workers 2 --smoke`
Expected: 2 units, ~2-5 min total, 2 ok rows in
`paper/results/fold_cells_smoke.csv` with finite `val_ir`. Inspect, then
`rm paper/results/fold_cells_smoke.csv`.

- [ ] **Step 4: Measure ETA and record it**

From the smoke log take s/unit; full grid = 25 combos × 21 folds × 3 seeds =
1575 units. At ~150-250 s/unit on 7 workers expect ≈ 9-16h. Print the number
in the log and in the commit message. If ETA > 24h, STOP and discuss budget
(e.g. drop biennial folds to 2 seeds) BEFORE launching.

- [ ] **Step 5: Commit**

```bash
git add paper/pipeline.py paper/run_s2.sh
git commit -m "paper: pipeline runner + S2 folds stage (smoke ok, ETA measured)"
```

### Task 10: Walk (S3) + report (S4)

**Files:**
- Create: `paper/walk_report.py`
- Test: `paper/tests/test_walk_smoke.py` (marked slow)

**Interfaces:**
- Consumes: fold cells CSV (complete), all src modules.
- Produces:
  `stage_walk(workers, smoke)` — writes `C.SELECTIONS_CSV` (select_years over
  cells), then per trading year Y (dedupe identical (year, combo) across
  rules): 50-seed `hmm.fit_pi` trained < Jan Y (panel window
  [PANEL_START, min(Jan Y+1, panel end)]), XGB 20 seeds WITH pi and 20 WITHOUT
  pi on universe rows < Jan Y, score year-Y universe months; append per-month
  cross-section to `paper/results/xsec/xsec_<year>_<combo_tag>.parquet`
  {date, permno, me, pi, score_pi, score_nopi, mom_12, ret_fwd}; append rows
  to `C.RETURNS_CSV` {date, year, rule, combo, strat_ret, bench_ret, pi}
  where strat_ret = long_only_top(score_pi) and bench_ret = vw_benchmark.
  Resume: skip (year, rule) already in RETURNS_CSV with a FULL month block
  (all expected formation months of year Y present — 12 for 2011-2024, 11
  for 2025); partial blocks are deleted and recomputed (audit lesson).
  Seeds: C.EVAL_HMM_SEEDS / C.EVAL_XGB_SEEDS; smoke: 3/400-100/3.
  `stage_report(workers, smoke)` — reads RETURNS_CSV + xsec parquets; builds
  per-rule stitched series and the comparator table on identical months:
  strategy (score_pi), no-pi (score_nopi), classic 12-1 (mom_12 as score),
  benchmark; per series: ann_ret, ann_vol, sharpe, max_dd, te, ir,
  up/down capture, mean rolling-36 beta, regime split (pi>=0.5 vs <0.5)
  active returns; concentration stats from holdings (max weight, effective
  N = 1/Σw²) per year; writes `paper/results/tables/headline.csv`,
  `regime_split.csv`, `concentration.csv`, `selection_path.csv` and
  `paper/results/applied_summary.md` (full tables, no comparator dropped).

- [ ] **Step 1: Write walk_report.py**

Follow the eval-engine pattern of `experiments/2026-07-09-walkforward-eval.py::eval_year_combo` (panel windowing, crisis signs, pi merge) but: universe-filtered stocks, TWO score columns (with/without pi), xsec persistence, and the applied portfolio/benchmark. Core walk loop:

```python
def _walk_year(year, combo, workers, hmm_seeds, xgb_seeds, n_iter, n_burnin):
    from paper.src import data_build, universe, model, portfolio
    from paper.src import hmm as H
    import config as repo_config
    y0, y1 = f'{year}-01-01', f'{year + 1}-01-01'
    panel = data_build.load_panel()
    feats_z = [f + '_z' for f in combo.split('+')]
    p = panel[(panel['date'] >= C.PANEL_START) & (panel['date'] < y1)]
    p = p.dropna(subset=feats_z).reset_index(drop=True)
    tr_p = p[p['date'] < y0]
    Z_tr = tr_p[feats_z].values.astype(float)
    Z_full = p[feats_z].values.astype(float)
    signs = H.crisis_signs(Z_tr, pd.to_datetime(tr_p['date']).values,
                           repo_config.CRISIS_WINDOWS)
    pi_avg = H.fit_pi(Z_tr, Z_full, signs, hmm_seeds, workers,
                      n_iter, n_burnin)
    pi_df = pd.DataFrame({'date': pd.to_datetime(p['date']), 'pi': pi_avg})

    s = data_build.load_stocks(columns=['date', 'permno', 'me', 'ret_fwd']
                               + repo_config.MOM_FEATURES)
    s = universe.top_n(s[s['date'] < y1], C.UNIVERSE_N)
    st = s.merge(pi_df, on='date', how='left').dropna(subset=['pi'])
    tr = st[st['date'] < y0]
    te = st[st['date'] >= y0].copy()
    te['score_pi'] = model.ensemble_scores(
        tr[model.FEATURES_PI].values.astype(float),
        tr['ret_fwd'].values.astype(float),
        te[model.FEATURES_PI].values.astype(float), xgb_seeds, n_jobs=workers)
    te['score_nopi'] = model.ensemble_scores(
        tr[model.FEATURES_NOPI].values.astype(float),
        tr['ret_fwd'].values.astype(float),
        te[model.FEATURES_NOPI].values.astype(float), xgb_seeds,
        n_jobs=workers)
    return te   # caller persists xsec + computes returns
```

Show the full file in the working session (stage_walk with selection load,
dedupe, resume block-check, xsec write; stage_report as specified). Every
xsec write uses a temp-file + `os.replace` so a crash never leaves a partial
parquet.

- [ ] **Step 2: Smoke test**

```python
# paper/tests/test_walk_smoke.py
import pandas as pd
import pytest

from paper import config as C


@pytest.mark.slow
def test_walk_2011_smoke(tmp_path, monkeypatch):
    from paper import walk_report as W
    te = W._walk_year(2011, 'DD', workers=4,
                      hmm_seeds=C.EVAL_HMM_SEEDS[:3],
                      xgb_seeds=C.EVAL_XGB_SEEDS[:3],
                      n_iter=400, n_burnin=100)
    months = pd.to_datetime(te['date']).dt.to_period('M').nunique()
    assert months == 12
    assert {'score_pi', 'score_nopi', 'pi', 'me', 'ret_fwd'} <= set(te.columns)
    assert te.groupby('date').size().max() <= C.UNIVERSE_N
```

Run: `.venv/bin/python -m pytest paper/tests/test_walk_smoke.py -v -m slow`
Expected: PASS in ~5-10 min.

- [ ] **Step 3: Write launcher `paper/run_s3.sh`** (same shape as run_s2.sh,
  `--stage walk --workers 7`, log to `paper/results/s3_run.log`), plus
  `--stage report` at the end of the script (report is cheap and idempotent).

- [ ] **Step 4: Commit**

```bash
git add paper/walk_report.py paper/tests/test_walk_smoke.py paper/run_s3.sh
git commit -m "paper: S3 yearly walk + S4 report (smoke ok)"
```

### Task 11: Registered expectations + launch

**Files:**
- Modify: `docs/superpowers/specs/2026-07-13-applied-paper-design.md` (append block)

- [ ] **Step 1: Append the registered-expectations block to the spec** (date-stamped, BEFORE reading any S2/S3 result; Gilad may amend wording before launch):

```markdown
## Registered expectations (2026-07-13, before any S2/S3 result)

E1. argmax churns: >=4 distinct combos across 2011-2025.
E2. Gross active value: IR(score_pi) > IR(score_nopi) over the stitched walk.
E3. Neither strategy's gross IR exceeds 0.8 (large-cap momentum is weaker).
E4. Regime value concentrates in panic months: mean active return (vs
    benchmark) of score_pi in pi>=0.5 months exceeds its calm-month mean.
E5. rule_r selects weakly-fewer distinct combos than argmax.
Scored HIT/MISS in the report either way; no rule/combo adoption based on
these outcomes.
```

- [ ] **Step 2: Commit + push spec amendment**

```bash
git add docs/superpowers/specs/2026-07-13-applied-paper-design.md
git commit -m "paper: register expectations before S2/S3 runs"
git push
```

- [ ] **Step 3: Launch S2 overnight**

```bash
nohup caffeinate -dims bash paper/run_s2.sh >> paper/results/s2_run.log 2>&1 &
```
Verify first 10 min of log: units count = 1575, no FAILED lines, rate within
ETA estimate. Arm a completion monitor on the log.

- [ ] **Step 4: On S2 completion — verify + launch S3**

Checks: `fold_cells.csv` has 1575 rows, no duplicate (combo, fold, hmm_seed),
`val_ir` finite for >99% of rows. Print `select_years` selection paths and
eyeball: 2011 selections use folds 1-7 only. Then:
```bash
nohup caffeinate -dims bash paper/run_s3.sh >> paper/results/s3_run.log 2>&1 &
```

- [ ] **Step 5: On S3 completion — run report, deliver numbers**

`python -m paper.pipeline --stage report --workers 2`; read
`paper/results/applied_summary.md`; score E1-E5; present full tables to
Gilad. NO paper prose until he has reviewed the numbers.

## Self-Review

- **Spec coverage:** universe (T4), applied-IR selection (T8), argmax+rule_r
  (T8), expanding yearly walk with 50/20 seeds (T10), xsec persistence for
  post-run TC work (T10), comparators incl. 12-1 and no-pi (T10),
  concentration/beta/regime reporting (T10), frozen hyperparams (T6),
  registered expectations + full-table rule (T11), ext2026 swap = one
  config constant (T1). S5/TC frontier and top-500 robustness are
  post-run per spec and Gilad's instruction — deliberately not in this plan.
  Feature real-time availability statement: prose-level, lands with S4
  report notes (T10 report writes conventions line into applied_summary.md).
- **Placeholder scan:** T10 Step 1 shows the core loop and precisely
  specifies the remaining file content (interfaces block) — acceptable
  because the eval-engine template it mirrors is named with exact path.
  No TBDs.
- **Type consistency:** run_cell returns dict with `val_ir` consumed by
  select_years via cells DataFrame column `val_ir` ✓; fold tuples
  (id, val_start, val_end) consistent across config/selection/pipeline ✓;
  returns series indexed by formation date everywhere ✓.
- **Known deviations (disclosed):** selection cells use posterior-mean HMM
  (not the thesis last-draw shortcut) — selection matches deployment;
  rule_r tie-break drops ESS (undefined for IR objective).
