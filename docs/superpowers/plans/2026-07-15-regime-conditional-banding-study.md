# Regime-Conditional Banding Study — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether regime-conditional banding is a general cost-mitigation technique: for 7 strategies (momentum, reversal, low-vol, XGB, value, profitability, investment), compare no-band vs static-optimal band vs regime-conditional band (2-state and 3-state crash/recovery), net of measured and stressed transaction costs.

**Architecture:** Every strategy is reduced to a stock-level score panel with the same schema as the existing XGB xsec files, so the existing `paper/execution.py::simulate()` engine runs all of them unchanged. New code: signal builders (`paper/src/signals.py`, `paper/src/fundamentals.py`, `paper/src/ivol.py`), a 3-state band policy in `execution.py`, a pi-history backfill, and a driver (`paper/banding_study.py`) that sweeps (strategy × band policy × cost level), prices trades from the stock-month half-spread panel, and evaluates pre-registered gates G1–G3.

**Tech Stack:** Python 3.12 (`.venv`), pandas/numpy, statsmodels (NW t), existing `paper/` engine (`execution.simulate`, `paired_block_bootstrap`, `src/portfolio.py`, `src/metrics.py`, `src/hmm.py`), WRDS (`wrds` package, gated), Ken French public data library.

**Spec:** `docs/superpowers/specs/2026-07-15-regime-conditional-banding-study-design.md`

## Global Constraints

- Branch `paper/applied-study`. All new code under `paper/`. NOTHING under `scripts/`, `results/`, `tables/`, `latex/`, `data/` may be modified (reading `data/ff_factors.parquet` is allowed).
- **WRDS HARD GATE:** Task 5 opens a WRDS session. Do NOT run it without notifying Gilad and receiving an explicit "yes" for that session. All other tasks must run without WRDS.
- All paths/knobs go through `paper/config.py` ("ALL knobs live here"). No hardcoded paths in modules.
- TDD: write the failing test first, watch it fail, implement, watch it pass, commit. Tests live in `paper/tests/`, runnable via `.venv/bin/python -m pytest paper/tests/ -q` (pytest.ini sets rootdir).
- Features at time t use only information available at t (PIT). Fundamentals carry a 6-month reporting lag (already stamped in `funda_linked.parquet::avail_date`). Time-ordered evaluation only.
- Long-only, cap-weighted top-decile portfolios (`DECILE_FRAC = 0.10`), universe = top-`UNIVERSE_N` (1000) by `me` per month.
- Return convention: series indexed by FORMATION month t, value = `ret_fwd` earned over t+1 (repo-wide).
- Cost convention (S5/S6): `net = gross − Σ_i |Δw_i|·hs_i/1e4`; the benchmark pays its own reconstitution costs; `turnover = traded/2`.
- Stochastic runs log seeds; HMM backfill uses `SEL_HMM_SEEDS` (= `repo_config.HMM_SEEDS[:3]`), `N_ITER=2000, N_BURNIN=500`.
- Commit after every task with a `paper:` prefixed message.
- Long runs (Task 4 backfill, Task 11 full sweep) go to background with `nohup caffeinate -dims`; don't block; don't poll.

## Data reality (verified 2026-07-15 — the plan is built around these facts)

| Input | Coverage | Where |
|---|---|---|
| `stocks.parquet` (permno, date, me, mom_1..mom_12, ret_fwd) | 1991-01 → 2025-11 | `paper/results/data/` |
| `panel.parquet` (date, DD, vwretd, + z-features) | 1970-12 → 2025-11 | `paper/results/data/` |
| XGB scores (`xsec_YYYY_*.parquet`: date, permno, me, pi, score_pi, score_nopi, mom_12, ret_fwd) | 2011 → 2025 only | `paper/results/xsec/` |
| `funda_linked.parquet` (24 cols, avail_date = datadate+6mo exactly) | fyear 1985 → 2025, 260,890 rows | `paper/results/data/` |
| Daily CRSP `dsf_v2_YYYY.parquet` (permno, dlycaldt, dlybid, dlyask, dlyhigh, dlylow, dlyprc, …) | **2010-12 → 2025 only** (Task 5 backfills 1990–2010) | `experiments/results/spreads/` |
| `half_spreads.parquet` (permno, ym, qhs, cs_hs, hs) | 2010-12 → 2025-12 (Task 7 extends) | `paper/results/s5/` |
| `ff_long_legs.parquet` (size, value, robust, conservative, winner) | 1926-07 → 2026-05 | `paper/results/data/` |
| `data/ff_factors.parquet` (Mkt-RF, RF, …) — READ ONLY | full history | thesis `data/` |

Consequences encoded below: the study window is **1992-01 → 2025-11** (momentum needs 12m of 1991 history); XGB runs **2011 → 2025** only (inherent, reported as such); low-vol and measured spreads before 2011 depend on Task 5.

## File structure

- `paper/crash_recovery.py` (new) — promote the 2026-07-15 scratch decomposition; writes `paper/results/banding_study/crash_recovery.csv`.
- `paper/src/fundamentals.py` (new) — funda_linked → BE, OP, COP, asset growth per (permno, avail_date).
- `paper/src/regime_labels.py` (new) — 3-state calm/crash/recovery labeling from vwretd + pi.
- `paper/build_pi_history.py` (new) — monthly pi 1991–2025 (backfill 1991–2010 by expanding annual DD-only HMM refits; reuse xsec pi 2011+).
- `paper/pull_daily_history.py` (new, **WRDS-gated**) — daily CRSP 1990–2010 → `experiments/results/spreads/dsf_v2_1990.parquet` … `dsf_v2_2009.parquet` (existing naming so `paper/spreads.py` glob picks them up; 2010 exists).
- `paper/fetch_ff_daily.py` (new) — public French daily FF3 factors → `paper/results/data/ff_daily.parquet`.
- `paper/src/ivol.py` (new) — trailing-90-trading-day FF3-residual vol, monthly snapshots → `paper/results/data/ivol_monthly.parquet`.
- `paper/src/signals.py` (new) — 7 signal builders, common xsec schema `(date, permno, me, pi, score, mom_12, ret_fwd, state3)`.
- `paper/execution.py` (modify) — add `regime3_band` policy to `_members()`.
- `paper/banding_study.py` (new) — the sweep driver + gates + report.
- `paper/config.py` (modify) — one `BANDING STUDY` block of knobs.
- Tests: `paper/tests/test_crash_recovery.py`, `test_fundamentals.py`, `test_regime_labels.py`, `test_ivol.py`, `test_signals.py`, `test_regime3_band.py`, `test_banding_study.py`.

---

### Task 0: Register expectations G1–G3 (pre-run, doc only)

**Files:**
- Modify: `paper/results/PAPER_NOTES.md` (append a section)

**Interfaces:** none (documentation). Must land BEFORE any sweep results exist (spec §2 discipline).

- [ ] **Step 1: Append the registration section**

Append to `paper/results/PAPER_NOTES.md`:

```markdown
## N. Banding-study registered expectations (2026-07-15, PRE-RUN)

Registered before paper/banding_study.py produced any cells. Spec:
docs/superpowers/specs/2026-07-15-regime-conditional-banding-study-design.md.

- G1: regime-conditional banding (best 2- or 3-state cell, selected on net IR
  at measured spreads) beats the best STATIC band (best diagonal) on net IR
  for a MAJORITY of the 7 strategies; per-strategy paired block-bootstrap CI
  of the net-active difference excludes 0 for at least the majority winners.
- G2: the improvement d(net IR) is LARGER for high-cost strategies (low-vol,
  short-term reversal) than for low-cost ones (value): Spearman rank corr
  between cost intensity (monthly-tier TO x mean hs) and d(net IR) > 0.
- G3: the IR-weighted combined book across the 7 strategies has a higher net
  Sharpe under regime-conditional banding than under static banding.

A MISS on any gate is reported as such (null results are reportable).
Direction expectation from S6 + the 2026-07-15 crash/recovery decomposition:
wide band in crash, tighter band in recovery; but the grid explores all
directions and the frontier decides.
```

- [ ] **Step 2: Commit**

```bash
git add paper/results/PAPER_NOTES.md
git commit -m "paper: register banding-study gates G1-G3 (pre-run)"
```

---

### Task 1: Promote the crash/recovery decomposition into `paper/`

**Files:**
- Create: `paper/crash_recovery.py`
- Test: `paper/tests/test_crash_recovery.py`

**Interfaces:**
- Consumes: `C.RETURNS_CSV` (walk_returns.csv: date, year, rule, combo, strat_ret, bench_ret, pi).
- Produces: `decompose(walk_df) -> pd.DataFrame` with columns `bucket ∈ {calm, panic_crash, panic_recovery}, n_mo, strat_mo, bench_mo, active_mo, active_t, sum_active, share_of_total_active`; CLI writes `paper/results/banding_study/crash_recovery.csv`. Task 3/driver reuse the bucket rule via `paper/src/regime_labels.py` (Task 3 is the shared implementation; this task inlines the same rule and Task 3's test cross-checks equality).

- [ ] **Step 1: Write the failing test**

`paper/tests/test_crash_recovery.py`:

```python
import numpy as np
import pandas as pd

from paper.crash_recovery import decompose


def _walk_fixture():
    # 6 months: 2 calm, 2 panic-crash (market down), 2 panic-recovery (up)
    dates = pd.date_range('2020-01-31', periods=6, freq='ME')
    return pd.DataFrame({
        'date': dates,
        'strat_ret': [0.01, 0.02, -0.10, -0.05, 0.08, 0.09],
        'bench_ret': [0.01, 0.01, -0.08, -0.04, 0.06, 0.07],
        'pi':        [0.1,  0.2,   0.9,   0.8,  0.9,  0.7],
    })


def test_buckets_and_shares():
    out = decompose(_walk_fixture()).set_index('bucket')
    assert out.loc['calm', 'n_mo'] == 2
    assert out.loc['panic_crash', 'n_mo'] == 2
    assert out.loc['panic_recovery', 'n_mo'] == 2
    # active means, hand-computed
    assert np.isclose(out.loc['calm', 'active_mo'], 0.005)
    assert np.isclose(out.loc['panic_crash', 'active_mo'], -0.015)
    assert np.isclose(out.loc['panic_recovery', 'active_mo'], 0.02)
    # shares sum to 100%
    assert np.isclose(out['share_of_total_active'].sum(), 1.0)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest paper/tests/test_crash_recovery.py -q`
Expected: FAIL (`ModuleNotFoundError: paper.crash_recovery`).

- [ ] **Step 3: Implement `paper/crash_recovery.py`**

```python
"""Crash/recovery decomposition of the applied model's active return.

Promoted from the 2026-07-15 scratch analysis (spec §1). Splits each month
into calm / panic_crash / panic_recovery on the market's own path: panic
(pi >= 0.5) months are RECOVERY when the market drawdown is healing
(bench_ret > 0 this month, equivalently dd_t > dd_{t-1}), else CRASH.

Usage: .venv/bin/python -m paper.crash_recovery
Output: paper/results/banding_study/crash_recovery.csv
"""
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from paper import config as C                                   # noqa: E402


def decompose(walk):
    m = walk.sort_values('date').copy()
    m['active'] = m['strat_ret'] - m['bench_ret']
    panic = m['pi'] >= 0.5
    m['bucket'] = np.where(~panic, 'calm',
                           np.where(m['bench_ret'] > 0,
                                    'panic_recovery', 'panic_crash'))
    tot = m['active'].sum()
    rows = []
    for b in ['calm', 'panic_crash', 'panic_recovery']:
        g = m[m['bucket'] == b]
        n = len(g)
        std = g['active'].std()
        rows.append({
            'bucket': b, 'n_mo': n,
            'strat_mo': g['strat_ret'].mean(),
            'bench_mo': g['bench_ret'].mean(),
            'active_mo': g['active'].mean(),
            'active_t': (g['active'].mean() / std * np.sqrt(n)
                         if n > 1 and std > 0 else np.nan),
            'sum_active': g['active'].sum(),
            'share_of_total_active': g['active'].sum() / tot,
        })
    return pd.DataFrame(rows)


def main():
    walk = pd.read_csv(C.RETURNS_CSV, parse_dates=['date'])
    m = walk[(walk['rule'] == 'rule_r') & (walk['combo'] == 'DD')]
    out = decompose(m)
    os.makedirs(C.BANDING_DIR, exist_ok=True)
    path = os.path.join(C.BANDING_DIR, 'crash_recovery.csv')
    out.round(6).to_csv(path, index=False)
    print(out.round(4).to_string(index=False))
    print('Saved:', path)


if __name__ == '__main__':
    main()
```

Add to `paper/config.py` (bottom, new block — Tasks 2/4/6/10 extend this same block):

```python
# ── banding study (spec 2026-07-15) ──
BANDING_DIR = os.path.join(RESULTS, 'banding_study')
```

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest paper/tests/test_crash_recovery.py -q`
Expected: PASS.

- [ ] **Step 5: Run on real data, verify against the known result**

Run: `.venv/bin/python -m paper.crash_recovery`
Expected (must reproduce the 2026-07-15 numbers): calm n=134 active_mo≈+0.0004; panic_crash n=16 active_mo≈−0.0096; panic_recovery n=29 active_mo≈+0.0165, active_t≈2.1, share≈1.27.

- [ ] **Step 6: Commit**

```bash
git add paper/crash_recovery.py paper/tests/test_crash_recovery.py paper/config.py
git commit -m "paper: promote crash/recovery decomposition (recovery carries 127% of active)"
```

---

### Task 2: Fundamental signal levels (`paper/src/fundamentals.py`)

**Files:**
- Create: `paper/src/fundamentals.py`
- Test: `paper/tests/test_fundamentals.py`
- Modify: `paper/config.py` (add `FUNDA_PARQUET`)

**Interfaces:**
- Consumes: `paper/results/data/funda_linked.parquet` (permno, gvkey, datadate, avail_date, fyear, at_lag, seq, ceq, txditc, pstkrv, pstkl, pstk, at, lt, revt, cogs, xsga, xint, rect, invt, ap, xacc, xpp, che).
- Produces: `load_fundamentals() -> pd.DataFrame` with columns `(permno, datadate, avail_date, fyear, be, op, cop, asset_growth)` — one row per (permno, avail_date); NaN where undefined. Task 8 consumes it.

Definitions (FF 2015 + Ball et al. 2016, DNMV appendix conventions):
- `be` = `seq` (fallback `ceq + pstk`, then `at − lt`) `+ txditc(0 if missing) − ps` where `ps` = `pstkrv` (fallback `pstkl`, then `pstk`, then 0). `be <= 0 → NaN`.
- `op_raw` = `revt − cogs − xsga(0) − xint(0)`; requires `revt` and `cogs` non-missing, else NaN. `op = op_raw / be`.
- `cop_raw` = `op_raw − Δrect − Δinvt − Δxpp + Δap + Δxacc` (each Δ within gvkey by fyear; missing Δ → 0). `cop = cop_raw / be`.
- `asset_growth` = `at / at_lag − 1`; NaN if either missing or `at_lag <= 0`.

- [ ] **Step 1: Write the failing test**

`paper/tests/test_fundamentals.py`:

```python
import numpy as np
import pandas as pd
import pytest

from paper.src.fundamentals import compute_signals


def _funda_fixture():
    # one firm, two fiscal years; hand-checkable numbers
    return pd.DataFrame({
        'permno': [111, 111], 'gvkey': ['001', '001'],
        'datadate': pd.to_datetime(['2014-12-31', '2015-12-31']),
        'avail_date': pd.to_datetime(['2015-06-30', '2016-06-30']),
        'fyear': [2014, 2015],
        'seq': [100.0, 120.0], 'ceq': [95.0, 115.0],
        'txditc': [10.0, np.nan], 'pstkrv': [5.0, np.nan],
        'pstkl': [np.nan, 4.0], 'pstk': [3.0, 3.0],
        'at': [500.0, 550.0], 'at_lag': [np.nan, 500.0], 'lt': [400.0, 430.0],
        'revt': [300.0, 330.0], 'cogs': [200.0, 210.0],
        'xsga': [50.0, np.nan], 'xint': [10.0, 12.0],
        'rect': [40.0, 44.0], 'invt': [30.0, 27.0], 'ap': [20.0, 26.0],
        'xacc': [5.0, 6.0], 'xpp': [2.0, 3.0], 'che': [15.0, 18.0],
    })


def test_book_equity_and_profitability():
    out = compute_signals(_funda_fixture()).set_index('fyear')
    # fy2014: be = 100 + 10 - 5 = 105; op_raw = 300-200-50-10 = 40
    assert np.isclose(out.loc[2014, 'be'], 105.0)
    assert np.isclose(out.loc[2014, 'op'], 40.0 / 105.0)
    # fy2014 has no prior year: deltas -> 0, cop == op
    assert np.isclose(out.loc[2014, 'cop'], 40.0 / 105.0)
    assert np.isnan(out.loc[2014, 'asset_growth'])
    # fy2015: be = 120 + 0 - 4 = 116; op_raw = 330-210-0-12 = 108
    assert np.isclose(out.loc[2015, 'be'], 116.0)
    assert np.isclose(out.loc[2015, 'op'], 108.0 / 116.0)
    # cop_raw = 108 - (44-40) - (27-30) - (3-2) + (26-20) + (6-5) = 113
    assert np.isclose(out.loc[2015, 'cop'], 113.0 / 116.0)
    assert np.isclose(out.loc[2015, 'asset_growth'], 550.0 / 500.0 - 1)


def test_pit_lag_is_six_months():
    out = compute_signals(_funda_fixture())
    lag = ((out['avail_date'].dt.year * 12 + out['avail_date'].dt.month)
           - (out['datadate'].dt.year * 12 + out['datadate'].dt.month))
    assert (lag == 6).all()


def test_negative_be_is_nan():
    f = _funda_fixture()
    f.loc[0, ['seq', 'ceq']] = np.nan
    f.loc[0, 'at'] = 390.0            # at - lt = -10 -> be <= 0 -> NaN
    out = compute_signals(f).set_index('fyear')
    assert np.isnan(out.loc[2014, 'be'])
    assert np.isnan(out.loc[2014, 'op'])
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest paper/tests/test_fundamentals.py -q`
Expected: FAIL (`ModuleNotFoundError: paper.src.fundamentals`).

- [ ] **Step 3: Implement `paper/src/fundamentals.py`**

```python
"""Fundamental signal levels from the (gated, already-run) Compustat pull.

be / op / cop / asset_growth per (permno, avail_date), PIT-safe: every row
carries avail_date = datadate + 6 months (stamped by paper/pull_fundamentals).
Definitions: FF (2015) book equity and operating profitability; Ball et al.
(2016) cash-based adjustments (missing deltas treated as 0); asset growth
at/at_lag - 1.
"""
import numpy as np
import pandas as pd

from paper import config as C

_KEEP = ['permno', 'datadate', 'avail_date', 'fyear',
         'be', 'op', 'cop', 'asset_growth']


def compute_signals(f):
    f = f.sort_values(['gvkey', 'fyear']).copy()

    seq = f['seq'].fillna(f['ceq'] + f['pstk'].fillna(0))
    seq = seq.fillna(f['at'] - f['lt'])
    ps = f['pstkrv'].fillna(f['pstkl']).fillna(f['pstk']).fillna(0)
    be = seq + f['txditc'].fillna(0) - ps
    f['be'] = be.where(be > 0)

    op_raw = f['revt'] - f['cogs'] - f['xsga'].fillna(0) - f['xint'].fillna(0)
    op_raw = op_raw.where(f['revt'].notna() & f['cogs'].notna())
    f['op'] = op_raw / f['be']

    g = f.groupby('gvkey')
    adj = (-g['rect'].diff().fillna(0) - g['invt'].diff().fillna(0)
           - g['xpp'].diff().fillna(0) + g['ap'].diff().fillna(0)
           + g['xacc'].diff().fillna(0))
    f['cop'] = (op_raw + adj) / f['be']

    ag = f['at'] / f['at_lag'] - 1
    f['asset_growth'] = ag.where(f['at'].notna() & (f['at_lag'] > 0))
    return f[_KEEP].reset_index(drop=True)


def load_fundamentals():
    return compute_signals(pd.read_parquet(C.FUNDA_PARQUET))
```

Add to the `BANDING STUDY` block in `paper/config.py`:

```python
FUNDA_PARQUET = os.path.join(DATA_OUT, 'funda_linked.parquet')
```

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest paper/tests/test_fundamentals.py -q`
Expected: 3 passed.

- [ ] **Step 5: Real-data sanity print (no assert, evidence for the log)**

Run:
```bash
.venv/bin/python -c "
from paper.src.fundamentals import load_fundamentals
f = load_fundamentals()
print(f.shape); print(f[['be','op','cop','asset_growth']].describe().round(3))
print('coverage:', f[['be','op','cop','asset_growth']].notna().mean().round(3).to_dict())"
```
Expected: ~260k rows; medians plausible (op/cop roughly 0.1–0.4, asset_growth ~0.0–0.1); coverage ≥ 0.8 for be/op/asset_growth.

- [ ] **Step 6: Commit**

```bash
git add paper/src/fundamentals.py paper/tests/test_fundamentals.py paper/config.py
git commit -m "paper: fundamental signal levels (BE, OP, COP, asset growth), PIT-lagged"
```

---

### Task 3: 3-state regime labels (`paper/src/regime_labels.py`)

**Files:**
- Create: `paper/src/regime_labels.py`
- Test: `paper/tests/test_regime_labels.py`

**Interfaces:**
- Consumes: a monthly frame with `date, pi` and a market return series (`vwretd` from `panel.parquet` — exogenous to every strategy).
- Produces: `label_states(dates, pi, mkt_ret) -> pd.Series` of int8 codes indexed by date: `0 = calm, 1 = panic_crash, 2 = panic_recovery`, with module constants `CALM, CRASH, RECOVERY = 0, 1, 2`. Recovery ⇔ panic month with `mkt_ret > 0` (drawdown healing); the equivalent drawdown-trajectory definition is exposed as `label_states_dd(dates, pi, mkt_ret)` for the robustness check. Tasks 8–10 consume `label_states`; Task 1's inline rule must match (cross-check test below).

- [ ] **Step 1: Write the failing test**

`paper/tests/test_regime_labels.py`:

```python
import numpy as np
import pandas as pd

from paper.src.regime_labels import (CALM, CRASH, RECOVERY,
                                     label_states, label_states_dd)


def _fixture():
    dates = pd.date_range('2020-01-31', periods=6, freq='ME')
    pi = pd.Series([0.1, 0.2, 0.9, 0.8, 0.9, 0.7], index=dates)
    mkt = pd.Series([0.01, 0.01, -0.08, -0.04, 0.06, 0.07], index=dates)
    return dates, pi, mkt


def test_three_state_codes():
    dates, pi, mkt = _fixture()
    s = label_states(dates, pi, mkt)
    assert list(s.values) == [CALM, CALM, CRASH, CRASH, RECOVERY, RECOVERY]
    assert s.dtype == np.int8


def test_dd_definition_matches_sign_definition_on_fixture():
    # dd heals exactly when mkt_ret > 0 while below peak -> identical here
    dates, pi, mkt = _fixture()
    assert (label_states(dates, pi, mkt)
            == label_states_dd(dates, pi, mkt)).all()
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest paper/tests/test_regime_labels.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `paper/src/regime_labels.py`**

```python
"""3-state regime labels: calm / panic-crash / panic-recovery.

Panic = pi >= 0.5 (main applied model, DD-only HMM). Within panic, RECOVERY
= the market's own drawdown is healing this month. Primary definition uses
the market-return sign (mkt_ret > 0); label_states_dd implements the
drawdown-trajectory version (dd_t > dd_{t-1}) for the robustness check.
Both use only information available at formation (contemporaneous month).
The 2026-07-15 decomposition found the two definitions identical on
2011-2025 walk data.
"""
import numpy as np
import pandas as pd

CALM, CRASH, RECOVERY = 0, 1, 2
PANIC_THR = 0.5


def label_states(dates, pi, mkt_ret):
    pi = pd.Series(np.asarray(pi), index=dates)
    mkt = pd.Series(np.asarray(mkt_ret), index=dates)
    out = np.where(pi < PANIC_THR, CALM,
                   np.where(mkt > 0, RECOVERY, CRASH))
    return pd.Series(out.astype(np.int8), index=dates, name='state3')


def label_states_dd(dates, pi, mkt_ret):
    pi = pd.Series(np.asarray(pi), index=dates)
    mkt = pd.Series(np.asarray(mkt_ret), index=dates)
    lvl = (1 + mkt).cumprod()
    dd = lvl / lvl.cummax() - 1
    healing = dd > dd.shift(1).fillna(0.0)
    out = np.where(pi < PANIC_THR, CALM,
                   np.where(healing, RECOVERY, CRASH))
    return pd.Series(out.astype(np.int8), index=dates, name='state3')
```

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest paper/tests/test_regime_labels.py -q`
Expected: 2 passed.

- [ ] **Step 5: Cross-check against Task 1 on real walk data**

Run:
```bash
.venv/bin/python -c "
import pandas as pd
from paper import config as C
from paper.src.regime_labels import label_states, CRASH, RECOVERY
w = pd.read_csv(C.RETURNS_CSV, parse_dates=['date'])
w = w[(w.rule=='rule_r') & (w.combo=='DD')].sort_values('date')
s = label_states(w['date'], w['pi'].values, w['bench_ret'].values)
print('crash months:', (s==CRASH).sum(), '| recovery months:', (s==RECOVERY).sum())"
```
Expected: `crash months: 16 | recovery months: 29` (matches the decomposition).

- [ ] **Step 6: Commit**

```bash
git add paper/src/regime_labels.py paper/tests/test_regime_labels.py
git commit -m "paper: 3-state regime labels (calm/crash/recovery), two equivalent definitions"
```

---

### Task 4: Monthly pi history 1991–2025 (`paper/build_pi_history.py`)

**Files:**
- Create: `paper/build_pi_history.py`
- Test: `paper/tests/test_pi_history.py` (smoke of the assembly logic; the HMM itself is already tested by `test_hmm_gate.py`)
- Modify: `paper/config.py` (add `PI_MONTHLY`)

**Interfaces:**
- Consumes: `panel.parquet` (DD feature, 1970→), `paper/src/hmm.fit_pi(Z_train, Z_full, signs, seeds, workers, n_iter, n_burnin)`, existing xsec pi for 2011–2025 via `paper/execution.load_xsec()`.
- Produces: `paper/results/data/pi_monthly.parquet` with columns `(date, pi, src)` where `src ∈ {'backfill', 'walk'}`; monthly, 1991-01 → 2025-11. `assemble(back, walk) -> pd.DataFrame` is the pure, testable piece.

Backfill construction (DD-only, expanding annual refits): for each year Y in 1991–2010, train the 1-feature HMM on `panel[DD_z]` rows with `date < Y-01-01` using the FULL panel history (from 1970 — the walk's `PANEL_START` floor of 1990-12 is deliberately NOT applied: a 1992 fit would otherwise have ~13 training months), infer smoothed pi for the 12 months of Y, mean over `SEL_HMM_SEEDS` (3 seeds), `N_ITER=2000, N_BURNIN=500`. Sign convention: the DD-anchored `signs = [-1.0]` (panic = deeper-drawdown state), exactly what `selection.run_cell` uses for DD-only combos — NOT `hmm.crisis_signs`, because `CRISIS_WINDOWS` = [2000–02, 2007–09] and training windows for backfill years 1991–2000 contain no crisis window (crisis_signs would be degenerate there; run_cell's comment documents this convention for exactly this case; for DD-only the two coincide whenever crisis windows exist). For 2011–2025 reuse the walk's pi verbatim (consistency with every published applied number). These two deliberate deviations from the walk recipe (history floor, sign rule) are quantified by the 2011-overlap check in Step 6.

- [ ] **Step 1: Write the failing test (assembly logic only — no HMM)**

`paper/tests/test_pi_history.py`:

```python
import pandas as pd

from paper.build_pi_history import assemble


def test_assemble_prefers_walk_and_is_monotone():
    back = pd.DataFrame({
        'date': pd.to_datetime(['2010-11-30', '2010-12-31', '2011-01-31']),
        'pi': [0.2, 0.3, 0.99]})              # backfill overlaps 2011-01
    walk = pd.DataFrame({
        'date': pd.to_datetime(['2011-01-31', '2011-02-28']),
        'pi': [0.4, 0.5]})
    out = assemble(back, walk)
    assert list(out['src']) == ['backfill', 'backfill', 'walk', 'walk']
    assert out.loc[out['date'] == '2011-01-31', 'pi'].item() == 0.4
    assert out['date'].is_monotonic_increasing
    assert not out['date'].duplicated().any()
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest paper/tests/test_pi_history.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `paper/build_pi_history.py`**

```python
"""Monthly DD-only regime pi, 1991-2025.

2011-2025: reuse the applied walk's pi verbatim (from the xsec files via
execution.load_xsec) so every published applied number is untouched.
1991-2010: backfill with the same recipe the walk uses - expanding annual
refits of the 1-feature DD HMM (train < Jan-Y, infer year Y), mean pi over
SEL_HMM_SEEDS, N_ITER/N_BURNIN from config. Seeds logged in the output.

Usage: .venv/bin/python -m paper.build_pi_history [--start 1991 --end 2010]
Output: paper/results/data/pi_monthly.parquet  (date, pi, src)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from paper import config as C                                   # noqa: E402
from paper.src import hmm                                       # noqa: E402


def assemble(back, walk):
    back = back.assign(src='backfill')
    walk = walk.assign(src='walk')
    out = pd.concat([back[~back['date'].isin(set(walk['date']))], walk])
    return out.sort_values('date').reset_index(drop=True)


def _backfill(years, workers):
    panel = pd.read_parquet(C.PANEL_PARQUET)
    panel['date'] = pd.to_datetime(panel['date'])
    panel = (panel.dropna(subset=['DD_z'])
             .sort_values('date').reset_index(drop=True))
    # DD-anchored sign rule (panic = deeper-drawdown state): the
    # selection.run_cell convention for DD-only combos. crisis_signs is NOT
    # usable here - CRISIS_WINDOWS start in 2000, so 1991-2000 training
    # windows contain no crisis months.
    signs = np.array([-1.0])
    rows = []
    for y in years:
        # full history from 1970 (no PANEL_START floor - early years need it)
        sub = panel[panel['date'] < f'{y + 1}-01-01'].reset_index(drop=True)
        tr = sub['date'] < f'{y}-01-01'
        Z_tr = sub.loc[tr, ['DD_z']].values.astype(float)
        Z_full = sub[['DD_z']].values.astype(float)
        pi = hmm.fit_pi(Z_tr, Z_full, signs, C.SEL_HMM_SEEDS, workers,
                        n_iter=C.N_ITER, n_burnin=C.N_BURNIN)
        yr = (sub['date'].dt.year == y).values
        rows.append(pd.DataFrame({'date': sub.loc[yr, 'date'],
                                  'pi': np.asarray(pi)[yr]}))
        print(f'[pi] {y}: panic months '
              f'{(rows[-1]["pi"] >= 0.5).sum()}/12', flush=True)
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=1991)
    ap.add_argument('--end', type=int, default=2010)
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()

    from paper import execution as X
    x = X.load_xsec()
    walk = (x.groupby('date', as_index=False)['pi'].first())

    back = _backfill(range(a.start, a.end + 1), a.workers)
    out = assemble(back, walk)
    os.makedirs(C.DATA_OUT, exist_ok=True)
    out.to_parquet(C.PI_MONTHLY, index=False)
    print(f'Saved {C.PI_MONTHLY}: {len(out)} months '
          f'{out.date.min().date()} -> {out.date.max().date()} | '
          f'panic share {(out.pi >= 0.5).mean():.2%} | '
          f'seeds {C.SEL_HMM_SEEDS}')


if __name__ == '__main__':
    main()
```

Add to the `BANDING STUDY` block in `paper/config.py`:

```python
PI_MONTHLY = os.path.join(DATA_OUT, 'pi_monthly.parquet')
```

VERIFIED (2026-07-15): `hmm.fit_pi` returns the seed-mean smoothed-pi vector over `Z_full` (`np.vstack([...]).mean(axis=0)`) — the code above consumes it correctly as-is.

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest paper/tests/test_pi_history.py -q`
Expected: PASS.

- [ ] **Step 5: Smoke run (2 years, tiny iterations) — fast correctness check**

Run: `.venv/bin/python -c "
import sys; sys.argv = ['x', '--start', '2008', '--end', '2009', '--workers', '3']
import runpy; runpy.run_module('paper.build_pi_history', run_name='__main__')"`
after temporarily lowering iterations is NOT allowed (config is canonical) — instead just let the 2-year smoke run at full iterations (~2 fits, minutes).
Expected: 2008 and/or 2009 show panic months > 0 (GFC), file written.

- [ ] **Step 6: Full backfill run (background, ~20 fits x ~1-2 min)**

```bash
nohup caffeinate -dims .venv/bin/python -m paper.build_pi_history \
  >> paper/results/banding_study/pi_backfill.log 2>&1 &
```
Do not poll; on completion verify:
- 2011+ pi identical to the walk (`src == 'walk'` everywhere post-2011).
- Sanity: panic months cover 2008-09→2009-H1 and 2020-03/04; overall panic share between 15% and 35% (walk period was 45/179 ≈ 25%).
- **2011-overlap consistency check** (quantifies the two deliberate recipe deviations — 1970 history floor and `signs=[-1]` — plus 3-vs-50 seed noise): run `_backfill([2011], 3)` and compare its 12 months against the walk's 2011 pi. Require the panic/calm label (pi ≥ 0.5) to agree on ≥ 10 of 12 months; report the pi correlation in the log. If agreement < 10/12, STOP and surface to Gilad before using the backfill.

- [ ] **Step 7: Commit**

```bash
git add paper/build_pi_history.py paper/tests/test_pi_history.py paper/config.py
git commit -m "paper: monthly DD-only pi history 1991-2025 (expanding backfill + walk reuse)"
```

---

### Task 5: Daily CRSP history 1990–2010 — **WRDS HARD GATE**

**Files:**
- Create: `paper/pull_daily_history.py`
- Output: `experiments/results/spreads/dsf_v2_1990.parquet` … `dsf_v2_2009.parquet` (2010+ already exist; same schema/naming so `paper/spreads.py`'s glob and `paper/src/ivol.py` pick them up unchanged)

**Interfaces:**
- Consumes: WRDS `crsp.dsf_v2` (post-2022 CRSP daily format, same table the 2010–2025 files came from).
- Produces: per-year parquets with columns `(permno, dlycaldt, dlybid, dlyask, dlyhigh, dlylow, dlyprc, dlyret)` — superset of what `paper/spreads.py::build_spreads` reads plus `dlyret` for IVOL. NOTE: existing 2010–2025 files may lack `dlyret`; Task 6 therefore reads `dlyret` where present and computes returns from `dlyprc` otherwise (see Task 6).

> **GATE — STOP HERE.** Before running this task: tell Gilad exactly what will be pulled (crsp.dsf_v2 daily, 1990-01-01→2009-12-31, ~20 files, est. 10–30 min, a few GB on disk) and **wait for an explicit yes for this session**. The pull for fundamentals earlier does NOT cover this session.

- [ ] **Step 1: Confirm the schema of an existing file (drives the SELECT list)**

Run: `.venv/bin/python -c "import pandas as pd; print(pd.read_parquet('experiments/results/spreads/dsf_v2_2015.parquet').columns.tolist())"`
Record which of `dlyret`/`dlyprc` are present; keep the new files' schema a superset.

- [ ] **Step 2: Implement `paper/pull_daily_history.py`**

```python
"""Backfill daily CRSP (dsf_v2) 1990-2009 for IVOL + pre-2011 spreads.

WRDS-GATED: run only after Gilad's explicit per-session approval.
Writes per-year parquets into experiments/results/spreads/ with the same
naming as the existing 2010-2025 files so downstream globs pick them up.

Usage: .venv/bin/python -m paper.pull_daily_history [--start 1990 --end 2009]
"""
import argparse
import os
import sys
from pathlib import Path

import wrds

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

OUT_DIR = 'experiments/results/spreads'
COLS = ('permno, dlycaldt, dlybid, dlyask, dlyhigh, dlylow, dlyprc, dlyret')


def _connect():
    pgpass = Path.home() / '.pgpass'
    _u, _p = 'giladgang', None
    for line in pgpass.read_text().splitlines():
        parts = line.strip().split(':')
        if len(parts) == 5 and parts[0].startswith('wrds-pgdata'):
            _u, _p = parts[3], parts[4]
            break
    return wrds.Connection(wrds_username=_u, wrds_password=_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=1990)
    ap.add_argument('--end', type=int, default=2009)
    a = ap.parse_args()
    db = _connect()
    for y in range(a.start, a.end + 1):
        out = os.path.join(OUT_DIR, f'dsf_v2_{y}.parquet')
        if os.path.exists(out):
            print(f'[skip] {out} exists', flush=True)
            continue
        d = db.raw_sql(f"""
            SELECT {COLS}
            FROM crsp.dsf_v2
            WHERE dlycaldt BETWEEN '{y}-01-01' AND '{y}-12-31'
        """, date_cols=['dlycaldt'])
        d.to_parquet(out, index=False)
        print(f'[pull] {y}: {len(d):,} rows -> {out}', flush=True)
    db.close()


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: [after explicit approval] Run the pull**

Run: `.venv/bin/python -m paper.pull_daily_history` (resume-safe: skips existing files).
Expected: 20 files, each ~1–3M rows (universe is all CRSP; filtering happens downstream).

- [ ] **Step 4: Verify coverage**

Run:
```bash
.venv/bin/python -c "
import glob, pandas as pd
fs = sorted(glob.glob('experiments/results/spreads/dsf_v2_19*.parquet') +
            glob.glob('experiments/results/spreads/dsf_v2_200*.parquet'))
print(len(fs), 'files')
d = pd.read_parquet(fs[0])
print(fs[0], d['dlycaldt'].min(), d['dlycaldt'].max(), len(d))
print('bid/ask non-null:', d[['dlybid','dlyask']].notna().mean().round(3).to_dict())"
```
Expected: 1990 file spans Jan–Dec 1990. Note: pre-1993 NYSE/AMEX quoted bid/ask is sparse — that is expected; Task 7's Corwin–Schultz + cap-decile fills handle it (record the fill fractions there).

- [ ] **Step 5: Commit (script only; parquets are data)**

```bash
git add paper/pull_daily_history.py
git commit -m "paper: gated daily CRSP backfill 1990-2009 (ivol + pre-2011 spreads inputs)"
```

---

### Task 6: Daily FF3 factors + IVOL (`paper/fetch_ff_daily.py`, `paper/src/ivol.py`)

**Files:**
- Create: `paper/fetch_ff_daily.py`, `paper/src/ivol.py`
- Test: `paper/tests/test_ivol.py`
- Modify: `paper/config.py` (add `FF_DAILY`, `IVOL_MONTHLY`)

**Interfaces:**
- Consumes: Ken French "F-F_Research_Data_Factors_daily" zip (public, no WRDS; parse like `paper/fetch_long_legs.py` parses its zips); daily files `experiments/results/spreads/dsf_v2_*.parquet`.
- Produces: `paper/results/data/ff_daily.parquet` (index date; columns `mktrf, smb, hml, rf`, decimals); `paper/results/data/ivol_monthly.parquet` (`permno, date [month-end], ivol`) where `ivol` = std of FF3-regression residuals over the trailing 90 trading days (min 60), using only days ≤ month-end (PIT). `compute_ivol(daily, ff) -> pd.DataFrame` is the pure testable piece; Task 8 consumes the parquet.

- [ ] **Step 1: Write the failing test**

`paper/tests/test_ivol.py`:

```python
import numpy as np
import pandas as pd

from paper.src.ivol import compute_ivol


def _make_daily(n_days=130, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2020-01-01', periods=n_days)
    ff = pd.DataFrame({'mktrf': rng.normal(0, 0.01, n_days),
                       'smb': rng.normal(0, 0.005, n_days),
                       'hml': rng.normal(0, 0.005, n_days),
                       'rf': 0.0}, index=dates)
    # stock A: beta 1 on mkt + known idio sigma; stock B: half the idio sigma
    sig_a, sig_b = 0.02, 0.01
    ra = ff['mktrf'] + rng.normal(0, sig_a, n_days)
    rb = ff['mktrf'] + rng.normal(0, sig_b, n_days)
    daily = pd.concat([
        pd.DataFrame({'permno': 1, 'dlycaldt': dates, 'ret': ra.values}),
        pd.DataFrame({'permno': 2, 'dlycaldt': dates, 'ret': rb.values}),
    ], ignore_index=True)
    return daily, ff, sig_a, sig_b


def test_ivol_recovers_idio_sigma_ordering_and_scale():
    daily, ff, sig_a, sig_b = _make_daily()
    out = compute_ivol(daily, ff)
    last = out[out['date'] == out['date'].max()].set_index('permno')['ivol']
    assert last.loc[1] > last.loc[2]                     # ordering
    assert abs(last.loc[1] - sig_a) / sig_a < 0.30       # scale ~ sigma
    assert abs(last.loc[2] - sig_b) / sig_b < 0.30


def test_pit_no_future_days():
    daily, ff, *_ = _make_daily()
    out = compute_ivol(daily, ff)
    # first month-end has < 60 trailing days -> no row
    first_me = (pd.Series(pd.bdate_range('2020-01-01', periods=1))
                .iloc[0] + pd.offsets.MonthEnd(0))
    assert (out['date'] > first_me).all()
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest paper/tests/test_ivol.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `paper/src/ivol.py`**

```python
"""IVOL: trailing FF3-residual daily volatility, monthly snapshots (PIT).

ivol(permno, month-end m) = std of residuals from OLS of (ret - rf) on
(mktrf, smb, hml) over the trailing WINDOW=90 trading days ending at m
(minimum MIN_DAYS=60). Only days <= m are used. Low ivol = long side of the
low-volatility strategy (sign applied in signals.py).
"""
import numpy as np
import pandas as pd

WINDOW, MIN_DAYS = 90, 60


def _resid_std(y, X):
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return float((y - Xd @ beta).std(ddof=4))


def compute_ivol(daily, ff):
    d = daily.dropna(subset=['ret']).copy()
    d['dlycaldt'] = pd.to_datetime(d['dlycaldt'])
    ff = ff.sort_index()
    d = d.merge(ff, left_on='dlycaldt', right_index=True, how='inner')
    d['exret'] = d['ret'] - d['rf']
    d = d.sort_values(['permno', 'dlycaldt'])
    d['me_month'] = d['dlycaldt'] + pd.offsets.MonthEnd(0)

    rows = []
    for permno, g in d.groupby('permno', sort=False):
        g = g.reset_index(drop=True)
        ends = g.groupby('me_month').apply(lambda x: x.index[-1],
                                           include_groups=False)
        for m, i_end in ends.items():
            lo = i_end - WINDOW + 1
            if lo < 0 and i_end + 1 < MIN_DAYS:
                continue
            w = g.iloc[max(lo, 0): i_end + 1]
            if len(w) < MIN_DAYS:
                continue
            rows.append((permno, m,
                         _resid_std(w['exret'].values,
                                    w[['mktrf', 'smb', 'hml']].values)))
    return pd.DataFrame(rows, columns=['permno', 'date', 'ivol'])
```

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest paper/tests/test_ivol.py -q`
Expected: 2 passed.

- [ ] **Step 5: Implement `paper/fetch_ff_daily.py`** (mirror `fetch_long_legs.py`'s zip parsing; French file `F-F_Research_Data_Factors_daily_CSV.zip`; save decimals to `C.FF_DAILY`), then build the ivol panel:

`paper/fetch_ff_daily.py` core (adapt the existing `_read_french_zip` helper from `fetch_long_legs.py` if importable, else copy its ~15-line urllib+zipfile idiom):

```python
"""Fetch daily FF3 factors (public French library, no WRDS).
Usage: .venv/bin/python -m paper.fetch_ff_daily
Output: paper/results/data/ff_daily.parquet (mktrf, smb, hml, rf; decimals)
"""
import io
import os
import ssl
import sys
import urllib.request
import zipfile

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)
from paper import config as C                                   # noqa: E402

URL = ('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/'
       'ftp/F-F_Research_Data_Factors_daily_CSV.zip')


def main():
    ctx = ssl.create_default_context()
    raw = urllib.request.urlopen(URL, context=ctx).read()
    csv = zipfile.ZipFile(io.BytesIO(raw)).read(
        zipfile.ZipFile(io.BytesIO(raw)).namelist()[0]).decode('latin1')
    lines = [ln for ln in csv.splitlines()]
    start = next(i for i, ln in enumerate(lines)
                 if ln.strip()[:8].isdigit())
    end = next((i for i in range(start, len(lines))
                if not lines[i].strip()[:8].isdigit()), len(lines))
    df = pd.read_csv(io.StringIO('\n'.join(lines[start:end])), header=None,
                     names=['date', 'mktrf', 'smb', 'hml', 'rf'])
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.set_index('date').astype(float) / 100.0
    df.to_parquet(C.FF_DAILY)
    print(f'Saved {C.FF_DAILY}: {df.index.min().date()} -> '
          f'{df.index.max().date()} ({len(df)} days)')


if __name__ == '__main__':
    main()
```

Build step (append to `paper/src/ivol.py`):

```python
def build_ivol_panel():
    """Build ivol_monthly.parquet from all dsf_v2 files + ff_daily."""
    import glob
    from paper import config as C
    ff = pd.read_parquet(C.FF_DAILY)
    frames = []
    for f in sorted(glob.glob('experiments/results/spreads/dsf_v2_*.parquet')):
        cols = pd.read_parquet(f).columns
        use = ['permno', 'dlycaldt', 'dlyprc'] + \
              (['dlyret'] if 'dlyret' in cols else [])
        d = pd.read_parquet(f, columns=use)
        if 'dlyret' in d:
            d['ret'] = d['dlyret']
        else:                       # fall back to close-to-close price return
            d = d.sort_values(['permno', 'dlycaldt'])
            d['ret'] = (d.groupby('permno')['dlyprc']
                        .pct_change(fill_method=None))
        frames.append(d[['permno', 'dlycaldt', 'ret']])
    daily = pd.concat(frames, ignore_index=True)
    out = compute_ivol(daily, ff)
    out.to_parquet(C.IVOL_MONTHLY, index=False)
    print(f'Saved {C.IVOL_MONTHLY}: {len(out):,} rows, '
          f'{out.date.min().date()} -> {out.date.max().date()}')
```

Add to the `BANDING STUDY` block in `paper/config.py`:

```python
FF_DAILY = os.path.join(DATA_OUT, 'ff_daily.parquet')
IVOL_MONTHLY = os.path.join(DATA_OUT, 'ivol_monthly.parquet')
```

- [ ] **Step 6: Run the fetch, then the panel build (background — full CRSP daily is big)**

```bash
.venv/bin/python -m paper.fetch_ff_daily
nohup caffeinate -dims .venv/bin/python -c \
  "from paper.src.ivol import build_ivol_panel; build_ivol_panel()" \
  >> paper/results/banding_study/ivol_build.log 2>&1 &
```
On completion verify: coverage from ~1991 (or 2011 if Task 5 hasn't run — the build works either way and the driver restricts low-vol's sample to ivol coverage); median ivol ~1–3%/day×√? (raw daily residual std, ~0.01–0.04); top-1000 join coverage ≥ 90% post-1993.

- [ ] **Step 7: Commit**

```bash
git add paper/fetch_ff_daily.py paper/src/ivol.py paper/tests/test_ivol.py paper/config.py
git commit -m "paper: daily FF3 fetch + trailing-90d FF3-residual IVOL panel"
```

---

### Task 7: Extend the half-spread panel to 1991–2025

**Files:**
- Modify: none (pure rerun) — `paper/spreads.py` already builds from the glob.
- Output: regenerated `paper/results/s5/half_spreads.parquet` (+ its stats CSVs)

**Interfaces:**
- Consumes: all `dsf_v2_*.parquet` (now 1990–2025 after Task 5).
- Produces: `half_spreads.parquet` (permno, ym, qhs, cs_hs, hs) spanning 1990-01→2025-12. `hs` = winsorized [1, 200] bp quoted half-spread with Corwin–Schultz and cap-decile fills (registered S5 conventions, unchanged).

- [ ] **Step 1: Snapshot the current panel (regression baseline)**

```bash
cp paper/results/s5/half_spreads.parquet \
   paper/results/s5/half_spreads_pre_extension_$(date +%Y%m%d).parquet
```

- [ ] **Step 2: Rebuild**

`paper/spreads.py::build_spreads` returns the cached file if present — delete it first, then rerun:

```bash
rm paper/results/s5/half_spreads.parquet
nohup caffeinate -dims .venv/bin/python -m paper.spreads \
  >> paper/results/banding_study/spreads_rebuild.log 2>&1 &
```

- [ ] **Step 3: Regression gate — post-2011 rows must be unchanged**

```bash
.venv/bin/python -c "
import pandas as pd, glob
new = pd.read_parquet('paper/results/s5/half_spreads.parquet')
old = pd.read_parquet(sorted(glob.glob(
    'paper/results/s5/half_spreads_pre_extension_*.parquet'))[-1])
j = old.merge(new, on=['permno','ym'], suffixes=('_o','_n'))
err = (j['hs_o'] - j['hs_n']).abs().max()
print('rows old/new:', len(old), len(new), '| max |d hs| on overlap:', err)
assert err < 1e-12, 'REGRESSION: post-2011 spreads moved'"
```
Expected: max |Δ| = 0. If `paper/spreads.py` hard-codes its month range or `spread_verdict` inputs, extend the range only — the fill conventions must not change.

- [ ] **Step 4: Coverage report (pre-1993 quoted sparsity is expected)**

```bash
.venv/bin/python -c "
import pandas as pd
s = pd.read_parquet('paper/results/s5/half_spreads.parquet')
s['year'] = s['ym'].astype(str).str[:4].astype(int)
g = s.groupby('year').agg(qhs_cov=('qhs', lambda x: x.notna().mean()),
                          med_hs=('hs', 'median'))
print(g.loc[[1991, 1995, 2000, 2005, 2010, 2015, 2020, 2025]].round(3))"
```
Record the quoted-coverage-by-year table in the eventual report (no silent fills).

- [ ] **Step 5: Commit (log only — parquets are data, script unchanged)**

```bash
git add -A paper/results/banding_study/ 2>/dev/null || true
git commit -m "paper: half-spread panel extended to 1991-2025 (post-2011 bit-identical)" --allow-empty
```

---

### Task 8: Signal builders (`paper/src/signals.py`)

**Files:**
- Create: `paper/src/signals.py`
- Test: `paper/tests/test_signals.py`
- Modify: `paper/config.py` (add `STUDY_START`, `SPLIT_DATE`)

**Interfaces:**
- Consumes: `stocks.parquet`, `universe.top_n`, `pi_monthly.parquet` (Task 4), `label_states` (Task 3), `panel.parquet` (`vwretd`), `fundamentals.load_fundamentals()` (Task 2), `ivol_monthly.parquet` (Task 6), `execution.load_xsec()` (XGB).
- Produces: `build(strategy: str) -> pd.DataFrame` with EXACTLY the columns `(date, permno, me, pi, score, mom_12, ret_fwd, state3)`, sorted by (date, permno), one row per stock-month in the top-`UNIVERSE_N` universe, restricted to the strategy's valid sample. `STRATEGIES = ['momentum', 'reversal', 'lowvol', 'xgb', 'value', 'profitability', 'investment']`. Higher `score` = long side ALWAYS (signs pre-flipped). Tasks 9–11 consume `build`.

Signal definitions:
- `momentum`: score = `mom_12` (repo's 12-1 column; verified: the applied walk's `mom_12_1` comparator is `long_only_top(x, 'mom_12', ...)`).
- `reversal`: score = `−mom_1`.
- `lowvol`: score = `−ivol` (merge `ivol_monthly` on (permno, month-end); drop rows without ivol).
- `xgb`: reuse `execution.load_xsec()` frames (already the top-1000 universe with `score_pi`); rename `score_pi → score`; sample 2011–2025.
- `value`: B/M = `be` (latest `avail_date ≤ June-Y`, fiscal year ending in calendar Y−1) / ME(Dec Y−1); score constant July-Y…June-Y+1; higher B/M = long.
- `profitability`: score = `cop` (same June-Y timing); `op` variant behind a flag for robustness.
- `investment`: score = `−asset_growth` (same June-Y timing; low growth = long).

- [ ] **Step 1: Write the failing tests**

`paper/tests/test_signals.py`:

```python
import numpy as np
import pandas as pd
import pytest

from paper.src import signals

SCHEMA = ['date', 'permno', 'me', 'pi', 'score', 'mom_12', 'ret_fwd',
          'state3']


@pytest.mark.parametrize('strat', signals.STRATEGIES)
def test_schema_and_sorting(strat):
    x = signals.build(strat)
    assert list(x.columns) == SCHEMA
    assert x['date'].is_monotonic_increasing or (
        x.sort_values(['date', 'permno']).index == x.index).all()
    assert x['score'].notna().all()
    assert x['state3'].isin([0, 1, 2]).all()
    # pi constant within month
    assert (x.groupby('date')['pi'].nunique() == 1).all()


def test_momentum_reproduces_walk_comparator():
    # gate: momentum top-decile gross == the walk report's mom_12_1 series
    from paper.src import portfolio
    from paper import config as C
    x = signals.build('momentum')
    x11 = x[x['date'] >= '2011-01-01']
    r, _ = portfolio.long_only_top(x11, 'score', C.DECILE_FRAC)
    walk = pd.read_csv(C.RETURNS_CSV, parse_dates=['date'])
    walk = walk[(walk.rule == 'rule_r') & (walk.combo == 'DD')]
    # same months present
    assert set(r.index) == set(walk['date'])


def test_annual_signals_constant_between_junes():
    x = signals.build('value')
    one = x[x['permno'] == x['permno'].iloc[0]]
    yr = one[(one['date'] >= '2015-07-01') & (one['date'] <= '2016-06-30')]
    if len(yr) > 1:
        assert yr['score'].nunique() == 1


def test_reversal_sign():
    x = signals.build('reversal')
    m = x.merge(
        pd.read_parquet('paper/results/data/stocks.parquet',
                        columns=['permno', 'date', 'mom_1']),
        on=['permno', 'date'])
    assert np.corrcoef(m['score'], m['mom_1'])[0, 1] < -0.99
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `.venv/bin/python -m pytest paper/tests/test_signals.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `paper/src/signals.py`**

```python
"""Strategy signal builders -> common xsec schema for execution.simulate.

Every builder returns (date, permno, me, pi, score, mom_12, ret_fwd, state3)
over the top-UNIVERSE_N universe; HIGHER score = long side (signs
pre-flipped). Monthly cadence: momentum, reversal, lowvol, xgb. Annual
June-formation, held July..June: value, profitability, investment.
XGB sample is 2011-2025 (scores exist only for the walk years).
"""
import numpy as np
import pandas as pd

from paper import config as C
from paper.src import regime_labels, universe
from paper.src.fundamentals import load_fundamentals

STRATEGIES = ['momentum', 'reversal', 'lowvol', 'xgb', 'value',
              'profitability', 'investment']
SCHEMA = ['date', 'permno', 'me', 'pi', 'score', 'mom_12', 'ret_fwd',
          'state3']


def _base_universe():
    s = pd.read_parquet(C.STOCKS_PARQUET,
                        columns=['permno', 'date', 'me', 'mom_1', 'mom_12',
                                 'ret_fwd'])
    s['date'] = pd.to_datetime(s['date'])
    # top_n returns the FILTERED frame (all columns), verified 2026-07-15
    s = universe.top_n(s, C.UNIVERSE_N)
    return s[s['date'] >= C.STUDY_START]


def _attach_regime(x):
    pi = pd.read_parquet(C.PI_MONTHLY)
    pi['date'] = pd.to_datetime(pi['date'])
    panel = pd.read_parquet(C.PANEL_PARQUET, columns=['date', 'vwretd'])
    panel['date'] = pd.to_datetime(panel['date'])
    m = pi.merge(panel, on='date')
    st = regime_labels.label_states(m['date'], m['pi'].values,
                                    m['vwretd'].values)
    m = m.assign(state3=st.values)[['date', 'pi', 'state3']]
    out = x.merge(m, on='date', how='inner')
    return out


def _finalize(x):
    x = x.dropna(subset=['score', 'ret_fwd', 'me'])
    return (x[SCHEMA].sort_values(['date', 'permno'])
            .reset_index(drop=True))


def _annual_scores(field):
    """June-Y formation from fiscal years ending in calendar Y-1."""
    f = load_fundamentals()
    s = pd.read_parquet(C.STOCKS_PARQUET, columns=['permno', 'date', 'me'])
    s['date'] = pd.to_datetime(s['date'])
    dec = s[s['date'].dt.month == 12][['permno', 'date', 'me']]
    dec = dec.rename(columns={'me': 'me_dec'})
    dec['dec_year'] = dec['date'].dt.year

    f = f[f['datadate'].dt.year >= 1986].copy()
    f['form_year'] = f['datadate'].dt.year + 1        # June of Y
    # PIT guard: must be available by June-30 of form_year
    ok = f['avail_date'] <= (pd.to_datetime(f['form_year'].astype(str)
                                            + '-06-30'))
    f = f[ok].sort_values('datadate').drop_duplicates(
        ['permno', 'form_year'], keep='last')

    if field == 'value':
        f = f.merge(dec[['permno', 'dec_year', 'me_dec']],
                    left_on=['permno', f['form_year'] - 1],
                    right_on=['permno', 'dec_year'])
        f['sig'] = f['be'] / f['me_dec']
    elif field == 'profitability':
        f['sig'] = f['cop']
    else:                                             # investment
        f['sig'] = -f['asset_growth']
    return f[['permno', 'form_year', 'sig']].dropna()


def _annual(field):
    x = _base_universe()
    # formation-year key: July..Dec of Y and Jan..June of Y+1 use June-Y score
    yr = x['date'].dt.year - (x['date'].dt.month <= 6).astype(int)
    x = x.assign(form_year=yr)
    sc = _annual_scores(field)
    x = x.merge(sc, on=['permno', 'form_year'], how='inner')
    x = x.rename(columns={'sig': 'score'})
    return _finalize(_attach_regime(x))


def build(strategy):
    if strategy == 'momentum':
        x = _base_universe().assign(score=lambda d: d['mom_12'])
        return _finalize(_attach_regime(x))
    if strategy == 'reversal':
        x = _base_universe().assign(score=lambda d: -d['mom_1'])
        return _finalize(_attach_regime(x))
    if strategy == 'lowvol':
        iv = pd.read_parquet(C.IVOL_MONTHLY)
        iv['date'] = pd.to_datetime(iv['date'])
        x = _base_universe().merge(iv, on=['permno', 'date'], how='inner')
        x = x.assign(score=lambda d: -d['ivol'])
        return _finalize(_attach_regime(x))
    if strategy == 'xgb':
        from paper import execution as X
        x = X.load_xsec().rename(columns={'score_pi': 'score'})
        panel = pd.read_parquet(C.PANEL_PARQUET, columns=['date', 'vwretd'])
        panel['date'] = pd.to_datetime(panel['date'])
        m = (x[['date', 'pi']].drop_duplicates()
             .merge(panel, on='date'))
        st = regime_labels.label_states(m['date'], m['pi'].values,
                                        m['vwretd'].values)
        x = x.merge(m.assign(state3=st.values)[['date', 'state3']],
                    on='date')
        return _finalize(x)
    if strategy in ('value', 'profitability', 'investment'):
        return _annual(strategy)
    raise ValueError(strategy)
```

Add to the `BANDING STUDY` block in `paper/config.py`:

```python
STUDY_START = '1992-01-01'          # momentum needs 12m of 1991 history
SPLIT_DATE = '2011-01-01'           # in/out split for the selection honesty check
```

NOTE for the implementer: `universe.top_n`'s exact return shape (mask frame vs filtered frame) must be checked in `paper/src/universe.py` and `_base_universe` adjusted to match; `stage_walk` in `paper/walk_report.py` shows the canonical usage.

- [ ] **Step 4: Run tests, verify they pass**

Run: `.venv/bin/python -m pytest paper/tests/test_signals.py -q`
Expected: all pass. (`lowvol` skips months without ivol coverage by construction.)

- [ ] **Step 5: Evidence print — decile spreads have the right sign**

```bash
.venv/bin/python -c "
from paper.src import signals, portfolio
from paper import config as C
for s in signals.STRATEGIES:
    x = signals.build(s)
    r, _ = portfolio.long_only_top(x, 'score', C.DECILE_FRAC)
    b = portfolio.vw_benchmark(x)
    print(f'{s:14s} n_mo={len(r):4d} ann_act={(r-b).mean()*12:+.2%}')"
```
Expected: momentum/xgb positive; others recorded as found (no cherry-picking — these are the raw materials, not the result).

- [ ] **Step 6: Commit**

```bash
git add paper/src/signals.py paper/tests/test_signals.py paper/config.py
git commit -m "paper: 7 strategy signal builders on the common xsec schema"
```

---

### Task 9: 3-state band policy in `paper/execution.py`

**Files:**
- Modify: `paper/execution.py` (`_members()` — add one policy branch)
- Test: `paper/tests/test_regime3_band.py`

**Interfaces:**
- Consumes: frames that now carry `state3` (Task 8 schema).
- Produces: policy `('regime3_band', (E_calm, E_crash, E_recovery))` — NMV buy/hold semantics identical to `nmv_band` but the keep-band E switches on `g['state3'].iloc[0]`. Backward compatibility: `regime3_band(E, E, E)` ≡ `nmv_band(E, E)` ≡ static band.

- [ ] **Step 1: Write the failing test**

`paper/tests/test_regime3_band.py`:

```python
import numpy as np
import pandas as pd

from paper import execution as X


def _xsec_fixture():
    # 3 months x 10 stocks; scores fixed so ranks are stable; k = 1 (decile)
    # month 1 calm(0), month 2 crash(1), month 3 recovery(2)
    rows = []
    dates = pd.date_range('2020-01-31', periods=3, freq='ME')
    st3 = [0, 1, 2]
    for mi, d in enumerate(dates):
        for i in range(10):
            score = 10 - i if mi == 0 else (10 - i if i != 0 else 5.5)
            # month>=2: stock 0 (prev top) slips to rank 5 of 10 (50th pct)
            rows.append({'date': d, 'permno': 100 + i, 'me': 1.0,
                         'pi': 0.0 if mi == 0 else 0.9, 'score': score,
                         'mom_12': 0.0, 'ret_fwd': 0.0, 'state3': st3[mi]})
    return pd.DataFrame(rows)


def test_regime3_band_switches_on_state():
    x = _xsec_fixture()
    # month >= 2: stock 100 (score 5.5) sits at rank 5 of 10 = 50th pct.
    # E_crash = 70%: 0.50 <= 0.70 -> KEPT in the crash month;
    # E_recovery = 10%: 0.50 > 0.10 -> DROPPED in the recovery month.
    r, ledger = X.simulate(x, ('regime3_band', (10, 70, 10)),
                           score_col='score')
    # crash month: no forced exit of stock 100
    assert 100 not in set(
        ledger[(ledger['date'] == x['date'].unique()[1])
               & (ledger['cause'] == 'exit_rank')]['permno'])
    # recovery month: stock 100 (rank 6 > E=10%) is expelled
    assert 100 in set(
        ledger[(ledger['date'] == x['date'].unique()[2])
               & (ledger['cause'] == 'exit_rank')]['permno'])


def test_regime3_all_equal_reduces_to_nmv_band():
    x = _xsec_fixture()
    a, _ = X.simulate(x, ('regime3_band', (20, 20, 20)), score_col='score')
    b, _ = X.simulate(x, ('nmv_band', (20, 20)), score_col='score')
    assert np.allclose(a['gross'].values, b['gross'].values)
    assert np.allclose(a['turnover'].values, b['turnover'].values)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest paper/tests/test_regime3_band.py -q`
Expected: FAIL (`ValueError: regime3_band` from `_members`).

- [ ] **Step 3: Implement — add to `_members()` in `paper/execution.py`, right after the `nmv_band` branch**

```python
    if name == 'regime3_band':
        # 3-state NMV banding: keep-band E switches on state3
        # (0 calm, 1 crash, 2 recovery). Same buy/hold semantics as
        # nmv_band: enter at the decile, hold until rank falls out of E.
        ec, ecr, erec = arg
        s3 = int(g['state3'].iloc[0]) if 'state3' in g else 0
        e = {0: ec, 1: ecr, 2: erec}[s3] / 100.0
        keep = {p for p in prev if p in univ and rank_pct[p] <= e}
        return set(ranked[:k]) | keep
```

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest paper/tests/test_regime3_band.py -q`
Expected: 2 passed. Also run the existing suite to prove no regression:
`.venv/bin/python -m pytest paper/tests/ -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add paper/execution.py paper/tests/test_regime3_band.py
git commit -m "paper: regime3_band policy (E switches on calm/crash/recovery)"
```

---

### Task 10: The sweep driver (`paper/banding_study.py`)

**Files:**
- Create: `paper/banding_study.py`
- Test: `paper/tests/test_banding_study.py`
- Modify: `paper/config.py` (add grids/costs)

**Interfaces:**
- Consumes: `signals.build`, `execution.simulate`, `execution.paired_block_bootstrap`, `portfolio.vw_benchmark`, `half_spreads.parquet`, `ff_long_legs.parquet`, `data/ff_factors.parquet` (read-only), `metrics.ir`.
- Produces (in `C.BANDING_DIR`): `cells.csv` (one row per strategy × policy × params with all metrics), `best_cells.csv`, `delta_bootstrap.csv`, `frontier.csv`, `report.md` (G1–G3 verdicts + honesty check). Key functions: `run_strategy(strategy, workers) -> pd.DataFrame`, `price_cells(...)`, `evaluate_gates(cells) -> dict`, `main()` with `--smoke`.

Config additions (`BANDING STUDY` block):

```python
E_GRID = [10, 15, 20, 30, 40]                    # S6 continuity
STRESS_MULT = [3, 5]                             # panic-spread stress multiples
BS_STRATEGIES = ['momentum', 'reversal', 'lowvol', 'xgb', 'value',
                 'profitability', 'investment']
```

Policy families per strategy (all via `simulate`):
- `('nmv_band', (E, E))` for E in E_GRID — 5 diagonal cells = no-band (E=10 ≡ monthly, the S6 gate) + static bands.
- `('nmv_band', (Ec, Ep))` for all off-diagonal pairs — 20 two-state cells.
- `('regime3_band', (Ec, Ecr, Erec))` for the full 5³ minus the 25 already covered where Ecr == Erec (those equal the 2-state cells) — 100 three-state cells.
- Per strategy: 125 sims; 7 strategies ≈ 875 sims. Parallelize with a spawn pool over cells at `--workers 7` (mirror `pipeline.stage_folds`); full sweep ~30–60 min in background.

Pricing (per cell, from the trade ledger):
- Merge ledger with `half_spreads` on (permno, month): `cost_meas_t = Σ |dw|·hs/1e4`; missing hs → month cap-decile median (already the panel's fill), final fallback month median.
- Stress: multiply `hs` by `m ∈ STRESS_MULT` in months where `pi ≥ 0.5` only.
- Net actives: `net = gross − cost`; benchmark from `simulate(x, ('benchmark', None))` priced the same way (benchmark pays reconstitution, S6 convention).
- Cost columns: `net_ir_0` (= gross IR), `net_ir_meas`, `net_ir_stress3`, `net_ir_stress5`.

Metrics per cell: `gross_ir`, the 4 net IRs, `to_mo`, `to_calm`, `to_crash`, `to_recovery`, `breakeven_bp` (S6 formula vs benchmark turnover), `alpha_ff5lo`, `alpha_ff6lo` + NW t (net-of-measured-cost active regressed on long-leg factor set; `statsmodels` HAC lags=6).
Long-only factor set for alphas: `Mkt-RF` from `data/ff_factors.parquet` + excess long legs (`size, value, robust, conservative[, winner for FF6-LO]` from `ff_long_legs.parquet`, each minus RF).

Selection + gates:
- Per strategy: `static_best` = argmax `net_ir_meas` over diagonal; `regime_best` = argmax over ALL non-diagonal (2-state ∪ 3-state) cells.
- `delta_bootstrap.csv`: per strategy, paired CI of (regime_best − static_best) net active (measured cost), `paired_block_bootstrap`.
- G1: majority of strategies with Δ>0 and majority-winner CIs excluding 0. G2: Spearman(cost intensity, Δ net IR) > 0, where cost intensity = monthly-tier `to_mo × mean hs`. G3: IR-weighted combined book (weights ∝ max(net_ir_meas, 0) of each strategy under the given tier, renormalized; combined net active = Σ w·act): compare net Sharpe across tiers {monthly, static_best, regime_best}.
- Honesty check: re-select best cells on pre-`SPLIT_DATE` months only, evaluate on post-`SPLIT_DATE`; report both (XGB exempt — its sample starts at SPLIT_DATE).

- [ ] **Step 1: Write the failing smoke test**

`paper/tests/test_banding_study.py`:

```python
import numpy as np
import pandas as pd

from paper import banding_study as B


def _cells_fixture():
    # two strategies, diagonal + one off-diagonal cell each
    return pd.DataFrame({
        'strategy': ['a', 'a', 'a', 'b', 'b', 'b'],
        'family': ['diag', 'diag', 'regime2', 'diag', 'diag', 'regime2'],
        'params': ['(10, 10)', '(20, 20)', '(20, 10)',
                   '(10, 10)', '(20, 20)', '(20, 10)'],
        'net_ir_meas': [0.10, 0.20, 0.35, 0.30, 0.28, 0.29],
        'to_mo': [0.70, 0.50, 0.55, 0.20, 0.15, 0.16],
        'mean_hs_bp': [10.0, 10.0, 10.0, 1.0, 1.0, 1.0],
    })


def test_best_cell_selection():
    best = B.select_best(_cells_fixture())
    a = best[best['strategy'] == 'a'].iloc[0]
    assert a['static_params'] == '(20, 20)' and a['regime_params'] == '(20, 10)'
    assert np.isclose(a['delta_net_ir'], 0.15)


def test_gate2_spearman_direction():
    best = B.select_best(_cells_fixture())
    rho = B.gate2_spearman(best)
    # strategy a: high cost intensity, big delta; b: low, small -> rho = +1
    assert rho > 0
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest paper/tests/test_banding_study.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `paper/banding_study.py`**

Skeleton (the implementer fills the pricing/alpha helpers with the formulas above — every formula is specified; no invented conventions):

```python
"""Regime-conditional banding sweep: 7 strategies x band policies x costs.

Spec: docs/superpowers/specs/2026-07-15-regime-conditional-banding-study-design.md
Gates G1-G3 registered in paper/results/PAPER_NOTES.md (2026-07-15, pre-run).

Usage:
  .venv/bin/python -m paper.banding_study --stage sweep  --workers 7
  .venv/bin/python -m paper.banding_study --stage report
  (--smoke: 2 strategies x tiny grid)
Outputs in paper/results/banding_study/: cells.csv, best_cells.csv,
delta_bootstrap.csv, frontier.csv, report.md
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from paper import config as C                                   # noqa: E402
from paper import execution as X                                # noqa: E402
from paper.src import metrics as M                              # noqa: E402
from paper.src import signals                                   # noqa: E402


def cell_grid():
    diag = [('nmv_band', (e, e)) for e in C.E_GRID]
    two = [('nmv_band', (ec, ep))
           for ec in C.E_GRID for ep in C.E_GRID if ec != ep]
    three = [('regime3_band', (ec, ecr, erec))
             for ec, ecr, erec in itertools.product(C.E_GRID, repeat=3)
             if ecr != erec]
    return diag + two + three


def family(policy):
    name, arg = policy
    if name == 'nmv_band':
        return 'diag' if arg[0] == arg[1] else 'regime2'
    return 'regime3'


def select_best(cells):
    rows = []
    for s, g in cells.groupby('strategy'):
        d = g[g['family'] == 'diag']
        r = g[g['family'] != 'diag']
        bd = d.loc[d['net_ir_meas'].idxmax()]
        br = r.loc[r['net_ir_meas'].idxmax()]
        rows.append({'strategy': s,
                     'static_params': bd['params'],
                     'static_net_ir': bd['net_ir_meas'],
                     'regime_params': br['params'],
                     'regime_family': br['family'],
                     'regime_net_ir': br['net_ir_meas'],
                     'delta_net_ir': br['net_ir_meas'] - bd['net_ir_meas'],
                     'cost_intensity': (
                         g[g['params'] == '(10, 10)']['to_mo'].iloc[0]
                         * g['mean_hs_bp'].iloc[0])})
    return pd.DataFrame(rows)


def gate2_spearman(best):
    from scipy.stats import spearmanr
    return float(spearmanr(best['cost_intensity'],
                           best['delta_net_ir']).statistic)

# -- run_strategy / price_cells / alphas / frontier / report: implement per
#    the formulas in the plan (Task 10 header). Parallelism mirrors
#    pipeline.stage_folds (spawn pool, imap_unordered, resume-safe CSV
#    append keyed on (strategy, params)).
#    CONVENTION: cells.csv 'params' column = str(arg) of the policy tuple,
#    e.g. '(10, 10)' / '(40, 15)' / '(40, 40, 15)' - select_best and the
#    resume key both rely on this exact stringification.
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `.venv/bin/python -m pytest paper/tests/test_banding_study.py -q`
Expected: 2 passed.

- [ ] **Step 5: S6 reproduction gate (before any full sweep)**

XGB + `nmv_band` on the 2011–2025 sample must reproduce S6's grid: run the xgb strategy over the 5×5 `nmv_band` grid at a FLAT 10bp cost and diff against `paper/results/s6_banding/cells.csv` (`net_ir_10bp` column, tolerance 1e-3; small diffs from schema plumbing are a STOP-and-investigate).

- [ ] **Step 6: Smoke sweep**

Run: `.venv/bin/python -m paper.banding_study --stage sweep --smoke --workers 2`
Expected: completes in minutes; `cells.csv` written with 2 strategies × small grid; no NaN metrics.

- [ ] **Step 7: Commit**

```bash
git add paper/banding_study.py paper/tests/test_banding_study.py paper/config.py
git commit -m "paper: banding-study sweep driver (grid, pricing, gates, frontier)"
```

---

### Task 11: Full run, gates, report

**Files:**
- Output: `paper/results/banding_study/*` (cells.csv, best_cells.csv, delta_bootstrap.csv, frontier.csv, report.md)
- Modify: `paper/results/PAPER_NOTES.md` (results section, AFTER the pre-registered section)

- [ ] **Step 1: Full sweep in background**

```bash
nohup caffeinate -dims .venv/bin/python -m paper.banding_study \
  --stage sweep --workers 7 >> paper/results/banding_study/sweep.log 2>&1 &
```
(~875 sims; resume-safe append keyed on (strategy, params). Don't poll.)

- [ ] **Step 2: Report stage**

Run: `.venv/bin/python -m paper.banding_study --stage report`
Produces report.md with: per-strategy best cells + Δ + CIs; G1/G2/G3 HIT/MISS verdicts; stress-cost columns; the in/out honesty check; the quoted-spread coverage table from Task 7; explicit note that XGB's sample is 2011–2025.

- [ ] **Step 3: Verification (repo "strict" rules — evidence before claims)**

- Re-run the full test suite: `.venv/bin/python -m pytest paper/tests/ -q` → all pass.
- Spot-check 2 cells by hand (one diag, one regime3): recompute net IR from `simulate` output + ledger × spreads in a throwaway snippet; must match cells.csv to 1e-9.
- Red-flag scan: any strategy whose monthly-tier gross IR moved vs Task 8's evidence print; suspiciously clean gate sweeps (all 7 HIT with tight CIs) → stop and recheck before reporting.

- [ ] **Step 4: Append results to PAPER_NOTES.md + commit everything**

```bash
git add paper/results/banding_study/ paper/results/PAPER_NOTES.md
git commit -m "paper: banding study results — G1-G3 verdicts (see report.md)"
```

- [ ] **Step 5: Surface to Gilad**

Summarize: gate verdicts, the headline frontier table, the stress-cost survival of the regime band, and (for the Marc reply, due ~Aug 2) whether "wide-in-crash / tight-in-recovery" is where the frontier lives.

---

## Self-review + verification notes (issues found and fixed inline, 2026-07-15)

- Spec §5 WRDS gate → encoded twice (Global Constraints + Task 5 banner).
- Spec §9 said 1990–2025; data reality is stocks from 1991-01 → STUDY_START 1992-01 (momentum lookback). Recorded in Data reality + config.
- Spec §8 trading-diversification netting in the combined book only — the IR-weighted frontier operates on active returns (already netted at the book level); the MVE-with-netting variant is folded into the frontier robustness row, not per-strategy tables.
- XGB 2011–2025 asymmetry appears in: Data reality, Task 8 builder, Task 10 honesty-check exemption, Task 11 report note.

Verification pass against the repo (subagent workflow hit the session limit; verified inline by hand instead):
- **Look-ahead bug fixed in Task 4**: original draft inferred pi on the full 1970–2025 panel (FFBS smoother would see the future); now infers on data < Y+1 only, matching `_walk_year`.
- **Sign convention fixed in Task 4**: `crisis_signs` is degenerate for 1991–2000 training windows (CRISIS_WINDOWS start in 2000, verified in root config.py:52); switched to the DD-anchored `signs=[-1]` per `selection.run_cell`'s documented convention. Walk-recipe deviations (history floor, sign rule, 3-vs-50 seeds) quantified by a mandatory 2011-overlap check (≥10/12 label agreement or STOP).
- `hmm.fit_pi` verified: returns seed-mean pi vector (`.mean(axis=0)`) — consumed as-is.
- `universe.top_n` verified: returns the filtered frame (not a mask) — `_base_universe` simplified.
- statsmodels 0.14.6 / scipy 1.17.1 present in .venv; `FLAT_BPS` exists in execution.py; no hardcoded date ranges in spreads.py.
- Fixture arithmetic hand-executed: test_crash_recovery (0.005/−0.015/0.02, shares→1.0 ✓), test_fundamentals (BE 105/116, cop_raw 113 ✓), test_regime3_band (rank-5/50th-pct comment corrected; E=70 keeps, E=10 expels ✓; (E,E,E)≡nmv_band(E,E) ✓).
- Cell counts consistent: 5 diag + 20 two-state + 100 three-state = 125/strategy, 875 total.
- `params` stringification pinned to `str(tuple)` (`'(10, 10)'`) in fixture, select_best, and skeleton convention note.
