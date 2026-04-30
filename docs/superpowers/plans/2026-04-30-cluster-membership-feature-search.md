# Picked-Stock Cluster Feature Search Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cluster months by the long-leg pick fingerprint (raw mom level + shape + dispersion of the picks), then identify which month-level features best predict cluster membership — yielding the natural "group-by" variables for §5.2.

**Architecture:** Four small TDD-tested helpers (`pi_panic_freq`, `past_strategy_sharpe`, `cross_section_skew`, `picked_stock_fingerprint`) plus a single orchestration script (`picked_stock_cluster_search.py`) that runs the cluster sweep, predictor-feature build, univariate ranking, multinomial L1 logistic with LOO-CV, depth-3 decision tree, and per-cluster summary.

**Tech Stack:** Python, pandas, numpy, scikit-learn (KMeans, LogisticRegression, DecisionTreeClassifier, f_classif), pytest.

---

## File Structure

**New helpers (TDD-tested):**
- `scripts/cluster_feature_search_helpers.py` — four primitives:
  - `pi_panic_freq(pi_series, date, n_months, threshold=0.5)` — fraction of past N months with π > threshold (excluding the current month)
  - `past_strategy_sharpe(returns_series, date, n_months)` — annualised Sharpe over the prior N months
  - `cross_section_skew(monthly_panel, mom_cols)` — per-month skewness across stocks at each horizon
  - `picked_stock_fingerprint(picks_df, mom_panel, mom_cols)` — 15-d per-month fingerprint of long-leg picks

**Tests:**
- `scripts/test_cluster_feature_search_helpers.py`

**Main analysis:**
- `scripts/picked_stock_cluster_search.py` — orchestration:
  1. Load picks + cross-section panel + returns + π
  2. Compute 15-d fingerprint per month (helper)
  3. Cluster sweep K∈{2,3,4,5}, 6-seed stability, silhouette, pick K
  4. Build 14-d predictor feature panel
  5. Univariate ANOVA + multinomial L1 logistic (LOO-CV) + depth-3 decision tree
  6. Per-cluster summary + headline verdict

**Outputs (in `results/thesis/`):**
- `picked_stock_cluster_sweep.csv` — K vs silhouette + ARI
- `picked_stock_cluster_labels.csv` — month-level cluster labels at chosen K
- `picked_stock_cluster_centroids.csv` — fingerprint centroid per cluster
- `picked_stock_feature_panel.csv` — per-month predictor feature panel
- `picked_stock_feature_ranking.csv` — univariate F-stat per feature
- `picked_stock_logistic_coefficients.csv` — L1 multinomial coefficients
- `picked_stock_decision_tree.txt` — depth-3 tree readout
- `picked_stock_cluster_summary.csv` — per-cluster mean ± std for fingerprint + predictor features

---

## Task 1: TDD — `pi_panic_freq` helper

**Files:**
- Create: `scripts/cluster_feature_search_helpers.py`
- Create: `scripts/test_cluster_feature_search_helpers.py`

- [ ] **Step 1: Write the failing test**

```python
# scripts/test_cluster_feature_search_helpers.py
import pandas as pd
import numpy as np
import pytest

from cluster_feature_search_helpers import pi_panic_freq


def test_pi_panic_freq_all_panic():
    dates = pd.date_range('2020-01-31', periods=13, freq='ME')
    pi = pd.Series([1.0] * 12 + [0.0], index=dates)
    # Past 12 months all panic -> 1.0
    result = pi_panic_freq(pi, dates[-1], n_months=12)
    assert result == pytest.approx(1.0)


def test_pi_panic_freq_half():
    dates = pd.date_range('2020-01-31', periods=13, freq='ME')
    # 6 panic + 6 calm in past 12 months
    pi = pd.Series([1.0]*6 + [0.0]*6 + [0.5], index=dates)
    result = pi_panic_freq(pi, dates[-1], n_months=12)
    assert result == pytest.approx(0.5)


def test_pi_panic_freq_excludes_current():
    dates = pd.date_range('2020-01-31', periods=13, freq='ME')
    # Past 12 = 0, current = 1 — should return 0.0
    pi = pd.Series([0.0]*12 + [1.0], index=dates)
    result = pi_panic_freq(pi, dates[-1], n_months=12)
    assert result == pytest.approx(0.0)


def test_pi_panic_freq_insufficient_history():
    dates = pd.date_range('2020-01-31', periods=5, freq='ME')
    pi = pd.Series([0.5] * 5, index=dates)
    result = pi_panic_freq(pi, dates[-1], n_months=12)
    assert pd.isna(result)


def test_pi_panic_freq_custom_threshold():
    dates = pd.date_range('2020-01-31', periods=13, freq='ME')
    # All values 0.6 — at threshold 0.5 -> all panic; at threshold 0.7 -> none
    pi = pd.Series([0.6] * 13, index=dates)
    assert pi_panic_freq(pi, dates[-1], n_months=12, threshold=0.5) == pytest.approx(1.0)
    assert pi_panic_freq(pi, dates[-1], n_months=12, threshold=0.7) == pytest.approx(0.0)
```

- [ ] **Step 2: Run test, verify it fails**

```bash
cd /Users/giladgang/momentum_regime && pytest scripts/test_cluster_feature_search_helpers.py -v
```

Expected: ImportError because `cluster_feature_search_helpers` doesn't exist.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/cluster_feature_search_helpers.py
"""Feature-search primitives for picked-stock cluster prediction.

Helpers:
  pi_panic_freq(pi_series, date, n_months, threshold=0.5)
  past_strategy_sharpe(returns_series, date, n_months)
  cross_section_skew(monthly_panel, mom_cols)
  picked_stock_fingerprint(picks_df, mom_panel, mom_cols)
"""
import numpy as np
import pandas as pd


def pi_panic_freq(pi_series, date, n_months, threshold=0.5):
    """Fraction of past n_months months with pi > threshold (excluding `date`).

    Parameters
    ----------
    pi_series : pd.Series indexed by date
    date : pd.Timestamp or compatible
    n_months : int
    threshold : float (default 0.5)

    Returns
    -------
    float in [0, 1] or NaN if insufficient history.
    """
    if date not in pi_series.index:
        return float('nan')
    pos = pi_series.index.get_loc(date)
    if pos < n_months:
        return float('nan')
    window = pi_series.iloc[pos - n_months:pos]
    return float((window > threshold).mean())
```

- [ ] **Step 4: Run test, verify it passes**

```bash
cd /Users/giladgang/momentum_regime && pytest scripts/test_cluster_feature_search_helpers.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/cluster_feature_search_helpers.py scripts/test_cluster_feature_search_helpers.py
git commit -m "$(cat <<'EOF'
feature-search: pi_panic_freq helper with TDD tests

Fraction of past N months with pi > threshold. Encodes "regime in past
period" while respecting that pi is binary-ish in practice.

5/5 tests pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: TDD — `past_strategy_sharpe` helper

**Files:**
- Modify: `scripts/cluster_feature_search_helpers.py`
- Modify: `scripts/test_cluster_feature_search_helpers.py`

- [ ] **Step 1: Append failing tests**

```python
# Append to scripts/test_cluster_feature_search_helpers.py
from cluster_feature_search_helpers import past_strategy_sharpe


def test_past_strategy_sharpe_zero_std_returns_nan():
    dates = pd.date_range('2020-01-31', periods=13, freq='ME')
    rets = pd.Series([0.01] * 13, index=dates)
    result = past_strategy_sharpe(rets, dates[-1], n_months=12)
    assert pd.isna(result)


def test_past_strategy_sharpe_known_value():
    dates = pd.date_range('2020-01-31', periods=13, freq='ME')
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.01, 0.05, 13), index=dates)
    window = rets.iloc[:12]
    expected = (window.mean() / window.std(ddof=1)) * np.sqrt(12)
    result = past_strategy_sharpe(rets, dates[-1], n_months=12)
    assert result == pytest.approx(expected)


def test_past_strategy_sharpe_excludes_current():
    dates = pd.date_range('2020-01-31', periods=13, freq='ME')
    # 12 months of small variance + huge spike at current
    rng = np.random.default_rng(1)
    rets = pd.Series(np.r_[rng.normal(0.01, 0.02, 12), 10.0], index=dates)
    window = rets.iloc[:12]
    expected = (window.mean() / window.std(ddof=1)) * np.sqrt(12)
    result = past_strategy_sharpe(rets, dates[-1], n_months=12)
    assert result == pytest.approx(expected)


def test_past_strategy_sharpe_insufficient_history():
    dates = pd.date_range('2020-01-31', periods=5, freq='ME')
    rets = pd.Series(np.random.default_rng(2).standard_normal(5), index=dates)
    result = past_strategy_sharpe(rets, dates[-1], n_months=12)
    assert pd.isna(result)
```

- [ ] **Step 2: Run, verify failure**

```bash
cd /Users/giladgang/momentum_regime && pytest scripts/test_cluster_feature_search_helpers.py -v
```

Expected: ImportError on `past_strategy_sharpe`.

- [ ] **Step 3: Append implementation**

```python
# Append to scripts/cluster_feature_search_helpers.py
def past_strategy_sharpe(returns_series, date, n_months):
    """Annualised Sharpe over past n_months months (excluding `date`).

    Uses sample std (ddof=1) and sqrt(12) annualisation, matching the
    project's bootstrap_helpers convention.

    Returns
    -------
    float Sharpe ratio or NaN if insufficient history or zero std.
    """
    if date not in returns_series.index:
        return float('nan')
    pos = returns_series.index.get_loc(date)
    if pos < n_months:
        return float('nan')
    window = returns_series.iloc[pos - n_months:pos]
    sd = window.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float('nan')
    return float((window.mean() / sd) * np.sqrt(12))
```

- [ ] **Step 4: Run, verify pass**

```bash
cd /Users/giladgang/momentum_regime && pytest scripts/test_cluster_feature_search_helpers.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/cluster_feature_search_helpers.py scripts/test_cluster_feature_search_helpers.py
git commit -m "$(cat <<'EOF'
feature-search: past_strategy_sharpe helper with TDD tests

Annualised Sharpe over the prior N months (sample std ddof=1, sqrt(12)).
Used as a candidate predictor: 'has the strategy been winning lately?'

9/9 tests pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: TDD — `cross_section_skew` helper

**Files:**
- Modify: `scripts/cluster_feature_search_helpers.py`
- Modify: `scripts/test_cluster_feature_search_helpers.py`

- [ ] **Step 1: Append failing tests**

```python
# Append to scripts/test_cluster_feature_search_helpers.py
from cluster_feature_search_helpers import cross_section_skew


def test_cross_section_skew_returns_dataframe():
    rng = np.random.default_rng(0)
    rows = []
    for d in pd.date_range('2020-01-31', periods=3, freq='ME'):
        for s in range(5):
            row = {'date': d, 'permno': s}
            for h in range(1, 5):
                row[f'mom_{h}'] = rng.standard_normal()
            rows.append(row)
    panel = pd.DataFrame(rows)
    skew = cross_section_skew(panel, [f'mom_{h}' for h in range(1, 5)])
    assert isinstance(skew, pd.DataFrame)
    assert skew.shape == (3, 4)
    assert list(skew.columns) == ['mom_1', 'mom_2', 'mom_3', 'mom_4']


def test_cross_section_skew_zero_for_symmetric():
    rows = []
    for d in pd.date_range('2020-01-31', periods=2, freq='ME'):
        for v in [-2, -1, 0, 1, 2]:
            rows.append({'date': d, 'permno': v + 100, 'mom_1': v})
    panel = pd.DataFrame(rows)
    skew = cross_section_skew(panel, ['mom_1'])
    assert all(abs(s) < 1e-9 for s in skew['mom_1'])


def test_cross_section_skew_positive_for_right_tail():
    rows = []
    for d in pd.date_range('2020-01-31', periods=1, freq='ME'):
        for v in [0, 0, 0, 0, 10]:
            rows.append({'date': d, 'permno': v, 'mom_1': v})
    panel = pd.DataFrame(rows)
    skew = cross_section_skew(panel, ['mom_1'])
    assert skew['mom_1'].iloc[0] > 0
```

- [ ] **Step 2: Run, verify failure**

```bash
cd /Users/giladgang/momentum_regime && pytest scripts/test_cluster_feature_search_helpers.py -v
```

Expected: ImportError on `cross_section_skew`.

- [ ] **Step 3: Append implementation**

```python
# Append to scripts/cluster_feature_search_helpers.py
def cross_section_skew(monthly_panel, mom_cols):
    """Per-month skewness across stocks at each momentum horizon.

    Parameters
    ----------
    monthly_panel : DataFrame with column 'date' + each of mom_cols
    mom_cols : list of column names

    Returns
    -------
    DataFrame indexed by date, columns = mom_cols, values = skewness
    """
    return monthly_panel.groupby('date')[mom_cols].skew()
```

- [ ] **Step 4: Run, verify pass**

```bash
cd /Users/giladgang/momentum_regime && pytest scripts/test_cluster_feature_search_helpers.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/cluster_feature_search_helpers.py scripts/test_cluster_feature_search_helpers.py
git commit -m "$(cat <<'EOF'
feature-search: cross_section_skew helper with TDD tests

Per-month cross-sectional skewness across stocks at each horizon.
Captures asymmetry in the cross-section beyond mean and std.

12/12 tests pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: TDD — `picked_stock_fingerprint` helper

**Files:**
- Modify: `scripts/cluster_feature_search_helpers.py`
- Modify: `scripts/test_cluster_feature_search_helpers.py`

- [ ] **Step 1: Append failing tests**

```python
# Append to scripts/test_cluster_feature_search_helpers.py
from cluster_feature_search_helpers import picked_stock_fingerprint


def test_picked_stock_fingerprint_columns_and_index():
    # 2 months, full panel of 5 stocks; pick 3 of 5 each month.
    dates = pd.date_range('2020-01-31', periods=2, freq='ME')
    rng = np.random.default_rng(0)
    panel_rows = []
    for d in dates:
        for permno in range(5):
            row = {'date': d, 'permno': permno}
            for h in range(1, 13):
                row[f'mom_{h}'] = rng.standard_normal()
            panel_rows.append(row)
    panel = pd.DataFrame(panel_rows)
    picks = pd.DataFrame([
        {'date': d, 'permno': permno}
        for d in dates for permno in [0, 1, 2]
    ])
    mom_cols = [f'mom_{h}' for h in range(1, 13)]
    fp = picked_stock_fingerprint(picks, panel, mom_cols)
    assert list(fp.index) == list(dates)
    expected_cols = (
        [f'pick_mom_{h}' for h in range(1, 13)]
        + ['pick_disp_short', 'pick_disp_mid', 'pick_disp_long']
    )
    assert list(fp.columns) == expected_cols


def test_picked_stock_fingerprint_mean_correct():
    # Single month, 3 picks, known mom values.
    d = pd.Timestamp('2020-01-31')
    panel = pd.DataFrame([
        {'date': d, 'permno': 0, **{f'mom_{h}': 1.0 for h in range(1, 13)}},
        {'date': d, 'permno': 1, **{f'mom_{h}': 2.0 for h in range(1, 13)}},
        {'date': d, 'permno': 2, **{f'mom_{h}': 3.0 for h in range(1, 13)}},
        {'date': d, 'permno': 3, **{f'mom_{h}': 99.0 for h in range(1, 13)}},  # not picked
    ])
    picks = pd.DataFrame([{'date': d, 'permno': p} for p in [0, 1, 2]])
    mom_cols = [f'mom_{h}' for h in range(1, 13)]
    fp = picked_stock_fingerprint(picks, panel, mom_cols)
    # Mean of picks at every horizon = 2.0
    for h in range(1, 13):
        assert fp[f'pick_mom_{h}'].iloc[0] == pytest.approx(2.0)


def test_picked_stock_fingerprint_dispersion_correct():
    # Picks have known std at short horizon; mid and long different.
    d = pd.Timestamp('2020-01-31')
    panel_rows = []
    for permno, vals in [
        (0, [1, 1, 1, 1,  10, 10, 10, 10,  0, 0, 0, 0]),
        (1, [3, 3, 3, 3,  20, 20, 20, 20,  1, 1, 1, 1]),
    ]:
        row = {'date': d, 'permno': permno}
        for h, v in zip(range(1, 13), vals):
            row[f'mom_{h}'] = float(v)
        panel_rows.append(row)
    panel = pd.DataFrame(panel_rows)
    picks = pd.DataFrame([{'date': d, 'permno': p} for p in [0, 1]])
    mom_cols = [f'mom_{h}' for h in range(1, 13)]
    fp = picked_stock_fingerprint(picks, panel, mom_cols)
    # short tertile (h=1..4): values are [(1,3),(1,3),(1,3),(1,3)] -> std per horizon = sqrt(2),
    #   mean of those 4 stds = sqrt(2)
    assert fp['pick_disp_short'].iloc[0] == pytest.approx(np.sqrt(2.0))
    # mid tertile (h=5..8): values [(10,20)] -> std per horizon = sqrt(50),
    #   mean = sqrt(50)
    assert fp['pick_disp_mid'].iloc[0] == pytest.approx(np.sqrt(50.0))
    # long tertile (h=9..12): values [(0,1)] -> std per horizon = sqrt(0.5),
    #   mean = sqrt(0.5)
    assert fp['pick_disp_long'].iloc[0] == pytest.approx(np.sqrt(0.5))
```

- [ ] **Step 2: Run, verify failure**

```bash
cd /Users/giladgang/momentum_regime && pytest scripts/test_cluster_feature_search_helpers.py -v
```

Expected: ImportError on `picked_stock_fingerprint`.

- [ ] **Step 3: Append implementation**

```python
# Append to scripts/cluster_feature_search_helpers.py
SHORT_H = [1, 2, 3, 4]
MID_H   = [5, 6, 7, 8]
LONG_H  = [9, 10, 11, 12]


def picked_stock_fingerprint(picks_df, mom_panel, mom_cols):
    """Per-month 15-d fingerprint of long-leg picks.

    Parameters
    ----------
    picks_df : DataFrame with columns 'date', 'permno' (long-leg picks).
    mom_panel : DataFrame with columns 'date', 'permno', and each of mom_cols.
    mom_cols : list of length 12 — momentum horizon columns in order
        (mom_1, mom_2, ..., mom_12).

    Returns
    -------
    DataFrame indexed by date with 15 columns:
      pick_mom_1 .. pick_mom_12  : mean of picks at each horizon
      pick_disp_short            : mean of per-horizon std at horizons 1..4
      pick_disp_mid              : mean of per-horizon std at horizons 5..8
      pick_disp_long             : mean of per-horizon std at horizons 9..12
    """
    if len(mom_cols) != 12:
        raise ValueError(f'expected 12 mom columns, got {len(mom_cols)}')
    merged = picks_df[['date', 'permno']].merge(
        mom_panel[['date', 'permno'] + list(mom_cols)],
        on=['date', 'permno'], how='inner',
    )
    grp = merged.groupby('date')[list(mom_cols)]
    means = grp.mean()
    stds = grp.std(ddof=1)

    fp = means.rename(columns={c: f'pick_{c}' for c in mom_cols})

    short_cols = [mom_cols[h - 1] for h in SHORT_H]
    mid_cols   = [mom_cols[h - 1] for h in MID_H]
    long_cols  = [mom_cols[h - 1] for h in LONG_H]
    fp['pick_disp_short'] = stds[short_cols].mean(axis=1)
    fp['pick_disp_mid']   = stds[mid_cols].mean(axis=1)
    fp['pick_disp_long']  = stds[long_cols].mean(axis=1)
    return fp
```

- [ ] **Step 4: Run, verify pass**

```bash
cd /Users/giladgang/momentum_regime && pytest scripts/test_cluster_feature_search_helpers.py -v
```

Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/cluster_feature_search_helpers.py scripts/test_cluster_feature_search_helpers.py
git commit -m "$(cat <<'EOF'
feature-search: picked_stock_fingerprint helper with TDD tests

Per-month 15-d fingerprint of long-leg picks: mean momentum at each of
12 horizons + std-based dispersion at short/mid/long tertiles. Captures
both level and shape of the basket (vs prior z-scored 'shape only').

15/15 tests pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Cluster sweep + pick K + save labels and centroids

**Files:**
- Create: `scripts/picked_stock_cluster_search.py`

- [ ] **Step 1: Write the script (sweep + label save)**

```python
# scripts/picked_stock_cluster_search.py
"""Picked-stock cluster + feature search.

Stage 1 (this task): cluster months on the 15-d picked-stock fingerprint;
sweep K=2..5 with 6-seed stability and silhouette; pick smallest stable K
with sane size distribution; save labels + centroids + sweep CSV.

Stages 2-3 (later tasks) extend this script with predictor features,
ANOVA, multinomial logistic, decision tree, and per-cluster summary.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_feature_search_helpers import picked_stock_fingerprint

RES_DIR = 'results/thesis'
HORIZONS = list(range(1, 13))
MOM_COLS = [f'mom_{h}' for h in HORIZONS]
K_GRID = [2, 3, 4, 5]
STABILITY_SEEDS = [42, 123, 456, 789, 1011, 1213]
N_INIT = 20
ARI_THRESHOLD = 0.90
MIN_CLUSTER_SIZE_FRAC = 0.05  # smallest cluster must be ≥5% of months


def cluster_sweep(X_std):
    """Run KMeans for each K in K_GRID, with 6-seed stability + silhouette."""
    rows = []
    labels_by_k = {}
    for k in K_GRID:
        fits = []
        sils = []
        for s in STABILITY_SEEDS:
            km = KMeans(n_clusters=k, random_state=s, n_init=N_INIT).fit(X_std)
            fits.append(km.labels_)
            sils.append(silhouette_score(X_std, km.labels_))
        pair_aris = []
        for i in range(len(STABILITY_SEEDS)):
            for j in range(i + 1, len(STABILITY_SEEDS)):
                pair_aris.append(adjusted_rand_score(fits[i], fits[j]))
        sizes = sorted(np.bincount(fits[0], minlength=k).tolist())
        rows.append({
            'k': k,
            'mean_silhouette': round(float(np.mean(sils)), 4),
            'mean_ari': round(float(np.mean(pair_aris)), 4),
            'min_size_frac': round(min(sizes) / len(X_std), 4),
            'sizes_seed42': str(tuple(sizes)),
            'stable': bool(np.mean(pair_aris) >= ARI_THRESHOLD),
        })
        labels_by_k[k] = fits[0]
        print(f'  k={k}: silhouette={rows[-1]["mean_silhouette"]:.3f}, '
              f'ari={rows[-1]["mean_ari"]:.3f}, sizes={sizes}, '
              f'stable={rows[-1]["stable"]}', flush=True)
    return pd.DataFrame(rows), labels_by_k


def pick_k(sweep_df):
    """Smallest K that is stable AND whose smallest cluster is >=5%."""
    eligible = sweep_df[
        sweep_df['stable']
        & (sweep_df['min_size_frac'] >= MIN_CLUSTER_SIZE_FRAC)
    ].sort_values('k')
    if len(eligible) == 0:
        # Fall back to most stable K
        return int(sweep_df.sort_values('mean_ari', ascending=False).iloc[0]['k'])
    return int(eligible.iloc[0]['k'])


def main():
    print('Loading data...', flush=True)
    picks = pd.read_csv(f'{RES_DIR}/rule_path_labels.csv',
                        parse_dates=['date'])[['date', 'permno']]
    with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
        art = pickle.load(f)
    panel = art['test'].copy()
    panel['date'] = pd.to_datetime(panel['date'])

    print('Computing fingerprint...', flush=True)
    fp = picked_stock_fingerprint(picks, panel, MOM_COLS)
    fp = fp.dropna()
    print(f'  fingerprint shape: {fp.shape}', flush=True)

    X = fp.values.astype(np.float64)
    X_std = StandardScaler().fit_transform(X)

    print('\n=== Cluster sweep K=2..5 ===', flush=True)
    sweep_df, labels_by_k = cluster_sweep(X_std)
    sweep_df.to_csv(f'{RES_DIR}/picked_stock_cluster_sweep.csv', index=False)
    print(f'\nSaved: {RES_DIR}/picked_stock_cluster_sweep.csv', flush=True)
    print(sweep_df.to_string(index=False))

    chosen_k = pick_k(sweep_df)
    print(f'\n>>> CHOSEN K = {chosen_k}', flush=True)

    labels = labels_by_k[chosen_k]
    label_df = pd.DataFrame({
        'date': fp.index,
        'cluster': labels,
    })
    label_df.to_csv(f'{RES_DIR}/picked_stock_cluster_labels.csv', index=False)
    print(f'Saved: {RES_DIR}/picked_stock_cluster_labels.csv', flush=True)

    # Centroids on raw (un-standardised) fingerprint
    centroids = fp.assign(cluster=labels).groupby('cluster').mean()
    centroids['n_months'] = pd.Series(labels).value_counts().sort_index().values
    centroids.to_csv(f'{RES_DIR}/picked_stock_cluster_centroids.csv')
    print(f'Saved: {RES_DIR}/picked_stock_cluster_centroids.csv', flush=True)
    print('\nCentroids (raw mom) per cluster:')
    print(centroids.round(3).to_string())


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run**

```bash
cd /Users/giladgang/momentum_regime && python scripts/picked_stock_cluster_search.py
```

Expected: prints sweep table, chosen K, centroids; writes 3 CSVs to `results/thesis/`.

- [ ] **Step 3: Inspect**

```bash
cat /Users/giladgang/momentum_regime/results/thesis/picked_stock_cluster_sweep.csv
head /Users/giladgang/momentum_regime/results/thesis/picked_stock_cluster_centroids.csv
wc -l /Users/giladgang/momentum_regime/results/thesis/picked_stock_cluster_labels.csv
```

Expected:
- Sweep CSV has 4 rows (K=2..5), columns include `mean_ari`, `mean_silhouette`, `stable`.
- Centroids CSV has K rows, 16 columns (15 fingerprint + n_months).
- Labels CSV has ≈170 rows (one per month).

- [ ] **Step 4: Commit**

```bash
git add scripts/picked_stock_cluster_search.py \
        results/thesis/picked_stock_cluster_sweep.csv \
        results/thesis/picked_stock_cluster_labels.csv \
        results/thesis/picked_stock_cluster_centroids.csv
git commit -m "$(cat <<'EOF'
picked-stock cluster: K sweep with stability gate + chosen-K labels

Clusters months on the 15-d picked-stock fingerprint (raw mom level + shape
+ tertile dispersion). Sweep K=2..5 with 6-seed ARI stability and
silhouette; pick smallest stable K with min-cluster-size >=5%.

Outputs sweep CSV, per-month labels, per-cluster centroids.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Build predictor feature panel

**Files:**
- Modify: `scripts/picked_stock_cluster_search.py`

- [ ] **Step 1: Append the feature-panel builder**

Insert this function after the imports (above `cluster_sweep`) and update `main()` as shown in Step 2.

```python
# Append after imports in scripts/picked_stock_cluster_search.py
from cluster_feature_search_helpers import (
    pi_panic_freq, past_strategy_sharpe, cross_section_skew,
)

SHORT, MID, LONG = [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]


def tertile_mean(df_row_or_frame, group):
    cols = [f'mom_{h}' for h in group]
    return df_row_or_frame[cols].mean(axis=1)


def build_feature_panel(panel, m2_returns, dates):
    """Build the 14-feature predictor panel for the given dates.

    Parameters
    ----------
    panel : DataFrame full cross-section (date, permno, mom_1..mom_12, pi_filter)
    m2_returns : pd.Series M2 monthly returns indexed by date
    dates : iterable of pd.Timestamp — months to include (cluster labels' dates)
    """
    pi = panel.groupby('date')['pi_filter'].first().sort_index()
    agg_mean = panel.groupby('date')[MOM_COLS].mean()
    agg_std = panel.groupby('date')[MOM_COLS].std()
    skew_df = cross_section_skew(panel, MOM_COLS)

    rows = []
    for d in dates:
        rows.append({
            'date': d,
            # Group A — context
            'pi_panic': int(pi.loc[d] > 0.5),
            'cs_mom_short': float(tertile_mean(agg_mean.loc[[d]], SHORT).iloc[0]),
            'cs_mom_mid':   float(tertile_mean(agg_mean.loc[[d]], MID).iloc[0]),
            'cs_mom_long':  float(tertile_mean(agg_mean.loc[[d]], LONG).iloc[0]),
            'mom_overall':  float(agg_mean.loc[d, MOM_COLS].mean()),
            'cs_disp_short': float(tertile_mean(agg_std.loc[[d]], SHORT).iloc[0]),
            'cs_disp_mid':   float(tertile_mean(agg_std.loc[[d]], MID).iloc[0]),
            'cs_disp_long':  float(tertile_mean(agg_std.loc[[d]], LONG).iloc[0]),
            # Group B — time-series
            'pi_panic_freq_6mo':  pi_panic_freq(pi, d, n_months=6),
            'pi_panic_freq_12mo': pi_panic_freq(pi, d, n_months=12),
            'past_sharpe_12mo':   past_strategy_sharpe(m2_returns, d, n_months=12),
            # Group C — cross-section skew
            'cs_skew_short': float(tertile_mean(skew_df.loc[[d]], SHORT).iloc[0]),
            'cs_skew_mid':   float(tertile_mean(skew_df.loc[[d]], MID).iloc[0]),
            'cs_skew_long':  float(tertile_mean(skew_df.loc[[d]], LONG).iloc[0]),
        })
    return pd.DataFrame(rows)
```

Replace the bottom of `main()` (after the centroids print) with:

```python
    # ---- Predictor feature panel ----
    print('\nLoading M2 returns...', flush=True)
    with open(f'{RES_DIR}/fundamentals_returns.pkl', 'rb') as f:
        rets = pickle.load(f)
    m2_ret = rets['baseline_mom_pi']['returns']
    m2_ret.index = pd.to_datetime(m2_ret.index)

    print('Building predictor feature panel...', flush=True)
    feat = build_feature_panel(panel, m2_ret, list(fp.index))
    feat = feat.merge(label_df, on='date', how='left')
    feat.to_csv(f'{RES_DIR}/picked_stock_feature_panel.csv', index=False)
    print(f'Saved: {RES_DIR}/picked_stock_feature_panel.csv', flush=True)
    print(f'\nFeature panel shape: {feat.shape}')
    print('\nMissingness per feature:')
    print(feat.isna().sum())
```

- [ ] **Step 2: Run**

```bash
cd /Users/giladgang/momentum_regime && python scripts/picked_stock_cluster_search.py
```

Expected: completes through "Missingness per feature" section. The first 12 months will have NaN on `pi_panic_freq_12mo` and `past_sharpe_12mo`; the first 6 will have NaN on `pi_panic_freq_6mo`. All other features should have 0 NaN.

- [ ] **Step 3: Inspect**

```bash
head -3 /Users/giladgang/momentum_regime/results/thesis/picked_stock_feature_panel.csv
wc -l /Users/giladgang/momentum_regime/results/thesis/picked_stock_feature_panel.csv
```

Expected: ~170 rows + header. Columns: date, 14 features, cluster.

- [ ] **Step 4: Commit**

```bash
git add scripts/picked_stock_cluster_search.py \
        results/thesis/picked_stock_feature_panel.csv
git commit -m "$(cat <<'EOF'
picked-stock cluster: predictor feature panel (14 features per month)

8 context features (pi_panic, cs_mom S/M/L, mom_overall, cs_disp S/M/L)
+ 3 time-series (pi_panic_freq 6/12mo, past_sharpe_12mo)
+ 3 cross-section skewness (S/M/L). Note: pi-derived features encoded
as binary / fraction-in-panic to reflect that pi is binary-ish.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Univariate ANOVA + multinomial L1 logistic with LOO-CV

**Files:**
- Modify: `scripts/picked_stock_cluster_search.py`

- [ ] **Step 1: Append univariate + logistic functions**

Append after `build_feature_panel`:

```python
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score


PRED_COLS = [
    'pi_panic',
    'cs_mom_short', 'cs_mom_mid', 'cs_mom_long', 'mom_overall',
    'cs_disp_short', 'cs_disp_mid', 'cs_disp_long',
    'pi_panic_freq_6mo', 'pi_panic_freq_12mo', 'past_sharpe_12mo',
    'cs_skew_short', 'cs_skew_mid', 'cs_skew_long',
]


def univariate_ranking(feat_df):
    """ANOVA F-statistic per feature against cluster labels."""
    valid = feat_df[PRED_COLS + ['cluster']].dropna()
    F, p = f_classif(valid[PRED_COLS].values, valid['cluster'].values)
    rank = pd.DataFrame({
        'feature': PRED_COLS,
        'F_statistic': F,
        'p_value': p,
    }).sort_values('F_statistic', ascending=False)
    return rank


def multinomial_l1_loo(feat_df, C_grid=(0.05, 0.1, 0.3, 1.0, 3.0)):
    """Multinomial L1 logistic regression, LOO-CV across C grid."""
    valid = feat_df[PRED_COLS + ['cluster']].dropna()
    X = valid[PRED_COLS].values
    y = valid['cluster'].values
    X_std = StandardScaler().fit_transform(X)

    print('\n  LOO-CV across C grid:')
    best_acc = -1.0
    best_C = None
    for C in C_grid:
        lr = LogisticRegression(
            penalty='l1', solver='saga', C=C,
            multi_class='multinomial', max_iter=10000,
        )
        y_pred = cross_val_predict(lr, X_std, y, cv=LeaveOneOut(), n_jobs=-1)
        acc = accuracy_score(y, y_pred)
        print(f'    C={C:>5.2f}: acc={acc:.3f}', flush=True)
        if acc > best_acc:
            best_acc = acc
            best_C = C

    lr = LogisticRegression(
        penalty='l1', solver='saga', C=best_C,
        multi_class='multinomial', max_iter=10000,
    ).fit(X_std, y)
    coef = pd.DataFrame(lr.coef_.T, index=PRED_COLS, columns=lr.classes_)
    coef.index.name = 'feature'
    baseline = float(pd.Series(y).value_counts(normalize=True).max())
    return coef, best_acc, baseline, best_C
```

Append the call inside `main()` (after feature-panel block):

```python
    # ---- Univariate ranking ----
    print('\n=== Univariate ANOVA ranking ===', flush=True)
    rank = univariate_ranking(feat)
    rank.to_csv(f'{RES_DIR}/picked_stock_feature_ranking.csv', index=False)
    print(rank.to_string(index=False, float_format='%.3f'))
    print(f'Saved: {RES_DIR}/picked_stock_feature_ranking.csv')

    # ---- Multinomial L1 logistic with LOO-CV ----
    print('\n=== Multinomial L1 logistic (LOO-CV) ===', flush=True)
    coef, loo_acc, baseline, best_C = multinomial_l1_loo(feat)
    coef.to_csv(f'{RES_DIR}/picked_stock_logistic_coefficients.csv')
    print(f'\n  best C = {best_C}, LOO-CV acc = {loo_acc:.3f}, baseline (max class) = {baseline:.3f}')
    print('  Coefficients:')
    print(coef.round(3).to_string())
    print(f'Saved: {RES_DIR}/picked_stock_logistic_coefficients.csv')
```

- [ ] **Step 2: Run**

```bash
cd /Users/giladgang/momentum_regime && python scripts/picked_stock_cluster_search.py
```

Expected: prints univariate ranking and LOO-CV accuracy at each C, picks best, prints coefficient matrix.

- [ ] **Step 3: Inspect**

```bash
cat /Users/giladgang/momentum_regime/results/thesis/picked_stock_feature_ranking.csv
cat /Users/giladgang/momentum_regime/results/thesis/picked_stock_logistic_coefficients.csv
```

Expected:
- Ranking has 14 rows ordered by F-statistic desc.
- Coefficients has 14 rows × K columns; many entries should be exactly 0 (L1).

- [ ] **Step 4: Commit**

```bash
git add scripts/picked_stock_cluster_search.py \
        results/thesis/picked_stock_feature_ranking.csv \
        results/thesis/picked_stock_logistic_coefficients.csv
git commit -m "$(cat <<'EOF'
picked-stock cluster: univariate ANOVA + multinomial L1 logistic (LOO-CV)

ANOVA F-stat per feature against cluster labels.
Multinomial L1 logistic, C selected by leave-one-out CV across
{0.05,0.1,0.3,1.0,3.0}. Reports sparse coefficient matrix and
class-imbalance baseline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Decision tree + per-cluster summary + headline verdict

**Files:**
- Modify: `scripts/picked_stock_cluster_search.py`

- [ ] **Step 1: Append decision-tree + summary functions**

Append after `multinomial_l1_loo`:

```python
from sklearn.tree import DecisionTreeClassifier, export_text


def decision_tree_readout(feat_df, max_depth=3):
    """Fit a depth-3 tree, return text rule + LOO accuracy."""
    valid = feat_df[PRED_COLS + ['cluster']].dropna()
    X = valid[PRED_COLS].values
    y = valid['cluster'].values
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42).fit(X, y)
    text = export_text(tree, feature_names=PRED_COLS)
    train_acc = accuracy_score(y, tree.predict(X))
    y_loo = cross_val_predict(
        DecisionTreeClassifier(max_depth=max_depth, random_state=42),
        X, y, cv=LeaveOneOut(), n_jobs=-1,
    )
    loo_acc = accuracy_score(y, y_loo)
    return text, train_acc, loo_acc


def per_cluster_summary(feat_df, fp_df):
    """Mean ± std of every fingerprint dim AND every predictor feature, per cluster."""
    fp_with_cluster = fp_df.merge(
        feat_df[['date', 'cluster']], left_index=True, right_on='date',
    ).drop(columns='date')
    fp_summary = fp_with_cluster.groupby('cluster').agg(['mean', 'std'])
    feat_summary = (
        feat_df[PRED_COLS + ['cluster']]
        .groupby('cluster').agg(['mean', 'std'])
    )
    full = pd.concat([fp_summary, feat_summary], axis=1)
    return full
```

Append the call inside `main()` (after the logistic block):

```python
    # ---- Decision tree ----
    print('\n=== Depth-3 decision tree ===', flush=True)
    tree_text, tree_train_acc, tree_loo_acc = decision_tree_readout(feat, max_depth=3)
    with open(f'{RES_DIR}/picked_stock_decision_tree.txt', 'w') as f:
        f.write(tree_text)
    print(tree_text)
    print(f'  train acc = {tree_train_acc:.3f}, LOO acc = {tree_loo_acc:.3f}')
    print(f'Saved: {RES_DIR}/picked_stock_decision_tree.txt')

    # ---- Per-cluster summary ----
    print('\n=== Per-cluster summary ===', flush=True)
    summary = per_cluster_summary(feat, fp)
    summary.to_csv(f'{RES_DIR}/picked_stock_cluster_summary.csv')
    print(f'Saved: {RES_DIR}/picked_stock_cluster_summary.csv')

    # ---- Headline verdict ----
    print('\n' + '=' * 60)
    print('  HEADLINE')
    print('=' * 60)
    print(f'  Chosen K:                     {chosen_k}')
    print(f'  Logistic LOO-CV accuracy:     {loo_acc:.3f}')
    print(f'  Decision tree LOO-CV accuracy: {tree_loo_acc:.3f}')
    print(f'  Class-imbalance baseline:     {baseline:.3f}')
    if loo_acc >= 0.70:
        print('\n  STRONG: features cleanly recover cluster structure.')
        print('  -> Use top features as the natural group-by variables for §5.2.')
    elif loo_acc >= 0.60:
        print('\n  WEAK: marginal predictability; check decision tree for partial split.')
    else:
        print('\n  NEGATIVE: no month-level feature predicts cluster membership.')
        print('  -> Picks are determined below the monthly aggregate (stock-level).')
        print('  -> Establishes a boundary for the §5.2 narrative.')
```

- [ ] **Step 2: Run**

```bash
cd /Users/giladgang/momentum_regime && python scripts/picked_stock_cluster_search.py
```

Expected: completes end-to-end with HEADLINE block printed.

- [ ] **Step 3: Inspect**

```bash
cat /Users/giladgang/momentum_regime/results/thesis/picked_stock_decision_tree.txt
head -3 /Users/giladgang/momentum_regime/results/thesis/picked_stock_cluster_summary.csv
```

Expected:
- Tree text shows an indented rule like `|--- pi_panic <= 0.5 / |   |--- mom_overall <= 0.05 / ...`
- Summary CSV is a wide table: rows = clusters, columns = (feature, stat) pairs.

- [ ] **Step 4: Commit**

```bash
git add scripts/picked_stock_cluster_search.py \
        results/thesis/picked_stock_decision_tree.txt \
        results/thesis/picked_stock_cluster_summary.csv
git commit -m "$(cat <<'EOF'
picked-stock cluster: decision tree + per-cluster summary + verdict

Depth-3 decision tree (with LOO-CV accuracy alongside training acc).
Per-cluster mean+/-std of every fingerprint dim and predictor feature
for interpretation. Headline prints chosen K, accuracies, and
SUCCESS / WEAK / NEGATIVE verdict.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Final verification

**Files:**
- No code changes; verification only.

- [ ] **Step 1: Re-run helper tests**

```bash
cd /Users/giladgang/momentum_regime && pytest scripts/test_cluster_feature_search_helpers.py -v
```

Expected: 15 passed.

- [ ] **Step 2: Re-run main script end-to-end**

```bash
cd /Users/giladgang/momentum_regime && python scripts/picked_stock_cluster_search.py
```

Expected: completes; HEADLINE block prints; same numbers as the earlier runs (deterministic — random_state=42 throughout).

- [ ] **Step 3: Confirm all outputs exist**

```bash
ls -la /Users/giladgang/momentum_regime/results/thesis/picked_stock_*
```

Expected: 8 files —
`picked_stock_cluster_sweep.csv`,
`picked_stock_cluster_labels.csv`,
`picked_stock_cluster_centroids.csv`,
`picked_stock_feature_panel.csv`,
`picked_stock_feature_ranking.csv`,
`picked_stock_logistic_coefficients.csv`,
`picked_stock_decision_tree.txt`,
`picked_stock_cluster_summary.csv`.

- [ ] **Step 4: Quick sanity checks against domain knowledge**

Inspect `picked_stock_cluster_sweep.csv` and `picked_stock_cluster_centroids.csv` and confirm:
- At least one K is stable (mean_ari ≥ 0.90).
- The cluster centroids on `pick_mom_*` differ meaningfully across clusters (ranges in raw mom of ≥ 5–10 percentage points between clusters).
- If `pi_panic` is the top univariate feature with F >> any other, that's expected (binary-ish π is a strong axis).

If a check fails, do NOT mark the task complete — investigate (e.g., is the fingerprint sane? does the picks merge produce the expected stock count per month?).

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| Picked-stock fingerprint definition (15-d) | Task 4 (helper) + Task 5 (use) |
| Cluster sweep K=2..5 with stability + silhouette | Task 5 |
| K choice rule | Task 5 (`pick_k`) |
| Predictor library Group A (8: pi_panic, cs_mom S/M/L, mom_overall, cs_disp S/M/L) | Task 6 |
| Predictor library Group B (3: pi_panic_freq 6mo/12mo, past_sharpe_12mo) | Tasks 1, 2, 6 |
| Predictor library Group C (3: cs_skew S/M/L) | Tasks 3, 6 |
| pi binary treatment | Tasks 1, 6 (`pi_panic` indicator + `pi_panic_freq`) |
| Univariate ANOVA ranking | Task 7 |
| Multinomial L1 logistic with LOO-CV | Task 7 |
| Depth-3 decision tree (train + LOO accuracy) | Task 8 |
| Per-cluster summary | Task 8 |
| Output CSVs (8 files) | Tasks 5, 6, 7, 8 |
| Headline verdict (success / weak / negative) | Task 8 |
| Validation re-run | Task 9 |

All spec items covered.

**Placeholder scan:** every code step has runnable code; no TBDs; expected output is concrete on every Step 2 (run).

**Type consistency:** `cluster` column used consistently in labels CSV → feature panel → all downstream. `PRED_COLS` defined once in Task 7, reused in Task 8. Helper signatures in Task 4 (`picked_stock_fingerprint(picks_df, mom_panel, mom_cols)`) match the call site in Task 5. Mom-tertile groups (`SHORT`/`MID`/`LONG` = 1..4 / 5..8 / 9..12) consistent across helpers and the orchestration script.

**Scope:** 9 tasks, ~4–5 hours. Single implementation plan.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-30-cluster-membership-feature-search.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
