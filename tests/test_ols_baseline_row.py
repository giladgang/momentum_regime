"""
test_ols_baseline_row.py
========================
Tests for `scripts/ols_baseline_row.py` (Step 22). The script computes the
plain OLS baseline row for tab:performance and inserts it into
tables/table_performance.tex directly below the LR row.

Tests pin the pure helpers (row formatting, idempotent row insertion,
significance stars, Newey-West t) on synthetic fixtures; no artefacts
needed.
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)


def _load_script():
    path = os.path.join(REPO, 'scripts', 'ols_baseline_row.py')
    spec = importlib.util.spec_from_file_location('ols_baseline_row', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def m():
    return _load_script()


SAMPLE_TEX = r"""\begin{table}[htbp]
\begin{tabular}{l r r r r r r r}
\toprule
 & Ann.\ Ret & Ann.\ Vol & Sharpe & Max DD & $\beta$ & NW $t$ & Final \$ \\
\midrule
Market & 11.9\% & 14.6\% & 0.84 & -24.7\% & 1.00 & -- & 4.8 \\
\midrule
Fixed 12-mo mom & -1.9\% & 26.1\% & 0.06 & -72.2\% & $-$0.66 & 0.24 & 0.8 \\
LR & -1.3\% & 15.2\% & -0.01 & -42.4\% & 0.05 & -0.04 & 0.8 \\
XGB & 21.7\% & 19.5\% & 1.11 & -22.8\% & 0.45 & 4.37*** & 15.3 \\
\bottomrule
\end{tabular}
\end{table}
"""


# ── format_ols_row ────────────────────────────────────────────────────────────

def test_format_row_escapes_percents_and_matches_step3_style(m):
    row = m.format_ols_row(ann_ret=-0.111, ann_vol=0.174, sharpe=-0.58,
                           mdd=-0.820, beta=0.05, nw_t=-1.23, nw_p=0.22,
                           final=0.2)
    assert row == r'OLS & -11.1\% & 17.4\% & -0.58 & -82.0\% & 0.05 & -1.23 & 0.2 \\'


def test_format_row_negative_beta_uses_math_minus(m):
    row = m.format_ols_row(ann_ret=0.01, ann_vol=0.1, sharpe=0.1,
                           mdd=-0.1, beta=-0.66, nw_t=0.24, nw_p=0.81,
                           final=1.0)
    assert '$-$0.66' in row


def test_format_row_appends_significance_stars(m):
    row = m.format_ols_row(ann_ret=0.217, ann_vol=0.195, sharpe=1.11,
                           mdd=-0.228, beta=0.45, nw_t=4.37, nw_p=0.001,
                           final=15.3)
    assert '4.37***' in row


# ── insert_ols_row ────────────────────────────────────────────────────────────

def test_insert_places_row_directly_below_lr(m):
    row = r'OLS & -11.1\% & 17.4\% & -0.58 & -82.0\% & 0.05 & -1.23 & 0.2 \\'
    out = m.insert_ols_row(SAMPLE_TEX, row)
    lines = out.splitlines()
    lr_idx = next(i for i, l in enumerate(lines) if l.startswith('LR &'))
    assert lines[lr_idx + 1] == row
    assert lines[lr_idx + 2].startswith('XGB &')


def test_insert_is_idempotent(m):
    row1 = r'OLS & -11.1\% & 17.4\% & -0.58 & -82.0\% & 0.05 & -1.23 & 0.2 \\'
    row2 = r'OLS & -12.0\% & 17.0\% & -0.60 & -83.0\% & 0.06 & -1.30 & 0.2 \\'
    once = m.insert_ols_row(SAMPLE_TEX, row1)
    twice = m.insert_ols_row(once, row2)
    assert twice.count('OLS &') == 1
    assert row2 in twice
    assert row1 not in twice


def test_insert_preserves_all_other_rows(m):
    row = r'OLS & -11.1\% & 17.4\% & -0.58 & -82.0\% & 0.05 & -1.23 & 0.2 \\'
    out = m.insert_ols_row(SAMPLE_TEX, row)
    for label in ('Market &', 'Fixed 12-mo mom &', 'LR &', 'XGB &'):
        assert out.count(label) == SAMPLE_TEX.count(label)


def test_insert_raises_without_lr_row(m):
    with pytest.raises(ValueError):
        m.insert_ols_row(r'Market & 11.9\% \\', 'OLS & 1 \\\\')


# ── sig_stars / newey_west_t ─────────────────────────────────────────────────

def test_sig_stars_thresholds(m):
    assert m.sig_stars(0.005) == '***'
    assert m.sig_stars(0.03) == '**'
    assert m.sig_stars(0.07) == '*'
    assert m.sig_stars(0.5) == ''


def test_newey_west_t_short_series_is_nan(m):
    t, p = m.newey_west_t(pd.Series(np.ones(5)))
    assert np.isnan(t) and np.isnan(p)


def test_newey_west_t_positive_mean_series(m):
    rng = np.random.default_rng(0)
    r = pd.Series(0.02 + 0.01 * rng.standard_normal(167))
    t, p = m.newey_west_t(r)
    assert t > 2
    assert p < 0.05
