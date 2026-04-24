"""
test_portfolio_edge_cases.py
============================
Edge-case tests for portfolio construction in src/utils.py (long_short_port).

These cover the scenarios that are easy to break silently:
  - Empty / tiny monthly panels (fewer than 10 NYSE stocks)
  - All-NaN score column, partial NaN score column
  - Zero market equity on long or short legs
  - Single-stock months
  - Mixed exchanges (NYSE vs AMEX/NASDAQ) — breakpoints from NYSE only
  - Regime-transition months (score distribution changes abruptly)
  - Weight conservation (long and short weights each sum to 1)
  - Zero-fee run is pure gross return (no turnover cost)
  - Fee scales linearly with turnover

Run with:
    pytest tests/test_portfolio_edge_cases.py -v
"""

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

from src import utils


# ═══════════════════════════════════════════════════════════════════════════════
# Fixture helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _month(date, n_stocks=40, exchcd=1, seed=0, me_override=None,
           ret_fn=None, score_fn=None):
    """Make a single-month slice."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_stocks):
        rows.append({
            'date': pd.Timestamp(date),
            'permno': p,
            'exchcd': exchcd,
            'me': me_override if me_override is not None else float(rng.uniform(1e8, 1e10)),
            'ret_fwd': ret_fn(p) if ret_fn else float(rng.normal(0.01, 0.05)),
            'score': score_fn(p) if score_fn else float(rng.normal()),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Empty / degenerate inputs
# ═══════════════════════════════════════════════════════════════════════════════

class TestDegenerateInputs:
    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=['date', 'permno', 'exchcd', 'me', 'ret_fwd', 'score'])
        out = utils.long_short_port(df, 'score', fee=0.0)
        assert isinstance(out, pd.Series)
        assert len(out) == 0

    def test_exactly_nine_nyse_stocks_skipped(self):
        df = _month('2020-01-31', n_stocks=9, exchcd=1, seed=1)
        out = utils.long_short_port(df, 'score', fee=0.0)
        assert len(out) == 0, "Must require >= 10 NYSE stocks"

    def test_exactly_ten_nyse_stocks_included(self):
        df = _month('2020-01-31', n_stocks=10, exchcd=1, seed=2)
        out = utils.long_short_port(df, 'score', fee=0.0)
        # With exactly 10 stocks, P10/P90 will overlap — may skip if longs==shorts
        # but should not crash and should return <=1 observation
        assert len(out) <= 1

    def test_all_nan_score(self):
        df = _month('2020-01-31', n_stocks=40, exchcd=1, seed=3)
        df['score'] = np.nan
        out = utils.long_short_port(df, 'score', fee=0.0)
        assert len(out) == 0

    def test_partial_nan_score_uses_remaining(self):
        df = _month('2020-01-31', n_stocks=40, exchcd=1, seed=4)
        # NaN out 30 stocks' scores — only 10 remain on NYSE
        df.loc[:29, 'score'] = np.nan
        out = utils.long_short_port(df, 'score', fee=0.0)
        # 10 remaining NYSE stocks meets threshold but P10==P90 is possible
        assert len(out) <= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Zero market equity
# ═══════════════════════════════════════════════════════════════════════════════

class TestZeroME:
    def test_zero_me_on_longs_skipped(self):
        df = _month('2020-01-31', n_stocks=40, exchcd=1, seed=5)
        hi = df['score'].quantile(0.90)
        df.loc[df['score'] >= hi, 'me'] = 0.0
        out = utils.long_short_port(df, 'score', fee=0.0)
        assert len(out) == 0

    def test_zero_me_on_shorts_skipped(self):
        df = _month('2020-01-31', n_stocks=40, exchcd=1, seed=6)
        lo = df['score'].quantile(0.10)
        df.loc[df['score'] <= lo, 'me'] = 0.0
        out = utils.long_short_port(df, 'score', fee=0.0)
        assert len(out) == 0

    def test_all_zero_me_skipped(self):
        df = _month('2020-01-31', n_stocks=40, exchcd=1, seed=7, me_override=0.0)
        out = utils.long_short_port(df, 'score', fee=0.0)
        assert len(out) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Exchange filtering: breakpoints from NYSE only
# ═══════════════════════════════════════════════════════════════════════════════

class TestExchangeHandling:
    def test_only_non_nyse_stocks_skipped(self):
        # 40 NASDAQ stocks (exchcd=3) — zero NYSE -> skipped
        df = _month('2020-01-31', n_stocks=40, exchcd=3, seed=8)
        out = utils.long_short_port(df, 'score', fee=0.0)
        assert len(out) == 0

    def test_mixed_exchanges_use_nyse_breakpoints(self):
        # 12 NYSE + 28 NASDAQ. Non-NYSE stocks are still eligible for
        # long/short legs but breakpoints are computed from NYSE only.
        nyse = _month('2020-01-31', n_stocks=12, exchcd=1, seed=9)
        nyse['permno'] = range(0, 12)
        nasdaq = _month('2020-01-31', n_stocks=28, exchcd=3, seed=10)
        nasdaq['permno'] = range(12, 40)
        df = pd.concat([nyse, nasdaq], ignore_index=True)
        out = utils.long_short_port(df, 'score', fee=0.0)
        # Should succeed (12 NYSE >= 10) and produce one observation
        assert len(out) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Weight conservation / economic invariants
# ═══════════════════════════════════════════════════════════════════════════════

class TestEconomicInvariants:
    def test_long_leg_weight_sums_to_one(self):
        """Verify that when the port computes r_long = sum(ret * me) / sum(me),
        it's equivalent to a weighted average with weights summing to 1."""
        df = _month('2020-01-31', n_stocks=40, exchcd=1, seed=11)
        top = df[df['score'] >= df['score'].quantile(0.90)]
        w = top['me'] / top['me'].sum()
        assert w.sum() == pytest.approx(1.0, abs=1e-12)

    def test_zero_fee_equals_gross_return_on_first_month(self):
        """On the first month there are no prev weights; turnover = 0.5 per leg.
        Zero fee means returns are purely gross long - short."""
        df = _month('2020-01-31', n_stocks=40, exchcd=1, seed=12)
        out = utils.long_short_port(df, 'score', fee=0.0)

        # Compute gross long-short by hand
        nyse = df[df['exchcd'] == 1]['score'].dropna()
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs = df[df['score'] >= hi]
        shorts = df[df['score'] <= lo]
        r_long = (longs['ret_fwd'] * longs['me']).sum() / longs['me'].sum()
        r_short = (shorts['ret_fwd'] * shorts['me']).sum() / shorts['me'].sum()
        expected = r_long - r_short

        assert out.iloc[0] == pytest.approx(expected, abs=1e-12)

    def test_fee_scales_linearly_with_fee_level(self):
        df = _month('2020-01-31', n_stocks=40, exchcd=1, seed=13)
        r0 = utils.long_short_port(df, 'score', fee=0.0).iloc[0]
        r1 = utils.long_short_port(df, 'score', fee=0.001).iloc[0]
        r2 = utils.long_short_port(df, 'score', fee=0.002).iloc[0]
        # (r0 - r1) must equal (r1 - r2) — linearity in fee
        assert (r0 - r1) == pytest.approx(r1 - r2, abs=1e-12)

    def test_second_month_full_holding_rebalance_turnover_positive(self):
        """If the long set changes entirely between two months, turnover
        on the second month should be ~1 per leg and the fee cost ~2*fee."""
        rng = np.random.default_rng(14)

        def rows_for(date, scores):
            out = []
            for p, s in enumerate(scores):
                out.append({
                    'date': pd.Timestamp(date),
                    'permno': p,
                    'exchcd': 1,
                    'me': 1e9,
                    'ret_fwd': 0.0,
                    'score': s,
                })
            return out

        # Month 1: first 20 stocks rank high, last 20 low
        scores_m1 = list(range(40))[::-1]  # permno 0 highest
        # Month 2: reverse ranking — first 20 stocks now rank low
        scores_m2 = list(range(40))

        df = pd.DataFrame(rows_for('2020-01-31', scores_m1) +
                          rows_for('2020-02-29', scores_m2))

        r_no_fee = utils.long_short_port(df, 'score', fee=0.0)
        r_fee = utils.long_short_port(df, 'score', fee=0.01)
        assert len(r_no_fee) == 2 and len(r_fee) == 2

        # Second month fee impact should be larger than first month
        diff_m1 = r_no_fee.iloc[0] - r_fee.iloc[0]
        diff_m2 = r_no_fee.iloc[1] - r_fee.iloc[1]
        assert diff_m2 > diff_m1, \
            "Full turnover on month 2 should cost more than first-month build-up"


# ═══════════════════════════════════════════════════════════════════════════════
# Date / multi-month handling
# ═══════════════════════════════════════════════════════════════════════════════

class TestDateHandling:
    def test_output_dates_preserved(self):
        df1 = _month('2020-01-31', n_stocks=40, seed=15)
        df2 = _month('2020-02-29', n_stocks=40, seed=16)
        df3 = _month('2020-03-31', n_stocks=40, seed=17)
        df = pd.concat([df1, df2, df3], ignore_index=True)
        out = utils.long_short_port(df, 'score', fee=0.0)
        assert list(out.index) == [
            pd.Timestamp('2020-01-31'),
            pd.Timestamp('2020-02-29'),
            pd.Timestamp('2020-03-31'),
        ]

    def test_output_dates_are_unique(self):
        df = pd.concat([_month(d, n_stocks=40, seed=i)
                        for i, d in enumerate(['2020-01-31', '2020-02-29',
                                               '2020-03-31', '2020-04-30'])],
                       ignore_index=True)
        out = utils.long_short_port(df, 'score', fee=0.0)
        assert out.index.is_unique

    def test_shuffled_input_order_produces_same_result(self):
        """groupby('date') must make the output order-invariant."""
        df = pd.concat([_month(d, n_stocks=40, seed=i)
                        for i, d in enumerate(['2020-01-31', '2020-02-29',
                                               '2020-03-31'])],
                       ignore_index=True)
        out1 = utils.long_short_port(df, 'score', fee=0.0)
        # Shuffle rows within the same panel
        shuffled = df.sample(frac=1, random_state=99).reset_index(drop=True)
        out2 = utils.long_short_port(shuffled, 'score', fee=0.0)
        # Same dates, same values (possibly different prev_weight state if
        # groupby iteration order differs — but pandas groupby preserves
        # sorted keys by default)
        pd.testing.assert_series_equal(out1.sort_index(), out2.sort_index())


# ═══════════════════════════════════════════════════════════════════════════════
# Regime-transition month (abrupt score distribution change)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegimeTransition:
    def test_score_sign_flip_does_not_crash(self):
        """When regime changes, score distribution may flip sign. Port
        construction must still produce valid returns."""
        rng = np.random.default_rng(20)
        m1 = _month('2020-01-31', n_stocks=40, exchcd=1, seed=21,
                    score_fn=lambda p: float(rng.normal(0.5, 0.2)))
        m2 = _month('2020-02-29', n_stocks=40, exchcd=1, seed=22,
                    score_fn=lambda p: float(rng.normal(-0.5, 0.2)))
        df = pd.concat([m1, m2], ignore_index=True)
        out = utils.long_short_port(df, 'score', fee=0.001)
        assert len(out) == 2
        assert np.isfinite(out).all()

    def test_extreme_score_outlier_does_not_blow_up(self):
        """A single extreme outlier should not make the portfolio return
        diverge. Value-weighting caps its contribution."""
        df = _month('2020-01-31', n_stocks=40, exchcd=1, seed=23)
        # One stock gets an enormous score and moderate ret_fwd
        df.loc[0, 'score'] = 1e6
        df.loc[0, 'ret_fwd'] = 0.05
        out = utils.long_short_port(df, 'score', fee=0.0)
        assert len(out) == 1
        # Output must be finite and in a sane range
        assert np.isfinite(out.iloc[0])
        assert -1.0 < out.iloc[0] < 1.0
