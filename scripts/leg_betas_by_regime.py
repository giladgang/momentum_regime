"""
leg_betas_by_regime.py
======================
Leg-level market betas across regimes, for comparison with Daniel & Moskowitz
(2016) Fig. 3.

D&M compute 126-day rolling betas of the winner decile and the loser decile
separately, and show that the loser-leg beta spikes to 4--5 around the 1932
and 2009 rebounds while the winner-leg beta stays near 1. The widening gap is
the proximate mechanism behind the momentum crash.

This script constructs the analogous picture for:
  (a) Fixed 12-month momentum  (benchmark; should reproduce D&M's direction)
  (b) Method 2 (XGB + pi_filter) (thesis strategy; expected to INVERT in panic
      because the selection direction flips)

For each strategy it computes:
  1. Monthly value-weighted returns of the long leg and the short leg
     separately (NYSE P90/P10 breakpoints, same construction as
     cross_sectional_model.py).
  2. Static CAPM beta of each leg, split by regime state (pi_filter >= 0.5 = panic).
  3. 24-month rolling CAPM beta of each leg (continuous time series).

Outputs:
  results/leg_betas_by_regime.csv        static table
  tables/table_leg_betas.tex             LaTeX table
  plots/leg_betas_rolling.pdf            rolling-beta figure

Usage:
  python scripts/leg_betas_by_regime.py

Reads from artefacts/cs_artefacts_data.pkl (produced by cross_sectional_model.py)
so the XGB model does not need to be re-trained.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

ROLLING_WINDOW = 24           # months
PANIC_CUTOFF   = 0.50         # same convention as main_results_analysis.py
MIN_OBS_REG    = 12           # min monthly obs for a regime-conditional beta


def build_leg_returns(test, score_col, fee=cfg.TRADING_FEE):
    """Monthly value-weighted returns for the LONG and SHORT legs separately.

    Returns a DataFrame indexed by date with columns ['r_long', 'r_short'].
    Net of one-way turnover fee on each leg. Uses NYSE P10/P90 breakpoints,
    same as cross_sectional_model.long_short_port.
    """
    rows = []
    prev_lw, prev_sw = {}, {}
    for date, grp in test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs  = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0:
            continue
        lme = longs['me'].sum()
        sme = shorts['me'].sum()
        new_lw = (longs.set_index('permno')['me'] / lme).to_dict()
        new_sw = (shorts.set_index('permno')['me'] / sme).to_dict()
        r_long  = (longs['ret_fwd']  * longs['me']).sum()  / lme
        r_short = (shorts['ret_fwd'] * shorts['me']).sum() / sme
        tl = sum(abs(new_lw.get(p, 0) - prev_lw.get(p, 0))
                 for p in set(new_lw) | set(prev_lw)) / 2
        ts = sum(abs(new_sw.get(p, 0) - prev_sw.get(p, 0))
                 for p in set(new_sw) | set(prev_sw)) / 2
        rows.append({
            'date':    date,
            'r_long':  r_long  - fee * tl,
            'r_short': r_short - fee * ts,
        })
        prev_lw, prev_sw = new_lw, new_sw
    return pd.DataFrame(rows).set_index('date')


def capm_beta(r, r_mkt):
    """OLS beta of leg excess return on market excess return.
    No risk-free subtraction: monthly T-bill impact on beta is negligible and
    matches the thesis's existing CAPM specification."""
    df = pd.concat([r, r_mkt], axis=1).dropna()
    df.columns = ['r', 'mkt']
    if len(df) < MIN_OBS_REG:
        return np.nan, np.nan, len(df)
    X = sm.add_constant(df['mkt'])
    res = sm.OLS(df['r'], X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    return res.params['mkt'], res.bse['mkt'], len(df)


def rolling_beta(r, r_mkt, window=ROLLING_WINDOW):
    """Monthly rolling beta; returns a Series aligned to r."""
    df = pd.concat([r, r_mkt], axis=1).dropna()
    df.columns = ['r', 'mkt']
    cov = df['r'].rolling(window).cov(df['mkt'])
    var = df['mkt'].rolling(window).var()
    return (cov / var).rename('beta')


def main():
    root = Path(__file__).resolve().parents[1]
    with open(root / cfg.ARTEFACTS_PATH, 'rb') as f:
        art = pickle.load(f)
    test  = art['test'].copy()
    r_mkt = art['r_mkt'].copy()
    r_mkt.index = pd.to_datetime(r_mkt.index)

    # Regime state per month (same cutoff as thesis)
    pi_by_date = (test[['date', 'pi_filter']]
                  .drop_duplicates('date')
                  .set_index('date')['pi_filter'])
    panic_mask = pi_by_date >= PANIC_CUTOFF

    strategies = {
        'Fixed 12-mo mom': 'score_mom12',
        'Method 2: XGB':   'score_xgb',
    }

    # ─── 1. Static leg-level betas, split by regime ───────────────────────────
    rows = []
    leg_returns = {}
    for name, col in strategies.items():
        legs = build_leg_returns(test, col)
        leg_returns[name] = legs
        for leg_name, leg_col in [('Long', 'r_long'), ('Short', 'r_short')]:
            r = legs[leg_col]
            r_calm  = r[r.index.isin(panic_mask[~panic_mask].index)]
            r_panic = r[r.index.isin(panic_mask[panic_mask].index)]
            for regime_name, r_sub in [('Full',  r),
                                       ('Calm',  r_calm),
                                       ('Panic', r_panic)]:
                b, se, n = capm_beta(r_sub, r_mkt)
                rows.append({
                    'strategy': name, 'leg': leg_name, 'regime': regime_name,
                    'beta': b, 'se': se, 'n_months': n,
                })

    df = pd.DataFrame(rows)
    out_csv = root / cfg.RESULTS_DIR / 'leg_betas_by_regime.csv'
    out_csv.parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False, float_format='%.3f')
    print(f"Wrote {out_csv}")
    print(df.to_string(index=False, float_format='%.3f'))

    # ─── 2. LaTeX table ───────────────────────────────────────────────────────
    tex = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Leg-level CAPM betas by regime, 2011--2025 OOS test period. '
        r'Panic months are those with $\pi_t^{\mathrm{filter}} \geq 0.5$. '
        r'Newey--West standard errors (3 lags) in parentheses.}',
        r'\label{tab:leg_betas}',
        r'\begin{tabular}{llccc}',
        r'\toprule',
        r'Strategy & Leg & Full & Calm & Panic \\',
        r'\midrule',
    ]
    for name in strategies:
        for leg in ['Long', 'Short']:
            sub = df[(df['strategy'] == name) & (df['leg'] == leg)]
            def cell(regime):
                row = sub[sub['regime'] == regime].iloc[0]
                if np.isnan(row['beta']):
                    return '--'
                return f"{row['beta']:.2f} ({row['se']:.2f})"
            tex.append(
                f"{name} & {leg} & {cell('Full')} & {cell('Calm')} & {cell('Panic')} \\\\"
            )
        tex.append(r'\midrule')
    tex[-1] = r'\bottomrule'
    tex += [r'\end{tabular}', r'\end{table}']
    out_tex = root / cfg.TABLES_DIR / 'table_leg_betas.tex'
    out_tex.write_text('\n'.join(tex))
    print(f"Wrote {out_tex}")

    # ─── 3. Rolling-beta figure (D&M Fig. 3 layout) ───────────────────────────
    # Panel A: Fixed 12-mom; Panel B: M2. Within each panel, long leg (solid)
    # vs short leg (dotted), matching D&M's winner-vs-loser-on-one-panel idea.
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)

    # Share y-limits across panels so the magnitude of any inversion is visible
    all_betas = []
    for legs in leg_returns.values():
        for col in ('r_long', 'r_short'):
            all_betas.append(rolling_beta(legs[col], r_mkt))
    b_concat = pd.concat(all_betas).dropna()
    ylo = float(np.floor(b_concat.min() * 2) / 2)
    yhi = float(np.ceil(b_concat.max() * 2) / 2)

    panels = [
        (axes[0], 'Panel A: Fixed 12-month momentum', 'Fixed 12-mo mom'),
        (axes[1], 'Panel B: Method 2 (XGB + $\\pi^{\\mathrm{filter}}$)', 'Method 2: XGB'),
    ]
    for ax, title, name in panels:
        legs = leg_returns[name]
        b_long  = rolling_beta(legs['r_long'],  r_mkt)
        b_short = rolling_beta(legs['r_short'], r_mkt)
        ax.plot(b_long.index,  b_long,  label='Long leg (winners in 12-mom)',
                color='k', ls='-',  lw=1.4)
        ax.plot(b_short.index, b_short, label='Short leg (losers in 12-mom)',
                color='k', ls=':',  lw=1.4)
        panic_dates = panic_mask[panic_mask].index
        for d in panic_dates:
            ax.axvspan(d - pd.Timedelta(days=15),
                       d + pd.Timedelta(days=15),
                       color='grey', alpha=0.18, lw=0)
        ax.axhline(1.0, color='k', lw=0.4, ls='--', alpha=0.6)
        ax.set_ylim(ylo, yhi)
        ax.set_ylabel(f'{ROLLING_WINDOW}-month rolling $\\beta$')
        ax.set_title(title, loc='left', fontsize=10)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.3)

    axes[1].set_xlabel('Date')
    fig.suptitle('Leg-level rolling CAPM betas, 2011--2025 '
                 '(grey bands = panic months, $\\pi^{\\mathrm{filter}} \\geq 0.5$)',
                 fontsize=11, y=0.995)
    fig.tight_layout()
    out_pdf = root / cfg.PLOTS_DIR / 'leg_betas_rolling.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Wrote {out_pdf}")


if __name__ == '__main__':
    main()
