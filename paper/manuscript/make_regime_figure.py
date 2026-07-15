"""Data-chapter regime figure + realized separation stats for the applied paper.

Faithful to production: the standardized drawdown feature is read from the built
panel, and the filtered panic probability is the ACTUAL out-of-sample signal that
traded in the expanding-window walk (walk_returns.csv, rule_r == DD-only). No model
is refit here; this script only reads saved outputs and plots/summarizes them.

Outputs:
  plots/dd_regime_signal.{pdf,png}   two panels: DD feature (full sample) + OOS pi
  prints realized separation statistics used in Section 4.3 prose.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
PANEL = os.path.join(REPO, 'paper', 'results', 'data', 'panel.parquet')
WALK = os.path.join(REPO, 'paper', 'results', 'walk_returns.csv')
OUTDIR = os.path.join(HERE, 'plots')
os.makedirs(OUTDIR, exist_ok=True)

# Standard NBER recession bands overlapping the sample (for shading only).
NBER = [('1990-07-01', '1991-03-31'), ('2001-03-01', '2001-11-30'),
        ('2007-12-01', '2009-06-30'), ('2020-02-01', '2020-04-30')]

# The macro panel extends back to 1970; the applied study's panel starts 1990-12
# (config.PANEL_START), so clip the figure to the window the model actually uses.
PANEL_START = pd.Timestamp('1990-12-01')
OOS_START = pd.Timestamp('2011-01-01')
panel = pd.read_parquet(PANEL, columns=['date', 'DD', 'DD_z'])
panel['date'] = pd.to_datetime(panel['date'])
panel = panel[panel['date'] >= PANEL_START].sort_values('date').reset_index(drop=True)

walk = pd.read_csv(WALK, parse_dates=['date'])
pi = walk[walk['rule'] == 'rule_r'][['date', 'pi']].sort_values('date').reset_index(drop=True)
assert (walk[walk['rule'] == 'rule_r']['combo'] == 'DD').all(), 'rule_r is not DD-only'

# ── realized separation: merge the traded pi onto the DD panel ────────────────
m = panel.merge(pi, on='date', how='inner')          # OOS months only (2011-2025)
panic = m['pi'] >= 0.5
n_oos, n_panic = len(m), int(panic.sum())
print('=== Realized DD-only regime signal (OOS walk, rule_r) ===')
print(f'OOS months (2011-01..2025-11): {n_oos}')
print(f'panic months (pi>=0.5):        {n_panic}  ({n_panic/n_oos:.1%})')
print(f'mean pi:                       {m["pi"].mean():.3f}   median {m["pi"].median():.3f}')
print(f'mean DD (raw)  panic vs calm:  {m.loc[panic,"DD"].mean():+.3f}  vs  {m.loc[~panic,"DD"].mean():+.3f}')
print(f'mean DD_z      panic vs calm:  {m.loc[panic,"DD_z"].mean():+.3f}  vs  {m.loc[~panic,"DD_z"].mean():+.3f}')
print(f'corr(pi, -DD_z):               {np.corrcoef(m["pi"], -m["DD_z"])[0,1]:+.3f}')
yr = m.assign(year=m['date'].dt.year).groupby('year').agg(
    mean_pi=('pi', 'mean'), panic_mo=('pi', lambda s: int((s >= 0.5).sum())))
print('\nby year (mean pi | # panic months):')
print(yr.to_string())
# panic episodes (contiguous runs of pi>=0.5)
episodes, run = [], []
for _, r in m.iterrows():
    if r['pi'] >= 0.5:
        run.append(r['date'])
    elif run:
        episodes.append((run[0], run[-1], len(run))); run = []
if run:
    episodes.append((run[0], run[-1], len(run)))
print('\npanic episodes (start, end, n_months):')
for s, e, n in episodes:
    print(f'  {s:%Y-%m} .. {e:%Y-%m}  ({n} mo)')

# ── figure ───────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 150})
ACCENT = '#b2182b'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True,
                               gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.15})


def shade(ax):
    for s, e in NBER:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color='0.85', lw=0, zorder=0)


# Panel A: the HMM input feature over the study window
shade(ax1)
ax1.plot(panel['date'], panel['DD_z'], color='0.25', lw=0.9)
ax1.axhline(0, color='0.6', lw=0.6, ls=':')
ax1.axvline(OOS_START, color='0.4', lw=0.8, ls='-.')
ax1.set_ylabel('Standardized\ndrawdown')
ax1.set_title('(a) HMM input feature: market drawdown, 1990--2025',
              loc='left', fontsize=10)

# Panel B: the OOS filtered panic probability that actually traded
shade(ax2)
ax2.fill_between(pi['date'], 0, pi['pi'], color=ACCENT, alpha=0.85, lw=0)
ax2.axhline(0.5, color='0.4', lw=0.7, ls='--')
ax2.axvline(OOS_START, color='0.4', lw=0.8, ls='-.')
ax2.text(OOS_START, 0.92, ' out-of-sample walk begins', fontsize=7.5,
         color='0.35', ha='left', va='top')
ax2.set_ylim(0, 1.02)
ax2.set_xlim(PANEL_START, panel['date'].max())
ax2.set_ylabel(r'Filtered $\pi_t$')
ax2.set_xlabel('Year')
ax2.set_title(r'(b) Out-of-sample filtered panic probability $\pi_t^{\mathrm{filter}}$ (drawdown-only), 2011--2025',
              loc='left', fontsize=10)
ax2.legend(handles=[Patch(facecolor='0.85', label='NBER recession')],
           loc='upper right', frameon=False, fontsize=8)

fig.savefig(os.path.join(OUTDIR, 'dd_regime_signal.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTDIR, 'dd_regime_signal.png'), bbox_inches='tight')
print('\nwrote plots/dd_regime_signal.{pdf,png}')
