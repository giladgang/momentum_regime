"""Regenerate ALL applied figures in the original thesis's visual formats.

Thesis style references:
- regime_probabilities.png: wide, palevioletred area fill, dotted 0.5 line,
  gray NBER bands, gray episode labels on top.
- cs_performance_regime_shaded.png: (12,7), log wealth, SR legend, red main
  line, gray market, dashed comparators, light red panic vspans, grid 0.3.
- zscore_l2_k4_panel_c*.png: one figure per cluster, steelblue members,
  marked centroid, bold title 'Cluster k (name) -- n=.., pi=.., Sharpe=..
  [lo, hi]' with block-bootstrap CI (block 6, 5000 reps), ylim symmetric.
- shap_per_horizon_by_leg.png: grouped % bars, steelblue/red, dotted y-grid.
- zscore_long_heatmap.png: RdBu_r, bold title, Jan y-ticks, rotated x labels.
- risk_aversion_dual_util.pdf: serif font, 2x2, white-faced square markers,
  red highlight dots + bold red annotations at gamma in {0,0.1,0.5,1.0}.

All numbers computed from canonical applied outputs; no thesis data used.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

REPO = '/Users/giladgang/momentum_regime'
os.chdir(REPO)
sys.path.insert(0, REPO)
OUT = 'paper/manuscript/plots'
MOMS = [f'mom_{h}' for h in range(1, 13)]
NBER = [('2001-03-01','2001-11-30'), ('2007-12-01','2009-06-30'), ('2020-02-01','2020-04-30')]
NBER_FULL = [('1990-07-01','1991-03-31')] + NBER

plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})

w = pd.read_csv('paper/results/walk_returns.csv', parse_dates=['date'])
rr = w[w['rule'] == 'rule_r'].set_index('date').sort_index()
panic_months = rr.index[rr['pi'] >= 0.5]

# ── 1. DD feature series (ch4.2, features_hmm analog) ────────────────────────
panel = pd.read_parquet('paper/results/data/panel.parquet', columns=['date','DD_z'])
panel['date'] = pd.to_datetime(panel['date'])
panel = panel[panel['date'] >= '1990-12-01'].dropna().sort_values('date')
fig, ax = plt.subplots(figsize=(14, 4))
for s, e in NBER_FULL:
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color='lightgray', alpha=0.6, zorder=0)
ax.plot(panel['date'], panel['DD_z'], color='steelblue', lw=1.0)
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(pd.Timestamp('2011-01-01'), color='black', ls='--', lw=1.2, label='Walk start')
ax.set_xlabel('Date', fontsize=13); ax.set_ylabel('Standardized drawdown', fontsize=13)
ax.legend(loc='lower left', fontsize=10); ax.grid(True, alpha=0.3)
ax.margins(x=0.01)
fig.tight_layout()
fig.savefig(f'{OUT}/dd_feature_series.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/dd_feature_series.png', bbox_inches='tight'); plt.close(fig)

# ── 2. Filtered probability (ch4.3, regime_probabilities analog) ─────────────
fig, ax = plt.subplots(figsize=(14, 4))
x0, x1 = rr.index.min(), rr.index.max()
for s, e in NBER:
    s, e = pd.Timestamp(s), pd.Timestamp(e)
    if e >= x0 and s <= x1:
        ax.axvspan(max(s, x0), min(e, x1), color='lightgray', alpha=0.6, zorder=0)
ax.fill_between(rr.index, 0, rr['pi'], color='palevioletred', alpha=0.75, lw=1.0,
                edgecolor='crimson')
ax.axhline(0.5, color='black', lw=0.7, ls=':')
for lbl, x in [('2011 downgrade','2011-09-01'), ('2015--16','2015-11-01'),
               ('Late-2018','2018-11-01'), ('COVID','2020-03-01'),
               ('2022 hikes','2022-06-01'), ('Tariff','2025-03-01')]:
    ax.text(pd.Timestamp(x), 1.03, lbl.replace('--','–'), color='gray',
            fontsize=9, ha='center')
ax.set_ylim(0, 1.0)
ax.set_xlim(x0 - pd.offsets.MonthEnd(2), x1 + pd.offsets.MonthEnd(2))
ax.set_xlabel('Date', fontsize=13)
ax.set_ylabel(r'$\pi_t^{\mathrm{filter}}$', fontsize=13)
fig.tight_layout()
fig.savefig(f'{OUT}/applied_regime_probabilities.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/applied_regime_probabilities.png', bbox_inches='tight'); plt.close(fig)

# ── 3. Cumulative wealth with comparators (ch5, cs_performance analog) ───────
from paper.src import portfolio
sel = pd.read_csv('paper/results/selections.csv')
am_sel = sel[sel['rule'] == 'argmax']
nopi, m12 = [], []
for _, row in am_sel.iterrows():
    p = f"paper/results/xsec/xsec_{int(row['year'])}_{row['combo'].replace('+','_')}.parquet"
    if not os.path.exists(p): continue
    x = pd.read_parquet(p)
    nopi.append(portfolio.long_only_top(x, 'score_nopi', 0.10)[0])
    m12.append(portfolio.long_only_top(x, 'mom_12', 0.10)[0])
nopi = pd.concat(nopi).sort_index().reindex(rr.index)
m12 = pd.concat(m12).sort_index().reindex(rr.index)
def sh(x): return float(np.mean(x)/np.std(x, ddof=1)*np.sqrt(12))
series = [('Benchmark top-1000 VW', rr['bench_ret'], 'gray', '-', 1.8),
          ('Fixed 12-1 long-only', m12, 'darkred', '--', 1.2),
          ('XGB (mom only)', nopi, 'blue', '--', 1.2),
          (r'XGB (mom+$\pi$, DD)', rr['strat_ret'], 'red', '-', 2.4)]
fig, ax = plt.subplots(figsize=(12, 7))
for d in panic_months:
    ax.axvspan(d, d + pd.offsets.MonthEnd(1), alpha=0.1, color='red', zorder=0)
for name, r, c, ls, lw in series:
    cum = (1 + r).cumprod()
    ax.plot(cum.index, cum, label=f'{name} (SR={sh(r):.2f})', color=c, linestyle=ls, lw=lw)
ax.set_yscale('log')
ax.set_ylabel('Cumulative Wealth (from \\$1)', fontsize=13)
ax.set_title('Long-Only Portfolio Performance (gross)', fontsize=13)
ax.legend(fontsize=9, loc='upper left'); ax.grid(True, alpha=0.3)
ax.axhline(1, color='black', lw=0.5, ls=':')
ax.margins(x=0.01)
fig.tight_layout()
fig.savefig(f'{OUT}/applied_cumulative_ownpi.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/applied_cumulative_ownpi.png', bbox_inches='tight'); plt.close(fig)

# ── 4. SHAP per-horizon grouped bars (ch5, by-leg analog) ────────────────────
s = pd.read_csv('paper/results/extras/shap_sums_by_year.csv')
all_sh = s[s['scope']=='overall'][MOMS].sum(); all_sh = 100*all_sh/all_sh.sum()
lg_sh = s[s['scope']=='long_leg'][MOMS].sum(); lg_sh = 100*lg_sh/lg_sh.sum()
x = np.arange(1, 13); width = 0.38
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, all_sh.values, width, label='All stock-months', color='tab:blue')
ax.bar(x + width/2, lg_sh.values, width, label='Long-leg picks', color='tab:red')
ax.set_xticks(x)
ax.set_xlabel('Momentum horizon (months)', fontsize=13)
ax.set_ylabel('Share of momentum $|$SHAP$|$ (\\%)', fontsize=13)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.4, axis='y', linestyle=':')
fig.tight_layout()
fig.savefig(f'{OUT}/shap_per_horizon_applied.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/shap_per_horizon_applied.png', bbox_inches='tight'); plt.close(fig)

# ── 5. z-curve heatmap (ch5, thesis heatmap style) ───────────────────────────
z = pd.read_csv('paper/results/tables/applied_zscore_long_by_month.csv', parse_dates=['date'])
z = z.sort_values('date').set_index('date')
fig, ax = plt.subplots(figsize=(10, 12))
vmax = np.nanmax(np.abs(z[MOMS].values))
im = ax.imshow(z[MOMS].values, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
jan = [i for i, d in enumerate(z.index) if d.month == 1]
ax.set_yticks(jan); ax.set_yticklabels([z.index[i].strftime('%Y-%m') for i in jan])
ax.set_xticks(range(12)); ax.set_xticklabels(MOMS, rotation=30, ha='right')
ax.set_xlabel('Momentum lookback horizon', fontsize=13)
ax.set_ylabel('Month (chronological)', fontsize=13)
ax.set_title(f'Long-leg z-curves over the test period ({len(z)} months)',
             fontsize=14, fontweight='bold')
cb = fig.colorbar(im, ax=ax, shrink=0.6)
cb.set_label('long-leg cross-sectional z-score', fontsize=12)
fig.tight_layout()
fig.savefig(f'{OUT}/applied_zscore_long_heatmap.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/applied_zscore_long_heatmap.png', bbox_inches='tight'); plt.close(fig)

# ── 6. Four cluster panels with bootstrap CIs (thesis panel style) ───────────
lab = pd.read_csv('paper/results/tables/applied_cluster_k4_labels.csv', parse_dates=['date'])
d = lab.merge(z.reset_index(), on='date').merge(
    rr[['strat_ret','pi']].reset_index(), on='date')
NAMES = ['strong continuation', 'calm, benchmark-like', 'mild reversal', 'deep crisis']
rng = np.random.default_rng(42)
def block_boot_sharpe(r, reps=5000, block=6):
    r = np.asarray(r); n = len(r); out = np.empty(reps)
    for k in range(reps):
        idx = []
        while len(idx) < n:
            s0 = rng.integers(0, n)
            idx.extend([(s0 + j) % n for j in range(block)])
        rs = r[np.array(idx[:n])]
        out[k] = rs.mean()/rs.std(ddof=1)*np.sqrt(12)
    return np.percentile(out, [2.5, 97.5])
for k in range(4):
    m = d[d['cluster'] == k]
    r = m['strat_ret'].values
    shp = r.mean()/r.std(ddof=1)*np.sqrt(12)
    lo, hi = block_boot_sharpe(r)
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in m.iterrows():
        ax.plot(range(1, 13), row[MOMS].values, color='steelblue', alpha=0.35, lw=0.9)
    ax.plot(range(1, 13), m[MOMS].mean().values, 'o-', color='steelblue', lw=3,
            ms=7, label=f'centroid ({len(m)} mo)')
    ax.axhline(0, color='gray', lw=0.8)
    lim = max(1.5, float(np.ceil(np.nanmax(np.abs(m[MOMS].values)) * 10) / 10))
    ax.set_ylim(-lim, lim)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel('Horizon (months)', fontsize=13)
    ax.set_ylabel('Cross-sectional z-score', fontsize=13)
    ax.set_title(f'Cluster {k+1} ({NAMES[k]}) -- n={len(m)}, '
                 f'$\\bar{{\\pi}}={m["pi"].mean():.2f}$, '
                 f'Sharpe={shp:.2f} [{lo:.2f}, {hi:.2f}]',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{OUT}/applied_zscore_k4_panel_c{k}.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/applied_zscore_k4_panel_c{k}.png', bbox_inches='tight')
    plt.close(fig)
    print(f'C{k+1}: Sharpe {shp:.2f} [{lo:.2f}, {hi:.2f}] ylim {lim}')

# ── 7. Risk-aversion sweep (thesis dual_util style, serif) ───────────────────
sw = pd.read_csv('paper/results/tables/risk_aversion_sweep.csv')
with plt.rc_context({'font.family': 'serif',
                     'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'mathtext.fontset': 'cm', 'legend.fontsize': 10}):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [('sharpe', 'Sharpe Ratio', 'Sharpe', '#1f4e8c', '{:.2f}'),
              ('ann_ret', 'Annualised Return', 'Ann. Return', 'darkgreen', '{:.1%}'),
              ('ann_vol', 'Annualised Volatility', 'Ann. Vol', 'darkred', '{:.1%}'),
              ('max_dd', 'Maximum Drawdown', 'Max DD', 'rebeccapurple', '{:.1%}')]
    HI = [0.0, 0.1, 0.5, 1.0]
    for ax, (col, title, ylab, color, fmt) in zip(axes.flat, panels):
        ax.plot(sw['gamma'], sw[col], 's-', color=color, lw=2.0, ms=5,
                markerfacecolor='white', markeredgewidth=1.8)
        for g in HI:
            row = sw[np.isclose(sw['gamma'], g)]
            if not len(row): continue
            yi = float(row[col].iloc[0])
            ax.plot([g], [yi], 'o', color='crimson', ms=9, zorder=5)
            ax.annotate(fmt.format(yi), (g, yi), textcoords='offset points',
                        xytext=(8, 8), fontsize=9, color='crimson', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(r'Risk aversion $\gamma$'); ax.set_ylabel(ylab)
        ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)
    fig.suptitle(r'Risk-Aversion Sweep: Sharpe, Return, Volatility, Drawdown vs. $\gamma$',
                 fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()
    fig.savefig(f'{OUT}/risk_aversion_applied.pdf', bbox_inches='tight',
                facecolor='white')
    fig.savefig(f'{OUT}/risk_aversion_applied.png', bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
print('ALL FIGURES DONE')
