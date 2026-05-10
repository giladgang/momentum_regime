"""
selection_rank_analysis.py
==========================
Analyzes what stocks XGB selects by computing the cross-sectional percentile
rank of long/short leg stocks at each of the 12 momentum horizons, split by
calm vs panic regime.

Pipeline:
1. XGBoost outputs predicted return for every stock every month
2. Rank stocks by score, assign top/bottom decile to long/short legs (NYSE breakpoints)
3. For each long/short stock, compute its percentile rank at each momentum horizon
   within that month's full cross-section
4. Average percentile ranks across all stocks in that leg for each month
5. Aggregate by year, then compute mean and 25th-75th percentile bands across years

Outputs:
- selection_rank_analysis.csv: full table (horizon x regime x leg)
- chart2_rank_by_horizon_v2.png: long leg percentile rank, monthly bands
- chart_ls_rank_yearly.png: long + short leg percentile rank, yearly bands

Usage:
    python scripts/selection_rank_analysis.py
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_FEE

# ── Load data ──
print("Loading artefacts ...")
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
FEATURES = art['FEATURES']

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

test_pi = test.merge(pi_monthly.reset_index()[['date', 'pi_filter']].rename(
    columns={'pi_filter': 'pi_month'}), on='date', how='left')
test_pi['regime'] = np.where(test_pi['pi_month'] >= 0.5, 'Panic', 'Calm')

# ── Step 2: Assign long/short legs using production scores ──
print("Assigning portfolio legs ...")
test_pi['leg'] = 'middle'
for date, grp in test_pi.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score_xgb'].dropna()
    if len(nyse) < 10:
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    test_pi.loc[grp[grp['score_xgb'] >= hi].index, 'leg'] = 'long'
    test_pi.loc[grp[grp['score_xgb'] <= lo].index, 'leg'] = 'short'

horizons = list(range(1, 13))

# ── Steps 3-5: Compute percentile ranks ──
print("Computing percentile ranks ...")
rows = []
monthly_data = {}  # cache for charts

for h in horizons:
    col = f'mom_{h}'
    for regime in ['Calm', 'Panic']:
        for leg in ['long', 'short']:
            mask = test_pi['regime'] == regime
            monthly = []
            for date, grp in test_pi[mask].groupby('date'):
                leg_mask = grp['leg'] == leg
                if leg_mask.sum() == 0:
                    continue
                ranks = grp[col].rank(pct=True)
                monthly.append({
                    'year': date.year,
                    'pctile': ranks[leg_mask].mean() * 100
                })

            mdf = pd.DataFrame(monthly)

            # Monthly stats
            all_monthly = mdf['pctile'].values
            monthly_data[(h, regime, leg)] = all_monthly

            # Yearly stats
            yearly = mdf.groupby('year')['pctile'].mean()
            rows.append({
                'Horizon': h,
                'Regime': regime,
                'Leg': leg,
                'Mean (yearly)': yearly.mean(),
                'Std (yearly)': yearly.std(),
                'P25 (yearly)': yearly.quantile(0.25),
                'P75 (yearly)': yearly.quantile(0.75),
                'Min (yearly)': yearly.min(),
                'Max (yearly)': yearly.max(),
                'N_years': len(yearly),
                'Pct_above_50 (yearly)': (yearly > 50).mean() * 100,
                'Mean (monthly)': np.mean(all_monthly),
                'P25 (monthly)': np.percentile(all_monthly, 25),
                'P75 (monthly)': np.percentile(all_monthly, 75),
            })

df = pd.DataFrame(rows)
df.to_csv('results/thesis/selection_rank_analysis.csv', index=False, float_format='%.1f')
print(f"Saved: selection_rank_analysis.csv ({len(df)} rows)")

# ── Print summary ──
for leg in ['long', 'short']:
    print(f"\n{'='*80}")
    print(f"  {leg.upper()} LEG")
    print(f"{'='*80}")
    print(f"{'Horizon':>8s} {'Calm Mean':>10s} {'Calm P25':>9s} {'Calm P75':>9s} "
          f"{'Panic Mean':>11s} {'Panic P25':>10s} {'Panic P75':>10s} {'Spread':>8s}")
    print("-" * 80)
    for h in horizons:
        c = df[(df['Horizon'] == h) & (df['Regime'] == 'Calm') & (df['Leg'] == leg)].iloc[0]
        p = df[(df['Horizon'] == h) & (df['Regime'] == 'Panic') & (df['Leg'] == leg)].iloc[0]
        spread = c['Mean (yearly)'] - p['Mean (yearly)']
        print(f"  mom_{h:<3d} {c['Mean (yearly)']:>9.1f}% {c['P25 (yearly)']:>8.1f}% "
              f"{c['P75 (yearly)']:>8.1f}% {p['Mean (yearly)']:>10.1f}% "
              f"{p['P25 (yearly)']:>9.1f}% {p['P75 (yearly)']:>9.1f}% {spread:>+7.1f}%")

# ── Chart 1: Long leg with monthly bands ──
print("\nGenerating charts ...")
x = np.arange(1, 13)

fig, ax = plt.subplots(figsize=(11, 6))
for regime, color, marker in [('Calm', 'steelblue', 'o'), ('Panic', '#E53935', 's')]:
    means = [np.mean(monthly_data[(h, regime, 'long')]) for h in horizons]
    p25s = [np.percentile(monthly_data[(h, regime, 'long')], 25) for h in horizons]
    p75s = [np.percentile(monthly_data[(h, regime, 'long')], 75) for h in horizons]
    ax.plot(x, means, f'{marker}-', color=color, linewidth=2.5, markersize=8, label=regime, zorder=5)
    ax.fill_between(x, p25s, p75s, color=color, alpha=0.15,
                    label=f'{regime} 25th-75th pctile')
ax.axhline(50, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.text(12.3, 50.5, '50%\n(neutral)', fontsize=8, color='grey', va='bottom')
ax.set_xticks(x)
ax.set_xticklabels([f'{h}' for h in horizons], fontsize=10)
ax.set_xlabel('Momentum lookback horizon (months)', fontsize=11)
ax.set_ylabel('Average percentile rank of selected stocks (%)', fontsize=11)
ax.set_title('Are the selected stocks winners or losers at each horizon?\n'
             'Above 50% = winners relative to cross-section, below 50% = losers',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)
ax.set_ylim(25, 70)
plt.tight_layout()
fig.savefig('plots/chart2_rank_by_horizon_v2.png', dpi=150, bbox_inches='tight')
fig.savefig('plots/chart2_rank_by_horizon_v2.pdf', bbox_inches='tight')
plt.close(fig)

# ── Chart 2: L/S with yearly bands ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
for i, (leg, title) in enumerate([('long', 'Long leg: what XGB buys'),
                                    ('short', 'Short leg: what XGB sells')]):
    ax = axes[i]
    for regime, color, marker in [('Calm', 'steelblue', 'o'), ('Panic', '#E53935', 's')]:
        sub = df[(df['Regime'] == regime) & (df['Leg'] == leg)].sort_values('Horizon')
        ax.plot(x, sub['Mean (yearly)'].values, f'{marker}-', color=color,
                linewidth=2.5, markersize=8, label=regime, zorder=5)
        ax.fill_between(x, sub['P25 (yearly)'].values, sub['P75 (yearly)'].values,
                        color=color, alpha=0.12)
    ax.axhline(50, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}' for h in horizons], fontsize=10)
    ax.set_xlabel('Momentum lookback horizon (months)', fontsize=11)
    if i == 0:
        ax.set_ylabel('Percentile rank of selected stocks (%)\n(>50 = winners, <50 = losers)',
                      fontsize=10)
axes[0].set_ylim(20, 75)
plt.suptitle('Long-short stock selection by momentum horizon and regime\n'
             '(yearly averages, bands = 25th-75th percentile across years)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig('plots/chart_ls_rank_yearly.png', dpi=150, bbox_inches='tight')
fig.savefig('plots/chart_ls_rank_yearly.pdf', bbox_inches='tight')
plt.close(fig)

print("Saved: chart2_rank_by_horizon_v2.png + .pdf")
print("Saved: chart_ls_rank_yearly.png + .pdf")

# ── Chart 3: Z-score + |SHAP| (4-panel figure for thesis) ──
print("Computing z-scores and SHAP for term structure figure ...")

shap_values = art['shap_values']
FEATURES = art['FEATURES']
mom_indices = [FEATURES.index(f'mom_{h}') for h in horizons]
regime_arr = test_pi['regime'].values
leg_arr = test_pi['leg'].values

# Z-scores (mean and std across stocks)
z_data = {}
z_std = {}
for regime_val in ['Calm', 'Panic']:
    for leg_val in ['long', 'short']:
        zs_mean = []
        zs_std = []
        for h in horizons:
            col = f'mom_{h}'
            mask = test_pi['regime'] == regime_val
            monthly_z_mean = []
            monthly_z_std = []
            for date, grp in test_pi[mask].groupby('date'):
                leg_mask = grp['leg'] == leg_val
                if leg_mask.sum() == 0: continue
                mean = grp[col].mean()
                std = grp[col].std()
                if std > 0:
                    stock_zs = (grp.loc[leg_mask, col] - mean) / std
                    monthly_z_mean.append(stock_zs.mean())
                    monthly_z_std.append(stock_zs.std() / np.sqrt(len(stock_zs)))
            zs_mean.append(np.mean(monthly_z_mean))
            zs_std.append(np.mean(monthly_z_std))
        z_data[(regime_val, leg_val)] = zs_mean
        z_std[(regime_val, leg_val)] = zs_std

# |SHAP| per leg, each leg sums to 100%
pct_shap = {}
for regime_val in ['Calm', 'Panic']:
    for leg_val in ['long', 'short']:
        mask = (regime_arr == regime_val) & (leg_arr == leg_val)
        abs_shap = np.abs(shap_values[mask][:, mom_indices]).mean(axis=0)
        pct_shap[(regime_val, leg_val)] = abs_shap / abs_shap.sum() * 100

w = 0.35
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)

# Top row: Z-scores (shared y-axis)
axes[0, 0].sharey(axes[0, 1])
for i, regime_val in enumerate(['Calm', 'Panic']):
    ax = axes[0, i]
    ax.plot(x, z_data[(regime_val, 'long')], 'o-', color='#2196F3', linewidth=2.5,
            markersize=8, label='Long leg', zorder=5)
    ax.plot(x, z_data[(regime_val, 'short')], 's-', color='#E53935', linewidth=2.5,
            markersize=8, label='Short leg', zorder=5)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    long_mean = np.array(z_data[(regime_val, 'long')])
    long_sd = np.array(z_std[(regime_val, 'long')])
    short_mean = np.array(z_data[(regime_val, 'short')])
    short_sd = np.array(z_std[(regime_val, 'short')])
    ax.fill_between(x, long_mean - long_sd, long_mean + long_sd, alpha=0.12, color='#2196F3')
    ax.fill_between(x, short_mean - short_sd, short_mean + short_sd, alpha=0.12, color='#E53935')
    n = int((test_pi.drop_duplicates('date')['regime'] == regime_val).sum())
    ax.set_title(f'{regime_val} ({n} months)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    if i == 0:
        ax.set_ylabel('Z-score vs cross-section', fontsize=11)

# Bottom row: |SHAP| percentage (shared y-axis, each leg sums to 100%)
axes[1, 0].sharey(axes[1, 1])
for i, regime_val in enumerate(['Calm', 'Panic']):
    ax = axes[1, i]
    ax.bar(x - w/2, pct_shap[(regime_val, 'long')], w, label='Long leg',
           color='#2196F3', alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, pct_shap[(regime_val, 'short')], w, label='Short leg',
           color='#E53935', alpha=0.85, edgecolor='white')
    ax.set_title(f'{regime_val}',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}' for h in horizons], fontsize=10)
    ax.set_xlabel('Momentum lookback horizon (months)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    if i == 0:
        ax.set_ylabel('Share of momentum |SHAP| (%)', fontsize=11)

plt.suptitle('Momentum term structure by regime: stock selection and feature importance',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig('plots/thesis/zscore_and_absshap_v3.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: zscore_and_absshap_v3.png")

# ── Export LaTeX table: table_zscore_shap_detail.tex ──
TABLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

tex_lines = []
tex_lines.append(r'\begin{table}[H]')
tex_lines.append(r'\centering')
tex_lines.append(r'\small')
tex_lines.append(r'\begin{tabular}{r rr rr rr rr}')
tex_lines.append(r'\toprule')
tex_lines.append(r' & \multicolumn{4}{c}{Z-score} & \multicolumn{4}{c}{SHAP share (\%)} \\')
tex_lines.append(r'\cmidrule(lr){2-5} \cmidrule(lr){6-9}')
tex_lines.append(r'Horizon & \multicolumn{2}{c}{Calm} & \multicolumn{2}{c}{Panic} & \multicolumn{2}{c}{Calm} & \multicolumn{2}{c}{Panic} \\')
tex_lines.append(r' & Long & Short & Long & Short & Long & Short & Long & Short \\')
tex_lines.append(r'\midrule')

for h in horizons:
    i = h - 1  # index into z_data / pct_shap lists
    z_cl = z_data[('Calm', 'long')][i]
    z_cs = z_data[('Calm', 'short')][i]
    z_pl = z_data[('Panic', 'long')][i]
    z_ps = z_data[('Panic', 'short')][i]
    s_cl = pct_shap[('Calm', 'long')][i]
    s_cs = pct_shap[('Calm', 'short')][i]
    s_pl = pct_shap[('Panic', 'long')][i]
    s_ps = pct_shap[('Panic', 'short')][i]
    tex_lines.append(
        f'{h} & {z_cl:+.2f} & {z_cs:+.2f} & {z_pl:+.2f} & {z_ps:+.2f} '
        f'& {s_cl:.1f} & {s_cs:.1f} & {s_pl:.1f} & {s_ps:.1f} \\\\'
    )

tex_lines.append(r'\bottomrule')
tex_lines.append(r'\end{tabular}')
tex_lines.append(r"\caption{Z-score and SHAP share by momentum horizon, regime, and portfolio leg. Z-scores measure how far above or below the cross-sectional average the selected stocks are at each lookback. SHAP share measures each horizon's contribution to the model's momentum decision (each leg sums to 100\%).}")
tex_lines.append(r'\label{tab:zscore_shap_detail}')
tex_lines.append(r'\end{table}')

tex_path = os.path.join(TABLES_DIR, 'table_zscore_shap_detail.tex')
with open(tex_path, 'w') as f:
    f.write('\n'.join(tex_lines) + '\n')
print(f"Saved: {tex_path}")

print("Done.")
