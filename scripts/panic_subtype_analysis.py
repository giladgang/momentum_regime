"""
panic_subtype_analysis.py
=========================
Split panic months into sub-types based on market behavior and compute
z-score profiles for each. Tests whether the z-score convergence at
months 6-12 is an averaging artefact across different panic phases.

NOTE FOR AI ASSISTANTS: Do NOT modify the main pipeline
(cross_sectional_model.py). This is a standalone analysis script.

Sub-types (based on concurrent market return):
  Crash:    panic month with negative market return
  Recovery: panic month with positive market return

Outputs:
  - plots/zscore_panic_subtypes.png
  - tables/table_panic_subtypes.tex
  - Console: z-score profiles, performance by subtype

Usage:
    python scripts/panic_subtype_analysis.py
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pickle, sys, os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_FEE, TABLES_DIR

os.makedirs(TABLES_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

print("[ 1/5 ] Loading artefacts ...")
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)

test     = art['test'].copy()
FEATURES = art['FEATURES']
r_mkt    = art['r_mkt']

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

test = test.merge(
    pi_monthly.reset_index()[['date', 'pi_filter']].rename(columns={'pi_filter': 'pi_month'}),
    on='date', how='left'
)
test['regime'] = np.where(test['pi_month'] >= 0.5, 'Panic', 'Calm')

# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO LEG ASSIGNMENT (production method)
# ══════════════════════════════════════════════════════════════════════════════

print("[ 2/5 ] Assigning portfolio legs ...")
test['leg'] = 'middle'
for date, grp in test.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score_xgb'].dropna()
    if len(nyse) < 10:
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    test.loc[grp[grp['score_xgb'] >= hi].index, 'leg'] = 'long'
    test.loc[grp[grp['score_xgb'] <= lo].index, 'leg'] = 'short'

# ══════════════════════════════════════════════════════════════════════════════
# PANIC SUBTYPE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

print("[ 3/5 ] Classifying panic subtypes ...")

panic_dates = test[test['regime'] == 'Panic'][['date']].drop_duplicates().sort_values('date')
panic_dates = panic_dates.merge(
    r_mkt.reset_index().rename(columns={r_mkt.name if hasattr(r_mkt, 'name') and r_mkt.name else 'ret_next': 'r_mkt'}),
    on='date', how='left'
)

# If the merge brought in the wrong column name, fix it
if 'r_mkt' not in panic_dates.columns:
    # r_mkt is a Series, its values are the market returns
    mkt_df = pd.DataFrame({'date': r_mkt.index, 'r_mkt': r_mkt.values})
    panic_dates = panic_dates.drop(columns=[c for c in panic_dates.columns if c != 'date'])
    panic_dates = panic_dates.merge(mkt_df, on='date', how='left')

panic_dates['subtype'] = np.where(panic_dates['r_mkt'] >= 0, 'Recovery', 'Crash')

n_crash = (panic_dates['subtype'] == 'Crash').sum()
n_recovery = (panic_dates['subtype'] == 'Recovery').sum()

print(f"  Total panic months: {len(panic_dates)}")
print(f"  Crash (market down):  {n_crash}")
print(f"  Recovery (market up): {n_recovery}")

# Merge subtype into test
test = test.merge(panic_dates[['date', 'subtype']], on='date', how='left')
test.loc[test['regime'] == 'Calm', 'subtype'] = 'Calm'

# ══════════════════════════════════════════════════════════════════════════════
# Z-SCORE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

print("[ 4/5 ] Computing z-scores ...")

horizons = list(range(1, 13))
mom_cols = [f'mom_{h}' for h in horizons]
subtypes = ['Calm', 'Crash', 'Recovery']

# Precompute cross-sectional mean and std per date (once, not per subtype)
cs_stats = {}
for date, grp in test.groupby('date'):
    stats = {}
    for col in mom_cols:
        vals = grp[col]
        stats[col] = (vals.mean(), vals.std())
    cs_stats[date] = stats

z_data = {}
z_se = {}

for subtype in subtypes:
    for leg in ['long', 'short']:
        zs_by_horizon = [[] for _ in horizons]
        sub_mask = test['subtype'] == subtype
        sub_data = test[sub_mask]

        for date, grp in sub_data.groupby('date'):
            leg_mask = grp['leg'] == leg
            if leg_mask.sum() == 0:
                continue
            for i, col in enumerate(mom_cols):
                cs_mean, cs_std = cs_stats[date][col]
                if cs_std > 0:
                    stock_zs = (grp.loc[leg_mask, col] - cs_mean) / cs_std
                    zs_by_horizon[i].append(stock_zs.mean())

        means = []
        ses = []
        for i in range(len(horizons)):
            arr = zs_by_horizon[i]
            means.append(np.mean(arr) if arr else 0)
            ses.append(np.std(arr) / np.sqrt(len(arr)) if len(arr) > 1 else 0)

        z_data[(subtype, leg)] = np.array(means)
        z_se[(subtype, leg)] = np.array(ses)

# Print z-scores
for subtype in subtypes:
    n = len(test[test['subtype'] == subtype]['date'].unique())
    print(f"\n  {subtype} ({n} months):")
    print(f"  {'Horizon':>8s} {'Long z':>8s} {'Short z':>8s} {'Spread':>8s}")
    print(f"  " + "-" * 36)
    for i, h in enumerate(horizons):
        lz = z_data[(subtype, 'long')][i]
        sz = z_data[(subtype, 'short')][i]
        print(f"  mom_{h:>2d}   {lz:>+7.3f}  {sz:>+7.3f}  {lz - sz:>+7.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE BY SUBTYPE (production portfolio construction)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[ 5/5 ] Computing performance by subtype ...")


def long_short_port_subset(df_data, score_col, date_set, fee=TRADING_FEE):
    """Production long-short portfolio on a subset of dates, with turnover costs."""
    monthly = []
    prev_lw, prev_sw = {}, {}
    for date, grp in df_data.groupby('date'):
        if date not in date_set:
            continue
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0:
            continue
        lme = longs['me'].sum()
        new_lw = (longs.set_index('permno')['me'] / lme).to_dict()
        r_long = (longs['ret_fwd'] * longs['me']).sum() / lme
        sme = shorts['me'].sum()
        new_sw = (shorts.set_index('permno')['me'] / sme).to_dict()
        r_short = (shorts['ret_fwd'] * shorts['me']).sum() / sme
        tl = sum(abs(new_lw.get(p, 0) - prev_lw.get(p, 0))
                 for p in set(new_lw) | set(prev_lw)) / 2
        ts = sum(abs(new_sw.get(p, 0) - prev_sw.get(p, 0))
                 for p in set(new_sw) | set(prev_sw)) / 2
        monthly.append({
            'date': date,
            'ret': r_long - r_short - fee * (tl + ts),
            'r_long': r_long,
            'r_short': r_short,
        })
        prev_lw, prev_sw = new_lw, new_sw
    return pd.DataFrame(monthly).set_index('date') if monthly else pd.DataFrame()


perf_rows = []
print(f"\n  {'Subtype':>10s} {'Mean ret':>10s} {'Sharpe':>8s} {'N':>5s}")
print(f"  " + "-" * 38)

for subtype in subtypes:
    date_set = set(test[test['subtype'] == subtype]['date'].unique())
    result = long_short_port_subset(test, 'score_xgb', date_set)
    if result.empty:
        continue
    r = result['ret']
    mean_r = r.mean()
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = mean_r / r.std() * np.sqrt(12) if r.std() > 0 else 0
    print(f"  {subtype:>10s} {mean_r:>+9.2%}/mo {sharpe:>7.2f} {len(r):>5d}")
    perf_rows.append({
        'subtype': subtype,
        'mean_ret': mean_r,
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'n_months': len(r),
    })

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: 3-panel z-score by subtype
# ══════════════════════════════════════════════════════════════════════════════

x = np.arange(1, 13)

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

panel_configs = [
    ('Calm', f'Calm'),
    ('Crash', f'Panic: Crash (market down)'),
    ('Recovery', f'Panic: Recovery (market up)'),
]

for i, (subtype, title) in enumerate(panel_configs):
    ax = axes[i]
    n = len(test[test['subtype'] == subtype]['date'].unique())

    for leg, color, marker in [('long', '#2196F3', 'o'), ('short', '#E53935', 's')]:
        mean = z_data[(subtype, leg)]
        se = z_se[(subtype, leg)]
        ax.plot(x, mean, f'{marker}-', color=color, linewidth=2.5,
                markersize=8, label=f'{leg.title()} leg', zorder=5)
        ax.fill_between(x, mean - se, mean + se, alpha=0.12, color=color)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_title(f'{title}\n({n} months)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(x)
    ax.set_xlabel('Momentum lookback horizon (months)', fontsize=11)
    if i == 0:
        ax.set_ylabel('Z-score vs cross-section', fontsize=11)

plt.suptitle('Momentum term structure: Calm vs Panic sub-types',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig('plots/thesis/zscore_panic_subtypes.png', dpi=150, bbox_inches='tight')
fig.savefig('plots/zscore_panic_subtypes.pdf', bbox_inches='tight')
plt.close(fig)
print("\nSaved: plots/zscore_panic_subtypes.png, .pdf")

# ══════════════════════════════════════════════════════════════════════════════
# LATEX TABLE
# ══════════════════════════════════════════════════════════════════════════════

import re

def write_tex(filename, content):
    content = re.sub(r'(\d)%', r'\1\\%', content)
    path = os.path.join(TABLES_DIR, filename)
    with open(path, 'w') as f:
        f.write(content)
    print(f"Saved: {path}")


tex_lines = []
tex_lines.append(r'\begin{table}[H]')
tex_lines.append(r'\centering')
tex_lines.append(r'\small')
tex_lines.append(r'\begin{tabular}{l r r r r}')
tex_lines.append(r'\toprule')
tex_lines.append(r' & Ann.\ Ret & Ann.\ Vol & Sharpe & Months \\')
tex_lines.append(r'\midrule')

for row in perf_rows:
    name = row['subtype']
    if name == 'Crash':
        name = 'Panic: Crash'
    elif name == 'Recovery':
        name = 'Panic: Recovery'
    ar = row['ann_ret']
    av = row['ann_vol']
    sh = row['sharpe']
    n = row['n_months']

    ar_str = f"$-${abs(ar):.1%}" if ar < 0 else f"{ar:.1%}"
    sh_str = f"$-${abs(sh):.2f}" if sh < 0 else f"{sh:.2f}"

    tex_lines.append(f"{name} & {ar_str} & {av:.1%} & {sh_str} & {n} \\\\")

tex_lines.append(r'\bottomrule')
tex_lines.append(r'\end{tabular}')
tex_lines.append(r'\caption{Long-short performance by panic sub-type. Panic months are '
                 r'split by concurrent market return: crash (negative) vs recovery (positive). '
                 r'Net of 10\,bps one-way transaction costs.}')
tex_lines.append(r'\label{tab:panic_subtypes}')
tex_lines.append(r'\end{table}')

write_tex('table_panic_subtypes.tex', '\n'.join(tex_lines))

print("\nDone.")
