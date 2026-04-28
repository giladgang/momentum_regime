"""
zscore_subregime_analysis.py
============================
Decomposes the HMM-classified panic regime into three cross-sectionally
distinct sub-regimes by clustering the long-leg cross-sectional z-profile
(12 momentum horizons) of each panic month. Produces:

  1. A 4-panel z-profile figure (calm + 3 panic sub-regimes, long & short).
  2. A LaTeX table summarising n, pi, Sharpe, and SHAP shares per sub-regime.
  3. A backing CSV with the underlying numbers for cross-checks.

Inputs:
  - results/thesis/zscore_long_by_month.csv
  - results/thesis/zscore_short_by_month.csv
  - data/panel_with_regimes.parquet
  - artefacts/cs_artefacts_data.pkl
  - results/thesis/fundamentals_returns.pkl   (for M2 monthly returns)

Outputs:
  - plots/thesis/zscore_subregimes.png
  - plots/thesis/zscore_subregimes.pdf
  - tables/table_subregime_shap.tex
  - results/thesis/zscore_subregime_summary.csv
"""

import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TABLES_DIR, RESULTS_THESIS_DIR, PLOTS_THESIS_DIR,
                    ARTEFACTS_PATH, PANEL_WITH_REGIMES_PATH)

# ----------------------------------------------------------------------
# Reproducibility / config
# ----------------------------------------------------------------------
RNG_SEED = 42
N_INIT = 20
HORIZONS = list(range(1, 13))
COLS = [f'mom_{h}' for h in HORIZONS]
MIN_NYSE_FOR_DECILES = 10
SEEDS_FOR_STABILITY = [42, 123, 456, 789, 1011, 1213]

PLOT_DIR = PLOTS_THESIS_DIR
RES_DIR = RESULTS_THESIS_DIR
TAB_DIR = TABLES_DIR
for d in (PLOT_DIR, RES_DIR, TAB_DIR):
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------
# Load and merge
# ----------------------------------------------------------------------
long_df = pd.read_csv(f'{RES_DIR}/zscore_long_by_month.csv', parse_dates=['date'])
short_df = pd.read_csv(f'{RES_DIR}/zscore_short_by_month.csv', parse_dates=['date'])
panel = pd.read_parquet(PANEL_WITH_REGIMES_PATH)
panel['date'] = pd.to_datetime(panel['date'])
pi = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').sort_values('date')

L = (long_df.merge(pi, on='date')
              .dropna(subset=['pi_filter'])
              .sort_values('date')
              .reset_index(drop=True))
S = (short_df.merge(pi, on='date')
              .dropna(subset=['pi_filter'])
              .sort_values('date')
              .reset_index(drop=True))
assert (L['date'].values == S['date'].values).all()

# ----------------------------------------------------------------------
# Cluster panic months (k=3 on the long-leg 12-d z-profile)
# ----------------------------------------------------------------------
panic_mask = L['pi_filter'] > 0.5
X_panic = L.loc[panic_mask, COLS].values
km = KMeans(n_clusters=3, random_state=RNG_SEED, n_init=N_INIT).fit(X_panic)
SILHOUETTE = float(silhouette_score(X_panic, km.labels_))
L['cluster'] = -1
L.loc[panic_mask, 'cluster'] = km.labels_
S['cluster'] = L['cluster']

# Seed-stability check: refit at several seeds, record cluster sizes (sorted).
# Surfaced in the console + sidecar so the caption claim is auditable.
def _sorted_sizes(seed):
    k = KMeans(n_clusters=3, random_state=seed, n_init=N_INIT).fit(X_panic)
    return tuple(sorted(np.bincount(k.labels_, minlength=3).tolist()))
STABILITY_SIZES = {s: _sorted_sizes(s) for s in SEEDS_FOR_STABILITY}

# Relabel by mean profile depth: most-negative -> deep_crisis, most-positive -> mild_panic
mp = (L[L['cluster'] >= 0]
        .groupby('cluster')[COLS]
        .mean().mean(axis=1))
order = mp.sort_values().index.tolist()
relabel = {order[0]: 'deep_crisis', order[1]: 'transition', order[2]: 'mild_panic'}
L['ptype'] = L['cluster'].map(relabel).fillna('calm')
S['ptype'] = L['ptype']

# Display order for figure / table: increasing severity
SUBREGIMES = ['calm', 'mild_panic', 'transition', 'deep_crisis']
PRETTY = {
    'calm': 'Calm',
    'mild_panic': 'Mild panic',
    'transition': 'Transition',
    'deep_crisis': 'Deep crisis',
}

# ----------------------------------------------------------------------
# Strategy returns per sub-regime
# ----------------------------------------------------------------------
with open('results/thesis/fundamentals_returns.pkl', 'rb') as f:
    rets = pickle.load(f)
m2_ret = rets['baseline_mom_pi']['returns']
m2_ret.index = pd.to_datetime(m2_ret.index)
R = pd.DataFrame({'date': m2_ret.index, 'm2_ret': m2_ret.values})
M = L.merge(R, on='date', how='left')

def ann_sharpe(x):
    x = x.dropna()
    if len(x) == 0 or x.std() == 0:
        return np.nan
    return (x.mean() / x.std()) * np.sqrt(12)

# ----------------------------------------------------------------------
# SHAP per sub-regime per leg
# ----------------------------------------------------------------------
with open(ARTEFACTS_PATH, 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
test['date'] = pd.to_datetime(test['date'])
shap = art['shap_values']
features = list(art['FEATURES'])
PI_IDX = features.index('pi_filter')
MOM_IDXS = [features.index(f'mom_{h}') for h in HORIZONS]

test = test.merge(L[['date', 'ptype']], on='date', how='left')
test['leg'] = 'middle'
for date, grp in test.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score_xgb'].dropna()
    if len(nyse) < MIN_NYSE_FOR_DECILES:
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    test.loc[grp[grp['score_xgb'] >= hi].index, 'leg'] = 'long'
    test.loc[grp[grp['score_xgb'] <= lo].index, 'leg'] = 'short'

# SHAP rows are aligned with `test` row-for-row by construction in
# cross_sectional_model.py. Asserting catches any future divergence
# instead of silently misaligning labels with attributions.
assert len(test) == len(shap), \
    f'SHAP/test length mismatch: shap={len(shap)} test={len(test)}'
test_aligned = test.reset_index(drop=True)
abs_shap = np.abs(shap)

def shap_share(ptype, leg):
    """Return SHAP share (%) per feature for a given sub-regime x leg."""
    idx = ((test_aligned['ptype'] == ptype) & (test_aligned['leg'] == leg)).values
    if idx.sum() == 0:
        return None
    s = abs_shap[idx].mean(axis=0)
    return 100 * s / s.sum()

# ----------------------------------------------------------------------
# Build summary table (rows = metrics, cols = sub-regimes)
# ----------------------------------------------------------------------
rows = []
for sr in SUBREGIMES:
    grp = M[M['ptype'] == sr]
    long_z  = L[L['ptype'] == sr][COLS].mean().values
    short_z = S[S['ptype'] == sr][COLS].mean().values

    sh_long  = shap_share(sr, 'long')
    sh_short = shap_share(sr, 'short')

    rows.append({
        'subregime': sr,
        'n_months': len(grp),
        'pi_mean': grp['pi_filter'].mean(),
        'pi_median': grp['pi_filter'].median(),
        'mean_ret_pct': 100 * grp['m2_ret'].mean(),
        'sharpe_ann': ann_sharpe(grp['m2_ret']),
        'long_z_mom1':  long_z[0],
        'long_z_mom8':  long_z[7],
        'long_z_mom12': long_z[11],
        'long_z_mean':  long_z.mean(),
        'short_z_mom1':  short_z[0],
        'short_z_mom8':  short_z[7],
        'short_z_mom12': short_z[11],
        'short_z_mean':  short_z.mean(),
        'long_pi_shap_pct':  sh_long[PI_IDX]            if sh_long  is not None else np.nan,
        'long_mom_shap_pct': sh_long[MOM_IDXS].sum()    if sh_long  is not None else np.nan,
        'long_mom1_shap_pct':  sh_long[MOM_IDXS[0]]     if sh_long  is not None else np.nan,
        'long_mom8_shap_pct':  sh_long[MOM_IDXS[7]]     if sh_long  is not None else np.nan,
        'long_mom12_shap_pct': sh_long[MOM_IDXS[11]]    if sh_long  is not None else np.nan,
        'short_pi_shap_pct':  sh_short[PI_IDX]          if sh_short is not None else np.nan,
        'short_mom_shap_pct': sh_short[MOM_IDXS].sum()  if sh_short is not None else np.nan,
        'short_mom1_shap_pct':  sh_short[MOM_IDXS[0]]   if sh_short is not None else np.nan,
        'short_mom8_shap_pct':  sh_short[MOM_IDXS[7]]   if sh_short is not None else np.nan,
        'short_mom12_shap_pct': sh_short[MOM_IDXS[11]]  if sh_short is not None else np.nan,
    })
summary = pd.DataFrame(rows).set_index('subregime').loc[SUBREGIMES]
summary.to_csv(f'{RES_DIR}/zscore_subregime_summary.csv', float_format='%.4f')
print(f'Saved: {RES_DIR}/zscore_subregime_summary.csv')

# Sidecar with the cluster-quality stats the caption cites, so they are
# reproducible from the saved output without rerunning the script.
robust_path = f'{RES_DIR}/zscore_subregime_robustness.csv'
robust_rows = [{'metric': 'silhouette_seed42', 'value': SILHOUETTE}]
for s, sizes in STABILITY_SIZES.items():
    robust_rows.append({'metric': f'cluster_sizes_seed{s}',
                        'value': '|'.join(str(x) for x in sizes)})
pd.DataFrame(robust_rows).to_csv(robust_path, index=False)
print(f'Saved: {robust_path}')

# ----------------------------------------------------------------------
# 2x2 figure: long & short z-profiles per sub-regime
# ----------------------------------------------------------------------
COL_LONG = '#1f77b4'   # blue
COL_SHORT = '#d62728'  # red

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
axes_flat = axes.flatten()

# y-axis range: actual union of all profiles, with 5% headroom
_all_means = []
for sr in SUBREGIMES:
    _all_means.append(L[L['ptype'] == sr][COLS].mean().values)
    _all_means.append(S[S['ptype'] == sr][COLS].mean().values)
_yabs = float(np.nanmax(np.abs(np.concatenate(_all_means)))) * 1.05
ymin, ymax = -_yabs, _yabs
for ax, sr in zip(axes_flat, SUBREGIMES):
    long_z  = L[L['ptype'] == sr][COLS].mean().values
    short_z = S[S['ptype'] == sr][COLS].mean().values
    n = (L['ptype'] == sr).sum()
    sharpe = summary.loc[sr, 'sharpe_ann']
    pi_mean = summary.loc[sr, 'pi_mean']

    ax.axhline(0, color='gray', linewidth=0.6, alpha=0.5)
    ax.plot(HORIZONS, long_z,  color=COL_LONG,  marker='o', linewidth=1.8,
            markersize=5, label='Long leg')
    ax.plot(HORIZONS, short_z, color=COL_SHORT, marker='s', linewidth=1.8,
            markersize=5, label='Short leg')
    ax.fill_between(HORIZONS, long_z, short_z, alpha=0.08, color='gray')

    title = f'{PRETTY[sr]}  (n={n}, $\\bar\\pi$={pi_mean:.2f}, Sharpe={sharpe:.2f})'
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xticks(HORIZONS)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(ymin, ymax)
    ax.grid(axis='y', alpha=0.3)

# Common axis labels via fig
for ax in axes[1, :]:
    ax.set_xlabel('Momentum lookback (months)', fontsize=10)
for ax in axes[:, 0]:
    ax.set_ylabel('Cross-sectional z-score', fontsize=10)

# One legend at top
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2,
           bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=10)

fig.suptitle('Long- and short-leg cross-sectional z-profiles by sub-regime',
             y=1.04, fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{PLOT_DIR}/zscore_subregimes.png', dpi=200, bbox_inches='tight')
fig.savefig(f'{PLOT_DIR}/zscore_subregimes.pdf', bbox_inches='tight')
plt.close(fig)
print(f'Saved: {PLOT_DIR}/zscore_subregimes.png')
print(f'Saved: {PLOT_DIR}/zscore_subregimes.pdf')

# ----------------------------------------------------------------------
# LaTeX table: summary by sub-regime
# ----------------------------------------------------------------------
def fmt(x, fmt_str='{:.2f}'):
    return '--' if pd.isna(x) else fmt_str.format(x)

def fmt_pct(x):
    return '--' if pd.isna(x) else f'{x:.1f}\\%'

def fmt_int(x):
    return '--' if pd.isna(x) else f'{int(x)}'

cols_pretty = [PRETTY[s] for s in SUBREGIMES]

rows_tex = []
rows_tex.append(r'\midrule')
rows_tex.append(r'\multicolumn{5}{l}{\emph{Months and performance}} \\')
rows_tex.append(' & '.join(['Months ($n$)'] + [fmt_int(summary.loc[s, 'n_months']) for s in SUBREGIMES]) + r' \\')
rows_tex.append(' & '.join([r'Mean $\pi^{\text{filter}}$'] + [fmt(summary.loc[s, 'pi_mean']) for s in SUBREGIMES]) + r' \\')
rows_tex.append(' & '.join(['Mean monthly return (\\%)'] + [fmt(summary.loc[s, 'mean_ret_pct']) for s in SUBREGIMES]) + r' \\')
rows_tex.append(' & '.join(['Sharpe (annualised)'] + [fmt(summary.loc[s, 'sharpe_ann']) for s in SUBREGIMES]) + r' \\')

rows_tex.append(r'\midrule')
rows_tex.append(r'\multicolumn{5}{l}{\emph{Long-leg z-score}} \\')
rows_tex.append(' & '.join(['$z$ at mom\\_1'] + [fmt(summary.loc[s, 'long_z_mom1']) for s in SUBREGIMES]) + r' \\')
rows_tex.append(' & '.join(['$z$ at mom\\_8'] + [fmt(summary.loc[s, 'long_z_mom8']) for s in SUBREGIMES]) + r' \\')
rows_tex.append(' & '.join(['$z$ at mom\\_12'] + [fmt(summary.loc[s, 'long_z_mom12']) for s in SUBREGIMES]) + r' \\')
rows_tex.append(' & '.join(['$\\bar z$ across horizons'] + [fmt(summary.loc[s, 'long_z_mean']) for s in SUBREGIMES]) + r' \\')

rows_tex.append(r'\midrule')
rows_tex.append(r'\multicolumn{5}{l}{\emph{Long-leg SHAP share}} \\')
rows_tex.append(' & '.join([r'$\pi^{\text{filter}}$'] + [fmt_pct(summary.loc[s, 'long_pi_shap_pct']) for s in SUBREGIMES]) + r' \\')
rows_tex.append(' & '.join(['Momentum (12 horizons)'] + [fmt_pct(summary.loc[s, 'long_mom_shap_pct']) for s in SUBREGIMES]) + r' \\')
rows_tex.append(' & '.join([r'\quad mom\_1'] + [fmt_pct(summary.loc[s, 'long_mom1_shap_pct']) for s in SUBREGIMES]) + r' \\')
rows_tex.append(' & '.join([r'\quad mom\_8'] + [fmt_pct(summary.loc[s, 'long_mom8_shap_pct']) for s in SUBREGIMES]) + r' \\')
rows_tex.append(' & '.join([r'\quad mom\_12'] + [fmt_pct(summary.loc[s, 'long_mom12_shap_pct']) for s in SUBREGIMES]) + r' \\')

rows_tex.append(r'\midrule')
rows_tex.append(r'\multicolumn{5}{l}{\emph{Short-leg SHAP share}} \\')
rows_tex.append(' & '.join([r'$\pi^{\text{filter}}$'] + [fmt_pct(summary.loc[s, 'short_pi_shap_pct']) for s in SUBREGIMES]) + r' \\')
rows_tex.append(' & '.join(['Momentum (12 horizons)'] + [fmt_pct(summary.loc[s, 'short_mom_shap_pct']) for s in SUBREGIMES]) + r' \\')
rows_tex.append(' & '.join([r'\quad mom\_1'] + [fmt_pct(summary.loc[s, 'short_mom1_shap_pct']) for s in SUBREGIMES]) + r' \\')
rows_tex.append(' & '.join([r'\quad mom\_8'] + [fmt_pct(summary.loc[s, 'short_mom8_shap_pct']) for s in SUBREGIMES]) + r' \\')
rows_tex.append(' & '.join([r'\quad mom\_12'] + [fmt_pct(summary.loc[s, 'short_mom12_shap_pct']) for s in SUBREGIMES]) + r' \\')

header_top = r'\toprule'
header_cols = ' & '.join(['Metric'] + cols_pretty) + r' \\'

table_tex = '\n'.join([
    r'\begin{table}[H]',
    r'\centering',
    r'\small',
    r'\begin{tabular}{l rrrr}',
    header_top,
    header_cols,
    *rows_tex,
    r'\bottomrule',
    r'\end{tabular}',
    r'\caption{Sub-regime decomposition of the HMM-classified panic regime. The 59 panic months are clustered (K-means, $k=3$, random seed 42, silhouette 0.46) on the long-leg 12-horizon cross-sectional z-profile. Cluster sizes are stable across six independent random seeds. SHAP shares are computed on the production XGBoost ensemble (50 seeds, depth~4) and normalised to sum to 100\% within each leg. Monthly returns are net of 10\,bps transaction cost. \emph{Conditional Sharpes are post-hoc, in-sample diagnostics, not investable strategy returns: sub-regime labels are determined after the fact from each month''s realised z-profile.} The HMM filtered probability cannot distinguish the three panic clusters (mean $\pi^{\text{filter}} \geq 0.95$ in all three); what differentiates them is the realised cross-sectional momentum landscape.}',
    r'\label{tab:zscore_subregime}',
    r'\end{table}',
    '',
])

with open(f'{TAB_DIR}/table_subregime_shap.tex', 'w') as f:
    f.write(table_tex)
print(f'Saved: {TAB_DIR}/table_subregime_shap.tex')

# ----------------------------------------------------------------------
# Console summary for verification
# ----------------------------------------------------------------------
print('\n' + '=' * 72)
print('Summary by sub-regime')
print('=' * 72)
disp = summary[['n_months', 'pi_mean', 'mean_ret_pct', 'sharpe_ann',
                'long_pi_shap_pct', 'long_mom_shap_pct',
                'short_pi_shap_pct', 'short_mom_shap_pct']].round(2)
print(disp.to_string())
print(f'\nSilhouette (seed {RNG_SEED}): {SILHOUETTE:.3f}')
print('Cluster size stability (sorted) across seeds:')
for s, sizes in STABILITY_SIZES.items():
    marker = ' <-- production' if s == RNG_SEED else ''
    print(f'  seed {s:>4}: {sizes}{marker}')
print('\nDone.')
