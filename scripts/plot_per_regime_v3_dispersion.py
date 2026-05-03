"""Per-regime v3 dispersion plot: 5 panels, one per sub-rule.

Order (left -> right): calm-bulk, calm-boundary, mid-panic-profitable,
panic-loss-tail, deep-panic.

Output:
- plots/thesis/rule_path_per_regime_v3_dispersion.{png,pdf}
"""
import os, pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES_DIR = 'results/thesis'
PLOT_DIR = 'plots/thesis'
HORIZONS = list(range(1, 13))
MOM_COLS = [f'mom_{h}' for h in HORIZONS]

os.makedirs(PLOT_DIR, exist_ok=True)

# Display configuration:
# panels in order: calm-bulk, calm-boundary, mid-panic-profitable, panic-loss-tail, deep-panic
PANELS = [
    {'rule_id': 1, 'name': 'Calm-bulk',              'color': '#1f77b4'},
    {'rule_id': 0, 'name': 'Calm-boundary',          'color': '#9ecae1'},
    {'rule_id': 3, 'name': 'Mid-panic-profitable',   'color': '#ff7f0e'},
    {'rule_id': 4, 'name': 'Panic-loss-tail',        'color': '#9467bd'},
    {'rule_id': 2, 'name': 'Deep-panic',             'color': '#d62728'},
]


def main():
    labels = pd.read_csv(f'{RES_DIR}/rule_path_per_regime_v3_labels.csv',
                         parse_dates=['date'])
    long_z = pd.read_csv(f'{RES_DIR}/zscore_long_by_month.csv',
                         parse_dates=['date']).set_index('date')

    with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
        art = pickle.load(f)
    test = art['test'].copy()
    test['date'] = pd.to_datetime(test['date'])
    pi = test.groupby('date')['pi_filter'].first()
    calm_ref = long_z.loc[pi <= 0.5, MOM_COLS].mean().values

    with open('results/thesis/fundamentals_returns.pkl', 'rb') as f:
        rets = pickle.load(f)
    m2_ret = rets['baseline_mom_pi']['returns']
    m2_ret.index = pd.to_datetime(m2_ret.index)

    counts = labels.groupby(['date', 'rule_id']).size().reset_index(name='n')
    totals = labels.groupby('date').size().reset_index(name='total')
    counts = counts.merge(totals, on='date')
    counts['share'] = counts['n'] / counts['total']
    plurality = counts[counts['share'] > 0.5][['date', 'rule_id']]

    n_panels = len(PANELS)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5.5),
                              sharey=True)
    YLIM = (-1.5, 1.0)

    for ax, p in zip(axes, PANELS):
        rid = p['rule_id']
        color = p['color']
        rule_dates = plurality[plurality['rule_id'] == rid]['date']
        n_plurality = len(rule_dates)

        # If no plurality months, fall back to all unique dates in this rule's stock-months
        if n_plurality == 0:
            sub = labels[labels['rule_id'] == rid]
            member_dates = sorted(sub['date'].unique())
        else:
            member_dates = rule_dates

        Z = long_z.loc[member_dates, MOM_COLS].values
        if len(Z) == 0:
            ax.set_title(f'{p["name"]}\n(empty)')
            continue
        centroid = Z.mean(axis=0)

        # π̄ and Sharpe info
        pi_bar = pi.loc[member_dates].mean()
        rs = m2_ret.reindex(member_dates).dropna().values

        if len(rs) >= 3:
            sharpe = (rs.mean() / rs.std(ddof=1)) * np.sqrt(12)
            sharpe_str = f'Sharpe={sharpe:.2f}'
        else:
            sharpe_str = f'Sharpe (n={len(rs)} too small)'

        # Member-month thin lines
        for row in Z:
            ax.plot(HORIZONS, row, color=color, lw=0.8, alpha=0.4)
        # Centroid
        ax.plot(HORIZONS, centroid, color=color, lw=2.5, marker='o', ms=6,
                label=f'{p["name"]} centroid')
        # Calm reference
        ax.plot(HORIZONS, calm_ref, color='#333', lw=1.5, ls='--',
                label='All-calm centroid (reference)')

        ax.axhline(0, color='#888', lw=0.6)
        ax.set_xticks(HORIZONS)
        ax.set_xlim(0.5, 12.5)
        ax.set_ylim(*YLIM)
        ax.set_xlabel('Momentum lookback (months)')
        ax.set_title(
            f'{p["name"]}\n'
            f'(n_months={n_plurality}, $\\bar\\pi$={pi_bar:.2f})\n'
            f'{sharpe_str}',
            fontsize=10,
        )
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=8, frameon=True)

    axes[0].set_ylabel('Long-leg cross-sectional z-score')
    fig.suptitle('Per-regime v3: 5 sub-rules (calm k=2 + panic k=3)', y=1.00)
    fig.tight_layout()

    fig.savefig(f'{PLOT_DIR}/rule_path_per_regime_v3_dispersion.png',
                dpi=200, bbox_inches='tight')
    fig.savefig(f'{PLOT_DIR}/rule_path_per_regime_v3_dispersion.pdf',
                bbox_inches='tight')
    print(f'Saved: {PLOT_DIR}/rule_path_per_regime_v3_dispersion.{{png,pdf}}',
          flush=True)


if __name__ == '__main__':
    main()
