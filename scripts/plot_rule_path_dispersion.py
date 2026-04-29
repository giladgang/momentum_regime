"""Phase 5 dispersion plot: per-rule long-leg z-profile.

For each rule, identify plurality months (>50% of long-leg picks belong
to the rule) and plot:
  - thin line: each plurality month's long-leg z-profile
  - heavy line: rule centroid (mean across plurality months)
  - dashed black: calm-month reference (all calm months' average)

Annotated with n_plurality, mean pi, panic_share, Sharpe + CI.

Output:
- plots/thesis/rule_path_dispersion.{png,pdf}
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


def main():
    # Load
    labels_df = pd.read_csv(f'{RES_DIR}/rule_path_labels.csv',
                            parse_dates=['date'])
    long_z = pd.read_csv(f'{RES_DIR}/zscore_long_by_month.csv',
                         parse_dates=['date']).set_index('date')
    centroids = pd.read_csv(f'{RES_DIR}/rule_path_centroids.csv').set_index('rule_id')

    with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
        art = pickle.load(f)
    test = art['test'].copy()
    test['date'] = pd.to_datetime(test['date'])
    pi = test.groupby('date')['pi_filter'].first()
    calm_ref = long_z.loc[pi <= 0.5, MOM_COLS].mean().values

    # Plurality month per rule
    plurality_rows = []
    for date, sub in labels_df.groupby('date'):
        counts = sub['rule_id'].value_counts()
        top_share = counts.iloc[0] / counts.sum()
        if top_share > 0.5:
            plurality_rows.append({
                'date': date,
                'rule_id': counts.index[0],
                'share': top_share,
            })
    plurality = pd.DataFrame(plurality_rows)
    print(f'plurality coverage: {len(plurality)} of '
          f'{labels_df["date"].nunique()} months', flush=True)

    rules = sorted(labels_df['rule_id'].unique())
    n_rules = len(rules)

    # 1 x n_rules layout
    fig, axes = plt.subplots(1, n_rules, figsize=(5.0 * n_rules, 5.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n_rules))
    YLIM = (-1.5, 1.0)

    rule_names = {0: 'Rule 0 (panic-rule)', 1: 'Rule 1 (calm-rule)'}

    for i, rid in enumerate(rules):
        ax = axes[i]
        color = cmap[i]
        rule_dates = plurality[plurality['rule_id'] == rid]['date']
        Z = long_z.loc[rule_dates, MOM_COLS].values
        centroid = Z.mean(axis=0)
        info = centroids.loc[rid]

        # Member-month thin lines
        for row in Z:
            ax.plot(HORIZONS, row, color=color, lw=0.8, alpha=0.4)
        # Centroid heavy line
        ax.plot(HORIZONS, centroid, color=color, lw=2.5, marker='o', ms=6,
                label=f'Rule {rid} centroid (n={int(info["n_plurality_months"])})')
        # Calm reference
        ax.plot(HORIZONS, calm_ref, color='#333', lw=1.5, ls='--',
                label='Calm centroid (reference)')

        ax.axhline(0, color='#888', lw=0.6)
        ax.set_xticks(HORIZONS)
        ax.set_xlim(0.5, 12.5)
        ax.set_ylim(*YLIM)
        ax.set_xlabel('Momentum lookback (months)')
        title = rule_names.get(rid, f'Rule {rid}')
        ax.set_title(
            f'{title}\n'
            f'n_stockmonths={int(info["n_stockmonths"])}, '
            f'plurality_months={int(info["n_plurality_months"])}, '
            f'$\\bar\\pi$={info["mean_pi"]:.2f}, '
            f'panic={info["panic_share"]:.2f}\n'
            f'Sharpe={info["plurality_sharpe"]:.2f} '
            f'[{info["plurality_sharpe_lo95"]:.2f}, '
            f'{info["plurality_sharpe_hi95"]:.2f}]',
            fontsize=10,
        )
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=9, frameon=True)

    axes[0].set_ylabel('Long-leg cross-sectional z-score')
    fig.suptitle('Per-rule long-leg z-profile dispersion (leaf-value k=2 clustering)',
                 y=1.00)
    fig.tight_layout()

    png = f'{PLOT_DIR}/rule_path_dispersion.png'
    pdf = f'{PLOT_DIR}/rule_path_dispersion.pdf'
    fig.savefig(png, dpi=200, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    print(f'Saved: {png}')
    print(f'Saved: {pdf}')


if __name__ == '__main__':
    main()
