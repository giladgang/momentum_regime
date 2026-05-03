"""Hierarchical analysis: leaf-value architectural rule, then output sub-shape.

Step 1: take each unified k=2 rule's plurality months (calm-rule n=104,
        panic-rule n=63).
Step 2: within each rule's months, KMeans k=2 on the 12-d long-leg z-profile.
Step 3: characterise each of the 4 resulting cells (rule x output-shape):
        n_months, mean pi, cross-section landscape, picked-stock features,
        output z centroid, Sharpe + bootstrap CI.

Output:
  results/thesis/rule_path_hierarchical_centroids.csv
  results/thesis/rule_path_hierarchical_labels.csv
  plots/thesis/rule_path_hierarchical_dispersion.{png,pdf}
"""
import os, pickle, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_helpers import block_bootstrap_sharpe

RES_DIR = 'results/thesis'
PLOT_DIR = 'plots/thesis'
HORIZONS = list(range(1, 13))
MOM_COLS = [f'mom_{h}' for h in HORIZONS]


def main():
    # Load architectural rule labels (unified k=2)
    leaf_labels = pd.read_csv(f'{RES_DIR}/rule_path_labels.csv',
                              parse_dates=['date'])
    long_z = pd.read_csv(f'{RES_DIR}/zscore_long_by_month.csv',
                         parse_dates=['date']).set_index('date')

    with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
        art = pickle.load(f)
    test = art['test'].copy()
    test['date'] = pd.to_datetime(test['date'])
    pi = test.groupby('date')['pi_filter'].first()
    agg_mean = test.groupby('date')[MOM_COLS].mean()

    with open('results/thesis/fundamentals_returns.pkl', 'rb') as f:
        rets = pickle.load(f)
    m2_ret = rets['baseline_mom_pi']['returns']
    m2_ret.index = pd.to_datetime(m2_ret.index)

    # Plurality month per leaf-rule
    counts = leaf_labels.groupby(['date', 'rule_id']).size().reset_index(name='n')
    totals = leaf_labels.groupby('date').size().reset_index(name='total')
    counts = counts.merge(totals, on='date')
    counts['share'] = counts['n'] / counts['total']
    plurality = counts[counts['share'] > 0.5][['date', 'rule_id']]

    rule_dates = {
        0: plurality[plurality['rule_id'] == 0]['date'].sort_values().values,
        1: plurality[plurality['rule_id'] == 1]['date'].sort_values().values,
    }
    rule_names = {0: 'panic-rule', 1: 'calm-rule'}
    print(f'Rule 0 (panic-rule): {len(rule_dates[0])} plurality months')
    print(f'Rule 1 (calm-rule):  {len(rule_dates[1])} plurality months')

    # ---- Step 2: within each rule, KMeans k=2 on output z-profile ----
    cell_rows = []
    cell_labels = []   # (rule_id, output_subshape, dates)
    sub_label_per_month = {}   # date -> (rule_id, sub_id)

    for rid, dates in rule_dates.items():
        if len(dates) < 4:
            print(f'\n[skip rule {rid}: only {len(dates)} months]')
            continue
        Y = long_z.loc[dates, MOM_COLS].values
        km = KMeans(n_clusters=2, random_state=42, n_init=20).fit(Y)
        sil = silhouette_score(Y, km.labels_)
        sizes = sorted(np.bincount(km.labels_, minlength=2).tolist())
        print(f'\n=== Rule {rid} ({rule_names[rid]}) sub-clustering on '
              f'output z-profile ===')
        print(f'  k=2 silhouette = {sil:.3f}, sizes = {tuple(sizes)}')

        # Order sub-clusters by output centroid depth (most-negative first)
        depths = [Y[km.labels_ == s].mean() for s in [0, 1]]
        order = sorted([0, 1], key=lambda s: depths[s])
        relabel = {old: new for new, old in enumerate(order)}
        sub = pd.Series(km.labels_).map(relabel).values

        for date, sublabel in zip(dates, sub):
            sub_label_per_month[pd.Timestamp(date)] = (rid, int(sublabel))

        # Characterise each sub-cell
        for s in [0, 1]:
            mask = sub == s
            cell_dates = dates[mask]
            n = len(cell_dates)
            if n == 0:
                continue
            cell_label = f'R{rid}.{s}'

            # Cross-section landscape
            cs = agg_mean.loc[cell_dates, MOM_COLS].mean()
            # Output z-profile centroid
            zc = long_z.loc[cell_dates, MOM_COLS].mean()
            # pi distribution
            pi_arr = pi.loc[cell_dates]
            # Returns
            rs = m2_ret.reindex(cell_dates).dropna().values
            if len(rs) >= 4:
                ci = block_bootstrap_sharpe(rs, block_size=6, n_reps=5000, seed=42)
            else:
                ci = {'sharpe_point': float((rs.mean()/rs.std(ddof=1))*np.sqrt(12))
                      if len(rs) > 1 else float('nan'),
                      'sharpe_lo95': float('nan'),
                      'sharpe_hi95': float('nan'), 'n_reps_valid': 0}
            row = {
                'cell_label': cell_label,
                'rule_id': rid, 'rule_name': rule_names[rid],
                'output_subshape': s,
                'n_months': n,
                'mean_pi': round(float(pi_arr.mean()), 3),
                'panic_share': round(float((pi_arr > 0.5).mean()), 3),
                'land_short_pct': round(100 * cs[[f'mom_{h}' for h in [1, 2, 3, 4]]].mean(), 1),
                'land_long_pct': round(100 * cs[[f'mom_{h}' for h in [9, 10, 11, 12]]].mean(), 1),
                'output_z_short': round(float(zc[[f'mom_{h}' for h in [1, 2, 3, 4]]].mean()), 3),
                'output_z_long': round(float(zc[[f'mom_{h}' for h in [9, 10, 11, 12]]].mean()), 3),
                'sharpe': round(ci['sharpe_point'], 3) if not np.isnan(ci['sharpe_point']) else None,
                'sharpe_lo95': round(ci['sharpe_lo95'], 3) if not np.isnan(ci['sharpe_lo95']) else None,
                'sharpe_hi95': round(ci['sharpe_hi95'], 3) if not np.isnan(ci['sharpe_hi95']) else None,
                'mean_ret_pct': round(100 * rs.mean(), 3) if len(rs) > 0 else None,
                'hit_rate': round(float((rs > 0).mean()), 3) if len(rs) > 0 else None,
            }
            cell_rows.append(row)
            print(f'  cell {cell_label}: n={n}, pi_bar={pi_arr.mean():.3f}, '
                  f'land S/L={row["land_short_pct"]:+.1f}/{row["land_long_pct"]:+.1f}%, '
                  f'output z S/L={row["output_z_short"]:+.2f}/{row["output_z_long"]:+.2f}, '
                  f'Sharpe={row["sharpe"]} '
                  f'[{row["sharpe_lo95"]}, {row["sharpe_hi95"]}]')

    # Save tables
    out_df = pd.DataFrame(cell_rows)
    out_df.to_csv(f'{RES_DIR}/rule_path_hierarchical_centroids.csv', index=False)
    print(f'\nSaved: {RES_DIR}/rule_path_hierarchical_centroids.csv')

    # Per-month label CSV
    label_rows = []
    for date, (rid, sub) in sub_label_per_month.items():
        label_rows.append({'date': date, 'rule_id': rid,
                          'output_subshape': sub, 'cell': f'R{rid}.{sub}'})
    label_df = pd.DataFrame(label_rows).sort_values('date')
    label_df.to_csv(f'{RES_DIR}/rule_path_hierarchical_labels.csv', index=False)
    print(f'Saved: {RES_DIR}/rule_path_hierarchical_labels.csv')

    # ---- Plot: 4-panel dispersion ----
    cells = sorted(out_df['cell_label'].tolist(),
                   key=lambda x: out_df[out_df['cell_label'] == x]['output_z_long'].iloc[0])
    n_cells = len(cells)
    fig, axes = plt.subplots(1, n_cells, figsize=(4.0 * n_cells, 5.5),
                              sharey=True)

    calm_ref = long_z.loc[pi <= 0.5, MOM_COLS].mean().values
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n_cells))

    for ax, cell_label, color in zip(axes, cells, cmap):
        info = out_df[out_df['cell_label'] == cell_label].iloc[0]
        rid = int(info['rule_id'])
        sub = int(info['output_subshape'])
        cell_dates = [d for d, (r, s) in sub_label_per_month.items()
                     if r == rid and s == sub]
        Z = long_z.loc[cell_dates, MOM_COLS].values
        centroid = Z.mean(axis=0)

        for row in Z:
            ax.plot(HORIZONS, row, color=color, lw=0.8, alpha=0.4)
        ax.plot(HORIZONS, centroid, color=color, lw=2.5, marker='o', ms=6,
                label=f'{cell_label} centroid')
        ax.plot(HORIZONS, calm_ref, color='#333', lw=1.5, ls='--',
                label='All-calm centroid (reference)')
        ax.axhline(0, color='#888', lw=0.6)
        ax.set_xticks(HORIZONS)
        ax.set_xlim(0.5, 12.5)
        ax.set_ylim(-1.5, 1.0)
        ax.set_xlabel('Momentum lookback (months)')
        sharpe_str = (f'Sharpe={info["sharpe"]:.2f} '
                      f'[{info["sharpe_lo95"]}, {info["sharpe_hi95"]}]'
                      if info['sharpe_lo95'] is not None
                      else f'Sharpe={info["sharpe"]} (n={info["n_months"]} too small)')
        ax.set_title(
            f'{cell_label} — {info["rule_name"]} × subshape {sub}\n'
            f'(n={info["n_months"]}, $\\bar\\pi$={info["mean_pi"]:.2f})\n'
            f'{sharpe_str}',
            fontsize=10,
        )
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=8, frameon=True)

    axes[0].set_ylabel('Long-leg cross-sectional z-score')
    fig.suptitle('Hierarchical: leaf rule × output sub-shape '
                 '(calm-rule × 2, panic-rule × 2)', y=1.00)
    fig.tight_layout()
    fig.savefig(f'{PLOT_DIR}/rule_path_hierarchical_dispersion.png',
                dpi=200, bbox_inches='tight')
    fig.savefig(f'{PLOT_DIR}/rule_path_hierarchical_dispersion.pdf',
                bbox_inches='tight')
    print(f'Saved: {PLOT_DIR}/rule_path_hierarchical_dispersion.{{png,pdf}}')


if __name__ == '__main__':
    main()
