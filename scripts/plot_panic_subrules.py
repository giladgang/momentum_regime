"""Phase 5 plots: visualising the calm / mid-panic / deep-panic story.

Produces four plots:

1. Per-rule dispersion plot (k=3, three panels: calm, mid-panic, deep-panic)
   plots/thesis/rule_path_k3_dispersion.{png,pdf}

2. Cross-section landscape comparison (one panel, three lines):
   plots/thesis/rule_path_k3_landscape.{png,pdf}

3. Sharpe ladder with bootstrap 95% CIs (horizontal bar chart):
   plots/thesis/rule_path_sharpe_ladder.{png,pdf}

4. Calendar timeline of plurality rule per month (color strip 2011-2024):
   plots/thesis/rule_path_calendar.{png,pdf}
"""
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

RES_DIR = 'results/thesis'
PLOT_DIR = 'plots/thesis'
HORIZONS = list(range(1, 13))
MOM_COLS = [f'mom_{h}' for h in HORIZONS]

os.makedirs(PLOT_DIR, exist_ok=True)

# Colour scheme:
#   Calm = blue, Mid-panic = orange, Deep-panic = red, k=2 panic = dark red
COLORS = {
    'calm': '#1f77b4',       # blue
    'mid_panic': '#ff7f0e',  # orange
    'deep_panic': '#d62728', # red
    'panic_k2': '#7f0000',   # dark red
}

# Rule IDs in unified k=3:
#   0 = mid-panic, 1 = calm, 2 = deep-panic
RULE_NAMES = {0: 'Mid-panic', 1: 'Calm', 2: 'Deep-panic'}
RULE_COLOURS_K3 = {0: COLORS['mid_panic'], 1: COLORS['calm'], 2: COLORS['deep_panic']}

# Display order (left -> right): calm, mid-panic, deep-panic
DISPLAY_ORDER = [1, 0, 2]


def load_data():
    labels = pd.read_csv(f'{RES_DIR}/rule_path_k3_labels.csv',
                        parse_dates=['date'])
    long_z = pd.read_csv(f'{RES_DIR}/zscore_long_by_month.csv',
                        parse_dates=['date']).set_index('date')
    centroids_k3 = pd.read_csv(f'{RES_DIR}/rule_path_unified_k3_centroids.csv')
    centroids_k2 = pd.read_csv(f'{RES_DIR}/rule_path_unified_k2_centroids.csv')

    with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
        art = pickle.load(f)
    test = art['test'].copy()
    test['date'] = pd.to_datetime(test['date'])
    pi = test.groupby('date')['pi_filter'].first()
    agg_mean = test.groupby('date')[MOM_COLS].mean()

    # Plurality month per rule
    counts = labels.groupby(['date', 'rule_id']).size().reset_index(name='n')
    totals = labels.groupby('date').size().reset_index(name='total')
    counts = counts.merge(totals, on='date')
    counts['share'] = counts['n'] / counts['total']
    plurality = counts[counts['share'] > 0.5][['date', 'rule_id']]
    plurality = plurality.sort_values('date').reset_index(drop=True)

    return {
        'labels': labels,
        'long_z': long_z,
        'centroids_k3': centroids_k3.set_index('rule_id'),
        'centroids_k2': centroids_k2.set_index('rule_id'),
        'pi': pi,
        'agg_mean': agg_mean,
        'plurality': plurality,
    }


def plot_dispersion_k3(d):
    """Three-panel: long-leg z-profile dispersion per k=3 rule."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)

    calm_ref = d['long_z'].loc[d['pi'] <= 0.5, MOM_COLS].mean().values

    for i, rid in enumerate(DISPLAY_ORDER):
        ax = axes[i]
        color = RULE_COLOURS_K3[rid]
        rule_dates = d['plurality'][d['plurality']['rule_id'] == rid]['date']
        Z = d['long_z'].loc[rule_dates, MOM_COLS].values
        if len(Z) == 0:
            continue
        centroid = Z.mean(axis=0)
        info = d['centroids_k3'].loc[rid]

        # Member months
        for row in Z:
            ax.plot(HORIZONS, row, color=color, lw=0.8, alpha=0.4)
        # Centroid
        ax.plot(HORIZONS, centroid, color=color, lw=2.5, marker='o', ms=6,
                label=f'{RULE_NAMES[rid]} centroid')
        # Calm reference
        ax.plot(HORIZONS, calm_ref, color='#333', lw=1.5, ls='--',
                label='All-calm centroid (reference)')

        ax.axhline(0, color='#888', lw=0.6)
        ax.set_xticks(HORIZONS)
        ax.set_xlim(0.5, 12.5)
        ax.set_ylim(-1.5, 1.0)
        ax.set_xlabel('Momentum lookback (months)')
        ax.set_title(
            f'{RULE_NAMES[rid]} (n={int(info["n_plurality_months"])} months, '
            f'$\\bar\\pi$={info["mean_pi"]:.2f}, '
            f'panic={info["panic_share"]:.2f})\n'
            f'Sharpe = {info["sharpe"]:.2f} '
            f'[{info["sharpe_lo95"]:.2f}, {info["sharpe_hi95"]:.2f}]',
            fontsize=10,
        )
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=9, frameon=True)

    axes[0].set_ylabel('Long-leg cross-sectional z-score')
    fig.suptitle('Per-rule long-leg z-profile dispersion (leaf-value k=3 clustering)',
                 y=1.00)
    fig.tight_layout()

    fig.savefig(f'{PLOT_DIR}/rule_path_k3_dispersion.png',
                dpi=200, bbox_inches='tight')
    fig.savefig(f'{PLOT_DIR}/rule_path_k3_dispersion.pdf',
                bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {PLOT_DIR}/rule_path_k3_dispersion.{{png,pdf}}', flush=True)


def plot_landscape_comparison(d):
    """Single panel: typical-stock mom_h per regime — the 'market context' plot."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for rid in DISPLAY_ORDER:
        rule_dates = d['plurality'][d['plurality']['rule_id'] == rid]['date']
        landscape = 100 * d['agg_mean'].loc[rule_dates, MOM_COLS].mean()
        ax.plot(HORIZONS, landscape.values,
                color=RULE_COLOURS_K3[rid], lw=2.5, marker='o', ms=6,
                label=f'{RULE_NAMES[rid]} (n={len(rule_dates)} months)')

    ax.axhline(0, color='#888', lw=0.8, ls='--')
    ax.set_xticks(HORIZONS)
    ax.set_xlim(0.5, 12.5)
    ax.set_xlabel('Momentum lookback (months)')
    ax.set_ylabel('Typical-stock cumulative return (%)')
    ax.set_title('Cross-sectional landscape per rule — the typical stock\'s '
                 'momentum profile')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=10, frameon=True)

    # Annotation:
    ax.annotate(
        'Calm: rising tide\nMid-panic: flat market\nDeep-panic: uniformly down',
        xy=(0.98, 0.02), xycoords='axes fraction',
        ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray',
                  alpha=0.9),
        fontsize=9,
    )
    fig.tight_layout()

    fig.savefig(f'{PLOT_DIR}/rule_path_k3_landscape.png',
                dpi=200, bbox_inches='tight')
    fig.savefig(f'{PLOT_DIR}/rule_path_k3_landscape.pdf',
                bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {PLOT_DIR}/rule_path_k3_landscape.{{png,pdf}}', flush=True)


def plot_sharpe_ladder(d):
    """Horizontal bar chart of Sharpe + bootstrap 95% CIs, all rules."""
    rows = []
    # k=3 rules
    for rid in DISPLAY_ORDER:
        info = d['centroids_k3'].loc[rid]
        rows.append({
            'name': RULE_NAMES[rid],
            'sharpe': info['sharpe'],
            'lo95': info['sharpe_lo95'],
            'hi95': info['sharpe_hi95'],
            'color': RULE_COLOURS_K3[rid],
            'n_months': int(info['n_plurality_months']),
        })
    # k=2 panic-rule for context
    info = d['centroids_k2'].loc[0]
    rows.append({
        'name': 'Panic-rule (k=2 unified)',
        'sharpe': info['sharpe'],
        'lo95': info['sharpe_lo95'],
        'hi95': info['sharpe_hi95'],
        'color': COLORS['panic_k2'],
        'n_months': int(info['n_plurality_months']),
    })

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y_positions = list(range(len(rows)))[::-1]   # highest at top
    for i, r in enumerate(rows):
        ax.barh(y_positions[i], r['sharpe'], color=r['color'], height=0.6,
                edgecolor='black', lw=0.5)
        # CI
        ax.errorbar(r['sharpe'], y_positions[i],
                    xerr=[[r['sharpe'] - r['lo95']],
                          [r['hi95'] - r['sharpe']]],
                    fmt='none', color='black', lw=2, capsize=8)
        # Label
        ax.text(r['hi95'] + 0.1, y_positions[i],
                f'{r["sharpe"]:.2f} [{r["lo95"]:.2f}, {r["hi95"]:.2f}]   '
                f'(n={r["n_months"]} months)',
                va='center', fontsize=10)

    ax.axvline(0, color='red', lw=1.5, ls='--', alpha=0.6,
               label='Sharpe = 0')
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r['name'] for r in rows], fontsize=11)
    ax.set_xlabel('Annualised Sharpe ratio (with bootstrap 95% CI)')
    ax.set_title('Per-rule Sharpe ladder — all CIs exclude zero')
    ax.set_xlim(-0.5, 4.2)
    ax.grid(alpha=0.3, axis='x')
    fig.tight_layout()

    fig.savefig(f'{PLOT_DIR}/rule_path_sharpe_ladder.png',
                dpi=200, bbox_inches='tight')
    fig.savefig(f'{PLOT_DIR}/rule_path_sharpe_ladder.pdf',
                bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {PLOT_DIR}/rule_path_sharpe_ladder.{{png,pdf}}', flush=True)


def plot_calendar(d):
    """Time strip showing which rule fires in each OOS month."""
    plurality = d['plurality'].copy()
    # Fill gaps (months with no clear plurality) by using "no_plurality" id = -1
    all_dates = pd.date_range(plurality['date'].min(),
                              plurality['date'].max(), freq='ME')
    rule_per_month = plurality.set_index('date')['rule_id'].reindex(all_dates,
                                                                    fill_value=-1)

    fig, ax = plt.subplots(figsize=(14, 3.5))

    rule_colours = {
        1: COLORS['calm'], 0: COLORS['mid_panic'], 2: COLORS['deep_panic'],
        -1: '#cccccc',
    }

    for i, (date, rid) in enumerate(rule_per_month.items()):
        ax.axvspan(date - pd.DateOffset(days=15),
                   date + pd.DateOffset(days=15),
                   color=rule_colours[rid], alpha=0.85)

    # Annotate recognized crisis bands
    crisis_annotations = [
        ('2011-09', 'Eurozone'),
        ('2016-02', 'China devaluation'),
        ('2020-03', 'COVID'),
        ('2022-09', 'Rate hikes'),
    ]
    for date_str, label in crisis_annotations:
        d_anno = pd.to_datetime(date_str)
        ax.annotate(label, xy=(d_anno, 0.5), xytext=(d_anno, 0.95),
                    ha='center', va='top', fontsize=8,
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.7))

    ax.set_xlim(rule_per_month.index.min() - pd.DateOffset(days=15),
                rule_per_month.index.max() + pd.DateOffset(days=15))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Date')
    ax.set_title('Plurality rule per month, 2011-2024 OOS')

    legend_patches = [
        mpatches.Patch(color=COLORS['calm'], label='Calm-rule (Rule 1)'),
        mpatches.Patch(color=COLORS['mid_panic'], label='Mid-panic (Rule 0)'),
        mpatches.Patch(color=COLORS['deep_panic'], label='Deep-panic (Rule 2)'),
        mpatches.Patch(color='#cccccc', label='No plurality'),
    ]
    ax.legend(handles=legend_patches, loc='lower center',
              bbox_to_anchor=(0.5, -0.35), ncol=4, frameon=True)
    fig.tight_layout()

    fig.savefig(f'{PLOT_DIR}/rule_path_calendar.png',
                dpi=200, bbox_inches='tight')
    fig.savefig(f'{PLOT_DIR}/rule_path_calendar.pdf',
                bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {PLOT_DIR}/rule_path_calendar.{{png,pdf}}', flush=True)


def main():
    print('Loading data...', flush=True)
    d = load_data()
    print(f'  k=3 plurality coverage: {len(d["plurality"])} months', flush=True)

    print('\n[1/4] Per-rule dispersion plot...', flush=True)
    plot_dispersion_k3(d)

    print('\n[2/4] Cross-section landscape comparison...', flush=True)
    plot_landscape_comparison(d)

    print('\n[3/4] Sharpe ladder...', flush=True)
    plot_sharpe_ladder(d)

    print('\n[4/4] Calendar timeline...', flush=True)
    plot_calendar(d)

    print('\nAll 4 plots saved.', flush=True)


if __name__ == '__main__':
    main()
