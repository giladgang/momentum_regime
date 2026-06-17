"""
2026-05-31-features-plus-regime-combined.py
===========================================
Exploratory: stack the data-chapter Figure 3 (four standardised HMM input
features) and Figure 4 (filtered panic probability) into a single shared-axis
figure. Features on top as steelblue lines; pi_filter on the bottom as a
crimson fill. Grey bands mark hand-picked crisis windows (labelled at top),
matching the shading used in scripts/hmm_model.py's regime_probabilities.png.

Reads:  data/panel_with_regimes.parquet  (has DD_z, CS_z, DISP_z, REL_N_z, pi_filter)
Writes: plots/features_and_regime_combined.png

Not (yet) thesis-worthy: output stays in plots/ (root), script in experiments/.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PANEL_PATH = 'data/panel_with_regimes.parquet'
OUT_PATH = 'plots/features_and_regime_combined.png'

FEATURES_Z = ['DD_z', 'CS_z', 'DISP_z', 'REL_N_z']
FEATURE_LABELS = {
    'DD_z':    'DD (Drawdown)',
    'CS_z':    'CS (BAA-AAA Credit Spread)',
    'DISP_z':  'DISP (Return Dispersion)',
    'REL_N_z': 'REL_N (Market Participation)',
}

# Hand-picked crisis windows (matches scripts/hmm_model.py). Pre-sample
# windows are skipped via data_start.
CRISES = [
    ('1973-10-01', '1974-12-01', 'Oil shock'),
    ('1987-10-01', '1987-12-01', 'Black Monday'),
    ('2000-03-01', '2002-10-01', 'Dot-com'),
    ('2007-10-01', '2009-06-01', 'GFC'),
    ('2020-02-01', '2020-05-01', 'COVID'),
]
SPLIT = pd.Timestamp('2011-01-01')


def shade_crises(ax, data_start, label=False):
    for start, end, lbl in CRISES:
        s = pd.Timestamp(start)
        if s < pd.Timestamp(data_start):
            continue
        ax.axvspan(s, pd.Timestamp(end), alpha=0.12, color='grey', zorder=0)
        if label:
            ax.text(s, 1.02, lbl, fontsize=7, color='grey',
                    va='bottom', transform=ax.get_xaxis_transform())


def main():
    panel = pd.read_parquet(PANEL_PATH)
    panel = panel.dropna(subset=FEATURES_Z + ['pi_filter']).reset_index(drop=True)
    dates = panel['date']
    data_start = dates.min()

    # 4 feature panels + 1 probability panel, shared x-axis.
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1, 1, 1.1]})

    # --- top four: standardised features as steelblue lines ---
    for ax, feat in zip(axes[:4], FEATURES_Z):
        ax.plot(dates, panel[feat], linewidth=0.9, color='steelblue')
        ax.set_ylabel(FEATURE_LABELS[feat], fontsize=9)
        ax.axhline(0, color='black', linewidth=0.4, linestyle='--', alpha=0.5)
        ax.axvline(SPLIT, color='black', linewidth=1, linestyle='--')
        ax.grid(axis='y', alpha=0.2)
        shade_crises(ax, data_start, label=(ax is axes[0]))

    # --- bottom: filtered panic probability as crimson fill ---
    axp = axes[4]
    axp.fill_between(dates, panel['pi_filter'], alpha=0.5, color='crimson')
    axp.axhline(0.5, color='black', linewidth=0.5, linestyle=':')
    axp.axvline(SPLIT, color='black', linewidth=1, linestyle='--',
                label='Train/test split')
    axp.set_ylabel(r'$\pi_t^{\mathrm{filter}}$', fontsize=11)
    axp.set_ylim(0, 1)
    axp.legend(fontsize=8, loc='upper left')
    shade_crises(axp, data_start)

    axes[-1].set_xlabel('Date', fontsize=10)
    axes[-1].set_xlim(dates.min(), dates.max())
    axes[0].set_title('HMM Input Features and Filtered Panic Probability',
                      fontsize=12)
    fig.align_ylabels(axes)
    plt.subplots_adjust(hspace=0.08)
    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT_PATH}  ({len(panel)} months, '
          f'{dates.min().date()} -> {dates.max().date()})')


if __name__ == '__main__':
    main()
