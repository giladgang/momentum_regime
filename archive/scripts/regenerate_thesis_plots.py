"""
Regenerate two thesis figures:
1. regime_probabilities.png  -- single-panel filtered pi_panic
2. features_hmm.png          -- four HMM input features with NBER recession shading
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# ── Load data ───────────────────────────────────────────────────────────────

panel = pd.read_parquet('data/panel.parquet')
features_z = ['DD_z', 'CS_z', 'DISP_z', 'REL_N_z']
panel = panel.dropna(subset=features_z).reset_index(drop=True)

# Load the saved HMM artefacts to get pi_filter
# Try to load from the hmm model's saved outputs
try:
    with open('cs_artefacts_data.pkl', 'rb') as f:
        artefacts = pickle.load(f)
    # Check if pi_filter is in artefacts or in panel_with_regimes
    if 'pi_filter' in artefacts:
        pi_filter_full = artefacts['pi_filter']
    else:
        raise KeyError
except:
    # Load from panel_with_regimes
    pr = pd.read_parquet('data/panel_with_regimes.parquet')
    if 'pi_filter' in pr.columns:
        # Merge pi_filter into panel by date
        pf = pr.drop_duplicates('date')[['date', 'pi_filter']].dropna()
        panel = panel.merge(pf, on='date', how='left')
        pi_filter_full = panel['pi_filter'].values
    else:
        raise RuntimeError("Cannot find pi_filter in any data source")

dates = panel['date']

# ── NBER recession periods ──────────────────────────────────────────────────

nber_recessions = [
    ('1973-11-01', '1975-03-01'),
    ('1980-01-01', '1980-07-01'),
    ('1981-07-01', '1982-11-01'),
    ('1990-07-01', '1991-03-01'),
    ('2001-03-01', '2001-11-01'),
    ('2007-12-01', '2009-06-01'),
    ('2020-02-01', '2020-04-01'),
]

def shade_nber(ax):
    for start, end in nber_recessions:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.15, color='grey', zorder=0)

# ── Crisis labels for the regime plot ─────────────────────────────────────────

# (start, end, label, y_position, va_alignment)
crisis_labels = [
    ('2000-03-01', '2002-10-01', 'Dot-com', 1.08, 'bottom'),
    ('2007-10-01', '2009-06-01', 'GFC',     0.20, 'top'),
    ('2020-02-01', '2020-05-01', 'COVID',   1.08, 'bottom'),
    ('2022-01-01', '2022-10-01', '2022',    1.08, 'bottom'),
]

# ── Figure 1: regime_probabilities.png (single panel) ───────────────────────

fig, ax = plt.subplots(figsize=(14, 4))

ax.fill_between(dates, pi_filter_full, alpha=0.4, color='crimson',
                label=r'Panic ($\pi \geq 0.5$)')
ax.axvline(pd.Timestamp('2011-01-01'), color='black', linewidth=1.2,
           linestyle='--', label='Train/test split')
ax.axhline(0.5, color='black', linewidth=0.5, linestyle=':', alpha=0.5)
ax.set_ylabel(r'$\pi_t^{\mathrm{filter}}$', fontsize=11)
ax.set_xlabel('Date', fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_xlim(dates.min(), dates.max())
shade_nber(ax)

# Add crisis labels
for start, end, lbl, y_pos, va in crisis_labels:
    mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
    ax.annotate(lbl, xy=(mid, y_pos), fontsize=10, fontweight='bold',
                ha='center', va=va, color='#333333',
                annotation_clip=False)

ax.legend(fontsize=9, loc='upper left')
ax.set_title(r'Filtered Regime Probability $\pi_t^{\mathrm{filter}}$', fontsize=12)

plt.tight_layout()
fig.savefig('regime_probabilities.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved regime_probabilities.png (single panel)")

# ── Figure 2: features_hmm.png (4 features with NBER shading) ──────────────

feature_labels = {
    'DD_z':    'DD (Drawdown)',
    'CS_z':    'CS (BAA-AAA Credit Spread)',
    'DISP_z':  'DISP (Return Dispersion)',
    'REL_N_z': 'REL_N (Market Participation)',
}

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

for ax, feat in zip(axes, features_z):
    ax.plot(dates, panel[feat], linewidth=0.9, color='steelblue')
    ax.set_ylabel(feature_labels[feat], fontsize=9)
    ax.axhline(0, color='black', linewidth=0.4, linestyle='--', alpha=0.5)
    shade_nber(ax)
    ax.grid(axis='y', alpha=0.2)
    ax.set_xlim(dates.min(), dates.max())

axes[-1].set_xlabel('Date', fontsize=10)
axes[0].set_title('HMM Input Features (standardised, training-sample statistics)', fontsize=12)
fig.align_ylabels(axes)

plt.subplots_adjust(hspace=0.08)
plt.tight_layout()
fig.savefig('features_hmm.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved features_hmm.png (correct 4 features)")
