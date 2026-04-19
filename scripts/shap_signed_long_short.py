"""
Two charts: signed SHAP contribution by momentum horizon, calm vs panic.
One for long leg, one for short leg.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('shap_portfolio_analysis.csv')

x = np.arange(1, 13)
w = 0.35
horizons = [f'mom_{h}' for h in range(1, 13)]

# Figure 1: Long leg
fig1, ax = plt.subplots(figsize=(10, 5.5))
calm = df[(df['Regime'] == 'Calm') & (df['Leg'] == 'long') & (df['Horizon'].isin(horizons))]
panic = df[(df['Regime'] == 'Panic') & (df['Leg'] == 'long') & (df['Horizon'].isin(horizons))]

calm_vals = calm.sort_values('Horizon', key=lambda s: s.str.extract(r'(\d+)')[0].astype(int))['Signed SHAP (bps)'].values
panic_vals = panic.sort_values('Horizon', key=lambda s: s.str.extract(r'(\d+)')[0].astype(int))['Signed SHAP (bps)'].values

ax.bar(x - w/2, calm_vals, w, label='Calm', color='steelblue', alpha=0.85, edgecolor='white')
ax.bar(x + w/2, panic_vals, w, label='Panic', color='#E53935', alpha=0.85, edgecolor='white')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f'{h}' for h in range(1, 13)], fontsize=10)
ax.set_xlabel('Momentum lookback horizon (months)', fontsize=11)
ax.set_ylabel('Mean signed SHAP contribution (bps)', fontsize=11)
ax.set_title('Long leg', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig1.savefig('plots/shap_signed_long.png', dpi=150, bbox_inches='tight')
plt.close(fig1)

# Figure 2: Short leg
fig2, ax = plt.subplots(figsize=(10, 5.5))
calm = df[(df['Regime'] == 'Calm') & (df['Leg'] == 'short') & (df['Horizon'].isin(horizons))]
panic = df[(df['Regime'] == 'Panic') & (df['Leg'] == 'short') & (df['Horizon'].isin(horizons))]

calm_vals = calm.sort_values('Horizon', key=lambda s: s.str.extract(r'(\d+)')[0].astype(int))['Signed SHAP (bps)'].values
panic_vals = panic.sort_values('Horizon', key=lambda s: s.str.extract(r'(\d+)')[0].astype(int))['Signed SHAP (bps)'].values

ax.bar(x - w/2, calm_vals, w, label='Calm', color='steelblue', alpha=0.85, edgecolor='white')
ax.bar(x + w/2, panic_vals, w, label='Panic', color='#E53935', alpha=0.85, edgecolor='white')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f'{h}' for h in range(1, 13)], fontsize=10)
ax.set_xlabel('Momentum lookback horizon (months)', fontsize=11)
ax.set_ylabel('Mean signed SHAP contribution (bps)', fontsize=11)
ax.set_title('Short leg', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig2.savefig('plots/shap_signed_short.png', dpi=150, bbox_inches='tight')
plt.close(fig2)

from PIL import Image
for name in ['shap_signed_long', 'shap_signed_short']:
    img = Image.open(f'{name}.png')
    img.save(f'{name}.pdf', 'PDF', resolution=150)

print("Saved: shap_signed_long.png/pdf, shap_signed_short.png/pdf")
