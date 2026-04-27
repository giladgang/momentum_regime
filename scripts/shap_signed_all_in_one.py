"""
Two charts: Long leg and Short leg.
Each chart shows signed SHAP for all 12 momentum horizons + pi_filter,
with calm and panic side by side.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('shap_portfolio_analysis.csv')

features = [f'mom_{h}' for h in range(1, 13)] + ['pi_filter']
x_labels = [f'{h}' for h in range(1, 13)] + [r'$\pi^{filter}$']
x = np.arange(len(features))
w = 0.35

for leg, title in [('long', 'Long leg'), ('short', 'Short leg')]:
    fig, ax = plt.subplots(figsize=(12, 6))

    calm = df[(df['Regime'] == 'Calm') & (df['Leg'] == leg) & (df['Horizon'].isin(features))]
    panic = df[(df['Regime'] == 'Panic') & (df['Leg'] == leg) & (df['Horizon'].isin(features))]

    # Sort by feature order
    calm = calm.set_index('Horizon').loc[features]
    panic = panic.set_index('Horizon').loc[features]

    ax.bar(x - w/2, calm['Signed SHAP (bps)'].values, w,
           label='Calm', color='steelblue', alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, panic['Signed SHAP (bps)'].values, w,
           label='Panic', color='#E53935', alpha=0.85, edgecolor='white')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(11.5, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Mean signed SHAP contribution (bps)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_png = f'plots/diagnostic/shap_signed_{leg}_final.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_png}")

from PIL import Image
for leg in ['long', 'short']:
    src = f'plots/diagnostic/shap_signed_{leg}_final.png'
    dst = f'plots/diagnostic/shap_signed_{leg}_final.pdf'
    img = Image.open(src)
    img.save(dst, 'PDF', resolution=150)
    print(f"Saved: {dst}")
