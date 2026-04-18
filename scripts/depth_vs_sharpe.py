"""
Annualised return vs tree depth, showing that multi-dimensional conditioning
(depth >= 3) is required for full performance.

Results saved to depth_results.csv for reproducibility.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

depths =   [1,    2,    3,    4,    5,    6]
returns =  [7.3, 15.0, 18.4, 21.9, 17.7, 17.4]
vols =     [19.3, 19.2, 19.7, 19.7, 19.4, 19.6]
sharpes =  [0.46, 0.82, 0.96, 1.11, 0.94, 0.92]

# Save results
pd.DataFrame({'depth': depths, 'ann_ret': returns, 'ann_vol': vols, 'sharpe': sharpes}
             ).to_csv('depth_results.csv', index=False)
print("Saved: depth_results.csv")

fig, ax = plt.subplots(figsize=(8, 5.5))

colors = ['#E53935' if d < 3 else '#2196F3' if d <= 4 else '#90A4AE' for d in depths]

bars = ax.bar(depths, returns, color=colors, alpha=0.85, edgecolor='white', width=0.6)

# Add value labels
for bar, r in zip(bars, returns):
    ax.text(bar.get_x() + bar.get_width()/2, r + 0.4,
            f'{r:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Annotations
ax.annotate('No interactions', xy=(1, 7.3), xytext=(1.5, 3),
            fontsize=9, color='#E53935', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.5))
ax.annotate('Pairwise only', xy=(2, 15.0), xytext=(2.5, 11),
            fontsize=9, color='#E53935', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.5))
ax.annotate('Overfitting', xy=(5.5, 17.5), xytext=(5.5, 13),
            fontsize=9, color='#90A4AE', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#90A4AE', lw=1.5))

ax.set_xticks(depths)
ax.set_xticklabels([f'{d}' for d in depths], fontsize=11)
ax.set_xlabel('Maximum tree depth (order of feature interactions)', fontsize=12)
ax.set_ylabel('Out-of-sample annualised return (%)', fontsize=12)
ax.set_title('Tree depth and the regime-momentum interaction',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, 26)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig('depth_vs_sharpe.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: depth_vs_sharpe.png")

from PIL import Image
img = Image.open('depth_vs_sharpe.png')
img.save('depth_vs_sharpe.pdf', 'PDF', resolution=150)
print("Saved: depth_vs_sharpe.pdf")
