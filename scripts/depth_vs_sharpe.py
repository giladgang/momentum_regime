"""
Sharpe ratio vs tree depth, showing that multi-dimensional conditioning
(depth >= 3) is required for full performance.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

depths = [1, 2, 3, 4, 5, 6]
sharpes = [0.46, 0.82, 0.96, 1.11, 0.94, 0.92]

fig, ax = plt.subplots(figsize=(8, 5.5))

colors = ['#E53935' if d < 3 else '#2196F3' if d <= 4 else '#90A4AE' for d in depths]

bars = ax.bar(depths, sharpes, color=colors, alpha=0.85, edgecolor='white', width=0.6)

# Add value labels
for bar, s in zip(bars, sharpes):
    ax.text(bar.get_x() + bar.get_width()/2, s + 0.02,
            f'{s:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Annotations
ax.annotate('No interactions', xy=(1, 0.46), xytext=(1.5, 0.25),
            fontsize=9, color='#E53935', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.5))
ax.annotate('Pairwise only', xy=(2, 0.82), xytext=(2.5, 0.65),
            fontsize=9, color='#E53935', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.5))
ax.annotate('Overfitting', xy=(5.5, 0.93), xytext=(5.5, 0.78),
            fontsize=9, color='#90A4AE', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#90A4AE', lw=1.5))

ax.set_xticks(depths)
ax.set_xticklabels([f'{d}' for d in depths], fontsize=11)
ax.set_xlabel('Maximum tree depth (order of feature interactions)', fontsize=12)
ax.set_ylabel('Out-of-sample Sharpe ratio', fontsize=12)
ax.set_title('Tree depth and the regime-momentum interaction',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.3)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig('depth_vs_sharpe.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: depth_vs_sharpe.png")

from PIL import Image
img = Image.open('depth_vs_sharpe.png')
img.save('depth_vs_sharpe.pdf', 'PDF', resolution=150)
print("Saved: depth_vs_sharpe.pdf")
