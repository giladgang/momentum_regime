"""
Plot the CRRA dual-utility risk-aversion sweep.

Reads results/two_model_dual_util.csv and produces:
    plots/risk_aversion_dual_util.pdf  (vector, for thesis)
    plots/risk_aversion_dual_util.png  (raster, for preview)

Four panels: Sharpe, annualised return, annualised volatility, max drawdown vs gamma.
Highlights gamma = {0, 0.1, 0.5, 1.0} as reference points.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/two_model_dual_util.csv')
print(f"Loaded {len(df)} gamma points from results/two_model_dual_util.csv")

# Restrict to the useful range (gamma <= 1) for the thesis figure.
# The beyond-1 regime is discussed qualitatively in the text.
df = df[df['gamma'] <= 1.0].reset_index(drop=True)
print(f"Plotting {len(df)} points over gamma in [0, 1]")

HIGHLIGHTS = [0.0, 0.1, 0.5, 1.0]
HIGHLIGHT_COLOR = '#C62828'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 10,
})

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
gammas = df['gamma'].values

def panel(ax, y, color, title, ylabel, ypct=False):
    ax.plot(gammas, y, 's-', color=color, linewidth=2.0, markersize=5,
            markerfacecolor='white', markeredgewidth=1.8)
    # highlight points
    for hi in HIGHLIGHTS:
        if hi in gammas:
            idx = np.where(gammas == hi)[0][0]
            yi = y[idx]
            ax.plot([hi], [yi], 'o', color=HIGHLIGHT_COLOR, markersize=9,
                    markerfacecolor=HIGHLIGHT_COLOR, zorder=5)
            label = f"{yi:.1%}" if ypct else f"{yi:.2f}"
            ax.annotate(label, (hi, yi), textcoords='offset points',
                        xytext=(8, 10), fontsize=9, color=HIGHLIGHT_COLOR,
                        fontweight='bold')
    ax.set_xlabel(r'Risk aversion $\gamma$')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)

panel(axes[0, 0], df['sharpe'].values,  '#1B4F8A',
      'Sharpe Ratio', 'Sharpe')
panel(axes[0, 1], df['ann_ret'].values, '#2E6B4F',
      'Annualised Return', 'Ann. Return', ypct=True)
panel(axes[1, 0], df['ann_vol'].values, '#9B2226',
      'Annualised Volatility', 'Ann. Vol', ypct=True)
panel(axes[1, 1], df['mdd'].values,     '#5B3A8C',
      'Maximum Drawdown', 'Max DD', ypct=True)

fig.suptitle(r'Risk-Aversion Sweep: Sharpe, Return, Volatility, Drawdown '
             r'vs. $\gamma$',
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.975])
plt.savefig('plots/risk_aversion_dual_util.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('plots/risk_aversion_dual_util.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)
print("Saved: plots/risk_aversion_dual_util.pdf")
print("Saved: plots/risk_aversion_dual_util.png")
