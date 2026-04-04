"""
Generate a simple one-row pipeline diagram for the cross-sectional model.
Features → Model → Score & Rank → Top Decile Portfolio
Output: cs_model_diagram.png (and .pdf)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colours ──
BLUE = '#1B4F8A'
PURPLE = '#5B3A8C'
GREEN = '#2E6B4F'
ORANGE = '#E65100'
TEXT_DARK = '#1A1A1A'
GREY = '#666666'
EQ_COLOR = '#333333'

FEAT_FILL = '#EEF2FA'
FEAT_EDGE = '#6683AA'
MODEL_LR_FILL = '#E3ECFF'
MODEL_LR_EDGE = '#1B4F8A'
MODEL_XGB_FILL = '#EDE5F5'
MODEL_XGB_EDGE = '#5B3A8C'
RANK_FILL = '#FFF3E0'
RANK_EDGE = '#E65100'
PORT_FILL = '#E0F0E8'
PORT_EDGE = '#2E6B4F'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

fig, ax = plt.subplots(1, 1, figsize=(18, 5.5))
ax.set_xlim(-0.5, 22)
ax.set_ylim(-2.5, 3.2)
ax.set_aspect('equal')
ax.axis('off')

def arrow_right(ax, x1, y, x2, color=GREY, lw=2.2):
    ax.add_patch(FancyArrowPatch(
        posA=(x1, y), posB=(x2, y),
        arrowstyle='-|>', mutation_scale=18, linewidth=lw,
        color=color, zorder=5))

# ── Box positions (left to right) ──
bh = 3.8  # box height

# 1. FEATURES
fx, fw = 2.2, 4.2
feat_box = FancyBboxPatch((fx - fw/2, -bh/2), fw, bh,
                           boxstyle='round,pad=0.3', facecolor=FEAT_FILL,
                           edgecolor=FEAT_EDGE, linewidth=2, zorder=3)
ax.add_patch(feat_box)
ax.text(fx, 1.3, 'Features', fontsize=15, ha='center', va='center',
        color=FEAT_EDGE, fontweight='bold', zorder=4)
ax.text(fx, 0.55, r'mom$_{1}$...mom$_{12}$', fontsize=11, ha='center', va='center',
        color=TEXT_DARK, zorder=4)
ax.text(fx, 0.05, r'$\pi_{\mathrm{filter}}$  (from HMM)', fontsize=11, ha='center', va='center',
        color=PURPLE, fontweight='bold', zorder=4)
ax.text(fx, -0.5, 'bm, roe, leverage', fontsize=10.5, ha='center', va='center',
        color=GREY, zorder=4)
ax.text(fx, -0.95, 'log market cap, ...', fontsize=10.5, ha='center', va='center',
        color=GREY, zorder=4)

# Arrow 1
arrow_right(ax, fx + fw/2, 0.8, 6.8, FEAT_EDGE)
arrow_right(ax, fx + fw/2, -0.8, 6.8, FEAT_EDGE)

# 2. MODELS (two stacked)
mx, mw = 8.5, 3.2
mh_each = 1.5
# LR
lr_y = 0.8
lr_box = FancyBboxPatch((mx - mw/2, lr_y - mh_each/2), mw, mh_each,
                          boxstyle='round,pad=0.2', facecolor=MODEL_LR_FILL,
                          edgecolor=MODEL_LR_EDGE, linewidth=2, zorder=3)
ax.add_patch(lr_box)
ax.text(mx, lr_y + 0.2, 'LR', fontsize=14, ha='center', va='center',
        color=MODEL_LR_EDGE, fontweight='bold', zorder=4)
ax.text(mx, lr_y - 0.3, r'$\sigma(\mathbf{x}^{\top}\boldsymbol{\beta})$',
        fontsize=12, ha='center', va='center', color=EQ_COLOR, zorder=4)

# XGB
xgb_y = -0.8
xgb_box = FancyBboxPatch((mx - mw/2, xgb_y - mh_each/2), mw, mh_each,
                           boxstyle='round,pad=0.2', facecolor=MODEL_XGB_FILL,
                           edgecolor=MODEL_XGB_EDGE, linewidth=2, zorder=3)
ax.add_patch(xgb_box)
ax.text(mx, xgb_y + 0.2, 'XGBoost', fontsize=14, ha='center', va='center',
        color=MODEL_XGB_EDGE, fontweight='bold', zorder=4)
ax.text(mx, xgb_y - 0.3, r'$\sum_m f_m(\mathbf{x})$',
        fontsize=12, ha='center', va='center', color=EQ_COLOR, zorder=4)

# Arrow 2
arrow_right(ax, mx + mw/2, lr_y, 12.0, MODEL_LR_EDGE)
arrow_right(ax, mx + mw/2, xgb_y, 12.0, MODEL_XGB_EDGE)

# 3. SCORE & RANK
rx, rw = 13.8, 3.4
rank_box = FancyBboxPatch((rx - rw/2, -bh/2), rw, bh,
                           boxstyle='round,pad=0.3', facecolor=RANK_FILL,
                           edgecolor=RANK_EDGE, linewidth=2, zorder=3)
ax.add_patch(rank_box)
ax.text(rx, 1.1, 'Score & Rank', fontsize=14, ha='center', va='center',
        color=RANK_EDGE, fontweight='bold', zorder=4)
ax.text(rx, 0.4, 'Score every stock', fontsize=11, ha='center', va='center',
        color=TEXT_DARK, zorder=4)
ax.text(rx, -0.1, 'each month', fontsize=11, ha='center', va='center',
        color=TEXT_DARK, zorder=4)
ax.text(rx, -0.7, r'Rank by $\hat{P}_i$', fontsize=11.5, ha='center', va='center',
        color=EQ_COLOR, zorder=4)

# Arrow 3
arrow_right(ax, rx + rw/2, 0, 17.5, RANK_EDGE)

# 4. PORTFOLIO
px, pw = 19.3, 3.4
port_box = FancyBboxPatch((px - pw/2, -bh/2), pw, bh,
                           boxstyle='round,pad=0.3', facecolor=PORT_FILL,
                           edgecolor=PORT_EDGE, linewidth=2, zorder=3)
ax.add_patch(port_box)
ax.text(px, 1.1, 'Portfolio', fontsize=14, ha='center', va='center',
        color=PORT_EDGE, fontweight='bold', zorder=4)
ax.text(px, 0.35, 'Long top decile', fontsize=11, ha='center', va='center',
        color=TEXT_DARK, zorder=4)
ax.text(px, -0.2, 'NYSE breakpoints', fontsize=10.5, ha='center', va='center',
        color=GREY, zorder=4)
ax.text(px, -0.7, 'Value-weighted', fontsize=10.5, ha='center', va='center',
        color=GREY, zorder=4)

# ── HMM connection label ──
ax.annotate('', xy=(fx - 0.3, -bh/2 - 0.15), xytext=(fx - 0.3, -bh/2 - 0.8),
            arrowprops=dict(arrowstyle='-|>', color=PURPLE, lw=1.8), zorder=5)
ax.text(fx - 0.3, -bh/2 - 1.2, r'from HMM (Part 1)', fontsize=10.5,
        ha='center', va='center', color=PURPLE, fontstyle='italic', zorder=4,
        bbox=dict(boxstyle='round,pad=0.15', facecolor='#F3EDF7', edgecolor=PURPLE,
                  linewidth=1, alpha=0.9))

plt.savefig('cs_model_diagram.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.15)
plt.savefig('cs_model_diagram.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.15)
plt.close()
print('Saved cs_model_diagram.png and .pdf')
