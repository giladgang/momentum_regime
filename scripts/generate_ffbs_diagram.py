"""
Generate a clear FFBS (Forward-Filtering Backward-Sampling) flowchart.
Output: ffbs_diagram.png (and .pdf)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colours (matching Gibbs diagram) ──
BLUE = '#1B4F8A'
PURPLE = '#5B3A8C'
GREEN = '#2E6B4F'
TEXT_DARK = '#1A1A1A'
GREY = '#666666'
EQ_COLOR = '#333333'

INPUT_FILL = '#FFF3E0'
INPUT_EDGE = '#E65100'
FWD_FILL = '#E3ECFF'
FWD_EDGE = '#1B4F8A'
TERM_FILL = '#FCE4EC'
TERM_EDGE = '#9B2226'
BWD_FILL = '#EDE5F5'
BWD_EDGE = '#5B3A8C'
OUTPUT_FILL = '#E0F0E8'
OUTPUT_EDGE = '#2E6B4F'
LOOP_BG = '#FAFAFE'
LOOP_EDGE = '#AAAACC'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

fig, ax = plt.subplots(1, 1, figsize=(11, 20))
ax.set_xlim(-1.5, 13.0)
ax.set_ylim(-28, 2.5)
ax.set_aspect('equal')
ax.axis('off')

# ── Helpers ──
def add_box(ax, x, y, w, h, lines, fc, ec, lw=2, zorder=3):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle='round,pad=0.25', facecolor=fc,
                          edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    n = len(lines)
    spacing = 0.15
    positions = []
    current_y = y + (n - 1) * spacing * 0.5
    for i in range(n):
        positions.append(current_y - i * spacing * 2.2)
    mid = (positions[0] + positions[-1]) / 2
    offset = y - mid
    positions = [p + offset for p in positions]
    for (text, fs, color, bold), py in zip(lines, positions):
        fw = 'bold' if bold else 'normal'
        ax.text(x, py, text, fontsize=fs, ha='center', va='center',
                color=color, fontweight=fw, zorder=zorder+1)

def arrow_down(ax, x, y1, y2, color=GREY, lw=2):
    ax.add_patch(FancyArrowPatch(
        posA=(x, y1), posB=(x, y2),
        arrowstyle='-|>', mutation_scale=15, linewidth=lw,
        color=color, zorder=5))

cx = 5.5
sw = 9.5
gap = 1.5

# ═══════════════════════════════════════════
# INPUT
# ═══════════════════════════════════════════
input_y = 1.0
input_h = 2.0
add_box(ax, cx, input_y, 10.0, input_h, [
    ('Input', 15, INPUT_EDGE, True),
    ('', 3, TEXT_DARK, False),
    (r'Current parameters $\boldsymbol{\mu}_k,\; \boldsymbol{\Sigma}_k,\; \mathbf{P}$', 13, TEXT_DARK, False),
    (r'and observations $\mathbf{z}_{1:T}$', 13, TEXT_DARK, False),
], INPUT_FILL, INPUT_EDGE)

# Arrow: Input → forward loop (orange, ends at dotted line)
arrow_down(ax, cx, input_y - input_h/2, input_y - input_h/2 - gap + 0.1, INPUT_EDGE)

# ═══════════════════════════════════════════
# FORWARD PASS (with loop box)
# ═══════════════════════════════════════════
fwd_h = 4.2
fwd_loop_top = input_y - input_h/2 - gap - 0.1
fwd_y = fwd_loop_top - 1.4 - fwd_h/2
fwd_loop_bot = fwd_y - fwd_h/2 - 0.8

# Loop background
fwd_loop_box = FancyBboxPatch((0.2, fwd_loop_bot), 11.4, fwd_loop_top - fwd_loop_bot,
                                boxstyle='round,pad=0.3', facecolor=LOOP_BG,
                                edgecolor=LOOP_EDGE, linewidth=2, linestyle='--', zorder=0)
ax.add_patch(fwd_loop_box)
ax.text(0.7, fwd_loop_top - 0.3,
        r'Forward pass: for $t = 1, 2, \ldots, T$',
        fontsize=13, fontstyle='italic', color='#555577', va='top', fontweight='bold')

add_box(ax, cx, fwd_y, sw, fwd_h, [
    ('Compute filtered probability', 15, BLUE, True),
    ('', 3, TEXT_DARK, False),
    ('', 2, TEXT_DARK, False),
    (r'$\alpha_t(k) \;\propto\; f(\mathbf{z}_t \mid s_t\!=\!k) \cdot \sum_j \alpha_{t-1}(j) P_{jk}$', 14, EQ_COLOR, False),
    ('', 2, TEXT_DARK, False),
    (r'unnormalised $\;\propto\;$ emission density $\times$ predicted prob.', 11.5, GREY, False),
    ('', 3, TEXT_DARK, False),
    (r'Normalise: $\alpha_t(0) + \alpha_t(1) = 1$', 12, BLUE, False),
], FWD_FILL, FWD_EDGE)

# Loop-back arrow (right side)
fwd_bot_arrow = fwd_y - fwd_h/2 - 0.5
fwd_rx = cx + sw/2 + 0.6
ax.plot([cx, cx], [fwd_y - fwd_h/2, fwd_bot_arrow], color=GREY, lw=1.8, zorder=2)
ax.plot([cx, fwd_rx], [fwd_bot_arrow, fwd_bot_arrow], color=GREY, lw=1.8, zorder=2)
ax.plot([fwd_rx, fwd_rx], [fwd_bot_arrow, fwd_y + fwd_h/2 + 0.5], color=GREY, lw=1.8, zorder=2)
ax.plot([fwd_rx, cx], [fwd_y + fwd_h/2 + 0.5, fwd_y + fwd_h/2 + 0.5], color=GREY, lw=1.8, zorder=2)
ax.add_patch(FancyArrowPatch(
    posA=(cx, fwd_y + fwd_h/2 + 0.5), posB=(cx, fwd_y + fwd_h/2 + 0.05),
    arrowstyle='-|>', mutation_scale=14, linewidth=1.8, color=GREY, zorder=5))
ax.text(fwd_rx + 0.45, fwd_y, r'$t \to t\!+\!1$',
        fontsize=12, fontstyle='italic', color=GREY,
        ha='center', va='center', rotation=270,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))

# Arrow: forward loop → terminal state (blue, starts at dotted line)
arrow_down(ax, cx, fwd_loop_bot - 0.35, fwd_loop_bot - gap + 0.1, FWD_EDGE)

# ═══════════════════════════════════════════
# SAMPLE TERMINAL STATE
# ═══════════════════════════════════════════
term_h = 2.2
term_y = fwd_loop_bot - gap - term_h/2
add_box(ax, cx, term_y, sw, term_h, [
    ('Sample terminal state', 15, TERM_EDGE, True),
    ('', 3, TEXT_DARK, False),
    (r'$s_T \sim \mathrm{Cat}\;\!\left(\alpha_T(0),\;\; \alpha_T(1)\right)$', 14, EQ_COLOR, False),
], TERM_FILL, TERM_EDGE)

# Arrow: terminal state → backward loop (red, ends at dotted line)
arrow_down(ax, cx, term_y - term_h/2, term_y - term_h/2 - gap + 0.1, TERM_EDGE)

# ═══════════════════════════════════════════
# BACKWARD PASS (with loop box)
# ═══════════════════════════════════════════
bwd_h = 4.2
bwd_loop_top = term_y - term_h/2 - gap - 0.1
bwd_y = bwd_loop_top - 1.4 - bwd_h/2
bwd_loop_bot = bwd_y - bwd_h/2 - 0.8

# Loop background
bwd_loop_box = FancyBboxPatch((0.2, bwd_loop_bot), 11.4, bwd_loop_top - bwd_loop_bot,
                                boxstyle='round,pad=0.3', facecolor=LOOP_BG,
                                edgecolor=LOOP_EDGE, linewidth=2, linestyle='--', zorder=0)
ax.add_patch(bwd_loop_box)
ax.text(0.7, bwd_loop_top - 0.3,
        r'Backward pass: for $t = T\!-\!1,\; T\!-\!2,\; \ldots,\; 1$',
        fontsize=13, fontstyle='italic', color='#555577', va='top', fontweight='bold')

add_box(ax, cx, bwd_y, sw, bwd_h, [
    ('Sample state conditioned on future', 15, BWD_EDGE, True),
    ('', 3, TEXT_DARK, False),
    ('', 2, TEXT_DARK, False),
    (r'$P(s_t\!=\!k) \;\propto\; \alpha_t(k) \cdot P_{k,s_{t+1}}$', 14, EQ_COLOR, False),
    ('', 2, TEXT_DARK, False),
    (r'posterior $\;\propto\;$ filtered belief $\times$ transition to $s_{t+1}$', 11.5, GREY, False),
    ('', 3, TEXT_DARK, False),
    (r'Draw $s_t$ from the normalised distribution', 12, PURPLE, False),
], BWD_FILL, BWD_EDGE)

# Loop-back arrow (right side)
bwd_bot_arrow = bwd_y - bwd_h/2 - 0.5
bwd_rx = cx + sw/2 + 0.6
ax.plot([cx, cx], [bwd_y - bwd_h/2, bwd_bot_arrow], color=GREY, lw=1.8, zorder=2)
ax.plot([cx, bwd_rx], [bwd_bot_arrow, bwd_bot_arrow], color=GREY, lw=1.8, zorder=2)
ax.plot([bwd_rx, bwd_rx], [bwd_bot_arrow, bwd_y + bwd_h/2 + 0.5], color=GREY, lw=1.8, zorder=2)
ax.plot([bwd_rx, cx], [bwd_y + bwd_h/2 + 0.5, bwd_y + bwd_h/2 + 0.5], color=GREY, lw=1.8, zorder=2)
ax.add_patch(FancyArrowPatch(
    posA=(cx, bwd_y + bwd_h/2 + 0.5), posB=(cx, bwd_y + bwd_h/2 + 0.05),
    arrowstyle='-|>', mutation_scale=14, linewidth=1.8, color=GREY, zorder=5))
ax.text(bwd_rx + 0.45, bwd_y, r'$t \to t\!-\!1$',
        fontsize=12, fontstyle='italic', color=GREY,
        ha='center', va='center', rotation=270,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))

# Arrow: backward loop → output (purple, starts at dotted line)
arrow_down(ax, cx, bwd_loop_bot - 0.35, bwd_loop_bot - gap + 0.1, BWD_EDGE)

# ═══════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════
out_h = 2.2
out_y = bwd_loop_bot - gap - out_h/2
add_box(ax, cx, out_y, 10.0, out_h, [
    ('Output', 15, GREEN, True),
    ('', 3, TEXT_DARK, False),
    (r'A complete regime assignment for every month: $\{s_1, s_2, \ldots, s_T\}$', 12.5, TEXT_DARK, False),
    (r'Returned to the Gibbs sampler as the current state path', 12, GREY, False),
], OUTPUT_FILL, OUTPUT_EDGE)

# Adjust y limits
ax.set_ylim(out_y - out_h/2 - 0.8, 2.5)

plt.savefig('ffbs_diagram.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.15)
plt.savefig('ffbs_diagram.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.15)
plt.close()
print('Saved ffbs_diagram.png and .pdf')
