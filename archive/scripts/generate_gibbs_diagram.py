"""
Generate a clear, intuitive Gibbs sampling flowchart for the HMM estimation.
Output: gibbs_sampling_diagram.png (and .pdf)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colours ──
BLUE = '#1B4F8A'
PURPLE = '#5B3A8C'
GREEN = '#2E6B4F'
TEXT_DARK = '#1A1A1A'
GREY = '#666666'
INIT_FILL = '#FFF3E0'
INIT_EDGE = '#E65100'
STEP1_FILL = '#E3ECFF'
STEP1_EDGE = '#1B4F8A'
STEP2_FILL = '#EDE5F5'
STEP2_EDGE = '#5B3A8C'
STEP3_FILL = '#FCE4EC'
STEP3_EDGE = '#9B2226'
OUTPUT_FILL = '#E0F0E8'
OUTPUT_EDGE = '#2E6B4F'
LOOP_BG = '#FAFAFE'
LOOP_EDGE = '#AAAACC'
EQ_COLOR = '#333333'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

FS = 13  # uniform font size everywhere

fig, ax = plt.subplots(1, 1, figsize=(12, 14))
ax.set_xlim(-1.5, 13.5)
ax.set_ylim(-19, 2.5)
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
sw = 10.5      # step box width (wider so titles fit)
init_w = 10.5  # init/label/output width matches steps
step_gap = 1.0

# ═══════════════════════════════════════════
# INIT
# ═══════════════════════════════════════════
init_y = 1.0
init_h = 1.8
add_box(ax, cx, init_y, init_w, init_h, [
    ('Initialisation', FS, INIT_EDGE, True),
    ('', 3, TEXT_DARK, False),
    (r'Split months by volatility $\;\longrightarrow\;$ starting guesses for $\boldsymbol{\mu}_k,\, \boldsymbol{\Sigma}_k,\, \mathbf{P}$', FS, TEXT_DARK, False),
], INIT_FILL, INIT_EDGE)

arrow_down(ax, cx, init_y - init_h/2, init_y - init_h/2 - 1.5, INIT_EDGE)

# ═══════════════════════════════════════════
# Compute step positions
# ═══════════════════════════════════════════
loop_top = -1.2
sh = 2.4

s1_y = loop_top - 1.4 - sh/2     # extra headroom above Step 1 for the "Repeat" label and loop-back arrow
s2_y = s1_y - sh/2 - step_gap - sh/2
s3_y = s2_y - sh/2 - step_gap - sh/2

# Loop box fits tightly around content
loop_bot = s3_y - sh/2 - 0.6

# ═══════════════════════════════════════════
# LOOP BOX (drawn first as background)
# ═══════════════════════════════════════════
loop_box = FancyBboxPatch((-1.0, loop_bot), 13, loop_top - loop_bot,
                            boxstyle='round,pad=0.35', facecolor=LOOP_BG,
                            edgecolor=LOOP_EDGE, linewidth=2, linestyle='--', zorder=0)
ax.add_patch(loop_box)
loop_label_y = loop_top - 0.25
ax.text(cx, loop_label_y,
        r'Repeat for $M = 2{,}000$ iterations  (first $B = 500$ discarded as burn-in)',
        fontsize=FS, fontstyle='italic', color='#555577', va='top', ha='center', fontweight='bold')

# ═══════════════════════════════════════════
# STEP 1: FFBS
# ═══════════════════════════════════════════
add_box(ax, cx, s1_y, sw, sh, [
    (r'Step 1: Sample the hidden state path  (FFBS)', FS, BLUE, True),
    ('', 2, TEXT_DARK, False),
    (r'Holds $\boldsymbol{\mu}_k,\, \boldsymbol{\Sigma}_k,\, \mathbf{P}$ fixed', FS, PURPLE, False),
], STEP1_FILL, STEP1_EDGE)

arrow_down(ax, cx, s1_y - sh/2, s1_y - sh/2 - step_gap, BLUE)

# ═══════════════════════════════════════════
# STEP 2: NIW update
# ═══════════════════════════════════════════
add_box(ax, cx, s2_y, sw, sh, [
    (r'Step 2: Update the emission parameters  (NIW posterior)', FS, PURPLE, True),
    ('', 2, TEXT_DARK, False),
    (r'Holds $\{s_t\}$ and $\mathbf{P}$ fixed', FS, PURPLE, False),
], STEP2_FILL, STEP2_EDGE)

arrow_down(ax, cx, s2_y - sh/2, s2_y - sh/2 - step_gap, PURPLE)

# ═══════════════════════════════════════════
# STEP 3: Dirichlet update
# ═══════════════════════════════════════════
add_box(ax, cx, s3_y, sw, sh, [
    (r'Step 3: Update the transition matrix  (Dirichlet posterior)', FS, '#9B2226', True),
    ('', 2, TEXT_DARK, False),
    (r'Holds $\{s_t\},\, \boldsymbol{\mu}_k,\, \boldsymbol{\Sigma}_k$ fixed', FS, PURPLE, False),
], STEP3_FILL, STEP3_EDGE)

# ── Loop-back arrow ──
# Routed on the LEFT side of the loop box, with the horizontal feed-in placed
# between the "Repeat for M=..." label and the Step 1 box.
loop_bottom_y = s3_y - sh/2 - 0.7
loop_x = -0.2
step1_top = s1_y + sh/2
horiz_top_y = (loop_label_y - 0.55)   # well below the label baseline
# Down from Step 3
ax.plot([cx, cx], [s3_y - sh/2, loop_bottom_y], color=GREY, lw=1.8, zorder=2)
# Across to the left
ax.plot([cx, loop_x], [loop_bottom_y, loop_bottom_y], color=GREY, lw=1.8, zorder=2)
# Up along the left side
ax.plot([loop_x, loop_x], [loop_bottom_y, horiz_top_y], color=GREY, lw=1.8, zorder=2)
# Across to the right, ending above Step 1 center
ax.plot([loop_x, cx], [horiz_top_y, horiz_top_y], color=GREY, lw=1.8, zorder=2)
# Arrow down into the top of Step 1
ax.add_patch(FancyArrowPatch(
    posA=(cx, horiz_top_y), posB=(cx, step1_top + 0.05),
    arrowstyle='-|>', mutation_scale=14, linewidth=1.8,
    color=GREY, zorder=5))
ax.text(loop_x - 0.6, (loop_bottom_y + s1_y) / 2, 'next\niteration',
        fontsize=FS, fontstyle='italic', color=GREY,
        ha='center', va='center', rotation=90,
        bbox=dict(boxstyle='round,pad=0.2', facecolor=LOOP_BG, edgecolor='none'))

# ═══════════════════════════════════════════
# REGIME IDENTIFICATION
# ═══════════════════════════════════════════
LABEL_FILL = '#FBE9E7'
LABEL_EDGE = '#D84315'

label_y = loop_bot - 2.0
label_h = 1.6
arrow_down(ax, cx, loop_bot - 0.35, label_y + label_h/2 + 0.1, LABEL_EDGE)

add_box(ax, cx, label_y, init_w, label_h, [
    ('Regime identification', FS, LABEL_EDGE, True),
    ('', 2, TEXT_DARK, False),
    (r'Label switching: state with higher $\bar{\mu}_{\mathrm{VOL}}$ $\;\rightarrow\;$ panic', FS, TEXT_DARK, False),
], LABEL_FILL, LABEL_EDGE)

# ═══════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════
out_y = label_y - label_h/2 - 2.0
out_h = 2.4
arrow_down(ax, cx, label_y - label_h/2, out_y + out_h/2 + 0.1, GREEN)

add_box(ax, cx, out_y, init_w, out_h, [
    ('Output', FS, GREEN, True),
    ('', 3, TEXT_DARK, False),
    (r'Average post-burn-in draws: $\bar{\boldsymbol{\mu}}_k,\, \bar{\boldsymbol{\Sigma}}_k,\, \bar{\mathbf{P}}$', FS, TEXT_DARK, False),
    ('', 2, TEXT_DARK, False),
    (r'Apply forward filter with posterior means $\;\rightarrow\; \pi_t^{\,\mathrm{filter}}$', FS, TEXT_DARK, False),
], OUTPUT_FILL, OUTPUT_EDGE)

plt.savefig('plots/thesis/gibbs_sampling_diagram.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.15)
plt.savefig('plots/thesis/gibbs_sampling_diagram.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.15)
plt.close()
print('Saved plots/thesis/gibbs_sampling_diagram.png and .pdf')
