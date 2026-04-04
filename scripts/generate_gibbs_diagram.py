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

fig, ax = plt.subplots(1, 1, figsize=(11, 18))
ax.set_xlim(-1.5, 12.5)
ax.set_ylim(-26, 2.5)
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
sw = 9.0
step_gap = 1.1

# ═══════════════════════════════════════════
# INIT
# ═══════════════════════════════════════════
init_y = 1.0
init_h = 1.6
add_box(ax, cx, init_y, 10.0, init_h, [
    ('Initialisation', 13, INIT_EDGE, True),
    ('', 3, TEXT_DARK, False),
    (r'Split months by volatility $\;\longrightarrow\;$ starting guesses for $\boldsymbol{\mu}_k,\, \boldsymbol{\Sigma}_k,\, \mathbf{P}$', 11, TEXT_DARK, False),
], INIT_FILL, INIT_EDGE)

arrow_down(ax, cx, init_y - init_h/2, init_y - init_h/2 - 1.5, INIT_EDGE)

# ═══════════════════════════════════════════
# Compute step positions
# ═══════════════════════════════════════════
loop_top = -1.2
sh = 4.0

s1_y = loop_top - 1.2 - sh/2
s2_y = s1_y - sh/2 - step_gap - sh/2
s3_y = s2_y - sh/2 - step_gap - sh/2

# Loop box fits tightly around content
loop_bot = s3_y - sh/2 - 0.6

# ═══════════════════════════════════════════
# LOOP BOX (drawn first as background)
# ═══════════════════════════════════════════
loop_box = FancyBboxPatch((0.0, loop_bot), 11, loop_top - loop_bot,
                            boxstyle='round,pad=0.35', facecolor=LOOP_BG,
                            edgecolor=LOOP_EDGE, linewidth=2, linestyle='--', zorder=0)
ax.add_patch(loop_box)
ax.text(0.5, loop_top - 0.25,
        r'Repeat for $M = 2{,}000$ iterations  (first $B = 500$ discarded as burn-in)',
        fontsize=12, fontstyle='italic', color='#555577', va='top', fontweight='bold')

# ═══════════════════════════════════════════
# STEP 1: FFBS
# ═══════════════════════════════════════════
add_box(ax, cx, s1_y, sw, sh, [
    ('Step 1: Assign each month to a regime  (FFBS)', 14, BLUE, True),
    ('', 3, TEXT_DARK, False),
    ('', 2, TEXT_DARK, False),
    (r'$P(s_t \mid \mathbf{z}_{1:T}) \;\propto\; f(\mathbf{z}_t \mid s_t) \;\cdot\; P(s_t \mid s_{t-1})$', 13, EQ_COLOR, False),
    ('', 1, TEXT_DARK, False),
    (r'posterior $\;\;\;\;\;\propto\;\;\;\;$ likelihood $\times$ transition prior', 10.5, GREY, False),
    ('', 2, TEXT_DARK, False),
    (r'Holds $\boldsymbol{\mu}_k,\, \boldsymbol{\Sigma}_k,\, \mathbf{P}$ fixed', 10, PURPLE, False),
], STEP1_FILL, STEP1_EDGE)

arrow_down(ax, cx, s1_y - sh/2, s1_y - sh/2 - step_gap, BLUE)

# ═══════════════════════════════════════════
# STEP 2: NIW update
# ═══════════════════════════════════════════
add_box(ax, cx, s2_y, sw, sh, [
    ('Step 2: Learn what each regime looks like  (NIW)', 14, PURPLE, True),
    ('', 3, TEXT_DARK, False),
    ('', 2, TEXT_DARK, False),
    (r'$p(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k \mid \mathbf{z}_k) \;\propto\; \prod_{t:\,s_t=k} \mathcal{N}(\mathbf{z}_t;\, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \;\cdot\; \mathrm{NIW}(\mathbf{m}_0, \kappa_0, \nu_0, \boldsymbol{\Psi}_0)$', 11.5, EQ_COLOR, False),
    ('', 1, TEXT_DARK, False),
    (r'posterior $\;\;\;\;\;\;\;\propto\;\;\;\;\;\;$ likelihood $\times$ conjugate prior', 10.5, GREY, False),
    ('', 2, TEXT_DARK, False),
    (r'Holds $\{s_t\}$ and $\mathbf{P}$ fixed', 10, PURPLE, False),
], STEP2_FILL, STEP2_EDGE)

arrow_down(ax, cx, s2_y - sh/2, s2_y - sh/2 - step_gap, PURPLE)

# ═══════════════════════════════════════════
# STEP 3: Dirichlet update
# ═══════════════════════════════════════════
add_box(ax, cx, s3_y, sw, sh, [
    ('Step 3: Learn how often regimes switch  (Dirichlet)', 14, '#9B2226', True),
    ('', 3, TEXT_DARK, False),
    ('', 2, TEXT_DARK, False),
    (r'$p(\mathbf{P}_{i,\cdot} \mid \{s_t\}) \;\propto\; \prod_t \; P_{i,\,s_t}^{\;\,n_{ij}} \;\;\cdot\;\; \mathrm{Dir}(\boldsymbol{\alpha}_i)$', 12, EQ_COLOR, False),
    ('', 1, TEXT_DARK, False),
    (r'posterior $\;\;\propto\;\;\;\;\;\;$ likelihood $\;\;\times\;\;$ conjugate prior', 10.5, GREY, False),
    ('', 2, TEXT_DARK, False),
    (r'Holds $\{s_t\},\, \boldsymbol{\mu}_k,\, \boldsymbol{\Sigma}_k$ fixed', 10, PURPLE, False),
], STEP3_FILL, STEP3_EDGE)

# ── Loop-back arrow ──
loop_bottom_y = s3_y - sh/2 - 0.7
loop_x = 0.5
# Down from Step 3
ax.plot([cx, cx], [s3_y - sh/2, loop_bottom_y], color=GREY, lw=1.8, zorder=2)
# Across to the left
ax.plot([cx, loop_x], [loop_bottom_y, loop_bottom_y], color=GREY, lw=1.8, zorder=2)
# Up along the left side to above Step 1
step1_top = s1_y + sh/2
ax.plot([loop_x, loop_x], [loop_bottom_y, step1_top + 0.5], color=GREY, lw=1.8, zorder=2)
# Across to the right, ending above Step 1 center
ax.plot([loop_x, cx], [step1_top + 0.5, step1_top + 0.5], color=GREY, lw=1.8, zorder=2)
# Arrow down into the top of Step 1
ax.add_patch(FancyArrowPatch(
    posA=(cx, step1_top + 0.5), posB=(cx, step1_top + 0.05),
    arrowstyle='-|>', mutation_scale=14, linewidth=1.8,
    color=GREY, zorder=5))
ax.text(loop_x - 0.45, (loop_bottom_y + s1_y) / 2, 'next\niteration',
        fontsize=11, fontstyle='italic', color=GREY,
        ha='center', va='center', rotation=90,
        bbox=dict(boxstyle='round,pad=0.2', facecolor=LOOP_BG, edgecolor='none'))

# ═══════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════
out_y = loop_bot - 2.5
arrow_down(ax, cx, loop_bot - 0.35, out_y + 1.1, GREEN)

add_box(ax, cx, out_y, 10.0, 2.2, [
    ('Output', 13, GREEN, True),
    ('', 3, TEXT_DARK, False),
    (r'Average the post-burn-in draws: $\bar{\boldsymbol{\mu}}_k,\, \bar{\boldsymbol{\Sigma}}_k,\, \bar{\mathbf{P}}$', 11.5, TEXT_DARK, False),
    ('', 2, TEXT_DARK, False),
    (r'Apply forward filter $\longrightarrow$ $\pi_t^{\,\mathrm{filter}} = P(s_t = 1 \mid \mathbf{z}_{1:t})$', 11.5, TEXT_DARK, False),
], OUTPUT_FILL, OUTPUT_EDGE)

plt.savefig('gibbs_sampling_diagram.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.15)
plt.savefig('gibbs_sampling_diagram.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.15)
plt.close()
print('Saved gibbs_sampling_diagram.png and .pdf')
