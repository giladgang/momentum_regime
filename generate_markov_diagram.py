"""
Generate a professional two-state HMM diagram with hidden/observed distinction,
feature names, filtered probability output, and example state sequence.
Output: markov_chain_diagram.png (and .pdf)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

# ── Colours ──
CALM_BLUE = '#1B4F8A'
PANIC_RED = '#9B2226'
CALM_FILL = '#DAE5F5'
PANIC_FILL = '#F2D4D4'
TEXT_DARK = '#1A1A1A'
GREY = '#666666'
LIGHT_GREY = '#E8E8E8'
HIDDEN_BG = '#F7F7FB'
OBS_GREEN = '#2E6B4F'
OBS_FILL = '#E0F0E8'
OUTPUT_PURPLE = '#5B3A8C'
OUTPUT_FILL = '#EDE5F5'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

fig, axes = plt.subplots(2, 1, figsize=(12, 10),
                          gridspec_kw={'height_ratios': [5.0, 2.0], 'hspace': 0.05})

# ═══════════════════════════════════════════════════════════════
# Panel A: HMM diagram with hidden/observed layers
# ═══════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_xlim(-5, 15)
ax.set_ylim(-7.5, 5.5)
ax.set_aspect('equal')
ax.axis('off')

# Title removed -- figure caption in LaTeX provides this

# ── "Hidden" layer background ──
hidden_box = FancyBboxPatch((-4.2, -1.8), 18.4, 6.2,
                              boxstyle='round,pad=0.3', facecolor='#F7F7FB',
                              edgecolor='#AAAACC', linewidth=1.5, linestyle='--',
                              zorder=0)
ax.add_patch(hidden_box)
ax.text(-3.5, 4.0, 'Hidden Layer', fontsize=13, fontstyle='italic',
        color='#666688', va='center')
ax.text(-3.5, 3.4, '(unobserved)', fontsize=10, fontstyle='italic',
        color='#888899', va='center')

# ── State circles ──
calm_xy = np.array([2.0, 1.2])
panic_xy = np.array([8.0, 1.2])
R = 1.5

calm_circle = plt.Circle(calm_xy, R, facecolor=CALM_FILL,
                          edgecolor=CALM_BLUE, linewidth=2.5, zorder=3)
ax.add_patch(calm_circle)
ax.text(calm_xy[0], calm_xy[1] + 0.5, 'Calm', fontsize=14, fontweight='bold',
        ha='center', va='center', color=CALM_BLUE, zorder=4)
ax.text(calm_xy[0], calm_xy[1] + 0.05, r'$(s_t = 0)$', fontsize=10,
        ha='center', va='center', color=CALM_BLUE, zorder=4)
ax.text(calm_xy[0], calm_xy[1] - 0.55,
        r'$\mathbf{z}_t \sim \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)$',
        fontsize=9.5, ha='center', va='center', color=TEXT_DARK, zorder=4)

panic_circle = plt.Circle(panic_xy, R, facecolor=PANIC_FILL,
                           edgecolor=PANIC_RED, linewidth=2.5, zorder=3)
ax.add_patch(panic_circle)
ax.text(panic_xy[0], panic_xy[1] + 0.5, 'Panic', fontsize=14, fontweight='bold',
        ha='center', va='center', color=PANIC_RED, zorder=4)
ax.text(panic_xy[0], panic_xy[1] + 0.05, r'$(s_t = 1)$', fontsize=10,
        ha='center', va='center', color=PANIC_RED, zorder=4)
ax.text(panic_xy[0], panic_xy[1] - 0.55,
        r'$\mathbf{z}_t \sim \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$',
        fontsize=9.5, ha='center', va='center', color=TEXT_DARK, zorder=4)

# ── Curved transition arrows ──
akw = dict(arrowstyle='-|>', mutation_scale=18, linewidth=2.2, zorder=5)

# Calm -> Panic (lower)
start_cp = calm_xy + R * np.array([np.cos(np.radians(-35)), np.sin(np.radians(-35))])
end_cp = panic_xy + R * np.array([np.cos(np.radians(215)), np.sin(np.radians(215))])
ax.add_patch(FancyArrowPatch(posA=tuple(start_cp), posB=tuple(end_cp),
              connectionstyle='arc3,rad=0.25', color=PANIC_RED, **akw))
ax.text(5, -0.9, r'$P_{01}$', fontsize=13, ha='center', va='center',
        color=PANIC_RED, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor=HIDDEN_BG, edgecolor='none'))

# Panic -> Calm (upper)
start_pc = panic_xy + R * np.array([np.cos(np.radians(145)), np.sin(np.radians(145))])
end_pc = calm_xy + R * np.array([np.cos(np.radians(35)), np.sin(np.radians(35))])
ax.add_patch(FancyArrowPatch(posA=tuple(start_pc), posB=tuple(end_pc),
              connectionstyle='arc3,rad=0.25', color=CALM_BLUE, **akw))
ax.text(5, 3.3, r'$P_{10}$', fontsize=13, ha='center', va='center',
        color=CALM_BLUE, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor=HIDDEN_BG, edgecolor='none'))

# ── Self-loops ──
def draw_self_loop(ax, center, radius, side, color, label):
    lr = 0.65
    offset_x = radius * 0.75
    offset_y = radius + lr * 0.3
    if side == 'left':
        arc_cx = center[0] - offset_x
        arc_cy = center[1] + offset_y
        # Sweep counter-clockwise from bottom-right to bottom-left
        angles = np.linspace(np.radians(-40), np.radians(270), 100)
    else:
        # Mirror: sweep clockwise from bottom-left to bottom-right
        arc_cx = center[0] + offset_x
        arc_cy = center[1] + offset_y
        angles = np.linspace(np.radians(220), np.radians(-90), 100)

    lx = arc_cx + lr * np.cos(angles)
    ly = arc_cy + lr * np.sin(angles)
    dist = np.sqrt((lx - center[0])**2 + (ly - center[1])**2)
    outside = dist >= radius * 0.97

    segs = []
    in_seg = False
    for idx in range(len(outside)):
        if outside[idx] and not in_seg:
            start = idx; in_seg = True
        elif not outside[idx] and in_seg:
            segs.append((start, idx)); in_seg = False
    if in_seg:
        segs.append((start, len(outside)))

    if segs:
        best = max(segs, key=lambda s: s[1] - s[0])
        s, e = best
        ax.plot(lx[s:e-5], ly[s:e-5], color=color, linewidth=2.2, zorder=5,
                solid_capstyle='round')
        ax.annotate('', xy=(lx[e-5], ly[e-5]), xytext=(lx[e-10], ly[e-10]),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=2.2,
                                    mutation_scale=16), zorder=5)

    if side == 'left':
        ax.text(arc_cx - lr - 0.55, arc_cy + 0.25, label, fontsize=12,
                ha='center', va='center', color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor=HIDDEN_BG, edgecolor='none'))
    else:
        ax.text(arc_cx + lr + 0.55, arc_cy + 0.25, label, fontsize=12,
                ha='center', va='center', color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor=HIDDEN_BG, edgecolor='none'))

draw_self_loop(ax, calm_xy, R, 'left', CALM_BLUE, r'$P_{00}$')
draw_self_loop(ax, panic_xy, R, 'right', PANIC_RED, r'$P_{11}$')

# Constraints beside the chain
ax.text(-1.2, -0.1, r'$P_{00} + P_{01} = 1$', fontsize=10,
        ha='center', va='center', color=GREY, fontstyle='italic')
ax.text(11.2, -0.1, r'$P_{10} + P_{11} = 1$', fontsize=10,
        ha='center', va='center', color=GREY, fontstyle='italic')

# ── "Observed" layer: emission box with feature names ──
obs_y = -3.5
obs_box = FancyBboxPatch((0.3, obs_y - 0.7), 9.4, 1.4,
                           boxstyle='round,pad=0.2', facecolor=OBS_FILL,
                           edgecolor=OBS_GREEN, linewidth=2, zorder=3)
ax.add_patch(obs_box)
ax.text(5, obs_y + 0.3, r'Observed: $\mathbf{z}_t$', fontsize=12, fontweight='bold',
        ha='center', va='center', color=OBS_GREEN, zorder=4)
ax.text(5, obs_y - 0.25,
        'Drawdown     Volatility     Dispersion     Participation',
        fontsize=10.5, ha='center', va='center', color=TEXT_DARK, zorder=4)

# Emission arrows from states down to observation box
emit_kw = dict(arrowstyle='-|>', mutation_scale=14, linewidth=1.8,
               color=OBS_GREEN, linestyle='--', zorder=2)
# From calm - start just below circle, end just above green box
ax.add_patch(FancyArrowPatch(
    posA=(calm_xy[0], calm_xy[1] - R - 0.1),
    posB=(3.5, obs_y + 0.85),
    connectionstyle='arc3,rad=0.15', **emit_kw))
# From panic
ax.add_patch(FancyArrowPatch(
    posA=(panic_xy[0], panic_xy[1] - R - 0.1),
    posB=(6.5, obs_y + 0.85),
    connectionstyle='arc3,rad=-0.15', **emit_kw))

ax.text(1.8, -1.5, 'emit', fontsize=9, fontstyle='italic', color=OBS_GREEN,
        ha='center', va='center', rotation=40)
ax.text(8.2, -1.5, 'emit', fontsize=9, fontstyle='italic', color=OBS_GREEN,
        ha='center', va='center', rotation=-40)

# ── Output: filtered probability ──
out_y = -6.2
out_box = FancyBboxPatch((2.5, out_y - 0.55), 5, 1.1,
                           boxstyle='round,pad=0.2', facecolor=OUTPUT_FILL,
                           edgecolor=OUTPUT_PURPLE, linewidth=2, zorder=3)
ax.add_patch(out_box)
ax.text(5, out_y, r'Output: $\pi_t^{\,\mathrm{filter}} = P(s_t = 1 \mid \mathbf{z}_{1:t})$',
        fontsize=11, fontweight='bold',
        ha='center', va='center', color=OUTPUT_PURPLE, zorder=4)

# Arrow from observation to output - gap before both boxes
ax.add_patch(FancyArrowPatch(
    posA=(5, obs_y - 0.85),
    posB=(5, out_y + 0.7),
    arrowstyle='-|>', mutation_scale=14, linewidth=1.8,
    color=OUTPUT_PURPLE, zorder=2))
ax.text(5.5, -4.85, 'filter', fontsize=11, fontstyle='italic', color=OUTPUT_PURPLE,
        ha='left', va='center')


# ═══════════════════════════════════════════════════════════════
# Panel B: Example state sequence
# ═══════════════════════════════════════════════════════════════
ax2 = axes[1]
ax2.set_xlim(-0.3, 17.2)
ax2.set_ylim(-1.8, 2.5)
ax2.axis('off')

ax2.text(8.35, 2.2, 'Example: Hidden State Sequence', fontsize=15, fontweight='bold',
         ha='center', va='center', color=TEXT_DARK)

states = ['C','C','C','C','C','P','P','P','P','C','C','C','P','P','C','C']
bw = 0.95
bh = 0.72
gap = 0.08
box_y = 0.8

for i, s in enumerate(states):
    x = i * (bw + gap) + 0.5
    if s == 'C':
        fc, ec, tc = CALM_FILL, CALM_BLUE, CALM_BLUE
    else:
        fc, ec, tc = PANIC_FILL, PANIC_RED, PANIC_RED

    box = FancyBboxPatch((x - bw/2, box_y - bh/2), bw, bh,
                          boxstyle='round,pad=0.05', facecolor=fc,
                          edgecolor=ec, linewidth=1.5, zorder=3)
    ax2.add_patch(box)
    ax2.text(x, box_y, s, fontsize=12, fontweight='bold', ha='center', va='center',
             color=tc, zorder=4)
    ax2.text(x, box_y - bh/2 - 0.13, f'$t_{{{i+1}}}$', fontsize=7.5, ha='center',
             va='top', color=GREY)

# ── Uniform braces ──
brace_y = box_y - bh/2 - 0.55
brace_depth = 0.22

def regime_brace(ax, i_start, i_end, label, color):
    x_left = i_start * (bw + gap) + 0.5 - bw/2 + 0.03
    x_right = i_end * (bw + gap) + 0.5 + bw/2 - 0.03
    mid = (x_left + x_right) / 2
    n = 60
    t = np.linspace(0, 1, n)
    xl = x_left + t * (mid - x_left)
    yl = brace_y - brace_depth * (t ** 1.5)
    xr = x_right - t * (x_right - mid)
    yr = brace_y - brace_depth * (t ** 1.5)
    ax.plot(xl, yl, color=color, lw=1.3, solid_capstyle='round')
    ax.plot(xr, yr, color=color, lw=1.3, solid_capstyle='round')
    tick = 0.08
    ax.plot([x_left, x_left], [brace_y + tick, brace_y], color=color, lw=1.3)
    ax.plot([x_right, x_right], [brace_y + tick, brace_y], color=color, lw=1.3)
    ax.plot([mid, mid], [brace_y - brace_depth, brace_y - brace_depth - tick],
            color=color, lw=1.3)
    ax.text(mid, brace_y - brace_depth - 0.12, label, fontsize=9, ha='center',
            va='top', color=color, fontweight='bold')

regime_brace(ax2, 0, 4, 'Calm', CALM_BLUE)
regime_brace(ax2, 5, 8, 'Panic', PANIC_RED)
regime_brace(ax2, 9, 11, 'Calm', CALM_BLUE)
regime_brace(ax2, 12, 13, 'Panic', PANIC_RED)
regime_brace(ax2, 14, 15, 'Calm', CALM_BLUE)

# ── Transition markers ──
def transition_marker(ax, i, label, color):
    x = i * (bw + gap) + 0.5 + (bw + gap) / 2
    y_bot = box_y + bh/2 + 0.08
    y_top = y_bot + 0.40
    ax.annotate('', xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.3))
    ax.text(x, y_top + 0.06, label, fontsize=8, ha='center', va='bottom',
            color=color, fontweight='bold')

transition_marker(ax2, 4, r'$P_{01}$', PANIC_RED)
transition_marker(ax2, 8, r'$P_{10}$', CALM_BLUE)
transition_marker(ax2, 11, r'$P_{01}$', PANIC_RED)
transition_marker(ax2, 13, r'$P_{10}$', CALM_BLUE)

plt.savefig('markov_chain_diagram.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.12)
plt.savefig('markov_chain_diagram.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.12)
plt.close()
print('Saved markov_chain_diagram.png and .pdf')
