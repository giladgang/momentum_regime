"""
Generate a professional two-state Markov chain diagram with example state sequence.
Output: markov_chain_diagram.png (and .pdf)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ── Colours ──
CALM_BLUE = '#1B4F8A'
PANIC_RED = '#9B2226'
CALM_FILL = '#DAE5F5'
PANIC_FILL = '#F2D4D4'
TEXT_DARK = '#1A1A1A'
GREY = '#666666'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                          gridspec_kw={'height_ratios': [3.0, 2.0], 'hspace': 0.02})

# ═══════════════════════════════════════════════════════════════
# Panel A: Markov chain diagram
# ═══════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_xlim(-4.5, 14.5)
ax.set_ylim(-2.5, 4.5)
ax.set_aspect('equal')
ax.axis('off')

ax.text(5, 4.2, 'Two-State Hidden Markov Model', fontsize=18, fontweight='bold',
        ha='center', va='center', color=TEXT_DARK)

# ── State circles ──
calm_xy = np.array([2.0, 0.8])
panic_xy = np.array([8.0, 0.8])
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

# ── Curved transition arrows between states ──
akw = dict(arrowstyle='-|>', mutation_scale=18, linewidth=2.2, zorder=5)

# Calm -> Panic (lower path)
a1 = -35
start_cp = calm_xy + R * np.array([np.cos(np.radians(a1)), np.sin(np.radians(a1))])
end_cp = panic_xy + R * np.array([np.cos(np.radians(180 + 35)), np.sin(np.radians(180 + 35))])
arrow_cp = FancyArrowPatch(posA=tuple(start_cp), posB=tuple(end_cp),
                            connectionstyle='arc3,rad=0.25', color=PANIC_RED, **akw)
ax.add_patch(arrow_cp)
ax.text(5, -1.3, r'$P_{01}$', fontsize=13, ha='center', va='center',
        color=PANIC_RED, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none'))

# Panic -> Calm (upper path)
start_pc = panic_xy + R * np.array([np.cos(np.radians(145)), np.sin(np.radians(145))])
end_pc = calm_xy + R * np.array([np.cos(np.radians(35)), np.sin(np.radians(35))])
arrow_pc = FancyArrowPatch(posA=tuple(start_pc), posB=tuple(end_pc),
                            connectionstyle='arc3,rad=0.25', color=CALM_BLUE, **akw)
ax.add_patch(arrow_pc)
ax.text(5, 2.9, r'$P_{10}$', fontsize=13, ha='center', va='center',
        color=CALM_BLUE, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none'))

# ── Self-loops (clean arcs, labels well separated) ──

def draw_self_loop(ax, center, radius, side, color, label):
    """Draw a clean self-loop that clearly starts/ends on the circle."""
    lr = 0.65  # loop radius
    if side == 'left':
        # Position arc center above-left of circle
        arc_cx = center[0] - radius * 0.75
        arc_cy = center[1] + radius + lr * 0.3
        # Start angle (on circle, bottom-ish) to end angle (on circle, left-ish)
        angles = np.linspace(np.radians(-30), np.radians(250), 100)
    else:
        arc_cx = center[0] + radius * 0.75
        arc_cy = center[1] + radius + lr * 0.3
        angles = np.linspace(np.radians(210), np.radians(-70), 100)

    lx = arc_cx + lr * np.cos(angles)
    ly = arc_cy + lr * np.sin(angles)

    # Clip to points outside the circle
    dist = np.sqrt((lx - center[0])**2 + (ly - center[1])**2)
    outside = dist >= radius * 0.97

    # Find longest contiguous outside segment
    segs = []
    in_seg = False
    for idx in range(len(outside)):
        if outside[idx] and not in_seg:
            start = idx
            in_seg = True
        elif not outside[idx] and in_seg:
            segs.append((start, idx))
            in_seg = False
    if in_seg:
        segs.append((start, len(outside)))

    if segs:
        best = max(segs, key=lambda s: s[1] - s[0])
        s, e = best
        # Draw arc
        ax.plot(lx[s:e-5], ly[s:e-5], color=color, linewidth=2.2, zorder=5,
                solid_capstyle='round')
        # Arrowhead
        ax.annotate('', xy=(lx[e-5], ly[e-5]),
                    xytext=(lx[e-10], ly[e-10]),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=2.2,
                                    mutation_scale=16), zorder=5)

    # Label well away from arrow
    if side == 'left':
        ax.text(arc_cx - lr - 0.55, arc_cy + 0.25, label, fontsize=12,
                ha='center', va='center', color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none'))
    else:
        ax.text(arc_cx + lr + 0.55, arc_cy + 0.25, label, fontsize=12,
                ha='center', va='center', color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none'))

draw_self_loop(ax, calm_xy, R, 'left', CALM_BLUE, r'$P_{00}$')
draw_self_loop(ax, panic_xy, R, 'right', PANIC_RED, r'$P_{11}$')

# Constraints -- positioned to the sides of the diagram
ax.text(-1.2, -0.5, r'$P_{00} + P_{01} = 1$', fontsize=10,
        ha='center', va='center', color=GREY, fontstyle='italic')
ax.text(11.2, -0.5, r'$P_{10} + P_{11} = 1$', fontsize=10,
        ha='center', va='center', color=GREY, fontstyle='italic')


# ═══════════════════════════════════════════════════════════════
# Panel B: Example state sequence (pushed up, compact)
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

    # Left half curve
    xl = x_left + t * (mid - x_left)
    yl = brace_y - brace_depth * (t ** 1.5)
    # Right half curve
    xr = x_right - t * (x_right - mid)
    yr = brace_y - brace_depth * (t ** 1.5)

    ax.plot(xl, yl, color=color, lw=1.3, solid_capstyle='round')
    ax.plot(xr, yr, color=color, lw=1.3, solid_capstyle='round')
    # End ticks
    tick = 0.08
    ax.plot([x_left, x_left], [brace_y + tick, brace_y], color=color, lw=1.3)
    ax.plot([x_right, x_right], [brace_y + tick, brace_y], color=color, lw=1.3)
    # Center tick
    ax.plot([mid, mid], [brace_y - brace_depth, brace_y - brace_depth - tick],
            color=color, lw=1.3)
    # Label
    ax.text(mid, brace_y - brace_depth - 0.12, label, fontsize=9, ha='center',
            va='top', color=color, fontweight='bold')

regime_brace(ax2, 0, 4, 'Calm regime', CALM_BLUE)
regime_brace(ax2, 5, 8, 'Panic regime', PANIC_RED)
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
