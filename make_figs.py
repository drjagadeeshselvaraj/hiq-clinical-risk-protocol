"""Generate article figures as 300-dpi PNGs."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path('/tmp/claude-0/-home-claude/240e3105-e79d-52b9-becb-b336b6aa69c5/scratchpad/figs')
OUT.mkdir(exist_ok=True)

# ── Palette (accessible, same as dashboard) ──────────────────────
C = {
    'baseline': '#2a78d6',   # blue
    'l2lr':     '#d94c1a',   # red-orange
    'gbm':      '#0e9e63',   # green
    'nn':       '#d08b00',   # amber
    'grid':     '#e9e7f0',
    'axis':     '#ccc9d9',
    'text':     '#100a34',
    'text2':    '#534f74',
    'good':     '#0a7c2b',
    'bad':      '#b80020',
}
MODELS = ['Clinical\nBaseline', 'L2-LR', 'LightGBM', 'Neural Net']
COLS   = [C['baseline'], C['l2lr'], C['gbm'], C['nn']]

# ── Figure 1: HIQ* all-hospital grouped bar + CI ─────────────────
# (Document order: Figure 1 — p26)
def fig1():
    hiq  = [84.8,  86.3, 113.2,  10.1]
    lo   = [79.6,  79.6, 107.6,  -3.3]
    hi   = [89.9,  93.0, 118.8,  23.4]
    err_lo = [h - l for h, l in zip(hiq, lo)]
    err_hi = [h - l for h, l in zip(hi, hiq)]

    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=300)
    xs = np.arange(len(MODELS))
    bars = ax.bar(xs, hiq, color=COLS, width=0.55, zorder=3,
                  edgecolor='white', linewidth=0.8)
    ax.errorbar(xs, hiq, yerr=[err_lo, err_hi],
                fmt='none', color='#333', linewidth=1.4,
                capsize=5, capthick=1.4, zorder=4)

    # 100 reference line
    ax.axhline(100, color=C['text2'], linewidth=1, linestyle='--', zorder=2, alpha=0.6)
    ax.text(3.42, 101.5, '100 (panel mean)', fontsize=8, color=C['text2'], ha='right')

    # value labels
    for x, v, h in zip(xs, hiq, hi):
        ax.text(x, h + 1.5, f'{v:.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='600', color=C['text'])

    ax.set_xticks(xs)
    ax.set_xticklabels(MODELS, fontsize=9)
    ax.set_ylabel('HIQ*', fontsize=10)
    ax.set_ylim(-10, 130)
    ax.set_yticks(range(0, 131, 20))
    ax.yaxis.grid(True, color=C['grid'], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(C['axis'])
    ax.tick_params(colors=C['text2'])
    plt.tight_layout()
    fig.savefig(OUT / 'fig1_hiq_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('fig1 done')

# ── Figure 3 (document order): Naive BSS vs BSS* per hospital (LightGBM) ──────────
# (Document order: Figure 3 — p28; PNG saved as fig2_rank_reversal.png)
def fig2():
    hosps  = ['Academic', 'Community', 'Teaching', 'Regional']
    naive  = [0.1465, 0.1427, 0.1409, 0.1172]
    bstar  = [0.1294, 0.1485, 0.1217, 0.1495]

    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=300)
    x = np.arange(len(hosps))
    w = 0.33
    b1 = ax.bar(x - w/2, naive, width=w, color=C['baseline'], label='Naive BSS',
                zorder=3, edgecolor='white')
    b2 = ax.bar(x + w/2, bstar, width=w, color=C['gbm'], label='BSS* (corrected)',
                zorder=3, edgecolor='white')

    # Delta arrows for biggest reversals
    for i, (n, s) in enumerate(zip(naive, bstar)):
        d = s - n
        col = C['good'] if d > 0 else C['bad']
        ax.annotate('', xy=(x[i] + w/2, s + 0.001), xytext=(x[i] + w/2, n + 0.001),
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.5))
        label = f'+{d:.3f}' if d >= 0 else f'{d:.3f}'
        ax.text(x[i] + w/2 + 0.18, (n + s)/2, label, fontsize=8,
                color=col, va='center', fontweight='600')

    ax.set_xticks(x)
    ax.set_xticklabels(hosps, fontsize=9)
    ax.set_ylabel('Brier Skill Score', fontsize=10)
    ax.set_ylim(0, 0.185)
    ax.yaxis.grid(True, color=C['grid'], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(C['axis'])
    ax.tick_params(colors=C['text2'])
    ax.legend(fontsize=8, framealpha=0.8, loc='upper right')
    plt.tight_layout()
    fig.savefig(OUT / 'fig2_rank_reversal.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('fig2 done')

# ── Figure 2 (document order): Forest plot ─────────────────────────────────────────
# (Document order: Figure 2 — p27; PNG saved as fig3_forest.png)
def fig3():
    rows = [
        ('LightGBM',         113.2, 107.6, 118.8, C['gbm']),
        ('L2-LR',             86.3,  79.6,  93.0, C['l2lr']),
        ('Clinical Baseline', 84.8,  79.6,  89.9, C['baseline']),
        ('Neural Net',        10.1,  -3.3,  23.4, C['nn']),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 3.5), dpi=300)
    ys = np.arange(len(rows))

    for y, (name, val, lo, hi, col) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=col, linewidth=2.5, solid_capstyle='round', zorder=3)
        ax.plot([lo, lo], [y - 0.15, y + 0.15], color=col, linewidth=2, zorder=3)
        ax.plot([hi, hi], [y - 0.15, y + 0.15], color=col, linewidth=2, zorder=3)
        ax.scatter([val], [y], color=col, s=60, zorder=4,
                   edgecolors='white', linewidths=1.5)
        ax.text(hi + 1.5, y, f'{val:.1f}  [{lo:.1f}, {hi:.1f}]',
                va='center', fontsize=8.5, color=C['text'])

    ax.axvline(100, color=C['text2'], linewidth=1, linestyle='--', alpha=0.6, zorder=2)
    ax.text(100.5, -0.6, '100', fontsize=8, color=C['text2'])

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlabel('HIQ*', fontsize=10)
    ax.set_xlim(-15, 145)
    ax.xaxis.grid(True, color=C['grid'], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(C['axis'])
    ax.tick_params(colors=C['text2'])
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(OUT / 'fig3_forest.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('fig3 done')

# ── Figure 4 (document order): HIQ* vs Naive BSS scorecard ───────────────────────
# (Document order: Figure 4 — p53; PNG saved as figA1_scorecard.png)
def figA1():
    dims   = ['Estimator\nvalidity', 'Statistical\nrigour',
              'Patient-mix\ncorrection', 'Censoring\nhandling', 'Interpretive\nutility']
    hiq_sc = [5, 5, 4, 4, 4]
    bss_sc = [2, 2, 1, 1, 2]

    x = np.arange(len(dims))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=300)
    ax.bar(x - w/2, hiq_sc, width=w, color=C['gbm'],      label='HIQ*',      zorder=3, edgecolor='white')
    ax.bar(x + w/2, bss_sc, width=w, color=C['baseline'], label='Naive BSS', zorder=3, edgecolor='white')

    for xi, (h, b) in enumerate(zip(hiq_sc, bss_sc)):
        ax.text(xi - w/2, h + 0.08, str(h), ha='center', va='bottom', fontsize=9,
                fontweight='700', color=C['text'])
        ax.text(xi + w/2, b + 0.08, str(b), ha='center', va='bottom', fontsize=9,
                fontweight='700', color=C['text'])

    ax.set_xticks(x)
    ax.set_xticklabels(dims, fontsize=8.5)
    ax.set_yticks(range(0, 6))
    ax.set_ylabel('Score (1–5)', fontsize=10)
    ax.set_ylim(0, 6)
    ax.yaxis.grid(True, color=C['grid'], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(C['axis'])
    ax.tick_params(colors=C['text2'])
    ax.legend(fontsize=9, framealpha=0.8)

    # overall scores
    ax.text(0.98, 0.95, 'HIQ* overall: 4.4/5', transform=ax.transAxes,
            fontsize=9, ha='right', va='top', color=C['gbm'], fontweight='700')
    ax.text(0.98, 0.88, 'Naive BSS overall: 1.6/5', transform=ax.transAxes,
            fontsize=9, ha='right', va='top', color=C['baseline'], fontweight='700')

    plt.tight_layout()
    fig.savefig(OUT / 'figA1_scorecard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('figA1 done')

fig1()
fig2()
fig3()
figA1()
print('All figures written to', OUT)
