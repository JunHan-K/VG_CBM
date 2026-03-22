"""
plot_layer_comparison.py
------------------------
Generate figures for the layer2/3/4 selection analysis.
Results from layer_scan.py (02127.py) on CIFAR-100, ResNet50 frozen,
expansion=8, l1=2.37e-2.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'plots')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
layers   = ['layer2\n(28×28)', 'layer3\n(14×14)', 'layer4\n(7×7)']
baseline = [32.59, 61.59, 72.26]
sae_z    = [52.81, 75.02, 74.88]
delta    = [s - b for s, b in zip(sae_z, baseline)]

x = np.arange(len(layers))
highlight = 1          # layer3 index

BLUE   = '#4C8BE2'
ORANGE = '#F28B30'
GREEN  = '#4CAF7D'
GRAY   = '#AAAAAA'
RED    = '#E05C5C'

# ── Figure 1: Baseline vs SAE-Z accuracy (grouped bar) ───────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
w = 0.35
bars_b = ax.bar(x - w/2, baseline, w, label='Baseline (linear probe)',
                color=[BLUE if i != highlight else '#2860C0' for i in range(3)],
                edgecolor='white', linewidth=0.8, alpha=0.85)
bars_s = ax.bar(x + w/2, sae_z,    w, label='SAE-Z (after SAE)',
                color=[ORANGE if i != highlight else '#C86010' for i in range(3)],
                edgecolor='white', linewidth=0.8, alpha=0.85)

# value labels
for bar in list(bars_b) + list(bars_s):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.6, f'{h:.1f}%',
            ha='center', va='bottom', fontsize=9)

# highlight box around layer3
ax.axvspan(highlight - 0.45, highlight + 0.45, alpha=0.07,
           color='gold', zorder=0)
ax.text(highlight, 78.5, '✓ selected', ha='center', fontsize=9,
        color='#8B6914', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(layers, fontsize=10)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('Linear Probe Accuracy: Baseline vs SAE-Z\n'
             '(ResNet50 frozen, CIFAR-100, expansion=8)', fontsize=11)
ax.set_ylim(0, 85)
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'layer_baseline_vs_sae.png'), dpi=150)
plt.close()
print("Saved: layer_baseline_vs_sae.png")

# ── Figure 2: SAE boost (delta) bar chart ────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
colors = [GREEN if i == highlight else GRAY for i in range(3)]
bars = ax.bar(x, delta, color=colors, edgecolor='white', linewidth=0.8,
              width=0.5, alpha=0.9)

for bar, d in zip(bars, delta):
    ax.text(bar.get_x() + bar.get_width()/2, d + 0.3, f'+{d:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.axvspan(highlight - 0.35, highlight + 0.35, alpha=0.07,
           color='gold', zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(layers, fontsize=10)
ax.set_ylabel('SAE Accuracy Boost (Δ%)', fontsize=11)
ax.set_title('SAE Gain per Layer\n'
             '(SAE-Z acc − Baseline acc)', fontsize=11)
ax.set_ylim(0, 26)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'layer_sae_boost.png'), dpi=150)
plt.close()
print("Saved: layer_sae_boost.png")

# ── Figure 3: Combined summary (2 subplots side by side) ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: grouped bar
ax = axes[0]
bars_b = ax.bar(x - w/2, baseline, w, label='Baseline',
                color=[BLUE if i != highlight else '#2860C0' for i in range(3)],
                edgecolor='white', linewidth=0.8, alpha=0.85)
bars_s = ax.bar(x + w/2, sae_z,    w, label='SAE-Z',
                color=[ORANGE if i != highlight else '#C86010' for i in range(3)],
                edgecolor='white', linewidth=0.8, alpha=0.85)
for bar in list(bars_b) + list(bars_s):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.6, f'{h:.1f}',
            ha='center', va='bottom', fontsize=8)
ax.axvspan(highlight - 0.45, highlight + 0.45, alpha=0.07, color='gold', zorder=0)
ax.set_xticks(x); ax.set_xticklabels(layers, fontsize=9)
ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 85)
ax.set_title('Baseline vs SAE-Z Accuracy', fontsize=10)
ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)

# Right: delta bar
ax = axes[1]
bars = ax.bar(x, delta, color=colors, edgecolor='white', linewidth=0.8,
              width=0.5, alpha=0.9)
for bar, d in zip(bars, delta):
    ax.text(bar.get_x() + bar.get_width()/2, d + 0.3, f'+{d:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axvspan(highlight - 0.35, highlight + 0.35, alpha=0.07, color='gold', zorder=0)
ax.set_xticks(x); ax.set_xticklabels(layers, fontsize=9)
ax.set_ylabel('SAE Boost (Δ%)'); ax.set_ylim(0, 26)
ax.set_title('SAE Accuracy Gain per Layer', fontsize=10)
ax.spines[['top','right']].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)

patch_g = mpatches.Patch(color=GREEN, label='layer3 (selected)', alpha=0.9)
patch_gray = mpatches.Patch(color=GRAY, label='others', alpha=0.9)
axes[1].legend(handles=[patch_g, patch_gray], fontsize=8)

fig.suptitle('ResNet50 Layer Selection  |  CIFAR-100  |  Frozen Backbone',
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'layer_selection_summary.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: layer_selection_summary.png")

print(f"\nAll plots saved to: {OUT_DIR}")
