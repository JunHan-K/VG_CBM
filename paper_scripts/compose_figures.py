#!/usr/bin/env python3
"""
compose_figures.py — Paper figures A-E from visual strip outputs
=================================================================
Reads visual strip images from {run_dir}/visuals/final/ and composes
figures A (SLIC), B (deviation map), C (FG mask), D (masking effect),
and E (concept activation maps).

Visual strip structure (output of visualize_batch):
  [original | concept_0 | concept_1 | ... | concept_n | combined]
  Each panel: cell=224, pad=8, text label 20px below
  Total strip height: 244px

Usage:
    python paper_scripts/compose_figures.py --run_dir ./models/flowers --tag _flowers
    python paper_scripts/compose_figures.py --run_dir ./models/car     --tag _car
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from src.backbone   import build_backbone
from src.models     import FeatureNorm, SparseSAE, CBMHead, fg_z_pool
from src.slic_utils import compute_slic_224, extract_sp_features, slic224_to_14, compute_sp_fg

OUT_DIR = Path('./figures')
OUT_DIR.mkdir(exist_ok=True)

CELL = 224
PAD  = 8
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

CONCEPT_COLORS = [
    (220,  50,  50), ( 50, 180,  80), ( 50, 120, 220), (220, 160,   0),
    (160,  50, 220), (  0, 190, 190), (220, 100, 160), (100, 180,  50),
]


# ── Strip parsing ─────────────────────────────────────────────────────────────
def parse_filename(fname):
    """img00_y3_p3.png → (3, 3)"""
    try:
        parts = fname.replace('.png', '').split('_')
        y = int([p for p in parts if p.startswith('y')][0][1:])
        p = int([p for p in parts if p.startswith('p')][0][1:])
        return y, p
    except Exception:
        return None, None


def load_strips(visuals_dir, only_correct=True):
    """Load visual strip images, optionally filter for correctly predicted."""
    files = sorted(f for f in os.listdir(visuals_dir) if f.endswith('.png'))
    result = []
    for f in files:
        y, p = parse_filename(f)
        if only_correct and y != p:
            continue
        result.append((os.path.join(visuals_dir, f), y, p))
    # If not enough correct, include all
    if len(result) < 2:
        result = [(os.path.join(visuals_dir, f), *parse_filename(f))
                  for f in files]
    return result


def extract_panels(strip_path):
    """
    From a strip image, extract individual 224x224 panels.
    Returns list: [original, concept_0, concept_1, ..., combined]
    """
    img = Image.open(strip_path).convert('RGB')
    panels = []
    x = 0
    while x + CELL <= img.width:
        panel = img.crop((x, 0, x + CELL, CELL))
        panels.append(panel)
        x += CELL + PAD
    return panels  # [orig, c0, c1, ..., combined]


def draw_slic_boundary(img_pil, slic_224, color=(255, 220, 0)):
    seg = np.array(slic_224)
    img = np.array(img_pil.resize((224, 224))).copy()
    boundary = np.zeros(seg.shape, dtype=bool)
    boundary[:-1, :] |= (seg[:-1, :] != seg[1:,  :])
    boundary[:, :-1] |= (seg[:,  :-1] != seg[:,   1:])
    img[boundary] = color
    return Image.fromarray(img)


def add_colorbar(fig, ax, vmin, vmax, cmap='hot', label=''):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=label)


# ── Config loading ────────────────────────────────────────────────────────────
def load_cfg(run_dir):
    from dataclasses import dataclass, field
    from typing import List

    @dataclass
    class CFG:
        dataset: str = 'flowers102'; data_root: str = './data'
        run_dir: str = './runs'; backbone: str = 'resnet50'
        batch_size: int = 64; num_workers: int = 0
        train_limit: int = 0; test_limit: int = 0
        img_size: int = 224; seed: int = 42
        d_in: int = 1024; expansion: int = 2; num_classes: int = 102
        n_segments: int = 80; compactness: float = 10.0
        slic_cache: str = './slic_cache'; fg_ratio: float = 0.30
        sae_epochs: int = 20; sae_lr: float = 3e-4; stage2_start: int = 10
        topk: int = 20; w_recon: float = 1.0; w_l1: float = 0.05
        w_aux: float = 0.003; w_sp_orth: float = 0.20; w_div: float = 0.02
        head_epochs: int = 30; head_lr: float = 1e-3; head_l1: float = 0.06
        load_sae_from: str = ''; eval_only: bool = False
        vis_coverage: float = 0.80; vis_max_concepts: int = 8
        gallery_n_concepts: int = 30; gallery_n_images: int = 8
        intervention_ms: List[int] = field(default_factory=lambda: [10, 25, 50])

    cfg = CFG()
    cfg_path = os.path.join(run_dir, 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            saved = json.load(f)
        for k, v in saved.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    cfg.run_dir = run_dir
    return cfg


def load_model(cfg, device):
    backbone, hook, d_in = build_backbone(device, cfg.backbone)
    cfg.d_in = d_in
    K = cfg.d_in * cfg.expansion
    feat_norm = FeatureNorm(cfg.d_in).to(device)
    sae  = SparseSAE(cfg.d_in, K, topk=cfg.topk).to(device)
    head = CBMHead(K, cfg.num_classes).to(device)

    ckpt = torch.load(os.path.join(cfg.run_dir, 'sae_checkpoint.pt'),
                      map_location=device, weights_only=False)
    if 'feat_norm' in ckpt:
        feat_norm.load_state_dict(ckpt['feat_norm'], strict=False)
    sae.load_state_dict(ckpt['sae'])
    hckpt_path = os.path.join(cfg.run_dir, 'head_best.pt')
    if os.path.exists(hckpt_path):
        hckpt = torch.load(hckpt_path, map_location=device, weights_only=False)
        head.load_state_dict(hckpt['head'])

    backbone.eval(); feat_norm.eval(); sae.eval(); head.eval()
    return backbone, hook, feat_norm, sae, head


@torch.no_grad()
def infer(backbone, hook, feat_norm, sae, head, cfg, img_pil, device):
    val_tf = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = val_tf(img_pil).unsqueeze(0).to(device)
    backbone(x)
    f = hook.out
    f = feat_norm(f.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    slic_224 = compute_slic_224(img_pil, cfg.n_segments, cfg.compactness)
    slic_t   = torch.from_numpy(slic_224.astype(np.int64)).unsqueeze(0)
    slic_14  = slic224_to_14(slic_t.to(device))

    sp_feat_raw = extract_sp_features(f, slic_14, cfg.n_segments)
    fg          = compute_sp_fg(sp_feat_raw, cfg.fg_ratio)
    sp_feat_fg  = sp_feat_raw * fg.unsqueeze(-1)
    z, _        = sae(sp_feat_fg)
    z_pool      = fg_z_pool(z, fg)
    logits      = head(z_pool)
    pred        = int(logits.argmax(1).item())

    return (slic_224,
            sp_feat_raw.squeeze(0),
            fg.squeeze(0),
            z.squeeze(0),
            z_pool.squeeze(0),
            pred)


# ════════════════════════════════════════════════════════════════════════════
# Figure A — SLIC superpixel  (original + SLIC boundary overlay)
# ════════════════════════════════════════════════════════════════════════════
def fig_A(strip_data, cfg, tag):
    n = len(strip_data)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 6.5))
    if n == 1: axes = [[axes[0]], [axes[1]]]

    for col, (img_pil, slic_224) in enumerate(strip_data):
        slic_img = draw_slic_boundary(img_pil, slic_224)
        axes[0][col].imshow(img_pil); axes[0][col].axis('off')
        if col == 0: axes[0][col].set_title('Input Image', fontsize=10)
        axes[1][col].imshow(slic_img); axes[1][col].axis('off')
        if col == 0: axes[1][col].set_title(f'SLIC (N={cfg.n_segments})', fontsize=10)

    fig.suptitle('A. SLIC Superpixel Segmentation', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = OUT_DIR / f'fig_A_slic{tag}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [A] → {path}')


# ════════════════════════════════════════════════════════════════════════════
# Figure B — Deviation score heatmap
# ════════════════════════════════════════════════════════════════════════════
def fig_B(strip_data_feat, cfg, tag):
    n = len(strip_data_feat)
    fig, axes = plt.subplots(3, n, figsize=(3.0 * n, 9.5))
    if n == 1: axes = [[axes[0]], [axes[1]], [axes[2]]]

    for col, (img_pil, slic_224, sp_feat_raw) in enumerate(strip_data_feat):
        gm  = sp_feat_raw.mean(0)
        dev = (sp_feat_raw - gm).norm(dim=-1).cpu().numpy()
        dev_n = (dev - dev.min()) / (dev.max() - dev.min() + 1e-8)
        seg = slic_224.clip(0, cfg.n_segments - 1)
        dev_map = dev_n[seg]

        cmap_fn = plt.get_cmap('hot')
        dev_col = (cmap_fn(dev_map)[:, :, :3] * 255).astype(np.uint8)
        overlay = (np.array(img_pil).astype(np.float32) * 0.45 +
                   dev_col.astype(np.float32) * 0.55).clip(0, 255).astype(np.uint8)

        axes[0][col].imshow(img_pil); axes[0][col].axis('off')
        if col == 0: axes[0][col].set_title('Input', fontsize=10)

        im = axes[1][col].imshow(dev_map, cmap='hot', vmin=0, vmax=1)
        axes[1][col].axis('off')
        if col == 0: axes[1][col].set_title('||sᵢ − mean(s)||', fontsize=10)
        if col == n - 1: add_colorbar(fig, axes[1][col], 0, 1, 'hot', 'score')

        axes[2][col].imshow(overlay); axes[2][col].axis('off')
        if col == 0: axes[2][col].set_title('Score Overlay', fontsize=10)

    fig.suptitle('B. Superpixel Deviation Score', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = OUT_DIR / f'fig_B_deviation{tag}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [B] → {path}')


# ════════════════════════════════════════════════════════════════════════════
# Figure C — Top-30% foreground selection mask
# ════════════════════════════════════════════════════════════════════════════
def fig_C(strip_data_fg, cfg, tag):
    n = len(strip_data_fg)
    fig, axes = plt.subplots(3, n, figsize=(3.0 * n, 9.5))
    if n == 1: axes = [[axes[0]], [axes[1]], [axes[2]]]

    for col, (img_pil, slic_224, fg_mask) in enumerate(strip_data_fg):
        seg    = slic_224.clip(0, cfg.n_segments - 1)
        fg_np  = fg_mask.cpu().numpy()
        fg_pxl = fg_np[seg]
        img_np = np.array(img_pil).astype(np.float32)
        gray   = img_np.mean(2, keepdims=True) * np.ones((1, 1, 3))
        alpha  = fg_pxl[:, :, None]
        green  = np.array([80, 220, 100], dtype=np.float32)
        tint   = (img_np * (1 - alpha * 0.35) +
                  green[None, None, :] * alpha * 0.35).clip(0, 255).astype(np.uint8)

        axes[0][col].imshow(img_pil); axes[0][col].axis('off')
        if col == 0: axes[0][col].set_title('Input', fontsize=10)

        axes[1][col].imshow((fg_pxl * 255).astype(np.uint8), cmap='gray', vmin=0, vmax=255)
        axes[1][col].axis('off')
        if col == 0: axes[1][col].set_title(f'FG Mask (top {int(cfg.fg_ratio*100)}%)', fontsize=10)

        axes[2][col].imshow(tint); axes[2][col].axis('off')
        axes[2][col].set_xlabel(f'{int(fg_np.sum())}/{cfg.n_segments} sp', fontsize=8)
        if col == 0: axes[2][col].set_title('Selected FG Overlay', fontsize=10)

    fig.suptitle(f'C. Top-{int(cfg.fg_ratio*100)}% Foreground Selection', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = OUT_DIR / f'fig_C_fg_mask{tag}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [C] → {path}')


# ════════════════════════════════════════════════════════════════════════════
# Figure D — Before / after FG masking (PCA feature tiles)
# ════════════════════════════════════════════════════════════════════════════
def fig_D(strip_data_mask, cfg, tag):
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print('  [D] skipped (sklearn not available)')
        return

    n = len(strip_data_mask)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7.5))
    if n == 1: axes = [[axes[0]], [axes[1]]]

    for col, (img_pil, slic_224, sp_feat_raw, fg_mask) in enumerate(strip_data_mask):
        seg    = slic_224.clip(0, cfg.n_segments - 1)
        feat_np = sp_feat_raw.cpu().numpy()
        pca     = PCA(n_components=3)
        rgb     = pca.fit_transform(feat_np)
        for ch in range(3):
            mn, mx = rgb[:, ch].min(), rgb[:, ch].max()
            rgb[:, ch] = (rgb[:, ch] - mn) / (mx - mn + 1e-8)

        before = (rgb[seg] * 255).astype(np.uint8)
        fg_np  = fg_mask.cpu().numpy()
        fg_pxl = fg_np[seg]
        after  = np.full((224, 224, 3), 180, dtype=np.uint8)
        after[fg_pxl > 0.5] = before[fg_pxl > 0.5]

        img_np = np.array(img_pil)
        b_blend = (before * 0.6 + img_np * 0.4).clip(0, 255).astype(np.uint8)
        a_blend = (after  * 0.6 + img_np * 0.4).clip(0, 255).astype(np.uint8)

        axes[0][col].imshow(b_blend); axes[0][col].axis('off')
        if col == 0: axes[0][col].set_title('All superpixel features\n(PCA→RGB)', fontsize=10)

        axes[1][col].imshow(a_blend); axes[1][col].axis('off')
        if col == 0: axes[1][col].set_title('Foreground only\n(BG → gray)', fontsize=10)

    fig.suptitle('D. Foreground-Masked Superpixel Features', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = OUT_DIR / f'fig_D_masking{tag}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [D] → {path}')


# ════════════════════════════════════════════════════════════════════════════
# Figure E — Concept activation maps (directly from visual strips)
# ════════════════════════════════════════════════════════════════════════════
def fig_E_from_strips(strip_files, tag, n_show=6, concepts_per_img=3):
    """
    Extracts [original + top concept overlays] from each strip and arranges into a grid.
    n_show: number of images to show (rows)
    concepts_per_img: number of concept panels per image
    """
    strips = strip_files[:n_show]
    n_cols = 1 + concepts_per_img   # original + concepts

    fig, axes = plt.subplots(len(strips), n_cols,
                              figsize=(3.2 * n_cols, 3.2 * len(strips)))
    if len(strips) == 1:
        axes = [axes]

    for row, (strip_path, y, p) in enumerate(strips):
        panels = extract_panels(strip_path)
        # panels[0] = original, panels[1..] = concepts, panels[-1] = combined
        concepts = panels[1:-1]  # exclude original and combined

        axes[row][0].imshow(panels[0])
        axes[row][0].axis('off')
        if row == 0:
            axes[row][0].set_title('Input', fontsize=10)
        axes[row][0].set_ylabel(f'y={y} p={p}', fontsize=8, rotation=0,
                                 labelpad=40, va='center')

        for ci in range(concepts_per_img):
            ax = axes[row][1 + ci]
            if ci < len(concepts):
                ax.imshow(concepts[ci])
                if row == 0:
                    ax.set_title(f'Concept {ci+1}', fontsize=10)
            ax.axis('off')

    fig.suptitle('E. Concept Activation Maps', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = OUT_DIR / f'fig_E_concepts{tag}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [E] → {path}')


# ════════════════════════════════════════════════════════════════════════════
# Figure E2 — Concept gallery (from concept_gallery/)
# ════════════════════════════════════════════════════════════════════════════
def fig_E2_gallery(gallery_dir, tag, n_concepts=6):
    """Show top-N concept gallery strips in a vertical stack."""
    files = sorted(f for f in os.listdir(gallery_dir) if f.endswith('.png'))[:n_concepts]
    imgs  = [Image.open(os.path.join(gallery_dir, f)) for f in files]

    fig, axes = plt.subplots(len(imgs), 1, figsize=(14, 2.2 * len(imgs)))
    if len(imgs) == 1: axes = [axes]

    for i, (img, fname) in enumerate(zip(imgs, files)):
        axes[i].imshow(img)
        axes[i].axis('off')
        # Parse rank and mean activation from filename (rank00_k338_mean2.741.png)
        label = fname.replace('.png', '').replace('_', '  ')
        axes[i].set_ylabel(label, fontsize=8, rotation=0, labelpad=130, va='center')

    fig.suptitle('E2. Concept Gallery (Top Activated Images per Concept)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = OUT_DIR / f'fig_E2_gallery{tag}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [E2] → {path}')


# ════════════════════════════════════════════════════════════════════════════
# Figure F — Intervention  (intuitive: image + concept overlays + slope graph)
# ════════════════════════════════════════════════════════════════════════════
def fig_F(strip_infer_data, strip_files, head, class_names, tag, top_remove=5):
    """
    Layout per row:
      [original image] [top concept removed] [2nd concept removed] | [slope graph]
    Slope graph: before→after confidence for top classes.
    True class highlighted in red with annotated drop.
    """
    n = len(strip_infer_data)
    fig, axes = plt.subplots(
        n, 4, figsize=(16, 3.8 * n),
        gridspec_kw={'width_ratios': [1, 1, 1, 2.2]})
    if n == 1:
        axes = [axes]

    for row, ((img_pil, y, z_pool, pred), (strip_path, _, __)) in enumerate(
            zip(strip_infer_data, strip_files)):
        ax_img, ax_c1, ax_c2, ax_chart = axes[row]

        # ── Compute intervention ──────────────────────────────────────────
        z_pool_t = z_pool.unsqueeze(0)
        W_y      = F.relu(head.fc.weight[y])
        contrib  = (W_y * z_pool).cpu()
        top_m    = contrib.topk(min(top_remove, contrib.shape[0])).indices

        z_int = z_pool_t.clone()
        z_int[0, top_m] = 0.0

        with torch.no_grad():
            probs_orig = F.softmax(head(z_pool_t).squeeze(0), dim=0).cpu().numpy()
            probs_int  = F.softmax(head(z_int).squeeze(0),   dim=0).cpu().numpy()
        pred_int = int(probs_int.argmax())

        cn  = class_names[y][:16]        if y        < len(class_names) else str(y)
        pn  = class_names[pred][:16]     if pred     < len(class_names) else str(pred)
        pin = class_names[pred_int][:16] if pred_int < len(class_names) else str(pred_int)

        # ── Panel 1: original image ───────────────────────────────────────
        ax_img.imshow(img_pil); ax_img.axis('off')
        correct = (pred == y)
        fc = '#27ae60' if correct else '#e74c3c'
        ax_img.set_title(
            f'True: {cn}\nPred: {pn}\n{probs_orig[pred]:.1%}',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=fc, alpha=0.25))

        # ── Panels 2-3: concept overlays from visual strip ────────────────
        panels = extract_panels(strip_path)
        concept_panels = panels[1:-1]   # skip [original] and [combined]

        for ax_c, ci in [(ax_c1, 0), (ax_c2, 1)]:
            ax_c.axis('off')
            if ci < len(concept_panels):
                ax_c.imshow(concept_panels[ci])
            k_id = top_m[ci].item() if ci < len(top_m) else '?'
            act  = float(contrib[top_m[ci]]) if ci < len(top_m) else 0.0
            label = f'Concept #{k_id}  (act={act:.2f})'
            color = '#c0392b'
            ax_c.set_title(label, fontsize=7.5, color=color)
        if row == 0:
            ax_c1.set_title(f'Top concept removed\n{ax_c1.get_title()}',
                             fontsize=7.5, color='#c0392b')

        # ── Panel 4: slope graph  before → after ─────────────────────────
        # Show top-5 classes by original confidence + true class
        top_cls = list(np.argsort(probs_orig)[::-1][:5])
        if y not in top_cls:
            top_cls = top_cls[:4] + [y]

        ax_chart.set_xlim(-0.25, 1.25)
        ax_chart.set_ylim(-0.05, 1.08)
        ax_chart.set_xticks([0, 1])
        ax_chart.set_xticklabels(['Before removal', 'After removal'], fontsize=9)
        ax_chart.yaxis.set_tick_params(labelsize=8)
        ax_chart.set_ylabel('Confidence', fontsize=9)
        ax_chart.spines[['top', 'right']].set_visible(False)
        ax_chart.axvline(0, color='#ccc', lw=0.8); ax_chart.axvline(1, color='#ccc', lw=0.8)

        for cls_i in top_cls:
            b = float(probs_orig[cls_i])
            a = float(probs_int[cls_i])
            is_true  = (cls_i == y)
            is_pred  = (cls_i == pred)
            clr  = '#e74c3c' if is_true else ('#3498db' if is_pred else '#bdc3c7')
            lw   = 2.5 if (is_true or is_pred) else 1.0
            ms   = 9   if (is_true or is_pred) else 4
            zo   = 5   if is_true else (4 if is_pred else 2)
            ls   = '-' if (is_true or is_pred) else '--'

            ax_chart.plot([0, 1], [b, a], ls + 'o', color=clr,
                          lw=lw, ms=ms, zorder=zo)

            # Annotate left side
            cname = (class_names[cls_i][:12] if cls_i < len(class_names)
                     else str(cls_i))
            ax_chart.annotate(
                f'{cname}  {b:.0%}',
                xy=(0, b), xytext=(-6, 0), textcoords='offset points',
                ha='right', va='center',
                fontsize=7.5 if (is_true or is_pred) else 6.5,
                color=clr,
                fontweight='bold' if is_true else 'normal')

            # Annotate right side with delta for true class
            if is_true:
                delta = a - b
                sign  = '▼' if delta < 0 else '▲'
                ax_chart.annotate(
                    f'{a:.0%}  {sign}{abs(delta):.0%}',
                    xy=(1, a), xytext=(6, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=8,
                    color='#e74c3c', fontweight='bold')
                # Shade the drop
                ax_chart.fill_between(
                    [0, 1], [b, a], min(b, a),
                    alpha=0.10, color='#e74c3c', zorder=1)

        drop = float(probs_orig[y]) - float(probs_int[y])
        ax_chart.set_title(
            f'True class drop: {probs_orig[y]:.1%} → {probs_int[y]:.1%}  '
            f'(Δ = {drop:+.1%})',
            fontsize=9,
            color='#c0392b' if drop > 0.02 else '#27ae60')

    fig.suptitle(
        'F. Concept Intervention — Removing Key Concepts Drops Confidence',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = OUT_DIR / f'fig_F_intervention{tag}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [F] → {path}')


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--tag',     default='')
    parser.add_argument('--n',       type=int, default=4, help='# samples for A-D/F')
    parser.add_argument('--n_e',     type=int, default=6, help='# strips for E')
    parser.add_argument('--test_indices', type=int, nargs='+', default=None,
                        help='Specific test set indices to use for A-D instead of strips')
    args = parser.parse_args()

    cfg     = load_cfg(args.run_dir)
    tag     = args.tag or f'_{cfg.dataset}'
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vis_dir = os.path.join(args.run_dir, 'visuals', 'final')
    gal_dir = os.path.join(args.run_dir, 'concept_gallery')

    print(f'Device: {device}  Run: {args.run_dir}')
    print(f'Visual dir: {vis_dir}')

    if not os.path.isdir(vis_dir):
        print(f'ERROR: {vis_dir} not found'); return

    # ── Load strips ────────────────────────────────────────────────────────
    all_strips   = load_strips(vis_dir, only_correct=True)
    use_strips   = all_strips[:max(args.n, args.n_e)]
    print(f'Using {len(use_strips)} correctly-predicted strips '
          f'(of {len(load_strips(vis_dir, only_correct=False))} total)')

    # ── Load model ─────────────────────────────────────────────────────────
    print('Loading model...')
    backbone, hook, feat_norm, sae, head = load_model(cfg, device)

    # ── Build class names ──────────────────────────────────────────────────
    if cfg.dataset == 'flowers102':
        class_names = [str(i) for i in range(cfg.num_classes)]
    else:
        suffix_map = {'car_select': 'stanford_cars_hf_carselect',
                      'car_best':   'stanford_cars_hf_carbest',
                      'car2':       'stanford_cars_hf'}
        suffix = suffix_map.get(cfg.dataset, 'stanford_cars_hf')
        cars_root = os.path.join(cfg.data_root, suffix)
        if os.path.isdir(os.path.join(cars_root, 'test')):
            from torchvision.datasets import ImageFolder
            ds = ImageFolder(os.path.join(cars_root, 'test'), transform=None)
            class_names = ds.classes
        else:
            class_names = [str(i) for i in range(cfg.num_classes)]

    # ── Build image list for A-D: from test_indices or strips ─────────────
    print('Running inference on visual samples...')
    slic_data, feat_data, fg_data, mask_data, infer_data = [], [], [], [], []

    if args.test_indices:
        # Load directly from test dataset
        val_tf = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        def _unnorm(t):
            m = torch.tensor(IMAGENET_MEAN).view(3,1,1)
            s = torch.tensor(IMAGENET_STD).view(3,1,1)
            return (t * s + m).clamp(0, 1)

        if cfg.dataset == 'flowers102':
            import torchvision as _tv
            test_ds = _tv.datasets.Flowers102(cfg.data_root, split='test',
                                               download=False, transform=val_tf)
        else:
            suffix_map2 = {'car_select': 'stanford_cars_hf_carselect',
                           'car_best': 'stanford_cars_hf_carbest',
                           'car2': 'stanford_cars_hf'}
            sfx = suffix_map2.get(cfg.dataset, 'stanford_cars_hf')
            test_ds = ImageFolder(os.path.join(cfg.data_root, sfx, 'test'), transform=val_tf)

        img_sources = []
        for idx in args.test_indices[:args.n]:
            img_t, y = test_ds[idx]
            img_np = (_unnorm(img_t).permute(1,2,0).numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            img_sources.append((img_pil, y))
        print(f'Using {len(img_sources)} test-set images (indices: {args.test_indices[:args.n]})')
    else:
        img_sources = [(extract_panels(sp)[0], y) for sp, y, p in use_strips[:args.n]]

    for img_pil, y in img_sources:
        slic_224, sp_feat_raw, fg, z_sp, z_pool, pred = infer(
            backbone, hook, feat_norm, sae, head, cfg, img_pil, device)

        slic_data.append( (img_pil, slic_224))
        feat_data.append( (img_pil, slic_224, sp_feat_raw))
        fg_data.append(   (img_pil, slic_224, fg))
        mask_data.append( (img_pil, slic_224, sp_feat_raw, fg))
        infer_data.append((img_pil, y, z_pool, pred))

    # ── Generate figures ───────────────────────────────────────────────────
    print('\nGenerating figures...')
    fig_A(slic_data,  cfg, tag)
    fig_B(feat_data,  cfg, tag)
    fig_C(fg_data,    cfg, tag)
    fig_D(mask_data,  cfg, tag)
    fig_E_from_strips(use_strips[:args.n_e], tag, n_show=args.n_e, concepts_per_img=3)
    if os.path.isdir(gal_dir):
        fig_E2_gallery(gal_dir, tag, n_concepts=6)
    fig_F(infer_data, use_strips[:args.n], head, class_names, tag, top_remove=5)

    print(f'\nAll figures saved to {OUT_DIR.resolve()}')


if __name__ == '__main__':
    main()
