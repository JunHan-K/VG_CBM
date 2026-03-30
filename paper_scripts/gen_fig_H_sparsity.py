#!/usr/bin/env python3
"""
gen_fig_H_sparsity.py
---------------------
Figure H: Concept Sparsity — cumulative contribution curve.

For each test sample:
  1. Compute contribution[k] = W_y[k] * z_pool[k]  for all k in 4096
  2. Sort descending, compute cumulative sum / total
  3. Average across all correctly-predicted test samples

Output: curve showing what fraction of confidence mass is explained
        by the top-N concepts, for N = 1 .. 4096.
K_0.95 (where curve crosses 0.95) is annotated with a dashed line.
"""

import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms
import torchvision

_HERE    = os.path.dirname(os.path.abspath(__file__))
_VG_CBM  = os.path.join(_HERE, '..')
_CBM_SAE = os.path.join(_VG_CBM, '..', 'CBM_SAE')
sys.path.insert(0, _VG_CBM)
sys.path.insert(0, _CBM_SAE)
os.chdir(_CBM_SAE)   # data/ and models/ are relative to here

from src.backbone   import build_backbone
from src.models     import FeatureNorm, SparseSAE, CBMHead, fg_z_pool
from src.slic_utils import compute_slic_224, extract_sp_features, slic224_to_14, compute_sp_fg
from torchvision.datasets import ImageFolder

OUT_DIR = Path(_VG_CBM) / 'figures'
OUT_DIR.mkdir(exist_ok=True)

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
val_tf = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def unnorm(t):
    m = torch.tensor(MEAN).view(3,1,1)
    s = torch.tensor(STD).view(3,1,1)
    return (t * s + m).clamp(0, 1)

def pil_from_t(t):
    from PIL import Image
    return Image.fromarray((unnorm(t).permute(1,2,0).numpy()*255).astype('uint8'))


def load_model(run_dir, device):
    with open(os.path.join(run_dir, 'config.json')) as f:
        saved = json.load(f)
    class CFG: pass
    cfg = CFG()
    defaults = dict(backbone='resnet50', d_in=1024, expansion=4, topk=256,
                    num_classes=31, n_segments=80, compactness=10.0,
                    fg_ratio=0.30, pool_mode='max')
    for k, v in defaults.items(): setattr(cfg, k, v)
    for k, v in saved.items():    setattr(cfg, k, v)
    backbone, hook, d_in = build_backbone(device, cfg.backbone)
    cfg.d_in = d_in
    K = cfg.d_in * cfg.expansion
    feat_norm = FeatureNorm(cfg.d_in).to(device)
    sae       = SparseSAE(cfg.d_in, K, topk=cfg.topk).to(device)
    head      = CBMHead(K, cfg.num_classes).to(device)
    ckpt = torch.load(os.path.join(run_dir, 'sae_checkpoint.pt'),
                      map_location=device, weights_only=False)
    if 'feat_norm' in ckpt:
        feat_norm.load_state_dict(ckpt['feat_norm'], strict=False)
    sae.load_state_dict(ckpt['sae'])
    hckpt = torch.load(os.path.join(run_dir, 'head_best.pt'),
                       map_location=device, weights_only=False)
    head.load_state_dict(hckpt['head'])
    backbone.eval(); feat_norm.eval(); sae.eval(); head.eval()
    return backbone, hook, feat_norm, sae, head, cfg


@torch.no_grad()
def collect_curves(run_dir, dataset, device, max_samples=None):
    """
    Returns:
        curves : (N_samples, K) array — each row is the sorted cumulative
                 contribution fraction for one correctly-predicted sample
        k095_mean : float — mean K_0.95 across samples
    """
    backbone, hook, feat_norm, sae, head, cfg = load_model(run_dir, device)
    K = cfg.d_in * cfg.expansion

    cumulative_curves = []
    n_correct = 0

    total = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    print(f'  Running inference on {total} samples...')

    for i in range(total):
        img_t, label = dataset[i]
        img_pil = pil_from_t(img_t)
        x = img_t.unsqueeze(0).to(device)

        backbone(x)
        f       = feat_norm(hook.out.permute(0,2,3,1)).permute(0,3,1,2)
        slic_224 = compute_slic_224(img_pil, cfg.n_segments, cfg.compactness)
        slic_t  = torch.from_numpy(slic_224.astype('int64')).unsqueeze(0).to(device)
        slic_14 = slic224_to_14(slic_t)
        sp_raw  = extract_sp_features(f, slic_14, cfg.n_segments)
        fg      = compute_sp_fg(sp_raw, cfg.fg_ratio)
        z, _    = sae(sp_raw * fg.unsqueeze(-1))
        z_pool  = fg_z_pool(z, fg, getattr(cfg, 'pool_mode', 'max'))

        probs = F.softmax(head(z_pool), dim=1).squeeze(0)
        pred  = int(probs.argmax())
        if pred != label:
            continue

        W_y    = F.relu(head.fc.weight[pred]).cpu()
        z_pool_cpu = z_pool.squeeze(0).cpu()
        contrib = (W_y * z_pool_cpu).numpy()          # (K,)
        contrib = np.maximum(contrib, 0)

        total_mass = contrib.sum()
        if total_mass < 1e-8:
            continue

        sorted_c = np.sort(contrib)[::-1]             # descending
        cumsum   = np.cumsum(sorted_c) / total_mass   # (K,) cumulative fraction
        cumulative_curves.append(cumsum)
        n_correct += 1

        if (i + 1) % 200 == 0:
            print(f'    {i+1}/{total}  correct so far: {n_correct}')

    print(f'  Done. {n_correct} correctly predicted samples used.')
    curves = np.stack(cumulative_curves, axis=0)   # (N, K)

    # K_0.95 per sample: first index where cumsum >= 0.95, then +1 for count
    k095_per = (curves < 0.95).sum(axis=1) + 1
    k095_mean = float(k095_per.mean())
    print(f'  K_0.95 = {k095_mean:.1f}  (range {k095_per.min()} to {k095_per.max()})')

    return curves, k095_mean


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    os.chdir(os.path.join(_HERE, '..'))   # so relative data paths work

    CACHE = OUT_DIR / 'fig_H_curves_cache.npz'

    if CACHE.exists():
        print('Loading cached curves...')
        d = np.load(CACHE)
        car_curves, fl_curves = d['car_curves'], d['fl_curves']
    else:
        # ── Car-Best ──────────────────────────────────────────────────────────
        print('=== Car-Best ===')
        car_run  = os.path.join(_VG_CBM, 'models', 'car')
        car_data = os.path.join(_CBM_SAE, 'data', 'stanford_cars_hf_carbest', 'test')
        car_ds   = ImageFolder(car_data, transform=val_tf)
        car_curves, _ = collect_curves(car_run, car_ds, device)

        # ── Flowers102 ────────────────────────────────────────────────────────
        print('\n=== Flowers102 ===')
        fl_run  = os.path.join(_VG_CBM, 'models', 'flowers')
        fl_data = os.path.join(_CBM_SAE, 'data')
        fl_ds   = torchvision.datasets.Flowers102(fl_data, split='test',
                                                  download=False, transform=val_tf)
        fl_curves, _ = collect_curves(fl_run, fl_ds, device)

        np.savez(CACHE, car_curves=car_curves, fl_curves=fl_curves)
        print(f'Curves cached -> {CACHE}')

    # ── Derived stats ─────────────────────────────────────────────────────────
    K_TOTAL = car_curves.shape[1]
    THRESHOLDS = [0.60, 0.70, 0.80, 0.90, 0.95]

    def k_at(curves, thresh):
        """Mean number of concepts needed to reach `thresh` fraction."""
        per = (curves < thresh).sum(axis=1) + 1
        return float(per.mean())

    car_k08 = k_at(car_curves, 0.95)
    fl_k08  = k_at(fl_curves,  0.95)

    car_ks  = [k_at(car_curves, t) for t in THRESHOLDS]
    fl_ks   = [k_at(fl_curves,  t) for t in THRESHOLDS]

    x_rank   = np.arange(1, K_TOTAL + 1)
    car_mean = car_curves.mean(axis=0)
    fl_mean  = fl_curves.mean(axis=0)
    car_p10  = np.percentile(car_curves, 10, axis=0)
    car_p90  = np.percentile(car_curves, 90, axis=0)
    fl_p10   = np.percentile(fl_curves,  10, axis=0)
    fl_p90   = np.percentile(fl_curves,  90, axis=0)

    CAR_COLOR = '#2980b9'
    FLW_COLOR = '#27ae60'
    THRESH_COLORS = ['#bdc3c7', '#95a5a6', '#e74c3c', '#8e44ad', '#2c3e50']

    # ── Clip x-axis: 50% to where curve flattens (99%) ──────────────────────
    avg_mean = (car_mean + fl_mean) / 2
    x_start  = max(0, int(np.argmax(avg_mean >= 0.50)) - 10)
    x_end    = int(np.argmax(avg_mean >= 0.99)) + 40
    x_end    = min(x_end, K_TOTAL)

    # ── Single panel ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.subplots_adjust(left=0.10, right=0.95, top=0.85, bottom=0.15)

    # Draw individual sample curves (random subset) to show real data
    rng = np.random.default_rng(42)
    n_show = 60
    for curves, col in [(car_curves, CAR_COLOR), (fl_curves, FLW_COLOR)]:
        idx = rng.choice(len(curves), size=min(n_show, len(curves)), replace=False)
        for i in idx:
            ax.plot(x_rank, curves[i], color=col, linewidth=0.4, alpha=0.08)

    # Mean on top
    ax.plot(x_rank, car_mean, color=CAR_COLOR, linewidth=2.2,
            label=f'Car-Best   (K$_{{0.95}}$ = {car_k08:.0f})')
    ax.plot(x_rank, fl_mean,  color=FLW_COLOR, linewidth=2.2,
            label=f'Flowers102 (K$_{{0.95}}$ = {fl_k08:.0f})')

    # Threshold lines — labels placed just inside the right edge
    for thresh, tc in zip(THRESHOLDS, THRESH_COLORS):
        lw = 1.6 if thresh == 0.95 else 0.85
        ls = '--' if thresh == 0.95 else ':'
        ax.axhline(thresh, color=tc, linewidth=lw, linestyle=ls, zorder=1)
        ax.text(x_end * 0.985, thresh + 0.012,
                f'{int(thresh*100)}%', ha='right', va='bottom',
                fontsize=8.5, color=tc,
                fontweight='bold' if thresh == 0.95 else 'normal')

    # K_0.95 vertical markers
    ax.axvline(car_k08, color=CAR_COLOR, linewidth=1.2, linestyle=':')
    ax.axvline(fl_k08,  color=FLW_COLOR, linewidth=1.2, linestyle=':')

    # Flowers102 (green) is higher on the curve → label above
    ax.annotate(f'Flowers102: {fl_k08:.0f}  ({fl_k08/K_TOTAL*100:.1f}% of {K_TOTAL})',
                xy=(fl_k08, 0.95), xytext=(x_end * 0.42, 1.00),
                fontsize=9, color=FLW_COLOR,
                arrowprops=dict(arrowstyle='->', color=FLW_COLOR, lw=1.8),
                bbox=dict(boxstyle='round,pad=0.28', fc='white', ec=FLW_COLOR, lw=1.2))
    # Car-Best (blue) is lower on the curve → label below
    ax.annotate(f'Car-Best: {car_k08:.0f}  ({car_k08/K_TOTAL*100:.1f}% of {K_TOTAL})',
                xy=(car_k08, 0.95), xytext=(x_end * 0.55, 0.83),
                fontsize=9, color=CAR_COLOR,
                arrowprops=dict(arrowstyle='->', color=CAR_COLOR, lw=1.8),
                bbox=dict(boxstyle='round,pad=0.28', fc='white', ec=CAR_COLOR, lw=1.2))

    ax.set_xlabel('Concepts (ranked by contribution)', fontsize=10.5)
    ax.set_ylabel('Cumulative confidence fraction', fontsize=10.5)
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(0.48, 1.02)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(['50%', '60%', '70%', '80%', '90%', '100%'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.9, loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.28)
    ax.set_title('Cumulative contribution of top-N concepts to predicted class confidence',
                 fontsize=10, pad=8, color='#333')

    fig.suptitle('H. Concept Sparsity  (K$_{0.95}$)', fontsize=13, fontweight='bold', y=0.97)

    out_path = OUT_DIR / 'fig_H_sparsity.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved -> {out_path}')
