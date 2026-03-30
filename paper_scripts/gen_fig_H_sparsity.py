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
K_0.8 (where curve crosses 0.8) is annotated with a dashed line.
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
        k08_mean : float — mean K_0.8 across samples
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

    # K_0.8 per sample: first index where cumsum >= 0.8, then +1 for count
    k08_per = (curves < 0.8).sum(axis=1) + 1
    k08_mean = float(k08_per.mean())
    print(f'  K_0.8 = {k08_mean:.1f}  (range {k08_per.min()} to {k08_per.max()})')

    return curves, k08_mean


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    os.chdir(os.path.join(_HERE, '..'))   # so relative data paths work

    # ── Car-Best ──────────────────────────────────────────────────────────────
    print('=== Car-Best ===')
    car_run  = os.path.join(_VG_CBM, 'models', 'car')
    car_data = os.path.join(_CBM_SAE, 'data', 'stanford_cars_hf_carbest', 'test')
    car_ds   = ImageFolder(car_data, transform=val_tf)
    car_curves, car_k08 = collect_curves(car_run, car_ds, device)

    # ── Flowers102 ────────────────────────────────────────────────────────────
    print('\n=== Flowers102 ===')
    fl_run  = os.path.join(_VG_CBM, 'models', 'flowers')
    fl_data = os.path.join(_CBM_SAE, 'data')
    fl_ds   = torchvision.datasets.Flowers102(fl_data, split='test',
                                              download=False, transform=val_tf)
    fl_curves, fl_k08 = collect_curves(fl_run, fl_ds, device)

    # ── Plot ──────────────────────────────────────────────────────────────────
    K_TOTAL   = car_curves.shape[1]
    x_rank    = np.arange(1, K_TOTAL + 1)

    car_mean  = car_curves.mean(axis=0)
    fl_mean   = fl_curves.mean(axis=0)
    car_p10   = np.percentile(car_curves, 10, axis=0)
    car_p90   = np.percentile(car_curves, 90, axis=0)
    fl_p10    = np.percentile(fl_curves,  10, axis=0)
    fl_p90    = np.percentile(fl_curves,  90, axis=0)

    CAR_COLOR = '#2980b9'
    FLW_COLOR = '#27ae60'

    fig, ax = plt.subplots(figsize=(7, 4.8))

    # Shaded 10–90 percentile band
    ax.fill_between(x_rank, car_p10, car_p90, alpha=0.12, color=CAR_COLOR)
    ax.fill_between(x_rank, fl_p10,  fl_p90,  alpha=0.12, color=FLW_COLOR)

    # Mean curves
    ax.plot(x_rank, car_mean, color=CAR_COLOR, linewidth=2.0,
            label=f'Car-Best   (K₀.₈ = {car_k08:.0f})')
    ax.plot(x_rank, fl_mean,  color=FLW_COLOR, linewidth=2.0,
            label=f'Flowers102 (K₀.₈ = {fl_k08:.0f})')

    # 0.8 reference line
    ax.axhline(0.8, color='#888', linewidth=1.0, linestyle='--', zorder=1)
    ax.text(K_TOTAL * 0.98, 0.81, '80%', ha='right', va='bottom',
            fontsize=9, color='#666')

    # K_0.8 vertical markers
    ax.axvline(car_k08, color=CAR_COLOR, linewidth=1.2, linestyle=':')
    ax.axvline(fl_k08,  color=FLW_COLOR, linewidth=1.2, linestyle=':')

    # Annotation boxes
    for k08, col, ydelta in [(car_k08, CAR_COLOR, -0.09), (fl_k08, FLW_COLOR, -0.16)]:
        ax.annotate(f'{int(round(k08))}\n({k08/K_TOTAL*100:.1f}% of {K_TOTAL})',
                    xy=(k08, 0.8),
                    xytext=(k08 + K_TOTAL * 0.04, 0.8 + ydelta),
                    fontsize=8.5, color=col,
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.2),
                    bbox=dict(boxstyle='round,pad=0.3', fc='white',
                              ec=col, lw=0.8, alpha=0.9))

    ax.set_xlabel('Number of concepts (ranked by contribution)', fontsize=10)
    ax.set_ylabel('Cumulative confidence fraction', fontsize=10)
    ax.set_xlim(0, K_TOTAL)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=9.5, framealpha=0.9, loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.35)

    fig.suptitle('H. Concept Sparsity', fontsize=13, fontweight='bold', y=1.01)
    ax.set_title('Cumulative contribution of top-N concepts to predicted class confidence',
                 fontsize=9.5, color='#444', pad=6)

    plt.tight_layout()
    out_path = OUT_DIR / 'fig_H_sparsity.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved -> {out_path}')
