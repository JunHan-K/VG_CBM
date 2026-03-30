#!/usr/bin/env python3
"""
gen_fig_F_progressive.py  v2
Layout per row: [original] | [concept attribution map] | [progressive curve]
- concept attribution map: weighted heatmap of all active concepts (W_y * z_sp summed)
- progressive curve: greedy 1-by-1 removal, no overlapping labels
"""

import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
_VG_CBM = os.path.join(_HERE, '..')       # vg_cbm/
_CBM_SAE = os.path.join(_VG_CBM, '..', 'CBM_SAE')  # CBM_SAE/ (data lives here)
sys.path.insert(0, _HERE)
sys.path.insert(0, _VG_CBM)
sys.path.insert(0, _CBM_SAE)
os.chdir(_CBM_SAE)  # data_root = ./data

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision

from src.backbone   import build_backbone
from src.models     import FeatureNorm, SparseSAE, CBMHead, fg_z_pool
from src.slic_utils import compute_slic_224, extract_sp_features, slic224_to_14, compute_sp_fg

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
    return Image.fromarray((unnorm(t).permute(1,2,0).numpy() * 255).astype('uint8'))


def concept_attr_map(img_pil, slic_224, z_sp_cpu, W_y_cpu, n_seg, alpha=0.65):
    """
    Weighted attribution map: for each superpixel, sum(W_y * z_sp) across all concepts.
    Shows WHERE the model attends to make its prediction.
    """
    contrib_sp = (W_y_cpu.unsqueeze(0) * z_sp_cpu).sum(dim=1).numpy()  # (N_sp,)
    contrib_sp = contrib_sp.clip(0)
    norm = (contrib_sp - contrib_sp.min()) / (contrib_sp.max() - contrib_sp.min() + 1e-8)

    seg  = slic_224.clip(0, n_seg - 1)
    hmap = norm[seg]

    cmap  = plt.get_cmap('hot')
    hrgb  = (cmap(hmap)[:, :, :3] * 255).astype(np.uint8)
    img_np = np.array(img_pil).astype(np.float32)
    blend  = (img_np * (1 - alpha) + hrgb.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blend)


def load_model(run_dir, device):
    with open(os.path.join(run_dir, 'config.json')) as f:
        saved = json.load(f)

    class CFG: pass
    cfg = CFG()
    defaults = dict(backbone='resnet50', d_in=1024, expansion=4, topk=20,
                    num_classes=102, n_segments=80, compactness=10.0, fg_ratio=0.30,
                    dataset='flowers102')
    for k, v in defaults.items(): setattr(cfg, k, v)
    for k, v in saved.items():    setattr(cfg, k, v)

    backbone, hook, d_in = build_backbone(device, cfg.backbone)
    cfg.d_in = d_in
    K = cfg.d_in * cfg.expansion
    feat_norm = FeatureNorm(cfg.d_in).to(device)
    sae       = SparseSAE(cfg.d_in, K, topk=cfg.topk).to(device)
    head      = CBMHead(K, cfg.num_classes).to(device)

    ckpt = torch.load(os.path.join(run_dir, 'sae_checkpoint.pt'), map_location=device, weights_only=False)
    if 'feat_norm' in ckpt:
        feat_norm.load_state_dict(ckpt['feat_norm'], strict=False)
    sae.load_state_dict(ckpt['sae'])
    hckpt = torch.load(os.path.join(run_dir, 'head_best.pt'), map_location=device, weights_only=False)
    head.load_state_dict(hckpt['head'])

    backbone.eval(); feat_norm.eval(); sae.eval(); head.eval()
    return backbone, hook, feat_norm, sae, head, cfg


@torch.no_grad()
def run_infer(img_tensor, img_pil, backbone, hook, feat_norm, sae, head, cfg, device):
    x = img_tensor.unsqueeze(0).to(device)
    backbone(x)
    f = feat_norm(hook.out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    slic_224 = compute_slic_224(img_pil, cfg.n_segments, cfg.compactness)
    slic_t   = torch.from_numpy(slic_224.astype('int64')).unsqueeze(0).to(device)
    slic_14  = slic224_to_14(slic_t)

    sp_raw  = extract_sp_features(f, slic_14, cfg.n_segments)
    fg      = compute_sp_fg(sp_raw, cfg.fg_ratio)
    z, _    = sae(sp_raw * fg.unsqueeze(-1))
    z_pool  = fg_z_pool(z, fg, getattr(cfg, 'pool_mode', 'mean'))
    logits  = head(z_pool)
    pred    = int(logits.argmax(1).item())
    z_sp_cpu = z.squeeze(0).cpu()
    W_y_cpu  = F.relu(head.fc.weight[pred]).cpu()
    return slic_224, z_sp_cpu, W_y_cpu, z_pool.squeeze(0), pred


@torch.no_grad()
def progressive_removal(z_pool, head, label, max_remove, device):
    z = z_pool.unsqueeze(0).clone().to(device)
    confs   = []
    kid_seq = []
    removed = set()

    for step in range(max_remove + 1):
        probs = F.softmax(head(z).squeeze(0), dim=0).cpu().numpy()
        confs.append(float(probs[label]))
        if step < max_remove:
            W_y     = F.relu(head.fc.weight[label])
            contrib = (W_y * z.squeeze(0)).cpu()
            for r in removed:
                contrib[r] = -1e9
            kid = int(contrib.argmax().item())
            removed.add(kid)
            kid_seq.append(kid)
            z[0, kid] = 0.0

    return confs, kid_seq


def make_figure(samples, out_path, title, max_remove=8):
    n = len(samples)
    fig = plt.figure(figsize=(17, 5.5 * n))
    outer = gridspec.GridSpec(n, 1, figure=fig, hspace=0.50,
                              left=0.04, right=0.97, top=0.95, bottom=0.04)

    for row, s in enumerate(samples):
        # 3 columns: original | attr map | curve — wider gap before chart
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[row],
            width_ratios=[1, 1, 3.2], wspace=0.35)

        ax_img  = fig.add_subplot(inner[0])
        ax_attr = fig.add_subplot(inner[1])
        ax_plot = fig.add_subplot(inner[2])

        confs   = s['confs']
        kid_seq = s['kid_seq']
        label   = s['label']
        cn      = s['class_name']

        # ── Panel 1: original ────────────────────────────────────────────
        ax_img.imshow(s['img_pil'])
        ax_img.axis('off')
        ax_img.set_title(f'{cn}\n{confs[0]:.1%}', fontsize=8.5, fontweight='bold',
                         pad=4, color='#1a1a1a')

        # ── Panel 2: concept attribution map ─────────────────────────────
        ax_attr.imshow(s['attr_map'])
        ax_attr.axis('off')
        ax_attr.set_title('Concept attribution\n(weighted sum)', fontsize=8, pad=4, color='#555')

        # ── Panel 3: progressive curve ────────────────────────────────────
        steps     = list(range(len(confs)))
        conf_pct  = [c * 100 for c in confs]

        ax_plot.plot(steps, conf_pct, 'o-', color='#e74c3c', lw=2.5, ms=7, zorder=5)
        ax_plot.fill_between(steps, conf_pct, 0, alpha=0.09, color='#e74c3c')

        # Annotate start value (above first point, inside axes)
        ax_plot.text(0, conf_pct[0] + 2.5, f'{confs[0]:.0%}',
                     fontsize=10, fontweight='bold', color='#e74c3c',
                     ha='center', va='bottom')
        # Annotate end value (below last point, with total drop)
        ax_plot.text(len(confs)-1, conf_pct[-1] - 2.5,
                     f'{confs[-1]:.0%}  (▼{confs[0]-confs[-1]:.0%})',
                     fontsize=9, fontweight='bold', color='#c0392b',
                     ha='center', va='top')

        ax_plot.set_xlim(-0.5, len(steps) - 0.5)
        ax_plot.set_ylim(-8, min(108, conf_pct[0] + 12))
        ax_plot.set_xticks(steps)
        ax_plot.set_xticklabels(
            ['0\n(orig)'] + [str(i+1) for i in range(len(kid_seq))],
            fontsize=8.5)
        ax_plot.set_xlabel('# concepts removed (greedy)', fontsize=9)
        ax_plot.set_ylabel('True-class confidence (%)', fontsize=9)
        ax_plot.spines[['top', 'right']].set_visible(False)
        ax_plot.axhline(0, color='#ddd', lw=0.8, ls='--')
        ax_plot.grid(axis='y', alpha=0.20, lw=0.7)
        ax_plot.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=0))

    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out_path}')


CELL = 224
PAD  = 8

def load_strips(visuals_dir, only_correct=True):
    """Load correctly-predicted visual strip files."""
    files = sorted(f for f in os.listdir(visuals_dir) if f.endswith('.png'))
    result = []
    for f in files:
        try:
            parts = f.replace('.png','').split('_')
            y = int([p for p in parts if p.startswith('y')][0][1:])
            p = int([p for p in parts if p.startswith('p')][0][1:])
        except Exception:
            continue
        if only_correct and y != p:
            continue
        result.append((os.path.join(visuals_dir, f), y))
    return result


def extract_orig(strip_path):
    """Extract the first 224x224 panel (original image) from a strip."""
    img = Image.open(strip_path).convert('RGB')
    return img.crop((0, 0, CELL, CELL))


def build_samples_from_dataset(run_dir, dataset, indices, class_names_list, device, max_remove=8):
    backbone, hook, feat_norm, sae, head, cfg = load_model(run_dir, device)
    samples = []
    with torch.no_grad():
        for idx in indices:
            img_t, label = dataset[idx]
            img_pil = pil_from_t(img_t)
            slic_224, z_sp_cpu, W_y_cpu, z_pool, pred = run_infer(
                img_t, img_pil, backbone, hook, feat_norm, sae, head, cfg, device)
            confs, kid_seq = progressive_removal(z_pool, head, label, max_remove, device)
            attr_map = concept_attr_map(img_pil, slic_224, z_sp_cpu, W_y_cpu, cfg.n_segments)
            cn = class_names_list[label] if label < len(class_names_list) else str(label)
            print(f'  idx={idx:4d}  {cn[:35]:<35}  {confs[0]:.1%} → {confs[-1]:.1%}  (▼{confs[0]-confs[-1]:.1%})')
            samples.append(dict(
                img_pil=img_pil, attr_map=attr_map,
                z_pool=z_pool, label=label, pred=pred, class_name=cn,
                confs=confs, kid_seq=kid_seq))
    return samples


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    MAX_REMOVE = 8

    # ── Flowers — high-conf + high-drop test samples ───────────────────────────
    print('\n=== Flowers102 ===')
    flowers_run = os.path.join(_HERE, '..', 'models', 'flowers')
    flowers_ds  = torchvision.datasets.Flowers102('./data', split='test',
                                                   download=False, transform=val_tf)
    flowers_cls = [str(i) for i in range(102)]
    flowers_idx = [1244, 90, 913]   # high-conf + high-drop: 81.7%→7%, 86.5%→14%, 91.0%→24%

    flowers_samples = build_samples_from_dataset(
        flowers_run, flowers_ds, flowers_idx, flowers_cls, device, MAX_REMOVE)
    make_figure(
        flowers_samples,
        OUT_DIR / 'fig_G_progressive_flowers.png',
        'F. Concept Intervention — Flowers102  (acc = 82.9%)',
        max_remove=MAX_REMOVE)

    # ── Car — high-conf + high-drop test samples ───────────────────────────────
    print('\n=== Car Best ===')
    with open('./data/carbest_classes.json') as f:
        car_class_names = json.load(f)['class_names']
    car_run = os.path.join(_HERE, '..', 'models', 'car')
    car_ds  = ImageFolder('./data/stanford_cars_hf_carbest/test', transform=val_tf)
    car_idx = [279, 1059, 872]   # high-conf + high-drop: 93.2%→21%, 88.3%→16%, 80.8%→10%

    car_samples = build_samples_from_dataset(
        car_run, car_ds, car_idx, car_class_names, device, MAX_REMOVE)
    make_figure(
        car_samples,
        OUT_DIR / 'fig_G_progressive_car.png',
        'F. Concept Intervention — Car-Best  (acc = 73.7%)',
        max_remove=MAX_REMOVE)

    print('\nDone.')
