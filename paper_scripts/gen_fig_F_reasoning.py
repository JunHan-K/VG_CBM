#!/usr/bin/env python3
"""
gen_fig_F_reasoning.py
--------------------
CBM reasoning figure: for each selected image, shows
  [original] [concept 1] [concept 2] [concept 3] [concept 4] | [contribution bar chart]

Each concept panel is a heatmap overlay showing WHERE concept k activates spatially.
The bar chart shows HOW MUCH each top concept contributes to the predicted class (W_y[k] * z_pool[k]).

Usage:
  # Step 1: scan and find best candidate indices
  python paper_scripts/gen_fig_F_reasoning.py --scan

  # Step 2: generate final figure with chosen indices
  python paper_scripts/gen_fig_F_reasoning.py \
      --car_idx   279 1059 872 \
      --flower_idx 1244 90 913
"""

import os, sys, json, argparse
_HERE   = os.path.dirname(os.path.abspath(__file__))
_VG_CBM = os.path.join(_HERE, '..')
sys.path.insert(0, _VG_CBM)

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

FLOWERS_NAMES = [
    'pink primrose', 'hard-leaved pocket orchid', 'Canterbury bells', 'sweet pea',
    'English marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood',
    'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle',
    'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily',
    'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower',
    'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy',
    'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william',
    'carnation', 'garden phlox', 'love in the mist', 'mexican aster',
    'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort',
    'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily',
    'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup',
    'oxeye daisy', 'common dandelion', 'petunia', 'wild pansy', 'primula',
    'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium',
    'orange dahlia', 'pink-yellow dahlia', 'cautleya spicata', 'japanese anemone',
    'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum',
    'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania',
    'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory',
    'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani',
    'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow',
    'magnolia', 'cyclamen', 'watercress', 'canna lily', 'hippeastrum',
    'bee balm', 'pink quill', 'foxglove', 'bougainvillea', 'camellia',
    'mallow', 'mexican petunia', 'bromelia', 'blanket flower',
    'trumpet creeper', 'blackberry lily',
]


# ─────────────────────────────────────────────────────────────────────────────
def unnorm(t):
    m = torch.tensor(MEAN).view(3, 1, 1)
    s = torch.tensor(STD).view(3, 1, 1)
    return (t * s + m).clamp(0, 1)

def pil_from_t(t):
    return Image.fromarray((unnorm(t).permute(1, 2, 0).numpy() * 255).astype('uint8'))


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
    ckpt  = torch.load(os.path.join(run_dir, 'sae_checkpoint.pt'),
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
def run_infer(img_t, img_pil, backbone, hook, feat_norm, sae, head, cfg, device):
    x = img_t.unsqueeze(0).to(device)
    backbone(x)
    f        = feat_norm(hook.out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    slic_224 = compute_slic_224(img_pil, cfg.n_segments, cfg.compactness)
    slic_t   = torch.from_numpy(slic_224.astype('int64')).unsqueeze(0).to(device)
    slic_14  = slic224_to_14(slic_t)
    sp_raw   = extract_sp_features(f, slic_14, cfg.n_segments)
    fg       = compute_sp_fg(sp_raw, cfg.fg_ratio)
    z, _     = sae(sp_raw * fg.unsqueeze(-1))
    z_pool   = fg_z_pool(z, fg, getattr(cfg, 'pool_mode', 'max'))
    probs    = F.softmax(head(z_pool), dim=1).squeeze(0)
    pred     = int(probs.argmax().item())
    conf     = float(probs[pred].item())
    W_y      = F.relu(head.fc.weight[pred]).cpu()
    z_sp_cpu = z.squeeze(0).cpu()
    z_pool_cpu = z_pool.squeeze(0).cpu()
    return slic_224, z_sp_cpu, z_pool_cpu, W_y, pred, conf


def concept_heatmap_overlay(img_pil, slic_224, z_sp_cpu, concept_idx, n_seg, alpha=0.6):
    """Heatmap overlay for a single concept on the image."""
    act = z_sp_cpu[:, concept_idx].numpy().clip(0)
    seg = slic_224.clip(0, n_seg - 1)
    hmap = act[seg]
    if hmap.max() > 0:
        hmap = hmap / hmap.max()
    cmap  = plt.get_cmap('hot')
    hrgb  = (cmap(hmap)[:, :, :3] * 255).astype(np.uint8)
    img_np = np.array(img_pil).astype(np.float32)
    blend  = (img_np * (1 - alpha) + hrgb.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blend)


def semantic_score(z_sp_cpu, W_y, top_k=4):
    """
    Score how 'semantic' a sample is:
    - Top concepts should be spatially concentrated (low entropy per concept)
    - Top concepts should activate in different regions (low overlap between them)
    Higher score = more semantic.
    """
    contrib = (W_y * z_sp_cpu.mean(0))  # (K,)
    top_idx = contrib.topk(top_k).indices.tolist()

    maps = []
    for k in top_idx:
        act = z_sp_cpu[:, k].numpy().clip(0)
        if act.max() > 0:
            act = act / act.max()
        maps.append(act)

    # Concentration: low entropy = concentrated
    scores = []
    for m in maps:
        p = m / (m.sum() + 1e-8)
        entropy = -(p * np.log(p + 1e-8)).sum()
        scores.append(-entropy)  # higher = more concentrated

    # Diversity between top concepts (low cosine sim = diverse)
    diversity = 0
    count = 0
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            a, b = maps[i], maps[j]
            sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            diversity += (1 - sim)
            count += 1
    diversity = diversity / max(count, 1)

    return np.mean(scores) + diversity


# ─────────────────────────────────────────────────────────────────────────────
def scan_dataset(run_dir, dataset, class_names, device,
                 n_scan=300, conf_thresh=0.75):
    """Scan dataset to find semantically rich samples. Returns top candidates."""
    backbone, hook, feat_norm, sae, head, cfg = load_model(run_dir, device)
    indices = list(range(len(dataset)))
    np.random.seed(0)
    np.random.shuffle(indices)
    indices = indices[:n_scan]

    candidates = []
    print(f'  Scanning {n_scan} samples (conf >= {conf_thresh:.0%})...')
    for i, idx in enumerate(indices):
        img_t, label = dataset[idx]
        img_pil = pil_from_t(img_t)
        slic_224, z_sp_cpu, z_pool_cpu, W_y, pred, conf = run_infer(
            img_t, img_pil, backbone, hook, feat_norm, sae, head, cfg, device)
        if pred != label or conf < conf_thresh:
            continue
        score = semantic_score(z_sp_cpu, W_y)
        cn = class_names[label] if label < len(class_names) else str(label)
        candidates.append((score, idx, label, pred, conf, cn))
        if (i + 1) % 50 == 0:
            print(f'    {i+1}/{n_scan}  found {len(candidates)} candidates so far')

    # Sort by score, pick one per class (most semantic per class)
    candidates.sort(key=lambda x: -x[0])
    seen_classes = set()
    best = []
    for row in candidates:
        cls = row[2]
        if cls not in seen_classes:
            seen_classes.add(cls)
            best.append(row)

    print(f'\n  Top candidates (score, idx, class, conf):')
    for row in best[:15]:
        score, idx, label, pred, conf, cn = row
        print(f'    idx={idx:5d}  conf={conf:.1%}  score={score:.3f}  {cn}')

    return [row[1] for row in best[:9]]  # return top-9 indices


# ─────────────────────────────────────────────────────────────────────────────
def make_reasoning_figure(run_dir, dataset, indices, class_names, out_path,
                          title, device, n_concepts=3):
    """
    For each index: [original | concept1 | concept2 | concept3 | bar chart]
    Labels sit below each panel, not overlapping the image.
    """
    backbone, hook, feat_norm, sae, head, cfg = load_model(run_dir, device)
    n = len(indices)

    # Taller rows + more bottom margin so xlabel text has room
    fig = plt.figure(figsize=(16, 5.0 * n))
    outer = gridspec.GridSpec(n, 1, figure=fig, hspace=0.70,
                              left=0.03, right=0.97, top=0.93, bottom=0.04)

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for row, idx in enumerate(indices):
        img_t, label = dataset[idx]
        img_pil = pil_from_t(img_t)
        slic_224, z_sp_cpu, z_pool_cpu, W_y, pred, conf = run_infer(
            img_t, img_pil, backbone, hook, feat_norm, sae, head, cfg, device)

        cn = class_names[pred] if pred < len(class_names) else str(pred)
        short_cn = cn if len(cn) <= 24 else cn[:23] + '…'

        contrib     = (W_y * z_pool_cpu).numpy()
        top_idx     = np.argsort(contrib)[::-1][:max(n_concepts, 12)]
        top_contrib = contrib[top_idx]

        # Inner grid: [orig | c1 | c2 | c3 | bar]
        inner = gridspec.GridSpecFromSubplotSpec(
            1, n_concepts + 2, subplot_spec=outer[row],
            width_ratios=[1] * (n_concepts + 1) + [1.6],
            wspace=0.12)

        # ── Original image ────────────────────────────────────────────────────
        ax0 = fig.add_subplot(inner[0])
        ax0.imshow(img_pil)
        ax0.set_xticks([]); ax0.set_yticks([])
        for sp in ax0.spines.values():
            sp.set_visible(True); sp.set_edgecolor('#333'); sp.set_linewidth(1.5)
        # Label below the image via xlabel
        ax0.set_xlabel(f'{short_cn}\n{conf:.0%}',
                       fontsize=9, fontweight='bold', color='#1a1a1a',
                       labelpad=6)

        # ── Concept heatmap panels ─────────────────────────────────────────────
        for ci in range(n_concepts):
            kid  = int(top_idx[ci])
            cval = float(top_contrib[ci])
            c    = colors[ci % len(colors)]
            ax   = fig.add_subplot(inner[ci + 1])
            overlay = concept_heatmap_overlay(img_pil, slic_224, z_sp_cpu,
                                              kid, cfg.n_segments)
            ax.imshow(overlay)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor(c); sp.set_linewidth(2.5)
            ax.set_xlabel(f'Concept {ci + 1}  ({cval:.3f})',
                          fontsize=8.5, color=c, labelpad=6)

        # ── Contribution bar chart ─────────────────────────────────────────────
        ax_bar = fig.add_subplot(inner[n_concepts + 1])
        n_bars      = min(12, len(top_idx))
        bar_vals    = top_contrib[:n_bars]
        bar_labels  = [f'C{i+1}' for i in range(n_bars)]
        bar_colors  = [colors[i % len(colors)] if i < n_concepts else '#ddd'
                       for i in range(n_bars)]
        ax_bar.barh(range(n_bars)[::-1], bar_vals,
                    color=bar_colors, edgecolor='none', height=0.6)
        ax_bar.set_yticks(range(n_bars)[::-1])
        ax_bar.set_yticklabels(bar_labels, fontsize=8)
        ax_bar.set_xlabel('W · z  (contribution)', fontsize=8, labelpad=6)
        ax_bar.spines[['top', 'right']].set_visible(False)
        ax_bar.tick_params(axis='x', labelsize=7.5)
        ax_bar.set_title('Concept contributions', fontsize=8.5,
                         pad=8, color='#444')

        print(f'  row {row}: idx={idx}  {cn[:40]:<40}  conf={conf:.1%}')

    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.97)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan', action='store_true',
                        help='Scan dataset to find good candidate indices')
    parser.add_argument('--car_idx',    type=int, nargs='+', default=[279, 1059, 872])
    parser.add_argument('--flower_idx', type=int, nargs='+', default=[1244, 90, 913])
    parser.add_argument('--n_scan',     type=int, default=400)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Car-Best ──────────────────────────────────────────────────────────────
    print('\n=== Car-Best ===')
    car_run = os.path.join(_HERE, '..', 'models', 'car')
    with open(os.path.join(car_run, 'carbest_classes.json')) as f:
        car_class_names = json.load(f)['class_names']
    car_ds = ImageFolder(os.path.join(_VG_CBM, 'data', 'stanford_cars_hf_carbest', 'test'), transform=val_tf)

    if args.scan:
        car_idx = scan_dataset(car_run, car_ds, car_class_names, device,
                               n_scan=args.n_scan)
        print(f'\n  -> Use: --car_idx {" ".join(map(str, car_idx))}')
    else:
        car_idx = args.car_idx

    make_reasoning_figure(
        car_run, car_ds, car_idx, car_class_names,
        OUT_DIR / 'fig_F_reasoning_car.png',
        'F. CBM Reasoning — Car-Best  (acc = 73.7%)',
        device)

    # ── Flowers102 ────────────────────────────────────────────────────────────
    print('\n=== Flowers102 ===')
    flowers_run = os.path.join(_HERE, '..', 'models', 'flowers')
    flowers_ds  = torchvision.datasets.Flowers102(
        os.path.join(_VG_CBM, 'data'), split='test', download=False, transform=val_tf)

    if args.scan:
        flower_idx = scan_dataset(flowers_run, flowers_ds, FLOWERS_NAMES, device,
                                  n_scan=args.n_scan)
        print(f'\n  -> Use: --flower_idx {" ".join(map(str, flower_idx))}')
    else:
        flower_idx = args.flower_idx

    make_reasoning_figure(
        flowers_run, flowers_ds, flower_idx, FLOWERS_NAMES,
        OUT_DIR / 'fig_F_reasoning_flowers.png',
        'F. CBM Reasoning — Flowers102  (acc = 82.9%)',
        device)

    print('\nDone.')
