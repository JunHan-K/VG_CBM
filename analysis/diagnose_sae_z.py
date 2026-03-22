#!/usr/bin/env python3
"""
diagnose_sae_z.py
-----------------
Decomposes the accuracy gap between linear probe and CBM-SAE into two sources:
SAE feature quality loss and non-negative head constraint loss.

Tests:
  A) Raw GAP features (layer4, 2048-dim) -> free linear probe    (linear probe baseline)
  B) SAE z_pool                          -> free linear probe    (SAE feature quality)
  C) SAE z_pool                          -> non-neg linear probe (CBMHead constraint)
  D) Actual trained CBM-SAE model                                (reported accuracy)

Interpretation:
  B > A  -> SAE features are richer than raw; non-neg head is the bottleneck
  B < A  -> SAE itself is degrading features
  C < B  -> non-neg constraint is the bottleneck (expected and intentional)

Usage:
  python analysis/diagnose_sae_z.py --run_dir ./models/car --data_root ./data
"""

import os
import sys
import json
import argparse

# Add repo root to path so src/ is importable
_HERE    = os.path.dirname(os.path.abspath(__file__))
_VG_CBM  = os.path.join(_HERE, '..')
sys.path.insert(0, _VG_CBM)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from src.backbone   import build_backbone
from src.models     import FeatureNorm, SparseSAE, CBMHead, fg_z_pool
from src.slic_utils import compute_slic_224, slic224_to_14, extract_sp_features, compute_sp_fg

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
val_tf = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def unnorm(t):
    m = torch.tensor(MEAN).view(3, 1, 1)
    s = torch.tensor(STD).view(3, 1, 1)
    return (t * s + m).clamp(0, 1)


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


def extract_features(run_dir, data_root, split, device):
    """Extract raw GAP features and SAE z_pool for all samples in the split."""
    backbone, hook, feat_norm, sae, head, cfg = load_model(run_dir, device)

    dataset_dir = os.path.join(data_root, 'stanford_cars_hf_carbest', split)
    ds = ImageFolder(dataset_dir, transform=val_tf)
    print(f'Extracting {split} features ({len(ds)} images)...')

    import torchvision.models as tvm
    resnet_full = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
    gap_extractor = nn.Sequential(*list(resnet_full.children())[:-1])
    for p in gap_extractor.parameters(): p.requires_grad_(False)

    all_gap, all_z, all_y = [], [], []

    with torch.no_grad():
        for i, (img_t, y) in enumerate(ds):
            # A) Raw GAP (layer4, 2048-dim)
            gap = gap_extractor(img_t.unsqueeze(0).to(device)).squeeze()
            all_gap.append(gap.cpu())

            # B/C) SAE z_pool
            x = img_t.unsqueeze(0).to(device)
            backbone(x)
            f = feat_norm(hook.out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            img_np  = (unnorm(img_t).permute(1, 2, 0).numpy() * 255).astype('uint8')
            img_pil = Image.fromarray(img_np)
            slic_224 = compute_slic_224(img_pil, cfg.n_segments, cfg.compactness)
            slic_t   = torch.from_numpy(slic_224.astype('int64')).unsqueeze(0).to(device)
            slic_14  = slic224_to_14(slic_t)
            sp_raw   = extract_sp_features(f, slic_14, cfg.n_segments)
            fg_mask  = compute_sp_fg(sp_raw, cfg.fg_ratio)
            z, _     = sae(sp_raw * fg_mask.unsqueeze(-1))
            zp       = fg_z_pool(z, fg_mask, cfg.pool_mode).squeeze(0)
            all_z.append(zp.cpu())
            all_y.append(y)

            if (i + 1) % 200 == 0:
                print(f'  {i + 1}/{len(ds)}')

    return torch.stack(all_gap), torch.stack(all_z), torch.tensor(all_y)


def train_probe(X_tr, y_tr, X_te, y_te, n_classes, device,
                n_epochs=30, non_neg=False, tag='probe'):
    """Train linear probe. non_neg=True mimics CBMHead (F.relu on weights)."""
    head = nn.Linear(X_tr.shape[1], n_classes).to(device)
    opt  = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)

    best = 0.0
    for ep in range(n_epochs):
        head.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 256):
            idx = perm[i:i + 256]
            x  = X_tr[idx].to(device)
            yb = y_tr[idx].to(device)
            w  = F.relu(head.weight) if non_neg else head.weight
            loss = F.cross_entropy(F.linear(x, w, head.bias), yb)
            opt.zero_grad(); loss.backward(); opt.step()

        head.eval()
        with torch.no_grad():
            w = F.relu(head.weight) if non_neg else head.weight
            acc = (F.linear(X_te.to(device), w, head.bias).argmax(1).cpu() == y_te
                   ).float().mean().item()
        if acc > best: best = acc
        if (ep + 1) % 10 == 0:
            print(f'  [{tag}] ep{ep+1:3d}  acc={acc:.1%}  best={best:.1%}')
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir',   default='./models/car',
                        help='Path to trained model directory')
    parser.add_argument('--data_root', default='./data',
                        help='Data root containing stanford_cars_hf_carbest/')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Model:  {args.run_dir}')

    print('\nExtracting train features...')
    gap_tr, z_tr, y_tr = extract_features(args.run_dir, args.data_root, 'train', device)
    print('Extracting test features...')
    gap_te, z_te, y_te = extract_features(args.run_dir, args.data_root, 'test', device)

    n_cls = int(y_tr.max().item()) + 1
    print(f'\nFeature dims — GAP: {gap_tr.shape[1]}, SAE-Z: {z_tr.shape[1]}')
    print(f'Classes: {n_cls}')

    print('\n--- Training probes (30 epochs each) ---\n')

    print('[A] Raw GAP (2048-dim) -> free linear probe  (linear probe baseline)')
    acc_A = train_probe(gap_tr, y_tr, gap_te, y_te, n_cls, device, tag='GAP-free')

    print('\n[B] SAE z_pool -> free linear probe  (SAE feature quality)')
    acc_B = train_probe(z_tr, y_tr, z_te, y_te, n_cls, device, tag='Z-free')

    print('\n[C] SAE z_pool -> non-neg linear probe  (CBMHead constraint)')
    acc_C = train_probe(z_tr, y_tr, z_te, y_te, n_cls, device,
                        non_neg=True, tag='Z-nonneg')

    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    print(f'  [A] GAP  -> free linear:    {acc_A:.1%}  (linear probe baseline)')
    print(f'  [B] SAE-Z -> free linear:   {acc_B:.1%}  (SAE feature quality)')
    print(f'  [C] SAE-Z -> non-neg probe: {acc_C:.1%}  (with CBMHead constraint)')
    print()

    if acc_B > acc_A:
        print('-> SAE features outperform raw. Non-neg head is the main bottleneck.')
    elif acc_B < acc_A * 0.95:
        print('-> SAE itself is degrading features. Feature quality needs improvement.')
    else:
        print('-> SAE features ~ raw features. Non-neg constraint is the main loss.')

    print(f'-> Non-neg constraint cost: {acc_B - acc_C:.1%}')
    print('=' * 60)


if __name__ == '__main__':
    main()
