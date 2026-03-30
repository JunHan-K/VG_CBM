#!/usr/bin/env python3
"""
train.py — Superpixel-Native CBM-SAE (TopK + Non-negative Head)
================================================================
Trains a Sparse Autoencoder (SAE) on superpixel features extracted from a
frozen ResNet-50 backbone, followed by a non-negative linear CBMHead for
classification.

Supported datasets:
  flowers102  — auto-downloaded via torchvision
  car_best    — Stanford Cars subset (31 classes); see data/prepare_carbest.py

Usage:
  python train.py --dataset flowers102 --run_dir ./runs/flowers
  python train.py --dataset car_best   --data_root ./data --run_dir ./runs/car
"""

import os, json, time, random, argparse
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset

import torchvision
from torchvision import transforms

try:
    from skimage.segmentation import slic as _slic_fn
    SLIC_OK = True
except ImportError:
    SLIC_OK = False
    print("[WARNING] scikit-image not found. pip install scikit-image")

# ─── Concept colors ───────────────────────────────────────────────────────────
_COLORS = [
    (220,  50,  50),  # red
    ( 50, 180,  80),  # green
    ( 50, 120, 220),  # blue
    (220, 160,   0),  # orange
    (160,  50, 220),  # purple
    (  0, 190, 190),  # cyan
    (220, 100, 160),  # pink
    (100, 180,  50),  # lime
]

# ─── Utilities ────────────────────────────────────────────────────────────────
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def tensor_to_pil(img_t, mean, std):
    img = img_t.detach().cpu().float().clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return Image.fromarray((img.clamp(0,1) * 255).byte().permute(1,2,0).numpy())


# ─── SLIC cache ───────────────────────────────────────────────────────────────
def compute_slic_224(img_pil, n_segments, compactness):
    img_np = np.array(img_pil.resize((224, 224)))
    seg = _slic_fn(img_np, n_segments=n_segments,
                   compactness=compactness, start_label=0)
    return seg.astype(np.uint8)

def get_slic_224(img_pil, cache_path, n_segments, compactness):
    if cache_path.exists():
        arr = np.load(str(cache_path))
        if arr.shape == (224, 224):
            return torch.from_numpy(arr.astype(np.int64))
    arr = compute_slic_224(img_pil, n_segments, compactness)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), arr)
    return torch.from_numpy(arr.astype(np.int64))


# ─── Dataset ──────────────────────────────────────────────────────────────────
class SLICTrainDataset(Dataset):
    """Single augmented view + SLIC. Returns (x, slic_224, label)."""
    def __init__(self, base_ds, aug_tf, n_segments=80, compactness=10.0,
                 slic_cache_dir=None):
        self.base        = base_ds
        self.aug_tf      = aug_tf
        self.n_segments  = n_segments
        self.compactness = compactness
        self.cache_dir   = slic_cache_dir

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        item    = self.base[idx]
        img_pil = item[0] if isinstance(item[0], Image.Image) else Image.fromarray(item[0])
        label   = item[1]
        x       = self.aug_tf(img_pil)

        if SLIC_OK and self.cache_dir is not None:
            cache_path = self.cache_dir / f"{idx:07d}.npy"
            slic_224   = get_slic_224(img_pil, cache_path,
                                       self.n_segments, self.compactness)
        else:
            slic_224 = torch.zeros(224, 224, dtype=torch.long)
        return x, slic_224, label


class SLICValDataset(Dataset):
    """Val dataset with SLIC. Returns (x, slic_224, label)."""
    def __init__(self, base_ds, val_tf, n_segments=80, compactness=10.0,
                 slic_cache_dir=None):
        self.base        = base_ds
        self.val_tf      = val_tf
        self.n_segments  = n_segments
        self.compactness = compactness
        self.cache_dir   = slic_cache_dir

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        item    = self.base[idx]
        img_pil = item[0] if isinstance(item[0], Image.Image) else Image.fromarray(item[0])
        label   = item[1]
        x       = self.val_tf(img_pil)

        if SLIC_OK and self.cache_dir is not None:
            cache_path = self.cache_dir / f"{idx:07d}.npy"
            slic_224   = get_slic_224(img_pil, cache_path,
                                       self.n_segments, self.compactness)
        else:
            slic_224 = torch.zeros(224, 224, dtype=torch.long)
        return x, slic_224, label


# ─── Transforms ───────────────────────────────────────────────────────────────
def make_transforms(img_size=224):
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    aug = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return aug, val, mean, std


# ─── DataLoaders ──────────────────────────────────────────────────────────────
def build_loaders(cfg):
    aug_tf, val_tf, mean, std = make_transforms(cfg.img_size)
    root = cfg.data_root

    if cfg.dataset == 'flowers102':
        base_tr = torchvision.datasets.Flowers102(root, split='train',  download=True, transform=None)
        base_te = torchvision.datasets.Flowers102(root, split='test',   download=True, transform=None)
        cfg.num_classes = 102
    elif cfg.dataset == 'car_best':
        cars_root = os.path.join(root, 'stanford_cars_hf_carbest')
        if not os.path.isdir(cars_root):
            raise FileNotFoundError(
                f"Car-Best dataset not found at: {cars_root}\n"
                f"Run data/prepare_carbest.py first. See README for instructions.")
        from torchvision.datasets import ImageFolder
        base_tr = ImageFolder(os.path.join(cars_root, 'train'), transform=None)
        base_te = ImageFolder(os.path.join(cars_root, 'test'),  transform=None)
        cfg.num_classes = len(base_tr.classes)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}. Supported: 'flowers102', 'car_best'")

    # SLIC cache: shared with v1 if n_segments matches, else separate dir
    cache_tag  = f"n{cfg.n_segments}"
    cache_root = Path(cfg.slic_cache) / cfg.dataset / cache_tag
    tr_cache   = cache_root / 'train'
    te_cache   = cache_root / 'test'
    tr_cache.mkdir(parents=True, exist_ok=True)
    te_cache.mkdir(parents=True, exist_ok=True)

    train_ds = SLICTrainDataset(base_tr, aug_tf,
                                n_segments=cfg.n_segments,
                                compactness=cfg.compactness,
                                slic_cache_dir=tr_cache)
    test_ds  = SLICValDataset(base_te, val_tf,
                               n_segments=cfg.n_segments,
                               compactness=cfg.compactness,
                               slic_cache_dir=te_cache)

    if cfg.train_limit > 0:
        train_ds = Subset(train_ds, list(range(min(cfg.train_limit, len(train_ds)))))
    if cfg.test_limit > 0:
        test_ds  = Subset(test_ds,  list(range(min(cfg.test_limit,  len(test_ds)))))

    pin = (cfg.num_workers > 0)
    g   = torch.Generator(); g.manual_seed(cfg.seed)
    tr_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                           num_workers=cfg.num_workers, pin_memory=pin,
                           worker_init_fn=lambda w: np.random.seed(cfg.seed+w),
                           generator=g)
    te_loader = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=pin)
    return tr_loader, te_loader, mean, std


# ─── Backbone ─────────────────────────────────────────────────────────────────
class FeatureHook:
    def __init__(self): self.out = None
    def __call__(self, m, i, o): self.out = o

class ViTSpatialHook:
    def __init__(self): self.out = None
    def __call__(self, m, i, o):
        patches = o[:, 1:, :]
        B, N, C = patches.shape
        H = W = int(N ** 0.5)
        self.out = patches.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

def build_backbone(device, backbone='resnet50'):
    if backbone == 'vit_b16':
        try:    w = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        except: w = None
        m = (torchvision.models.vit_b_16(weights=w) if w else
             torchvision.models.vit_b_16(pretrained=True))
        m.eval()
        for p in m.parameters(): p.requires_grad_(False)
        hook = ViTSpatialHook()
        m.encoder.layers[-1].register_forward_hook(hook)
        return m.to(device), hook, 768
    else:
        try:    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        except: w = None
        m = (torchvision.models.resnet50(weights=w) if w else
             torchvision.models.resnet50(pretrained=True))
        m.eval()
        for p in m.parameters(): p.requires_grad_(False)
        hook = FeatureHook()
        m.layer3.register_forward_hook(hook)
        return m.to(device), hook, 1024


# ─── Superpixel feature extraction ────────────────────────────────────────────
def slic224_to_14(slic_224):
    return slic_224[:, 8::16, 8::16].contiguous()

def extract_sp_features(f, slic_14, n_segments):
    B, C, H, W = f.shape
    N        = H * W
    f_flat   = f.view(B, C, N).permute(0, 2, 1).contiguous()
    seg_flat = slic_14.view(B, N).long().clamp(0, n_segments - 1)
    sp_sum   = torch.zeros(B, n_segments, C, device=f.device, dtype=f.dtype)
    sp_cnt   = torch.zeros(B, n_segments,    device=f.device, dtype=f.dtype)
    seg_exp  = seg_flat.unsqueeze(-1).expand(-1, -1, C)
    sp_sum.scatter_add_(1, seg_exp, f_flat)
    sp_cnt.scatter_add_(1, seg_flat, torch.ones(B, N, device=f.device, dtype=f.dtype))
    return sp_sum / sp_cnt.unsqueeze(-1).clamp(min=1.0)


# ─── [V2-1] FG mask: deviation from image mean ────────────────────────────────
def compute_sp_fg(sp_feat, fg_ratio=0.45):
    """FG = superpixels whose features deviate most from the image mean.
    Background superpixels tend to be close to the image mean; foreground deviates.
    """
    global_mean = sp_feat.mean(dim=1, keepdim=True)          # (B, 1, C)
    deviation   = (sp_feat - global_mean).norm(dim=-1)        # (B, N_sp)
    thr = deviation.quantile(1.0 - fg_ratio, dim=-1, keepdim=True)
    return (deviation >= thr).float()


# ─── Models ───────────────────────────────────────────────────────────────────
class FeatureNorm(nn.Module):
    """Identity passthrough — LayerNorm removed to preserve feature magnitude info."""
    def __init__(self, C): super().__init__()
    def forward(self, x): return x

class SparseSAE(nn.Module):
    def __init__(self, C, K, topk=20):
        super().__init__()
        self.C = C; self.K = K; self.topk = topk
        self.enc_w = nn.Parameter(torch.empty(K, C))
        self.enc_b = nn.Parameter(torch.zeros(K))
        self.dec_w = nn.Parameter(torch.empty(C, K))
        self.dec_b = nn.Parameter(torch.zeros(C))
        nn.init.kaiming_normal_(self.enc_w, nonlinearity='relu')
        nn.init.normal_(self.dec_w, 0.0, 0.01)

    def forward(self, x):
        z_pre = x @ self.enc_w.t() + self.enc_b
        if self.topk > 0:
            # Hard TopK: exactly topk concepts active per superpixel
            topk_vals, topk_idx = z_pre.topk(self.topk, dim=-1)
            z = torch.zeros_like(z_pre)
            z.scatter_(-1, topk_idx, topk_vals.clamp(min=0))
        else:
            z = F.relu(z_pre)
        x_hat = z @ self.dec_w.t() + self.dec_b
        return z, x_hat

class CBMHead(nn.Module):
    """Non-negative linear head: weights constrained via ReLU at forward time.
    Ensures concept contributions are always non-negative → intervention monotonic.
    """
    def __init__(self, K, num_classes):
        super().__init__()
        self.fc = nn.Linear(K, num_classes)
        nn.init.xavier_uniform_(self.fc.weight.data.abs_())
        nn.init.zeros_(self.fc.bias)
    def forward(self, z_pool):
        return F.linear(z_pool, F.relu(self.fc.weight), self.fc.bias)


# ─── z_pool ───────────────────────────────────────────────────────────────────
def fg_z_pool(z_sp, fg, pool_mode='mean'):
    fg_k = fg.unsqueeze(-1)
    if pool_mode == 'max':
        # z_sp >= 0 (SAE ReLU output), bg superpixels zeroed → max over fg
        return (z_sp * fg_k).max(dim=1).values
    else:  # mean
        return (z_sp * fg_k).sum(1) / fg_k.sum(1).clamp(1)


# ─── [V2-3] Losses ────────────────────────────────────────────────────────────
def loss_sp_orth(z_sp, fg, n_pairs=128):
    """Spatial orthogonality via random concept pair sampling.
    Encourages different concepts to activate in different superpixels.
    Uses random pairs instead of full K×K matrix for memory efficiency.
    """
    B, N_sp, K = z_sp.shape
    z_fg = z_sp * fg.unsqueeze(-1)                    # (B, N_sp, K) BG zeroed

    # Random concept pairs
    idx_a = torch.randint(K, (n_pairs,), device=z_sp.device)
    idx_b = torch.randint(K, (n_pairs,), device=z_sp.device)

    z_a = z_fg[:, :, idx_a]                           # (B, N_sp, n_pairs)
    z_b = z_fg[:, :, idx_b]

    # Cosine similarity in superpixel space (dim=1 = N_sp dimension)
    # eps=1e-8 guards against zero-vector NaN for dead concepts
    na  = F.normalize(z_a, dim=1, eps=1e-8)
    nb  = F.normalize(z_b, dim=1, eps=1e-8)
    sim = (na * nb).sum(dim=1)                        # (B, n_pairs)
    loss = sim.clamp(min=0).mean()
    return torch.nan_to_num(loss, nan=0.0)            # NaN-safe guard


def loss_diversity(z_pool):
    """Cross-image diversity: concepts should vary across images.
    Low-variance concepts activate on every image (background-like) and are suppressed.
    unbiased=False prevents NaN when batch size is 1.
    """
    if z_pool.shape[0] < 2:
        return torch.tensor(0.0, device=z_pool.device)
    return -z_pool.var(dim=0, unbiased=False).mean()


# ─── Feature extraction helper ────────────────────────────────────────────────
@torch.no_grad()
def extract_features(backbone, hook, feat_norm, x, slic_224, cfg, device):
    backbone(x)
    f        = hook.out
    f        = feat_norm(f.permute(0,2,3,1)).permute(0,3,1,2)
    slic_14  = slic224_to_14(slic_224.to(device))
    sp_feat  = extract_sp_features(f, slic_14, cfg.n_segments)
    fg       = compute_sp_fg(sp_feat, cfg.fg_ratio)
    sp_feat  = sp_feat * fg.unsqueeze(-1)
    return sp_feat, fg


# ─── SAE training epoch (stage-wise) ──────────────────────────────────────────
def train_sae_epoch(backbone, hook, feat_norm, sae, opt, loader, cfg, device, epoch):
    sae.train(); feat_norm.train()
    stage2 = (epoch > cfg.stage2_start)
    tots   = {k: 0.0 for k in ('recon','l1','aux','sp_orth','div','total')}
    n_dead_total, n = 0, 0

    for x, slic_224, _ in loader:
        x = x.to(device)

        with torch.no_grad():
            backbone(x); f_raw = hook.out.clone()

        f_norm  = feat_norm(f_raw.permute(0,2,3,1)).permute(0,3,1,2)
        slic_14 = slic224_to_14(slic_224.to(device))
        sp_feat = extract_sp_features(f_norm, slic_14, cfg.n_segments)

        with torch.no_grad():
            fg = compute_sp_fg(sp_feat, cfg.fg_ratio)

        sp_in      = sp_feat * fg.unsqueeze(-1)
        z, sp_hat  = sae(sp_in)

        # ── Stage 1: Core SAE losses ──────────────────────────────────────────
        fg3     = fg.unsqueeze(-1)
        L_recon = ((sp_hat - sp_in)**2 * fg3).sum() / fg3.sum().clamp(1) * cfg.w_recon
        L_l1    = (z * fg3).sum() / fg3.sum().clamp(1) * cfg.w_l1

        mean_z  = z.mean(dim=(0,1))
        dead    = (mean_z < 1e-5).float()
        L_aux   = dead.mean() * cfg.w_aux

        loss = L_recon + L_l1 + L_aux

        # ── Stage 2: Spatial structure losses (added after stage2_start) ──────
        L_sp_orth = torch.tensor(0.0, device=device)
        L_div     = torch.tensor(0.0, device=device)
        if stage2:
            L_sp_orth = loss_sp_orth(z, fg) * cfg.w_sp_orth
            L_div     = loss_diversity(fg_z_pool(z, fg, cfg.pool_mode)) * cfg.w_div
            loss      = loss + L_sp_orth + L_div

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            n_dead_total += (z.detach().mean(dim=(0,1)) < 1e-5).sum().item()

        tots['recon']   += L_recon.item()
        tots['l1']      += L_l1.item()
        tots['aux']     += L_aux.item()
        tots['sp_orth'] += L_sp_orth.item()
        tots['div']     += L_div.item()
        tots['total']   += loss.item()
        n += 1

    dead_rate = n_dead_total / (n * sae.K) if n else 0
    return {k: v/n for k,v in tots.items()} | {'dead': dead_rate,
                                                'stage2': stage2}


# ─── Head training (z_pool cached) ───────────────────────────────────────────
@torch.no_grad()
def cache_z_pool(backbone, hook, feat_norm, sae, loader, cfg, device):
    sae.eval(); feat_norm.eval()
    all_z, all_y = [], []
    for x, slic_224, y in loader:
        sp_feat, fg = extract_features(backbone, hook, feat_norm,
                                        x.to(device), slic_224, cfg, device)
        z, _    = sae(sp_feat)
        z_pool  = fg_z_pool(z, fg, cfg.pool_mode)
        all_z.append(z_pool.cpu()); all_y.append(y.cpu())
    return torch.cat(all_z), torch.cat(all_y)


def train_head_epoch_cached(head, opt, z_cache, y_cache, cfg, device):
    head.train()
    perm    = torch.randperm(len(z_cache))
    z_cache = z_cache[perm]; y_cache = y_cache[perm]
    tot_ce, tot_acc, n = 0.0, 0.0, 0
    bs = cfg.batch_size
    for i in range(0, len(z_cache), bs):
        zp     = z_cache[i:i+bs].to(device)
        y      = y_cache[i:i+bs].to(device)
        logits = head(zp)
        loss   = F.cross_entropy(logits, y.long())
        if cfg.head_l1 > 0:
            loss = loss + cfg.head_l1 * head.fc.weight.abs().mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        tot_ce  += loss.item()
        tot_acc += (logits.argmax(1) == y).float().mean().item()
        n += 1
    return {'ce': tot_ce/n, 'acc': tot_acc/n}


@torch.no_grad()
def eval_head_cached(head, z_cache, y_cache, cfg, device):
    head.eval()
    tot, n = 0.0, 0
    for i in range(0, len(z_cache), cfg.batch_size):
        zp = z_cache[i:i+cfg.batch_size].to(device)
        y  = y_cache[i:i+cfg.batch_size].to(device)
        tot += (head(zp).argmax(1) == y).float().mean().item()
        n   += 1
    return {'acc': tot/n}


# ─── Eval metrics ─────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_acc(backbone, hook, feat_norm, sae, head, loader, cfg, device):
    sae.eval(); feat_norm.eval(); head.eval()
    tot, n = 0.0, 0
    for x, slic_224, y in loader:
        sp_feat, fg = extract_features(backbone, hook, feat_norm,
                                        x.to(device), slic_224, cfg, device)
        z, _    = sae(sp_feat)
        logits  = head(fg_z_pool(z, fg, cfg.pool_mode))
        tot += (logits.argmax(1) == y.to(device)).float().mean().item(); n += 1
    return tot / n


@torch.no_grad()
def compute_K095(backbone, hook, feat_norm, sae, head, loader, cfg, device, coverage=0.95):
    sae.eval(); feat_norm.eval(); head.eval()
    ks = []
    for x, slic_224, y in loader:
        sp_feat, fg = extract_features(backbone, hook, feat_norm,
                                        x.to(device), slic_224, cfg, device)
        z, _    = sae(sp_feat)
        zp      = fg_z_pool(z, fg, cfg.pool_mode)
        W       = head.fc.weight
        for i in range(zp.shape[0]):
            S        = (W[y[i].item()] * zp[i]).abs()
            S_sorted = S.sort(descending=True).values
            cumsum   = S_sorted.cumsum(0) / S_sorted.sum().clamp(1e-12)
            ks.append((cumsum < coverage).sum().item() + 1)
    return float(np.mean(ks))


@torch.no_grad()
def compute_intervention(backbone, hook, feat_norm, sae, head, loader, cfg, ms, device):
    sae.eval(); feat_norm.eval(); head.eval()
    results = {}
    for m in ms:
        orig_scores, drop_scores = [], []
        for x, slic_224, y in loader:
            sp_feat, fg = extract_features(backbone, hook, feat_norm,
                                            x.to(device), slic_224, cfg, device)
            z, _    = sae(sp_feat)
            zp      = fg_z_pool(z, fg, cfg.pool_mode)
            y_dev   = y.to(device)
            orig    = (head(zp).argmax(1) == y_dev).float().mean().item()
            W_y     = F.relu(head.fc.weight[y_dev])   # non-negative (matches head)
            contrib = W_y * zp                         # all >= 0
            top_m   = contrib.topk(m, dim=1).indices
            zp_int  = zp.clone().scatter_(1, top_m, 0.0)
            interv  = (head(zp_int).argmax(1) == y_dev).float().mean().item()
            orig_scores.append(orig); drop_scores.append(orig - interv)
        results[m] = {'orig': float(np.mean(orig_scores)),
                      'drop': float(np.mean(drop_scores))}
    return results


# ─── Visualization ────────────────────────────────────────────────────────────
def concept_map_slic(img_pil, z_sp, fg_sp, slic_224, color_rgb,
                     alpha_max=0.65, thresh=0.25):
    z_fg  = (z_sp * fg_sp).cpu().numpy().astype(np.float32)
    a_max = z_fg.max()
    if a_max < 1e-8:
        return img_pil.copy()
    z_norm  = z_fg / a_max
    seg     = slic_224.cpu().numpy().astype(np.int64).clip(0, len(z_norm) - 1)
    act_map = z_norm[seg]
    alpha   = np.clip((act_map - thresh) / (1.0 - thresh + 1e-8), 0, 1) * alpha_max
    img_np  = np.array(img_pil.resize((224, 224))).astype(np.float32)
    col     = np.array(color_rgb, dtype=np.float32)
    out     = img_np * (1 - alpha[:,:,None]) + col[None,None,:] * alpha[:,:,None]
    return Image.fromarray(out.clip(0,255).astype(np.uint8))


@torch.no_grad()
def visualize_batch(backbone, hook, feat_norm, sae, head, loader, cfg,
                    mean, std, device, tag='', n_images=8):
    sae.eval(); feat_norm.eval()
    if head is not None: head.eval()

    vis_dir = os.path.join(cfg.run_dir, 'visuals', tag)
    ensure_dir(vis_dir)

    for x, slic_224, y in loader: break

    x_vis = x[:n_images].to(device)
    s_vis = slic_224[:n_images]
    y_vis = y[:n_images]
    B     = x_vis.shape[0]

    sp_feat, fg = extract_features(backbone, hook, feat_norm,
                                    x_vis, s_vis, cfg, device)
    z, _    = sae(sp_feat)
    zp      = fg_z_pool(z, fg, cfg.pool_mode)
    preds   = head(zp).argmax(1) if head else y_vis.to(device)

    cell = 224; pad = 8
    from PIL import ImageDraw

    for i in range(B):
        img_pil = tensor_to_pil(x_vis[i].cpu(), mean, std).resize((cell, cell))
        slic_i  = s_vis[i]
        fg_i    = fg[i].cpu()

        if head is not None:
            W_y = head.fc.weight[preds[i].item()]
            S   = (W_y * zp[i]).abs()
            S_h = S if S.sum() > 1e-6 else zp[i]
        else:
            S_h = zp[i]

        S_total    = S_h.sum().clamp_min(1e-12).item()
        sorted_idx = S_h.argsort(descending=True).cpu().tolist()

        selected_ks, selected_pcts, cumsum = [], [], 0.0
        for k in sorted_idx:
            contrib = max(S_h[k].item(), 0.0)
            cumsum += contrib
            selected_ks.append(k)
            selected_pcts.append(contrib / S_total)
            if len(selected_ks) >= cfg.vis_max_concepts: break
            if cumsum / S_total >= cfg.vis_coverage:     break

        # Diversity filter
        div_ks, div_pcts, div_maps = [], [], []
        for k, pct in zip(selected_ks, selected_pcts):
            zm = z[i, :, k].cpu() * fg_i
            zn = F.normalize(zm.unsqueeze(0), dim=1).squeeze()
            if not any((zn * prev).sum().item() > 0.85 for prev in div_maps):
                div_ks.append(k); div_pcts.append(pct); div_maps.append(zn)
        selected_ks, selected_pcts = div_ks, div_pcts
        n_shown = len(selected_ks)

        n_cols = n_shown + 2
        grid   = Image.new('RGB', (cell * n_cols + pad * n_cols, cell + 20), (20,20,20))
        grid.paste(img_pil, (0, 0))
        draw = ImageDraw.Draw(grid)
        draw.text((2, cell+2), f"y={y_vis[i].item()} p={preds[i].item()} "
                               f"cov={cumsum/S_total:.0%}({n_shown}c)",
                  fill=(200,200,200))

        for ci, (k, pct) in enumerate(zip(selected_ks, selected_pcts)):
            z_sp  = z[i, :, k].cpu()
            color = _COLORS[ci % len(_COLORS)]
            blend = concept_map_slic(img_pil, z_sp, fg_i, slic_i, color)
            x_off = (ci + 1) * (cell + pad)
            grid.paste(blend, (x_off, 0))
            d2 = ImageDraw.Draw(grid)
            r,g2,b = color
            d2.rectangle([x_off, cell+1, x_off+12, cell+14], fill=(r,g2,b))
            d2.text((x_off+15, cell+2), f"k{k} {pct:.1%}", fill=(200,200,200))

        # Combined overlay
        combined = np.array(img_pil).astype(np.float32)
        for ci, k in enumerate(selected_ks):
            z_sp = z[i, :, k].cpu()
            z_fg = (z_sp * fg_i).numpy().astype(np.float32)
            a_mx = z_fg.max()
            if a_mx < 1e-8: continue
            z_n  = z_fg / a_mx
            seg  = slic_i.numpy().astype(np.int64).clip(0, len(z_n)-1)
            act  = z_n[seg]
            alpha = np.clip((act - 0.25) / 0.75, 0, 1) * 0.60
            col  = np.array(_COLORS[ci % len(_COLORS)], dtype=np.float32)
            combined = combined * (1 - alpha[:,:,None]) + col[None,None,:] * alpha[:,:,None]
        comb_pil = Image.fromarray(combined.clip(0,255).astype(np.uint8))
        x_off_c  = (n_shown + 1) * (cell + pad)
        grid.paste(comb_pil, (x_off_c, 0))
        ImageDraw.Draw(grid).text((x_off_c+2, cell+2), 'combined', fill=(180,180,180))

        fname = os.path.join(vis_dir, f"img{i:02d}_y{y_vis[i].item()}_p{preds[i].item()}.png")
        grid.save(fname)

    print(f"  [VIS] {B} images → {vis_dir}/")


@torch.no_grad()
def compute_gallery(backbone, hook, feat_norm, sae, loader, cfg, mean, std, device):
    sae.eval(); feat_norm.eval()
    print(f"  Building gallery ({cfg.gallery_n_concepts} concepts × {cfg.gallery_n_images} imgs)...")

    all_zp = []
    for x, slic_224, _ in loader:
        sp_feat, fg = extract_features(backbone, hook, feat_norm,
                                        x.to(device), slic_224, cfg, device)
        z, _ = sae(sp_feat)
        all_zp.append(fg_z_pool(z, fg, cfg.pool_mode).cpu())
    all_zp = torch.cat(all_zp, 0)

    mean_act  = all_zp.mean(0)
    top_k_ids = mean_act.topk(cfg.gallery_n_concepts).indices.tolist()
    needed    = set()
    for k in top_k_ids:
        needed.update(all_zp[:, k].topk(cfg.gallery_n_images).indices.tolist())

    img_cache = {}; z_cache = {}; fg_cache = {}; slic_cache = {}
    offset = 0
    for x, slic_224, _ in loader:
        B   = len(x)
        loc = [i for i in range(B) if (offset+i) in needed]
        if loc:
            sp_feat, fg = extract_features(backbone, hook, feat_norm,
                                            x[loc].to(device), slic_224[loc], cfg, device)
            z, _ = sae(sp_feat)
            for j, li in enumerate(loc):
                gid = offset + li
                img_cache[gid]  = x[li].cpu()
                z_cache[gid]    = z[j].cpu()
                fg_cache[gid]   = fg[j].cpu()
                slic_cache[gid] = slic_224[li].cpu()
        offset += B

    gal_dir = os.path.join(cfg.run_dir, 'concept_gallery')
    ensure_dir(gal_dir); cell = 140

    for rank, k in enumerate(top_k_ids):
        top_ids = all_zp[:, k].topk(cfg.gallery_n_images).indices.tolist()
        m       = cfg.gallery_n_images
        grid    = Image.new('RGB', (cell*m + 4*(m-1), cell), (255,255,255))
        color   = _COLORS[rank % len(_COLORS)]
        for ci, gid in enumerate(top_ids):
            if gid not in img_cache: continue
            img_pil = tensor_to_pil(img_cache[gid], mean, std).resize((cell, cell))
            blend   = concept_map_slic(img_pil, z_cache[gid][:, k],
                                        fg_cache[gid], slic_cache[gid], color)
            grid.paste(blend.resize((cell,cell)), (ci*(cell+4), 0))
        fname = os.path.join(gal_dir, f"rank{rank:02d}_k{k}_mean{mean_act[k]:.3f}.png")
        grid.save(fname)

    print(f"  [GALLERY] → {gal_dir}/")
    return top_k_ids


# ─── Config ───────────────────────────────────────────────────────────────────
@dataclass
class CFG:
    dataset:      str   = 'flowers102'
    data_root:    str   = './data'
    run_dir:      str   = './runs'
    backbone:     str   = 'resnet50'
    batch_size:   int   = 64
    num_workers:  int   = 0
    train_limit:  int   = 0
    test_limit:   int   = 0
    img_size:     int   = 224
    seed:         int   = 42
    # Architecture
    d_in:         int   = 1024
    expansion:    int   = 2
    num_classes:  int   = 102
    # SLIC
    n_segments:   int   = 80
    compactness:  float = 10.0
    slic_cache:   str   = './slic_cache_316'
    fg_ratio:     float = 0.30    # (0.45→0.30 stronger background suppression)
    # SAE training
    sae_epochs:   int   = 20
    sae_lr:       float = 3e-4
    stage2_start: int   = 10      # ep 1~10: Stage1 / ep 11~20: Stage2
    topk:         int   = 20      # TopK SAE: exactly k concepts active per superpixel (0=ReLU)
    w_recon:      float = 1.0
    w_l1:         float = 0.05    # magnitude penalty on active TopK values
    w_aux:        float = 0.003
    w_sp_orth:    float = 0.20    # Stage2 only (0.10→0.20 stronger spatial separation)
    w_div:        float = 0.02    # Stage2 only
    # Head training
    head_epochs:  int   = 30
    head_lr:      float = 1e-3
    head_l1:      float = 0.06
    pool_mode:    str   = 'mean'  # 'mean' or 'max' for fg_z_pool
    # Checkpoint
    load_sae_from:  str  = ''
    eval_only:      bool = False
    # Vis
    vis_coverage:       float    = 0.80
    vis_max_concepts:   int      = 8
    gallery_n_concepts: int      = 30
    gallery_n_images:   int      = 8
    intervention_ms:    List[int] = field(default_factory=lambda: [10, 25, 50])


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',       default='flowers102')
    parser.add_argument('--data_root',     default='./data')
    parser.add_argument('--run_dir',       default='./runs')
    parser.add_argument('--backbone',      default='resnet50')
    parser.add_argument('--batch_size',    type=int,   default=64)
    parser.add_argument('--num_workers',   type=int,   default=0)
    parser.add_argument('--train_limit',   type=int,   default=0)
    parser.add_argument('--test_limit',    type=int,   default=0)
    parser.add_argument('--sae_epochs',    type=int,   default=20)
    parser.add_argument('--head_epochs',   type=int,   default=30)
    parser.add_argument('--expansion',     type=int,   default=2)
    parser.add_argument('--n_segments',    type=int,   default=80)
    parser.add_argument('--fg_ratio',      type=float, default=0.30)
    parser.add_argument('--stage2_start',  type=int,   default=10)
    parser.add_argument('--topk',          type=int,   default=20)
    parser.add_argument('--w_l1',          type=float, default=0.05)
    parser.add_argument('--w_sp_orth',     type=float, default=0.20)
    parser.add_argument('--w_div',         type=float, default=0.02)
    parser.add_argument('--head_l1',       type=float, default=0.06)
    parser.add_argument('--pool_mode',     default='mean', choices=['mean', 'max'])
    parser.add_argument('--load_sae_from', default='')
    parser.add_argument('--eval_only',     action='store_true')
    args = parser.parse_args()

    cfg = CFG(**{k: v for k, v in vars(args).items() if hasattr(CFG, k)})
    cfg.slic_cache = './slic_cache_316'

    set_seed(cfg.seed)
    ensure_dir(cfg.run_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Dataset: {cfg.dataset}  |  Backbone: {cfg.backbone}")
    print(f"  Stage2 starts at ep {cfg.stage2_start+1}  |  n_segments={cfg.n_segments}  |  fg_ratio={cfg.fg_ratio}")

    tr_loader, te_loader, mean, std = build_loaders(cfg)
    print(f"  Train: {len(tr_loader.dataset)}  Test: {len(te_loader.dataset)}"
          f"  Classes: {cfg.num_classes}")

    backbone, hook, d_in = build_backbone(device, cfg.backbone)
    cfg.d_in = d_in

    feat_norm = FeatureNorm(cfg.d_in).to(device)
    K         = cfg.d_in * cfg.expansion
    sae       = SparseSAE(cfg.d_in, K, topk=cfg.topk).to(device)
    head      = CBMHead(K, cfg.num_classes).to(device)

    with open(os.path.join(cfg.run_dir, 'config.json'), 'w') as f:
        json.dump(asdict(cfg), f, indent=2)

    # ── Load checkpoints (eval_only or load_sae_from) ─────────────────────────
    if cfg.eval_only or cfg.load_sae_from:
        sae_path  = cfg.load_sae_from or os.path.join(cfg.run_dir, 'sae_checkpoint.pt')
        head_path = os.path.join(cfg.run_dir, 'head_best.pt')
        ckpt = torch.load(sae_path, map_location=device, weights_only=False)
        if 'feat_norm' in ckpt:
            feat_norm.load_state_dict(ckpt['feat_norm'], strict=False)
        sae.load_state_dict(ckpt['sae'])
        if os.path.exists(head_path):
            hckpt = torch.load(head_path, map_location=device, weights_only=False)
            head.load_state_dict(hckpt['head'])
        skip_sae  = True
        skip_head = cfg.eval_only
    else:
        skip_sae = skip_head = False

    # ── Phase 1: SAE (Stage-wise) ─────────────────────────────────────────────
    if not skip_sae:
        print(f"\n{'='*60}\nPhase 1: SAE Training (stage-wise)\n{'='*60}")
        sae_opt     = torch.optim.Adam(sae.parameters(), lr=cfg.sae_lr)
        # feat_norm has no learnable params (affine=False), so not included
        sae_history = []

        for ep in range(1, cfg.sae_epochs + 1):
            t0   = time.time()
            logs = train_sae_epoch(backbone, hook, feat_norm, sae, sae_opt,
                                   tr_loader, cfg, device, ep)
            stage_tag = "S2" if logs['stage2'] else "S1"
            print(f"[SAE {ep:02d}/{cfg.sae_epochs}|{stage_tag}] {time.time()-t0:.0f}s | "
                  f"recon={logs['recon']:.4f} l1={logs['l1']:.4f} "
                  f"sp_orth={logs['sp_orth']:.4f} div={logs['div']:.4f} "
                  f"dead={logs['dead']:.3f}")
            sae_history.append({'epoch': ep, **logs})

            if ep % 5 == 0:
                visualize_batch(backbone, hook, feat_norm, sae, None,
                                te_loader, cfg, mean, std, device, tag=f'sae_ep{ep:02d}')

        torch.save({'feat_norm': feat_norm.state_dict(), 'sae': sae.state_dict()},
                   os.path.join(cfg.run_dir, 'sae_checkpoint.pt'))
        with open(os.path.join(cfg.run_dir, 'sae_history.json'), 'w') as f:
            json.dump(sae_history, f, indent=2)

    # ── Phase 2: Head (cached) ────────────────────────────────────────────────
    if not skip_head:
        print(f"\n{'='*60}\nPhase 2: Head Training (cached z_pool)\n{'='*60}")
        for p in feat_norm.parameters(): p.requires_grad_(False)
        for p in sae.parameters():       p.requires_grad_(False)

        print("  Caching train z_pool...")
        tr_z, tr_y = cache_z_pool(backbone, hook, feat_norm, sae, tr_loader, cfg, device)
        print(f"  {len(tr_z)} samples  {tr_z.element_size()*tr_z.nelement()/1e6:.0f} MB")
        print("  Caching val z_pool...")
        val_z, val_y = cache_z_pool(backbone, hook, feat_norm, sae, te_loader, cfg, device)

        head_opt   = torch.optim.Adam(head.parameters(), lr=cfg.head_lr)
        head_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            head_opt, T_max=cfg.head_epochs, eta_min=1e-5)
        best_acc   = 0.0
        head_hist  = []

        for ep in range(1, cfg.head_epochs + 1):
            t0  = time.time()
            tr  = train_head_epoch_cached(head, head_opt, tr_z, tr_y, cfg, device)
            val = eval_head_cached(head, val_z, val_y, cfg, device)
            head_sched.step()
            print(f"[Head {ep:02d}/{cfg.head_epochs}] {time.time()-t0:.0f}s | "
                  f"ce={tr['ce']:.4f} tr_acc={tr['acc']:.4f} val_acc={val['acc']:.4f}")
            head_hist.append({'epoch': ep, **tr, **{f'val_{k}': v for k,v in val.items()}})
            if val['acc'] >= best_acc:
                best_acc = val['acc']
                torch.save({'head': head.state_dict()},
                           os.path.join(cfg.run_dir, 'head_best.pt'))

        with open(os.path.join(cfg.run_dir, 'head_history.json'), 'w') as f:
            json.dump(head_hist, f, indent=2)
        ckpt = torch.load(os.path.join(cfg.run_dir, 'head_best.pt'),
                          map_location=device, weights_only=False)
        head.load_state_dict(ckpt['head'])
        print(f"\n[Phase 2 Done] best_val_acc={best_acc:.4f}")

    # ── Final evaluation ───────────────────────────────────────────────────────
    print(f"\n{'='*60}\nFinal Evaluation\n{'='*60}")
    test_acc = eval_acc(backbone, hook, feat_norm, sae, head, te_loader, cfg, device)
    K095     = compute_K095(backbone, hook, feat_norm, sae, head, te_loader, cfg, device)
    interv   = compute_intervention(backbone, hook, feat_norm, sae, head,
                                     te_loader, cfg, cfg.intervention_ms, device)

    print(f"test_acc : {test_acc:.4f}")
    print(f"K_0.95   : {K095:.1f}")
    for m in cfg.intervention_ms:
        print(f"Interv@{m:2d}: drop={interv[m]['drop']:.4f}")

    summary = {'acc': test_acc, 'K_0.95': K095,
               'intervention': {str(m): interv[m] for m in cfg.intervention_ms},
               'n_segments': cfg.n_segments, 'fg_ratio': cfg.fg_ratio,
               'stage2_start': cfg.stage2_start}
    with open(os.path.join(cfg.run_dir, 'final_stats.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nGenerating final visualizations...")
    visualize_batch(backbone, hook, feat_norm, sae, head,
                    te_loader, cfg, mean, std, device, tag='final')
    compute_gallery(backbone, hook, feat_norm, sae, te_loader, cfg, mean, std, device)

    print(f"\n[DONE] → {cfg.run_dir}/")


if __name__ == '__main__':
    main()
