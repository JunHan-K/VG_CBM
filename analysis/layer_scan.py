#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAE as Concept/Feature Extractor (Backbone Frozen) — ResNet50 vs ViT-B/16 (CIFAR-100)

What this script does
- Backbone: frozen (ResNet50 ImageNet, or ViT-B/16 ImageNet)
- Dataset: CIFAR-100 (resize to 224)
- Extract intermediate features (layer2/3/4 for ResNet, block3/6/12 for ViT by default)
- Cache pooled features to disk
- Train Sparse Autoencoder (ReLU + L1) with fixed:
    expansion=8, K=d_in*8, l1_lambda=2.37e-02
- Linear probe on:
    (a) raw backbone pooled features
    (b) SAE code Z
- Save per-run artifacts + a summary CSV

Run examples
1) ResNet50 (default layers):
python sae_extractor_resnet_vit_cifar100.py --backbone resnet50 --expansion 8 --l1_lambda 2.37e-2

2) ViT-B/16 (default layers):
python sae_extractor_resnet_vit_cifar100.py --backbone vit_b16 --expansion 8 --l1_lambda 2.37e-2

3) Run BOTH backbones in one go:
python sae_extractor_resnet_vit_cifar100.py --backbones resnet50 vit_b16 --expansion 8 --l1_lambda 2.37e-2

Notes
- For ViT pooling, default is CLS token ("cls"). You can switch to mean over tokens ("mean").
"""

import os
import math
import json
import time
import random
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import torchvision
import torchvision.transforms as T


# -----------------------------
# Repro / utilities
# -----------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    seed: int = 0
    device: str = "cuda"
    num_workers: int = 4

    dataset: str = "cifar100"
    data_root: str = "./data"

    # backbone
    backbone: str = "resnet50"  # "resnet50" | "vit_b16"
    imagenet_pretrained: bool = True
    backbone_train: bool = False  # frozen by default

    # layers to probe (set by args; if None, choose defaults per backbone)
    layers: Tuple[str, ...] = ()

    # feature cache
    cache_root: str = "./feat_cache"
    cache_dtype: str = "float16"  # float16 saves disk; SAE trains in float32
    use_memmap: bool = True

    # feature pooling
    pool_resnet: str = "gap"   # "gap" or "flatten" (resnet blocks are (B,C,H,W))
    pool_vit: str = "cls"      # "cls" or "mean"   (vit blocks are (B,T,D))

    # standardization
    standardize: str = "zscore"  # "none" | "zscore"
    eps: float = 1e-6

    # SAE
    expansion: int = 8
    sae_epochs: int = 30
    sae_batch: int = 512
    sae_lr: float = 1e-3
    weight_decay: float = 0.0
    tied_decoder: bool = False
    bias_init: float = 0.0

    # Fixed L1 lambda (no sweep here; you asked to fix it)
    l1_lambda: float = 2.37e-2

    # linear probe on features / on Z
    probe_epochs: int = 10
    probe_lr: float = 5e-2
    probe_batch: int = 512
    probe_weight_decay: float = 0.0

    # metrics
    active_thr: float = 0.0
    dead_thr: float = 1e-5

    # artifact
    runs_root: str = "./runs_sae_extractor"
    save_topk_examples: bool = True
    topk_per_feature: int = 5


# -----------------------------
# Backbone + hooks
# -----------------------------
class ResNetFeatureExtractor(nn.Module):
    """
    ResNet50 backbone with forward hooks to capture intermediate layer outputs.
    Captures: layer1..layer4 outputs (B,C,H,W)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = torchvision.models.resnet50(weights=weights)
        self.backbone = m
        self.backbone.fc = nn.Identity()

        self._acts: Dict[str, torch.Tensor] = {}
        self._hooks = []

        layer_map = {
            "layer1": self.backbone.layer1,
            "layer2": self.backbone.layer2,
            "layer3": self.backbone.layer3,
            "layer4": self.backbone.layer4,
        }
        for lname, lmod in layer_map.items():
            self._hooks.append(lmod.register_forward_hook(self._make_hook(lname)))

    def _make_hook(self, name: str):
        def hook(_module, _inp, out):
            self._acts[name] = out
        return hook

    @torch.no_grad()
    def forward(self, x):
        _ = self.backbone(x)
        return self._acts

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


class ViTFeatureExtractor(nn.Module):
    """
    ViT-B/16 backbone with forward hooks to capture intermediate block outputs.
    Captures: block1..block12 outputs from encoder.layers[i] (B,T,D)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.vit_b_16(weights=weights)
        m.heads = nn.Identity()
        self.backbone = m

        self._acts: Dict[str, torch.Tensor] = {}
        self._hooks = []

        # torchvision ViT: encoder.layers is a ModuleList of TransformerEncoderLayer blocks
        for idx, blk in enumerate(self.backbone.encoder.layers):
            name = f"block{idx+1}"
            self._hooks.append(blk.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inp, out):
            # out is typically (B, T, D)
            self._acts[name] = out
        return hook

    @torch.no_grad()
    def forward(self, x):
        _ = self.backbone(x)
        return self._acts

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def pool_resnet_features(feat: torch.Tensor, mode: str = "gap") -> torch.Tensor:
    # feat: (B,C,H,W)
    if mode == "gap":
        return feat.mean(dim=(2, 3))  # (B,C)
    elif mode == "flatten":
        return feat.flatten(1)        # (B,C*H*W)
    else:
        raise ValueError(f"Unknown pool_resnet: {mode}")


def pool_vit_features(feat: torch.Tensor, mode: str = "cls") -> torch.Tensor:
    # feat: (B,T,D)
    if mode == "cls":
        return feat[:, 0, :]          # (B,D)
    elif mode == "mean":
        return feat.mean(dim=1)       # (B,D)
    else:
        raise ValueError(f"Unknown pool_vit: {mode}")


# -----------------------------
# Dataset
# -----------------------------
def build_cifar100_loaders(root: str, batch: int, num_workers: int):
    tfm_train = T.Compose([
        T.Resize(224),
        T.RandomCrop(224, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    tfm_test = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=tfm_train)
    test  = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=tfm_test)

    train_loader = DataLoader(train, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# -----------------------------
# Feature caching
# -----------------------------
def np_dtype(dtype_str: str):
    if dtype_str == "float16":
        return np.float16
    if dtype_str == "float32":
        return np.float32
    raise ValueError(dtype_str)


def cache_paths(cfg: Config, layer: str) -> Dict[str, str]:
    ensure_dir(cfg.cache_root)
    tag = f"{cfg.dataset}_{cfg.backbone}_{layer}_pool={cfg.pool_resnet if cfg.backbone=='resnet50' else cfg.pool_vit}"
    return {
        "train_x": os.path.join(cfg.cache_root, f"{tag}_train_x.npy"),
        "train_y": os.path.join(cfg.cache_root, f"{tag}_train_y.npy"),
        "test_x":  os.path.join(cfg.cache_root, f"{tag}_test_x.npy"),
        "test_y":  os.path.join(cfg.cache_root, f"{tag}_test_y.npy"),
        "stats":   os.path.join(cfg.cache_root, f"{tag}_stats.json"),
    }


@torch.no_grad()
def build_feature_cache(cfg: Config, model: nn.Module, layer: str,
                        train_loader: DataLoader, test_loader: DataLoader) -> Tuple[int, Dict[str, str]]:
    paths = cache_paths(cfg, layer)

    if os.path.exists(paths["train_x"]) and os.path.exists(paths["train_y"]) and os.path.exists(paths["stats"]):
        st = json.load(open(paths["stats"], "r", encoding="utf-8"))
        return int(st["d_in"]), paths

    dtype = np_dtype(cfg.cache_dtype)

    def _extract(loader: DataLoader, split: str):
        xs = []
        ys = []
        t0 = time.time()
        for x, y in loader:
            x = x.to(cfg.device, non_blocking=True)
            acts = model(x)

            if layer not in acts:
                raise KeyError(f"Layer '{layer}' not found in captured activations. Available: {list(acts.keys())}")

            raw = acts[layer]
            if cfg.backbone == "resnet50":
                feat = pool_resnet_features(raw, cfg.pool_resnet)
            elif cfg.backbone == "vit_b16":
                feat = pool_vit_features(raw, cfg.pool_vit)
            else:
                raise ValueError(cfg.backbone)

            feat = feat.detach().float().cpu().numpy()  # float32
            xs.append(feat)
            ys.append(y.numpy())

        X = np.concatenate(xs, axis=0)  # (N, d_in)
        Y = np.concatenate(ys, axis=0)  # (N,)
        dt = time.time() - t0
        print(f"[Cache] backbone={cfg.backbone} layer={layer} split={split} feats={X.shape} time={dt:.1f}s")
        return X.astype(dtype), Y.astype(np.int64)

    Xtr, Ytr = _extract(train_loader, "train")
    Xte, Yte = _extract(test_loader, "test")

    np.save(paths["train_x"], Xtr)
    np.save(paths["train_y"], Ytr)
    np.save(paths["test_x"], Xte)
    np.save(paths["test_y"], Yte)

    d_in = int(Xtr.shape[1])
    st = {
        "dataset": cfg.dataset,
        "backbone": cfg.backbone,
        "layer": layer,
        "pool": cfg.pool_resnet if cfg.backbone == "resnet50" else cfg.pool_vit,
        "d_in": d_in,
        "cache_dtype": cfg.cache_dtype,
        "created": now_str(),
    }
    json.dump(st, open(paths["stats"], "w", encoding="utf-8"), indent=2)
    return d_in, paths


def load_cached_arrays(cfg: Config, paths: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xtr = np.load(paths["train_x"], mmap_mode="r" if cfg.use_memmap else None)
    Ytr = np.load(paths["train_y"], mmap_mode="r" if cfg.use_memmap else None)
    Xte = np.load(paths["test_x"],  mmap_mode="r" if cfg.use_memmap else None)
    Yte = np.load(paths["test_y"],  mmap_mode="r" if cfg.use_memmap else None)
    return Xtr, Ytr, Xte, Yte


# -----------------------------
# Standardization (critical for L1 behaving sanely)
# -----------------------------
def fit_standardizer(cfg: Config, X: np.ndarray) -> Dict[str, np.ndarray]:
    if cfg.standardize == "none":
        return {"mode": "none"}
    if cfg.standardize == "zscore":
        X32 = np.array(X, dtype=np.float32, copy=False)
        mu = X32.mean(axis=0)
        sd = X32.std(axis=0)
        sd = np.maximum(sd, cfg.eps)
        return {"mode": "zscore", "mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}
    raise ValueError(cfg.standardize)


def apply_standardizer(cfg: Config, X: torch.Tensor, st: Dict[str, np.ndarray]) -> torch.Tensor:
    if st["mode"] == "none":
        return X
    if st["mode"] == "zscore":
        mu = torch.from_numpy(st["mu"]).to(X.device)
        sd = torch.from_numpy(st["sd"]).to(X.device)
        return (X - mu) / sd
    raise ValueError(st["mode"])


# -----------------------------
# SAE model
# -----------------------------
class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_latent: int, tied_decoder: bool = False, bias_init: float = 0.0):
        super().__init__()
        self.d_in = d_in
        self.d_latent = d_latent
        self.tied_decoder = tied_decoder

        self.enc = nn.Linear(d_in, d_latent, bias=True)
        self.dec = nn.Linear(d_latent, d_in, bias=False)

        nn.init.kaiming_uniform_(self.enc.weight, a=math.sqrt(5))
        nn.init.zeros_(self.enc.bias)
        if bias_init != 0.0:
            self.enc.bias.data.fill_(bias_init)
        nn.init.kaiming_uniform_(self.dec.weight, a=math.sqrt(5))

    def forward(self, x):
        z_pre = self.enc(x)
        z = F.relu(z_pre)
        if self.tied_decoder:
            x_hat = z @ self.enc.weight  # (B,K) @ (K,d_in) = (B,d_in)
        else:
            x_hat = self.dec(z)
        return x_hat, z, z_pre


@torch.no_grad()
def sparsity_metrics(z: torch.Tensor, active_thr: float, dead_thr: float) -> Dict[str, float]:
    active = (z > active_thr).float().mean().item()
    fr = (z > active_thr).float().mean(dim=0)
    dead = (fr <= dead_thr).float().mean().item()

    zero_frac = (z == 0).float().mean().item()
    zmin = z.min().item()
    zmax = z.max().item()
    q = torch.quantile(z.flatten(), torch.tensor([0.0, 0.5, 0.9, 0.99], device=z.device)).tolist()

    return {
        "active_sample": float(active),
        "dead_feature_batchproxy": float(dead),
        "zero_frac": float(zero_frac),
        "zmin": float(zmin),
        "zmax": float(zmax),
        "q0": float(q[0]),
        "q50": float(q[1]),
        "q90": float(q[2]),
        "q99": float(q[3]),
    }


@torch.no_grad()
def dead_feature_full(z_all: torch.Tensor, active_thr: float, dead_thr: float) -> float:
    fr = (z_all > active_thr).float().mean(dim=0)
    return float((fr <= dead_thr).float().mean().item())


# -----------------------------
# Linear probe
# -----------------------------
class LinearHead(nn.Module):
    def __init__(self, d_in: int, num_classes: int = 100):
        super().__init__()
        self.fc = nn.Linear(d_in, num_classes)

    def forward(self, x):
        return self.fc(x)


def make_tensor_ds(X: np.ndarray, Y: np.ndarray) -> TensorDataset:
    X32 = torch.from_numpy(np.array(X, dtype=np.float32, copy=False))
    Yt = torch.from_numpy(np.array(Y, dtype=np.int64, copy=False))
    return TensorDataset(X32, Yt)


def train_linear_probe(cfg: Config, Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, Yte: np.ndarray,
                       d_in: int, out_dir: str, tag: str) -> float:
    device = cfg.device
    tr_ds = make_tensor_ds(Xtr, Ytr)
    te_ds = make_tensor_ds(Xte, Yte)
    tr_ld = DataLoader(tr_ds, batch_size=cfg.probe_batch, shuffle=True, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=cfg.probe_batch, shuffle=False, num_workers=0)

    model = LinearHead(d_in, 100).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.probe_lr, momentum=0.9, weight_decay=cfg.probe_weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.probe_epochs)

    best = 0.0
    for ep in range(1, cfg.probe_epochs + 1):
        model.train()
        for x, y in tr_ld:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in te_ld:
                x = x.to(device); y = y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        acc = 100.0 * correct / total
        best = max(best, acc)
        print(f"[{tag}] ep{ep:03d}/{cfg.probe_epochs} acc={acc:.2f}%")

    torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}_linear.pt"))
    return float(best)


# -----------------------------
# SAE training + encoding
# -----------------------------
def train_sae_extractor(cfg: Config,
                        Xtr: np.ndarray, Xte: np.ndarray,
                        d_in: int, out_dir: str) -> Dict[str, float]:
    device = cfg.device
    K = d_in * cfg.expansion
    l1_lambda = cfg.l1_lambda

    tr_ds = make_tensor_ds(Xtr, np.zeros((len(Xtr),), dtype=np.int64))
    te_ds = make_tensor_ds(Xte, np.zeros((len(Xte),), dtype=np.int64))
    tr_ld = DataLoader(tr_ds, batch_size=cfg.sae_batch, shuffle=True, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=cfg.sae_batch, shuffle=False, num_workers=0)

    sae = SparseAutoencoder(d_in, K, tied_decoder=cfg.tied_decoder, bias_init=cfg.bias_init).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=cfg.sae_lr, weight_decay=cfg.weight_decay)

    st = fit_standardizer(cfg, Xtr)
    json.dump(
        {
            "standardize": cfg.standardize,
            "l1_lambda": l1_lambda,
            "expansion": cfg.expansion,
            "d_in": d_in,
            "K": K,
        },
        open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8"),
        indent=2,
    )

    best_recon = 1e9
    for ep in range(1, cfg.sae_epochs + 1):
        sae.train()
        run_recon = 0.0
        run_l1 = 0.0
        n = 0

        for xb, _ in tr_ld:
            xb = xb.to(device)
            xb = apply_standardizer(cfg, xb, st)

            x_hat, z, _ = sae(xb)
            recon = F.mse_loss(x_hat, xb)
            l1 = z.abs().mean()
            loss = recon + l1_lambda * l1

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = xb.size(0)
            run_recon += recon.item() * bs
            run_l1 += l1.item() * bs
            n += bs

        sae.eval()
        with torch.no_grad():
            xb, _ = next(iter(te_ld))
            xb = xb.to(device)
            xb = apply_standardizer(cfg, xb, st)
            x_hat, z, _ = sae(xb)
            recon_te = F.mse_loss(x_hat, xb).item()
            met = sparsity_metrics(z, cfg.active_thr, cfg.dead_thr)

        avg_recon = run_recon / max(n, 1)
        avg_l1 = run_l1 / max(n, 1)
        best_recon = min(best_recon, recon_te)

        print(
            f"[SAE] ep{ep:03d}/{cfg.sae_epochs} "
            f"loss={(avg_recon + l1_lambda*avg_l1):.4f} recon={recon_te:.4f} "
            f"l1={avg_l1:.4f} active={met['active_sample']:.4f} "
            f"zero_frac={met['zero_frac']:.4f} dead(batchproxy)={met['dead_feature_batchproxy']:.4f} "
            f"z[q50,q90,q99]=({met['q50']:.4f},{met['q90']:.4f},{met['q99']:.4f})"
        )

    torch.save(sae.state_dict(), os.path.join(out_dir, "sae.pt"))
    json.dump({"mode": st["mode"]}, open(os.path.join(out_dir, "standardizer_meta.json"), "w", encoding="utf-8"), indent=2)

    # full test dead/active
    sae.eval()
    Z_chunks = []
    with torch.no_grad():
        for xb, _ in te_ld:
            xb = xb.to(device)
            xb = apply_standardizer(cfg, xb, st)
            _, z, _ = sae(xb)
            Z_chunks.append(z.detach().cpu())
    Zall = torch.cat(Z_chunks, dim=0)

    dead_full = dead_feature_full(Zall, cfg.active_thr, cfg.dead_thr)
    active_full = float((Zall > cfg.active_thr).float().mean().item())

    stats = {
        "K": int(K),
        "recon_test_best": float(best_recon),
        "active_sample_test": float(active_full),
        "dead_feature_test": float(dead_full),
    }
    json.dump(stats, open(os.path.join(out_dir, "sae_stats.json"), "w", encoding="utf-8"), indent=2)
    return stats


@torch.no_grad()
def encode_with_sae(cfg: Config, sae: SparseAutoencoder, st: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    device = cfg.device
    ds = make_tensor_ds(X, np.zeros((len(X),), dtype=np.int64))
    ld = DataLoader(ds, batch_size=cfg.sae_batch, shuffle=False, num_workers=0)
    outs = []
    sae.eval()
    for xb, _ in ld:
        xb = xb.to(device)
        xb = apply_standardizer(cfg, xb, st)
        _, z, _ = sae(xb)
        outs.append(z.detach().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


# -----------------------------
# Top-k examples per feature
# -----------------------------
def save_topk_examples_np(Z: np.ndarray, y: np.ndarray, out_dir: str, topk: int = 5):
    # Z: (N,K)
    top_idx = np.argsort(-Z, axis=0)[:topk, :]  # (topk,K)
    np.save(os.path.join(out_dir, "topk_indices.npy"), top_idx.astype(np.int32))
    top_lbl = y[top_idx]
    np.save(os.path.join(out_dir, "topk_labels.npy"), top_lbl.astype(np.int64))


# -----------------------------
# Defaults
# -----------------------------
def default_layers_for(backbone: str) -> List[str]:
    if backbone == "resnet50":
        return ["layer2", "layer3", "layer4"]
    if backbone == "vit_b16":
        return ["block3", "block6", "block12"]
    raise ValueError(backbone)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")

    # Run one or more backbones
    ap.add_argument("--backbone", type=str, default=None, choices=["resnet50", "vit_b16"])
    ap.add_argument("--backbones", type=str, nargs="*", default=None, choices=["resnet50", "vit_b16"])

    # Layers
    ap.add_argument("--layers", type=str, nargs="*", default=None)

    # pooling
    ap.add_argument("--pool_resnet", type=str, default="gap", choices=["gap", "flatten"])
    ap.add_argument("--pool_vit", type=str, default="cls", choices=["cls", "mean"])

    # SAE fixed hyperparams
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--l1_lambda", type=float, default=2.37e-2)
    ap.add_argument("--sae_epochs", type=int, default=30)
    ap.add_argument("--sae_lr", type=float, default=1e-3)
    ap.add_argument("--sae_batch", type=int, default=512)

    # Probe
    ap.add_argument("--probe_epochs", type=int, default=10)
    ap.add_argument("--probe_lr", type=float, default=5e-2)
    ap.add_argument("--probe_batch", type=int, default=512)

    # paths
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--cache_root", type=str, default="./feat_cache")
    ap.add_argument("--runs_root", type=str, default="./runs_sae_extractor")

    args = ap.parse_args()

    # Decide which backbones to run
    if args.backbones is not None and len(args.backbones) > 0:
        backbones = list(args.backbones)
    elif args.backbone is not None:
        backbones = [args.backbone]
    else:
        backbones = ["resnet50"]  # default if user passes nothing

    # Device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # data loaders for caching features
    cache_batch = 256
    train_loader, test_loader = build_cifar100_loaders(args.data_root, cache_batch, num_workers=4)

    # Global run root for this invocation
    run_id = f"run_{now_str()}_cifar100_fixedlambda={args.l1_lambda:.2e}_exp={args.expansion}"
    out_root = os.path.join(args.runs_root, run_id)
    ensure_dir(out_root)

    # Summary CSV
    csv_path = os.path.join(out_root, f"summary_{now_str()}.csv")
    rows = []

    for bb in backbones:
        cfg = Config(
            seed=args.seed,
            device=device,
            backbone=bb,
            imagenet_pretrained=True,
            layers=tuple(args.layers) if args.layers is not None and len(args.layers) > 0 else tuple(default_layers_for(bb)),
            pool_resnet=args.pool_resnet,
            pool_vit=args.pool_vit,
            expansion=args.expansion,
            l1_lambda=args.l1_lambda,
            sae_epochs=args.sae_epochs,
            sae_lr=args.sae_lr,
            sae_batch=args.sae_batch,
            probe_epochs=args.probe_epochs,
            probe_lr=args.probe_lr,
            probe_batch=args.probe_batch,
            data_root=args.data_root,
            cache_root=args.cache_root,
            runs_root=args.runs_root,
        )

        set_seed(cfg.seed)
        print(f"\n[Run] backbone={cfg.backbone} device={cfg.device} layers={cfg.layers}")
        print(f"[Fixed SAE] expansion={cfg.expansion}  K=d_in*{cfg.expansion}  l1_lambda={cfg.l1_lambda:.2e}")
        print(f"[Pool] resnet={cfg.pool_resnet} vit={cfg.pool_vit}")

        bb_dir = os.path.join(out_root, f"backbone={cfg.backbone}")
        ensure_dir(bb_dir)
        json.dump(asdict(cfg), open(os.path.join(bb_dir, "config.json"), "w", encoding="utf-8"), indent=2)

        # backbone model
        if cfg.backbone == "resnet50":
            model = ResNetFeatureExtractor(pretrained=cfg.imagenet_pretrained).to(cfg.device)
        elif cfg.backbone == "vit_b16":
            model = ViTFeatureExtractor(pretrained=cfg.imagenet_pretrained).to(cfg.device)
        else:
            raise ValueError(cfg.backbone)

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        for layer in cfg.layers:
            print("\n" + "=" * 90)
            print(f"[Backbone {cfg.backbone}] Layer {layer}")
            print("=" * 90)

            d_in, paths = build_feature_cache(cfg, model, layer, train_loader, test_loader)
            Xtr, Ytr, Xte, Yte = load_cached_arrays(cfg, paths)

            layer_dir = os.path.join(bb_dir, f"layer={layer}_d={d_in}")
            ensure_dir(layer_dir)

            # Baseline probe on raw pooled features
            print(f"\n[Baseline] Linear probe on pooled features d_in={d_in}")
            probe_feat_acc = train_linear_probe(cfg, Xtr, Ytr, Xte, Yte, d_in, layer_dir, tag="probe_feat")

            # Train SAE (fixed lambda)
            sae_dir = os.path.join(layer_dir, f"sae_exp={cfg.expansion}_l1={cfg.l1_lambda:.2e}".replace("+", ""))
            ensure_dir(sae_dir)

            print("\n" + "-" * 90)
            print(f"[SAE Train] d_in={d_in}  K={d_in*cfg.expansion}  l1_lambda={cfg.l1_lambda:.2e}")
            print("-" * 90)

            stats = train_sae_extractor(cfg, Xtr, Xte, d_in, sae_dir)

            # Load SAE again for encoding
            sae = SparseAutoencoder(d_in, d_in * cfg.expansion, tied_decoder=cfg.tied_decoder, bias_init=cfg.bias_init).to(cfg.device)
            sae.load_state_dict(torch.load(os.path.join(sae_dir, "sae.pt"), map_location=cfg.device))
            sae.eval()

            # Recompute standardizer deterministically from Xtr
            st = fit_standardizer(cfg, Xtr)

            # Encode Z
            Ztr = encode_with_sae(cfg, sae, st, Xtr)
            Zte = encode_with_sae(cfg, sae, st, Xte)

            # Probe on Z
            print(f"\n[Probe] Linear probe on SAE code Z (K={Ztr.shape[1]})")
            probe_z_acc = train_linear_probe(cfg, Ztr, Ytr, Zte, Yte, Ztr.shape[1], sae_dir, tag="probe_z")

            if cfg.save_topk_examples:
                save_topk_examples_np(Zte, np.array(Yte), sae_dir, topk=cfg.topk_per_feature)

            row = {
                "backbone": cfg.backbone,
                "layer": layer,
                "d_in": int(d_in),
                "expansion": int(cfg.expansion),
                "K": int(Ztr.shape[1]),
                "l1_lambda": float(cfg.l1_lambda),
                "pool": (cfg.pool_resnet if cfg.backbone == "resnet50" else cfg.pool_vit),
                "probe_feat_acc": float(probe_feat_acc),
                "probe_z_acc": float(probe_z_acc),
                "active_sample_test": float(stats["active_sample_test"]),
                "dead_feature_test": float(stats["dead_feature_test"]),
                "recon_test_best": float(stats["recon_test_best"]),
                "run_dir": sae_dir,
            }
            rows.append(row)

            # write csv progressively
            with open(csv_path, "w", encoding="utf-8") as f:
                keys = list(rows[0].keys())
                f.write(",".join(keys) + "\n")
                for r in rows:
                    f.write(",".join(str(r[k]) for k in keys) + "\n")

            print("-" * 90)
            print(
                f"[ROW] bb={cfg.backbone} layer={layer} "
                f"feat={probe_feat_acc:.2f}  z={probe_z_acc:.2f}  "
                f"active={stats['active_sample_test']:.3f} dead={stats['dead_feature_test']:.3f} "
                f"recon={stats['recon_test_best']:.4f}"
            )
            print("-" * 90)

        model.close()

    print(f"\n[Done] Summary CSV => {csv_path}")
    print(f"[Done] Run dir     => {out_root}")


if __name__ == "__main__":
    main()

