"""
datasets.py
-----------
SLIC-aware dataset wrappers and DataLoader builders for Flowers102 and Car-Best.
"""
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import torchvision

from PIL import Image

from .slic_utils import get_slic_224, SLIC_OK


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def make_transforms(img_size: int = 224):
    """
    Returns (aug_transform, val_transform, mean, std).
    ImageNet normalization. Training uses random crop + color jitter.
    """
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    aug = transforms.Compose([
        transforms.Resize(img_size,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val = transforms.Compose([
        transforms.Resize(img_size,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return aug, val, mean, std


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------

class SLICTrainDataset(Dataset):
    """
    Training dataset that returns (x, slic_224, label).
    SLIC label maps are computed once and cached to disk.
    """
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
        img_pil = item[0] if isinstance(item[0], Image.Image) \
                  else Image.fromarray(item[0])
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
    """
    Validation/test dataset that returns (x, slic_224, label).
    SLIC label maps are computed once and cached to disk.
    """
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
        img_pil = item[0] if isinstance(item[0], Image.Image) \
                  else Image.fromarray(item[0])
        label   = item[1]
        x       = self.val_tf(img_pil)

        if SLIC_OK and self.cache_dir is not None:
            cache_path = self.cache_dir / f"{idx:07d}.npy"
            slic_224   = get_slic_224(img_pil, cache_path,
                                       self.n_segments, self.compactness)
        else:
            slic_224 = torch.zeros(224, 224, dtype=torch.long)
        return x, slic_224, label


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_loaders(cfg):
    """
    Build train and test DataLoaders for the specified dataset.

    Supported datasets:
        'flowers102' — auto-downloaded via torchvision
        'car_best'   — requires data/stanford_cars_hf_carbest/ prepared via
                       data/prepare_carbest.py
    """
    aug_tf, val_tf, mean, std = make_transforms(cfg.img_size)
    root = cfg.data_root

    if cfg.dataset == 'flowers102':
        base_tr = torchvision.datasets.Flowers102(
            root, split='train', download=True, transform=None)
        base_te = torchvision.datasets.Flowers102(
            root, split='test',  download=True, transform=None)
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
        raise ValueError(f"Unknown dataset: {cfg.dataset}. "
                         f"Supported: 'flowers102', 'car_best'")

    # SLIC cache directories
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
        train_ds = Subset(train_ds,
                          list(range(min(cfg.train_limit, len(train_ds)))))
    if cfg.test_limit > 0:
        test_ds  = Subset(test_ds,
                          list(range(min(cfg.test_limit,  len(test_ds)))))

    pin = (cfg.num_workers > 0)
    g   = torch.Generator(); g.manual_seed(cfg.seed)
    tr_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin,
        worker_init_fn=lambda w: np.random.seed(cfg.seed + w),
        generator=g)
    te_loader = DataLoader(
        test_ds,  batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin)
    return tr_loader, te_loader, mean, std
