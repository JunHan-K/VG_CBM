"""
slic_utils.py
-------------
SLIC superpixel computation, caching, and superpixel-level feature pooling.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from skimage.segmentation import slic as _slic_fn
    SLIC_OK = True
except ImportError:
    SLIC_OK = False
    print("[WARNING] scikit-image not found. Install with: pip install scikit-image")


# ---------------------------------------------------------------------------
# SLIC computation & disk cache
# ---------------------------------------------------------------------------

def compute_slic_224(img_pil, n_segments: int, compactness: float) -> np.ndarray:
    """Compute SLIC on a 224×224 PIL image. Returns uint8 label map (224, 224)."""
    img_np = np.array(img_pil.resize((224, 224)))
    seg = _slic_fn(img_np, n_segments=n_segments,
                   compactness=compactness, start_label=0)
    return seg.astype(np.uint8)


def get_slic_224(img_pil, cache_path: Path,
                 n_segments: int, compactness: float) -> torch.Tensor:
    """
    Load SLIC label map from disk cache, or compute and cache it.
    Returns int64 tensor of shape (224, 224).
    """
    if cache_path.exists():
        arr = np.load(str(cache_path))
        if arr.shape == (224, 224):
            return torch.from_numpy(arr.astype(np.int64))
    arr = compute_slic_224(img_pil, n_segments, compactness)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), arr)
    return torch.from_numpy(arr.astype(np.int64))


# ---------------------------------------------------------------------------
# Superpixel feature pooling
# ---------------------------------------------------------------------------

def slic224_to_14(slic_224: torch.Tensor) -> torch.Tensor:
    """
    Downsample SLIC label map from 224×224 to 14×14 by picking the
    center pixel of each 16×16 block (matches ResNet50 layer3 stride).

    Input : (B, 224, 224) int64
    Output: (B, 14, 14)   int64
    """
    return slic_224[:, 8::16, 8::16].contiguous()


def extract_sp_features(f: torch.Tensor,
                         slic_14: torch.Tensor,
                         n_segments: int) -> torch.Tensor:
    """
    Mean-pool CNN features within each superpixel region.

    Args:
        f        : (B, C, H, W) backbone feature map
        slic_14  : (B, H, W)    superpixel label map (same H, W as f)
        n_segments: number of superpixels

    Returns:
        sp_feat  : (B, n_segments, C) superpixel-mean features
    """
    B, C, H, W = f.shape
    N        = H * W
    f_flat   = f.view(B, C, N).permute(0, 2, 1).contiguous()   # (B, N, C)
    seg_flat = slic_14.view(B, N).long().clamp(0, n_segments - 1)

    sp_sum = torch.zeros(B, n_segments, C, device=f.device, dtype=f.dtype)
    sp_cnt = torch.zeros(B, n_segments,    device=f.device, dtype=f.dtype)
    seg_exp = seg_flat.unsqueeze(-1).expand(-1, -1, C)
    sp_sum.scatter_add_(1, seg_exp, f_flat)
    sp_cnt.scatter_add_(1, seg_flat,
                        torch.ones(B, N, device=f.device, dtype=f.dtype))
    return sp_sum / sp_cnt.unsqueeze(-1).clamp(min=1.0)


# ---------------------------------------------------------------------------
# Foreground mask
# ---------------------------------------------------------------------------

def compute_sp_fg(sp_feat: torch.Tensor, fg_ratio: float = 0.30) -> torch.Tensor:
    """
    Identify foreground superpixels as those whose features deviate most
    from the image-level mean (background tends to cluster near the mean).

    Args:
        sp_feat  : (B, N_sp, C)
        fg_ratio : fraction of superpixels to label as foreground

    Returns:
        fg_mask  : (B, N_sp) float, 1=foreground, 0=background
    """
    global_mean = sp_feat.mean(dim=1, keepdim=True)       # (B, 1, C)
    deviation   = (sp_feat - global_mean).norm(dim=-1)    # (B, N_sp)
    thr = deviation.quantile(1.0 - fg_ratio, dim=-1, keepdim=True)
    return (deviation >= thr).float()
