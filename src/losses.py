"""
losses.py
---------
Auxiliary SAE losses: spatial orthogonality and cross-image diversity.
"""
import torch
import torch.nn.functional as F


def loss_sp_orth(z_sp: torch.Tensor, fg: torch.Tensor,
                 n_pairs: int = 128) -> torch.Tensor:
    """
    Spatial orthogonality loss: different concepts should activate
    different superpixel regions (not spatially overlapping).

    Implemented efficiently via random concept-pair sampling instead of
    computing the full K×K Gram matrix.

    Args:
        z_sp   : (B, N_sp, K) sparse concept codes
        fg     : (B, N_sp)    foreground mask
        n_pairs: number of random concept pairs to sample

    Returns:
        scalar loss (mean cosine similarity of random pairs in superpixel space)
    """
    B, N_sp, K = z_sp.shape
    z_fg = z_sp * fg.unsqueeze(-1)          # (B, N_sp, K) — BG zeroed out

    idx_a = torch.randint(K, (n_pairs,), device=z_sp.device)
    idx_b = torch.randint(K, (n_pairs,), device=z_sp.device)

    z_a = z_fg[:, :, idx_a]                 # (B, N_sp, n_pairs)
    z_b = z_fg[:, :, idx_b]

    na  = F.normalize(z_a, dim=1)           # normalize over superpixel dim
    nb  = F.normalize(z_b, dim=1)
    sim = (na * nb).sum(dim=1)              # (B, n_pairs)
    return sim.clamp(min=0).mean()          # penalize positive spatial overlap


def loss_diversity(z_pool: torch.Tensor) -> torch.Tensor:
    """
    Cross-image diversity loss: concept activations should vary across images.

    Low-variance concepts activate similarly for all images → background /
    non-discriminative → suppressed by a negative variance penalty.

    Args:
        z_pool : (B, K) pooled concept activations

    Returns:
        scalar loss (negative mean variance across images per concept)
    """
    return -z_pool.var(dim=0).mean()
