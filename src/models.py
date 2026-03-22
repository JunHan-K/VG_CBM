"""
models.py
---------
SparseSAE (TopK sparse autoencoder) and CBMHead (linear concept bottleneck).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNorm(nn.Module):
    """Identity passthrough — preserves feature magnitude information."""
    def __init__(self, C): super().__init__()
    def forward(self, x): return x


class SparseSAE(nn.Module):
    """
    Sparse Autoencoder with hard TopK activation.

    For each superpixel token x ∈ R^C:
        z_pre = W_enc x + b_enc          (K-dim pre-activation)
        z     = TopK(z_pre)              (exactly topk entries active per token)
        x_hat = W_dec z + b_dec          (reconstruction)

    Args:
        C    : input feature dimension (e.g. 1024 for ResNet50 layer3)
        K    : number of dictionary atoms (= C × expansion)
        topk : number of active concepts per superpixel token
    """
    def __init__(self, C: int, K: int, topk: int = 20):
        super().__init__()
        self.C = C; self.K = K; self.topk = topk
        self.enc_w = nn.Parameter(torch.empty(K, C))
        self.enc_b = nn.Parameter(torch.zeros(K))
        self.dec_w = nn.Parameter(torch.empty(C, K))
        self.dec_b = nn.Parameter(torch.zeros(C))
        nn.init.kaiming_normal_(self.enc_w, nonlinearity='relu')
        nn.init.normal_(self.dec_w, 0.0, 0.01)

    def forward(self, x):
        """
        Args:
            x : (B, N_sp, C) superpixel features
        Returns:
            z     : (B, N_sp, K) sparse concept codes
            x_hat : (B, N_sp, C) reconstructed features
        """
        z_pre = x @ self.enc_w.t() + self.enc_b
        if self.topk > 0:
            topk_vals, topk_idx = z_pre.topk(self.topk, dim=-1)
            z = torch.zeros_like(z_pre)
            z.scatter_(-1, topk_idx, topk_vals.clamp(min=0))
        else:
            z = F.relu(z_pre)
        x_hat = z @ self.dec_w.t() + self.dec_b
        return z, x_hat


class CBMHead(nn.Module):
    """
    Non-negative linear classification head on pooled concept activations.

    z_pool = mean_{foreground superpixels}(z)    (B, K)
    logits = ReLU(W) z_pool + b                  (B, num_classes)

    Weights are constrained to be non-negative via ReLU at forward time,
    ensuring each concept can only positively support a class prediction.
    This makes concept intervention interpretable: removing top concepts
    always decreases (or maintains) accuracy.
    """
    def __init__(self, K: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(K, num_classes)
        nn.init.xavier_uniform_(self.fc.weight.data.abs_())
        nn.init.zeros_(self.fc.bias)

    def forward(self, z_pool):
        return F.linear(z_pool, F.relu(self.fc.weight), self.fc.bias)


def fg_z_pool(z_sp: torch.Tensor, fg: torch.Tensor, pool_mode: str = 'mean') -> torch.Tensor:
    """
    Foreground-masked pooling over superpixels.

    Args:
        z_sp      : (B, N_sp, K)
        fg        : (B, N_sp) foreground mask (1=fg, 0=bg)
        pool_mode : 'mean' or 'max'
    Returns:
        z_pool : (B, K)
    """
    fg_k = fg.unsqueeze(-1)                          # (B, N_sp, 1)
    if pool_mode == 'max':
        return (z_sp * fg_k).max(dim=1).values
    return (z_sp * fg_k).sum(1) / fg_k.sum(1).clamp(1)
