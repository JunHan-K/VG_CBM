"""
evaluate.py
-----------
Evaluation metrics: accuracy, K_0.8 (concept sparsity), and concept intervention.
"""
import numpy as np
import torch

from .training import extract_features
from .models import fg_z_pool


@torch.no_grad()
def eval_acc(backbone, hook, feat_norm, sae, head, loader, cfg, device) -> float:
    """Compute test accuracy."""
    sae.eval(); feat_norm.eval(); head.eval()
    tot, n = 0.0, 0
    for x, slic_224, y in loader:
        sp_feat, fg = extract_features(backbone, hook, feat_norm,
                                        x.to(device), slic_224, cfg, device)
        z, _   = sae(sp_feat)
        logits = head(fg_z_pool(z, fg))
        tot += (logits.argmax(1) == y.to(device)).float().mean().item()
        n   += 1
    return tot / n


@torch.no_grad()
def compute_K08(backbone, hook, feat_norm, sae, head, loader, cfg,
                device, coverage: float = 0.8) -> float:
    """
    Compute K_0.8: mean number of concepts needed to cover `coverage`
    fraction of the total prediction mass for the true class.

    Lower K_0.8 → sparser, more interpretable concept usage.
    Target range: 50–400 (see paper).
    """
    sae.eval(); feat_norm.eval(); head.eval()
    ks = []
    for x, slic_224, y in loader:
        sp_feat, fg = extract_features(backbone, hook, feat_norm,
                                        x.to(device), slic_224, cfg, device)
        z, _  = sae(sp_feat)
        zp    = fg_z_pool(z, fg)
        W     = head.fc.weight
        for i in range(zp.shape[0]):
            S        = (W[y[i].item()] * zp[i]).abs()
            S_sorted = S.sort(descending=True).values
            cumsum   = S_sorted.cumsum(0) / S_sorted.sum().clamp(1e-12)
            ks.append((cumsum < coverage).sum().item() + 1)
    return float(np.mean(ks))


@torch.no_grad()
def compute_intervention(backbone, hook, feat_norm, sae, head,
                          loader, cfg, ms, device) -> dict:
    """
    Concept intervention: zero out top-m positive-contributing concepts.

    Selects the m concepts with the highest W_y * zp contribution to the
    true-class logit and sets them to zero. Since CBMHead uses non-negative
    weights (ReLU), all contributions are >= 0 and removal always hurts.

    drop = interv_acc - orig_acc  (negative = accuracy decreased = concepts help)
    Larger m → larger |drop|  (monotonically more harmful removal).

    Args:
        ms : list of m values, e.g. [10, 25, 50]

    Returns:
        dict mapping m → {'orig': float, 'drop': float}
    """
    sae.eval(); feat_norm.eval(); head.eval()
    results = {}
    for m in ms:
        orig_scores, drop_scores = [], []
        for x, slic_224, y in loader:
            sp_feat, fg = extract_features(backbone, hook, feat_norm,
                                            x.to(device), slic_224, cfg, device)
            z, _    = sae(sp_feat)
            zp      = fg_z_pool(z, fg)
            y_dev   = y.to(device)
            orig    = (head(zp).argmax(1) == y_dev).float().mean().item()
            W_y     = head.fc.weight[y_dev]          # (B, K)
            contrib = (W_y * zp)                     # all non-negative (ReLU head)
            k       = min(m, contrib.shape[1])
            top_m   = contrib.topk(k, dim=1).indices
            zp_int  = zp.clone().scatter_(1, top_m, 0.0)
            interv  = (head(zp_int).argmax(1) == y_dev).float().mean().item()
            orig_scores.append(orig)
            drop_scores.append(interv - orig)         # negative when accuracy falls
        results[m] = {'orig': float(np.mean(orig_scores)),
                      'drop': float(np.mean(drop_scores))}
    return results
