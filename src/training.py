"""
training.py
-----------
SAE and CBM head training loops.

Training is split into two phases:
  Phase 1 — SAE training (stage-wise):
      Stage 1 (ep 1 .. stage2_start): L_recon + L_l1 + L_aux
      Stage 2 (ep stage2_start+1 ..): + L_sp_orth + L_div
  Phase 2 — Head training on cached z_pool (SAE frozen)
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .slic_utils import slic224_to_14, extract_sp_features, compute_sp_fg
from .models import fg_z_pool
from .losses import loss_sp_orth, loss_diversity


# ---------------------------------------------------------------------------
# Feature extraction helper (shared by SAE training + head caching)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(backbone, hook, feat_norm, x, slic_224, cfg, device):
    """
    Run backbone, pool to superpixels, apply FG mask.

    Returns:
        sp_feat : (B, N_sp, C) — FG-masked superpixel features
        fg      : (B, N_sp)    — foreground mask
    """
    backbone(x)
    f        = hook.out
    f        = feat_norm(f.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    slic_14  = slic224_to_14(slic_224.to(device))
    sp_feat  = extract_sp_features(f, slic_14, cfg.n_segments)
    fg       = compute_sp_fg(sp_feat, cfg.fg_ratio)
    sp_feat  = sp_feat * fg.unsqueeze(-1)
    return sp_feat, fg


# ---------------------------------------------------------------------------
# Phase 1: SAE training
# ---------------------------------------------------------------------------

def train_sae_epoch(backbone, hook, feat_norm, sae, opt,
                    loader, cfg, device, epoch):
    """
    One SAE training epoch.

    Stage 1 (epoch <= stage2_start): L_recon + L_l1 + L_aux
    Stage 2 (epoch >  stage2_start): + L_sp_orth + L_div
    """
    sae.train(); feat_norm.train()
    stage2 = (epoch > cfg.stage2_start)
    tots   = {k: 0.0 for k in ('recon', 'l1', 'aux', 'sp_orth', 'div', 'total')}
    n_dead_total, n = 0, 0

    for x, slic_224, _ in loader:
        x = x.to(device)

        with torch.no_grad():
            backbone(x)
            f_raw = hook.out.clone()

        f_norm  = feat_norm(f_raw.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        slic_14 = slic224_to_14(slic_224.to(device))
        sp_feat = extract_sp_features(f_norm, slic_14, cfg.n_segments)

        with torch.no_grad():
            fg = compute_sp_fg(sp_feat, cfg.fg_ratio)

        sp_in     = sp_feat * fg.unsqueeze(-1)
        z, sp_hat = sae(sp_in)

        # Stage 1 losses
        fg3     = fg.unsqueeze(-1)
        L_recon = ((sp_hat - sp_in) ** 2 * fg3).sum() \
                  / fg3.sum().clamp(1) * cfg.w_recon
        L_l1    = (z * fg3).sum() / fg3.sum().clamp(1) * cfg.w_l1

        mean_z  = z.mean(dim=(0, 1))
        dead    = (mean_z < 1e-5).float()
        L_aux   = dead.mean() * cfg.w_aux

        loss = L_recon + L_l1 + L_aux

        # Stage 2 losses
        L_sp_orth = torch.tensor(0.0, device=device)
        L_div     = torch.tensor(0.0, device=device)
        if stage2:
            L_sp_orth = loss_sp_orth(z, fg) * cfg.w_sp_orth
            L_div     = loss_diversity(fg_z_pool(z, fg)) * cfg.w_div
            loss      = loss + L_sp_orth + L_div

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            n_dead_total += (z.detach().mean(dim=(0, 1)) < 1e-5).sum().item()

        tots['recon']   += L_recon.item()
        tots['l1']      += L_l1.item()
        tots['aux']     += L_aux.item()
        tots['sp_orth'] += L_sp_orth.item()
        tots['div']     += L_div.item()
        tots['total']   += loss.item()
        n += 1

    dead_rate = n_dead_total / (n * sae.K) if n else 0
    return {k: v / n for k, v in tots.items()} | {
        'dead': dead_rate, 'stage2': stage2}


# ---------------------------------------------------------------------------
# Phase 2: Head training (cached z_pool)
# ---------------------------------------------------------------------------

@torch.no_grad()
def cache_z_pool(backbone, hook, feat_norm, sae, loader, cfg, device):
    """Pre-compute and cache z_pool for all samples (speeds up head training)."""
    sae.eval(); feat_norm.eval()
    all_z, all_y = [], []
    for x, slic_224, y in loader:
        sp_feat, fg = extract_features(backbone, hook, feat_norm,
                                        x.to(device), slic_224, cfg, device)
        z, _   = sae(sp_feat)
        z_pool = fg_z_pool(z, fg)
        all_z.append(z_pool.cpu()); all_y.append(y.cpu())
    return torch.cat(all_z), torch.cat(all_y)


def train_head_epoch_cached(head, opt, z_cache, y_cache, cfg, device):
    """One head training epoch using pre-cached z_pool."""
    head.train()
    perm    = torch.randperm(len(z_cache))
    z_cache = z_cache[perm]; y_cache = y_cache[perm]
    tot_ce, tot_acc, n = 0.0, 0.0, 0
    bs = cfg.batch_size
    for i in range(0, len(z_cache), bs):
        zp     = z_cache[i:i + bs].to(device)
        y      = y_cache[i:i + bs].to(device)
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
    return {'ce': tot_ce / n, 'acc': tot_acc / n}


@torch.no_grad()
def eval_head_cached(head, z_cache, y_cache, cfg, device):
    """Evaluate head accuracy on cached z_pool."""
    head.eval()
    tot, n = 0.0, 0
    for i in range(0, len(z_cache), cfg.batch_size):
        zp = z_cache[i:i + cfg.batch_size].to(device)
        y  = y_cache[i:i + cfg.batch_size].to(device)
        tot += (head(zp).argmax(1) == y).float().mean().item()
        n   += 1
    return {'acc': tot / n}
