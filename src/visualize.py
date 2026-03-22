"""
visualize.py
------------
Concept map visualization using SLIC superpixel boundaries.
Produces per-image concept overlays and a concept gallery.
"""
import os

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from .training import extract_features
from .models import fg_z_pool


# Concept overlay colors (RGB)
_COLORS = [
    (220,  50,  50),   # red
    ( 50, 180,  80),   # green
    ( 50, 120, 220),   # blue
    (220, 160,   0),   # orange
    (160,  50, 220),   # purple
    (  0, 190, 190),   # cyan
    (220, 100, 160),   # pink
    (100, 180,  50),   # lime
]


def tensor_to_pil(img_t, mean, std):
    """Denormalize an ImageNet-normalized tensor → PIL Image."""
    img = img_t.detach().cpu().float().clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return Image.fromarray((img.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy())


def concept_map_slic(img_pil, z_sp, fg_sp, slic_224, color_rgb,
                     alpha_max=0.65, thresh=0.25):
    """
    Blend a concept activation map onto an image using SLIC boundaries.

    Args:
        img_pil   : PIL Image (any size, resized to 224)
        z_sp      : (N_sp,) concept activations per superpixel
        fg_sp     : (N_sp,) foreground mask
        slic_224  : (224, 224) int64 label map
        color_rgb : (R, G, B) overlay color
        alpha_max : maximum overlay opacity
        thresh    : activation threshold below which no overlay is drawn

    Returns:
        PIL Image with concept overlay
    """
    z_fg   = (z_sp * fg_sp).cpu().numpy().astype(np.float32)
    a_max  = z_fg.max()
    if a_max < 1e-8:
        return img_pil.copy()
    z_norm  = z_fg / a_max
    seg     = slic_224.cpu().numpy().astype(np.int64).clip(0, len(z_norm) - 1)
    act_map = z_norm[seg]
    alpha   = np.clip((act_map - thresh) / (1.0 - thresh + 1e-8), 0, 1) * alpha_max
    img_np  = np.array(img_pil.resize((224, 224))).astype(np.float32)
    col     = np.array(color_rgb, dtype=np.float32)
    out     = img_np * (1 - alpha[:, :, None]) + col[None, None, :] * alpha[:, :, None]
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


@torch.no_grad()
def visualize_batch(backbone, hook, feat_norm, sae, head, loader, cfg,
                    mean, std, device, tag='', n_images=8):
    """
    For the first batch, render concept maps for n_images samples.

    Each output image shows:
      [original] [concept_1] [concept_2] ... [combined overlay]

    Saved to: {cfg.run_dir}/visuals/{tag}/
    """
    sae.eval(); feat_norm.eval()
    if head is not None: head.eval()

    vis_dir = os.path.join(cfg.run_dir, 'visuals', tag)
    os.makedirs(vis_dir, exist_ok=True)

    for x, slic_224, y in loader: break

    x_vis = x[:n_images].to(device)
    s_vis = slic_224[:n_images]
    y_vis = y[:n_images]
    B     = x_vis.shape[0]

    sp_feat, fg = extract_features(backbone, hook, feat_norm,
                                    x_vis, s_vis, cfg, device)
    z, _  = sae(sp_feat)
    zp    = fg_z_pool(z, fg)
    preds = head(zp).argmax(1) if head else y_vis.to(device)

    cell = 224; pad = 8

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

        # Select concepts up to vis_coverage
        selected_ks, selected_pcts, cumsum = [], [], 0.0
        for k in sorted_idx:
            contrib = max(S_h[k].item(), 0.0)
            cumsum += contrib
            selected_ks.append(k)
            selected_pcts.append(contrib / S_total)
            if len(selected_ks) >= cfg.vis_max_concepts: break
            if cumsum / S_total >= cfg.vis_coverage:     break

        # Diversity filter: remove spatially redundant concepts
        div_ks, div_pcts, div_maps = [], [], []
        for k, pct in zip(selected_ks, selected_pcts):
            zm = z[i, :, k].cpu() * fg_i
            zn = F.normalize(zm.unsqueeze(0), dim=1).squeeze()
            if not any((zn * prev).sum().item() > 0.85 for prev in div_maps):
                div_ks.append(k); div_pcts.append(pct); div_maps.append(zn)
        selected_ks, selected_pcts = div_ks, div_pcts
        n_shown = len(selected_ks)

        # ── Contribution bar chart panel ──────────────────────────────────
        bar_w   = 220   # width of explanation panel
        bar_h   = cell  # same height as image
        bar_img = Image.new('RGB', (bar_w, bar_h + 20), (15, 15, 25))
        bd      = ImageDraw.Draw(bar_img)
        bd.text((4, 2), 'Concept Contributions', fill=(220, 220, 220))
        row_h    = max(12, min(20, (bar_h - 16) // max(len(selected_ks), 1)))
        safe_pcts = [p if (p == p and p >= 0) else 0.0 for p in selected_pcts]
        max_pct  = max(safe_pcts[0], 1e-8) if safe_pcts else 1.0
        for ci, (k, pct) in enumerate(zip(selected_ks, safe_pcts)):
            y0   = 16 + ci * row_h
            if y0 + row_h > bar_h: break
            blen = int((pct / max_pct) * (bar_w - 90))
            r, g2, b = _COLORS[ci % len(_COLORS)]
            # colored bar
            bd.rectangle([4, y0 + 1, 4 + blen, y0 + row_h - 2], fill=(r, g2, b))
            # label: concept id + contribution %
            bd.text((blen + 8, y0 + 1), f"k{k}  {pct:.1%}", fill=(210, 210, 210))
        # cumulative coverage annotation
        correct = (preds[i].item() == y_vis[i].item())
        status  = 'CORRECT' if correct else 'WRONG'
        s_color = (80, 220, 80) if correct else (220, 80, 80)
        bd.text((4, bar_h + 3),
                f"[{status}] y={y_vis[i].item()} p={preds[i].item()}  "
                f"cov={cumsum/S_total:.0%}({n_shown}c)",
                fill=s_color)

        # ── Grid layout: orig | concepts... | combined | bar ──────────────
        n_cols    = n_shown + 3   # orig + concepts + combined + bar
        grid_w    = cell * (n_shown + 2) + pad * (n_shown + 2) + bar_w + pad
        grid      = Image.new('RGB', (grid_w, cell + 20), (20, 20, 20))
        grid.paste(img_pil, (0, 0))
        ImageDraw.Draw(grid).text((2, cell + 2),
                                  f"y={y_vis[i].item()} p={preds[i].item()}",
                                  fill=(200, 200, 200))

        for ci, (k, pct) in enumerate(zip(selected_ks, selected_pcts)):
            z_sp  = z[i, :, k].cpu()
            color = _COLORS[ci % len(_COLORS)]
            blend = concept_map_slic(img_pil, z_sp, fg_i, slic_i, color)
            x_off = (ci + 1) * (cell + pad)
            grid.paste(blend, (x_off, 0))
            d2 = ImageDraw.Draw(grid)
            r, g2, b = color
            d2.rectangle([x_off, cell + 1, x_off + 12, cell + 14], fill=(r, g2, b))
            d2.text((x_off + 15, cell + 2), f"k{k} {pct:.1%}",
                    fill=(200, 200, 200))

        # Combined overlay
        combined = np.array(img_pil).astype(np.float32)
        for ci, k in enumerate(selected_ks):
            z_sp = z[i, :, k].cpu()
            z_fg = (z_sp * fg_i).numpy().astype(np.float32)
            a_mx = z_fg.max()
            if a_mx < 1e-8: continue
            z_n   = z_fg / a_mx
            seg   = slic_i.numpy().astype(np.int64).clip(0, len(z_n) - 1)
            act   = z_n[seg]
            alpha = np.clip((act - 0.25) / 0.75, 0, 1) * 0.60
            col   = np.array(_COLORS[ci % len(_COLORS)], dtype=np.float32)
            combined = (combined * (1 - alpha[:, :, None])
                        + col[None, None, :] * alpha[:, :, None])
        comb_pil = Image.fromarray(combined.clip(0, 255).astype(np.uint8))
        x_off_c  = (n_shown + 1) * (cell + pad)
        grid.paste(comb_pil, (x_off_c, 0))
        ImageDraw.Draw(grid).text((x_off_c + 2, cell + 2), 'combined',
                                  fill=(180, 180, 180))

        # Paste bar chart
        grid.paste(bar_img, (x_off_c + cell + pad, 0))

        fname = os.path.join(vis_dir,
                             f"img{i:02d}_y{y_vis[i].item()}_p{preds[i].item()}.png")
        grid.save(fname)

    print(f"  [VIS] {B} images → {vis_dir}/")


@torch.no_grad()
def compute_gallery(backbone, hook, feat_norm, sae, loader, cfg,
                    mean, std, device):
    """
    Build a concept gallery: for the top-K concepts (by mean activation),
    show the gallery_n_images images that activate each concept most strongly.

    Saved to: {cfg.run_dir}/concept_gallery/
    """
    sae.eval(); feat_norm.eval()
    print(f"  Building gallery "
          f"({cfg.gallery_n_concepts} concepts × {cfg.gallery_n_images} imgs)...")

    all_zp = []
    for x, slic_224, _ in loader:
        sp_feat, fg = extract_features(backbone, hook, feat_norm,
                                        x.to(device), slic_224, cfg, device)
        z, _ = sae(sp_feat)
        all_zp.append(fg_z_pool(z, fg).cpu())
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
        loc = [i for i in range(B) if (offset + i) in needed]
        if loc:
            sp_feat, fg = extract_features(backbone, hook, feat_norm,
                                            x[loc].to(device), slic_224[loc],
                                            cfg, device)
            z, _ = sae(sp_feat)
            for j, li in enumerate(loc):
                gid = offset + li
                img_cache[gid]  = x[li].cpu()
                z_cache[gid]    = z[j].cpu()
                fg_cache[gid]   = fg[j].cpu()
                slic_cache[gid] = slic_224[li].cpu()
        offset += B

    gal_dir = os.path.join(cfg.run_dir, 'concept_gallery')
    os.makedirs(gal_dir, exist_ok=True)
    cell = 140

    for rank, k in enumerate(top_k_ids):
        top_ids = all_zp[:, k].topk(cfg.gallery_n_images).indices.tolist()
        m       = cfg.gallery_n_images
        grid    = Image.new('RGB', (cell * m + 4 * (m - 1), cell), (255, 255, 255))
        color   = _COLORS[rank % len(_COLORS)]
        for ci, gid in enumerate(top_ids):
            if gid not in img_cache: continue
            img_pil = tensor_to_pil(img_cache[gid], mean, std).resize((cell, cell))
            blend   = concept_map_slic(img_pil, z_cache[gid][:, k],
                                        fg_cache[gid], slic_cache[gid], color)
            grid.paste(blend.resize((cell, cell)), (ci * (cell + 4), 0))
        fname = os.path.join(gal_dir,
                             f"rank{rank:02d}_k{k}_mean{mean_act[k]:.3f}.png")
        grid.save(fname)

    print(f"  [GALLERY] → {gal_dir}/")
    return top_k_ids
