"""
backbone.py
-----------
Frozen backbone (ResNet50 / ViT-B/16) with forward-hook for spatial features.
"""
import torchvision
import torch.nn as nn


class FeatureHook:
    """Stores the output of a registered layer."""
    def __init__(self): self.out = None
    def __call__(self, m, i, o): self.out = o


class ViTSpatialHook:
    """Reshapes ViT patch tokens to (B, C, H, W) spatial format."""
    def __init__(self): self.out = None
    def __call__(self, m, i, o):
        patches = o[:, 1:, :]           # drop CLS token
        B, N, C = patches.shape
        H = W = int(N ** 0.5)
        self.out = patches.permute(0, 2, 1).reshape(B, C, H, W).contiguous()


def build_backbone(device, backbone='resnet50'):
    """
    Returns (model, hook, d_in).

    ResNet50  → hooks layer3  → d_in=1024, spatial 14×14
    ViT-B/16  → hooks last encoder layer → d_in=768,  spatial 14×14

    Layer-selection rationale: see analysis/layer_scan.py and
    analysis/results/resnet_layer_scan.csv — layer3 offers the best
    trade-off between semantic richness and spatial resolution (14×14).
    """
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

    else:  # resnet50
        try:    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        except: w = None
        m = (torchvision.models.resnet50(weights=w) if w else
             torchvision.models.resnet50(pretrained=True))
        m.eval()
        for p in m.parameters(): p.requires_grad_(False)
        hook = FeatureHook()
        m.layer3.register_forward_hook(hook)
        return m.to(device), hook, 1024
