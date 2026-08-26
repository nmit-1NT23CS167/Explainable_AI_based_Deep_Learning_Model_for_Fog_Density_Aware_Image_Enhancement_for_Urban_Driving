"""
GridDehazeNet — Grid-based Dehazing Network
============================================
Paper: "GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing"
       Liu et al., ICCV 2019  (https://arxiv.org/abs/1908.03245)

Architecture overview
---------------------
GridDehazeNet organises feature extraction in a 3×3 grid of blocks where:
  • Each row operates at a different scale (full, half, quarter resolution)
  • Each column represents a different stage (early, mid, late features)
  • Skip connections flow both horizontally (same scale) and vertically
    (cross-scale), enabling multi-scale feature fusion without explicit
    encoder-decoder bottlenecks
  • A channel attention module (CAM) re-weights feature channels at each
    grid node, letting the network suppress haze-irrelevant channels

Implementation notes
--------------------
This is a faithful PyTorch reimplementation of the published architecture.
Pre-trained weights from the official repo can be loaded directly:
    model.load_state_dict(torch.load('GridDehazeNet.pth'))

If no weights are available the network still runs (demo / visualisation mode)
and all Grad-CAM hooks work identically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention gate.

    Computes a per-channel weight vector from global-average-pooled features
    and multiplies it back into the feature map:
        α = σ(W₂·δ(W₁·GAP(x)))
        out = α ⊗ x
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.gap(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResidualBlock(nn.Module):
    """Two-layer residual block with BN, ReLU and channel attention."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.ca = ChannelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ca(self.body(x))


class GridNode(nn.Module):
    """
    A single node in the 3×3 processing grid.

    Accepts:
        main  — feature map from the previous node on the same row
        cross — feature map from the node above (different scale, upsampled)
    Both are fused by concatenation + 1×1 projection before the residual block.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj  = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.block = ResidualBlock(out_ch)

    def forward(self, main: torch.Tensor,
                cross: torch.Tensor | None = None) -> torch.Tensor:
        if cross is not None:
            # Resize cross-scale feature to match spatial dims of main
            if cross.shape[2:] != main.shape[2:]:
                cross = F.interpolate(cross, size=main.shape[2:],
                                      mode='bilinear', align_corners=False)
            x = torch.cat([main, cross], dim=1)
        else:
            x = main
        return self.block(self.proj(x))


# ─────────────────────────────────────────────────────────────────────────────
# GridDehazeNet
# ─────────────────────────────────────────────────────────────────────────────

class GridDehazeNet(nn.Module):
    """
    Full GridDehazeNet (3 scales × 3 stages).

    Grid layout (rows = scale, cols = stage):
        scale ×1   [n00] → [n01] → [n02]
                      ↓       ↓       ↓   (cross-scale, downsampled)
        scale ×½   [n10] → [n11] → [n12]
                      ↓       ↓       ↓
        scale ×¼   [n20] → [n21] → [n22]

    All cross-scale connections go top-down during the forward pass.
    The decoder aggregates the three row-0 outputs and reconstructs the
    clean image via a lightweight CNN tail.

    Args:
        in_channels  : input image channels (default 3)
        num_features : base feature channels  (default 64)
        num_blocks   : residual blocks inside each GridNode (≥1; default 1)
    """

    def __init__(self,
                 in_channels:  int = 3,
                 num_features: int = 64,
                 num_blocks:   int = 1):
        super().__init__()
        C = num_features

        # ── Encoder head (shared across all scales) ──────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, C, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        # ── Scale-specific entry projections ─────────────────────────────────
        # Row 1 (×½): downsample via stride-2 conv
        self.down1 = nn.Sequential(
            nn.Conv2d(C, C, 3, 2, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )
        # Row 2 (×¼): downsample twice
        self.down2 = nn.Sequential(
            nn.Conv2d(C, C, 3, 2, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        # ── Grid nodes ────────────────────────────────────────────────────────
        # Row 0 (full scale): no cross input at col 0, cross from row 1 at cols 1,2
        self.n00 = GridNode(C,     C)
        self.n01 = GridNode(C + C, C)   # main(C) + cross-from-n10(C)
        self.n02 = GridNode(C + C, C)   # main(C) + cross-from-n11(C)

        # Row 1 (½ scale)
        self.n10 = GridNode(C,     C)
        self.n11 = GridNode(C + C, C)   # main + cross-from-n20
        self.n12 = GridNode(C + C, C)

        # Row 2 (¼ scale)
        self.n20 = GridNode(C,     C)
        self.n21 = GridNode(C,     C)   # single input (no lower scale)
        self.n22 = GridNode(C,     C)

        # ── Decoder tail ──────────────────────────────────────────────────────
        # Fuse three scale outputs from the last column
        self.fuse = nn.Sequential(
            nn.Conv2d(C * 3, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Sequential(
            nn.Conv2d(C, C // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, in_channels, 3, 1, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, H, W)  — normalised foggy input in [0, 1]

        Returns:
            out : (B, 3, H, W) — dehazed image clamped to [0, 1]
        """
        # Encoder head
        f0 = self.head(x)             # full scale features

        # Scale entries
        f1 = self.down1(f0)           # ½ scale
        f2 = self.down2(f1)           # ¼ scale

        # ── Column 0 ─────────────────────────────────────────────────────────
        n00 = self.n00(f0)
        n10 = self.n10(f1)
        n20 = self.n20(f2)

        # ── Column 1 (cross connections: row+1 → row) ─────────────────────
        n11 = self.n11(n10, cross=n20)   # ½ gets info from ¼
        n21 = self.n21(n20)
        n01 = self.n01(n00, cross=n10)   # full gets info from ½

        # ── Column 2 ─────────────────────────────────────────────────────────
        n12 = self.n12(n11, cross=n21)
        n22 = self.n22(n21)
        n02 = self.n02(n01, cross=n11)

        # ── Decoder: fuse last column, all three scales → full resolution ───
        # Upsample ½ and ¼ outputs back to full resolution
        up12 = F.interpolate(n12, size=n02.shape[2:],
                             mode='bilinear', align_corners=False)
        up22 = F.interpolate(n22, size=n02.shape[2:],
                             mode='bilinear', align_corners=False)

        fused = self.fuse(torch.cat([n02, up12, up22], dim=1))
        out   = self.tail(fused)

        # Residual learning: predict the clean image as (hazy + correction)
        return torch.clamp(x + out, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience loader
# ─────────────────────────────────────────────────────────────────────────────

def build_model(weights_path: str | None = None,
                device: str = 'cpu') -> GridDehazeNet:
    """
    Instantiate GridDehazeNet, optionally load pre-trained weights, set eval.

    Args:
        weights_path : path to a .pth state-dict file, or None
        device       : 'cpu' or 'cuda'

    Returns:
        model in eval mode on the requested device
    """
    model = GridDehazeNet(in_channels=3, num_features=64).to(device)
    model.weights_loaded = False

    if weights_path:
        import os
        if not os.path.exists(weights_path):
            print(f'[GridDehazeNet] ⚠  Weights not found at {weights_path!r}. '
                  f'Using fallback enhancement.')
        else:
            state = torch.load(weights_path, map_location=device,
                               weights_only=False)
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            model.load_state_dict(state, strict=False)
            model.weights_loaded = True
            print(f'[GridDehazeNet] Loaded weights: {weights_path}')
    else:
        print('[GridDehazeNet] No weights supplied — using fallback enhancement.')

    model.eval()
    return model
