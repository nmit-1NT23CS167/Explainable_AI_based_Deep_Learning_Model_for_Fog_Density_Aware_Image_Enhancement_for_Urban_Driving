"""
AOD-Net: All-in-One Dehazing Network
Paper: "AOD-Net: All-in-One Dehazing Network" (Li et al., 2017)

Architecture: Learns K(x) directly from the hazy image where
    J(x) = K(x) * I(x) - K(x) + b   (simplified atmospheric scattering)

This is integrated as STEP 1 of every frame in the pipeline:
    raw foggy frame  →  AOD-Net  →  dehazed frame  →  YOLOv11 detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class AODNet(nn.Module):
    def __init__(self):
        super(AODNet, self).__init__()
        self.relu   = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3,  3,  1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(3,  3,  3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(6,  3,  5, 1, 2, bias=True)
        self.e_conv4 = nn.Conv2d(6,  3,  7, 1, 3, bias=True)
        self.e_conv5 = nn.Conv2d(12, 3,  3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(torch.cat([x1, x2], dim=1)))
        x4 = self.relu(self.e_conv4(torch.cat([x2, x3], dim=1)))
        k  = self.relu(self.e_conv5(torch.cat([x1, x2, x3, x4], dim=1)))
        # Physics-based dehaze: J = K*I - K + b  (b ≈ atmospheric light)
        out = k * x - k + 1.0
        return torch.clamp(out, 0.0, 1.0)


class AODNetEnhancer:
    """
    Convenience wrapper for per-frame enhancement.
    Handles BGR↔tensor conversion, device placement, and optional
    downscale-for-speed before inference.

    Usage:
        enhancer = AODNetEnhancer(weights_path="models/aod_net.pth")
        enhanced_bgr = enhancer.enhance(raw_bgr_frame)
    """

    def __init__(self, weights_path: str = None,
                 device: str = "cpu",
                 inference_scale: float = 1.0):
        """
        Args:
            weights_path    : path to .pth weights; None → untrained (still
                              applies the dehazing formula with random K)
            device          : "cpu" or "cuda"
            inference_scale : resize factor before inference (0.5 = half size,
                              faster but lower quality). Output is resized back.
        """
        self.device = torch.device(device)
        self.scale  = inference_scale
        self.model  = AODNet().to(self.device)

        if weights_path:
            import os
            if os.path.exists(weights_path):
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state)
                print(f"[AOD-Net] Loaded weights: {weights_path}")
            else:
                print(f"[AOD-Net] ⚠  Weights not found at '{weights_path}'. "
                      f"Using untrained model — enhancement effect will be subtle.")
        else:
            print("[AOD-Net] No weights path given. Running untrained model.")

        self.model.eval()

    # ── Public ────────────────────────────────────────────────────────────────

    def enhance(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Enhance a single BGR frame through AOD-Net.

        Args:
            frame_bgr : (H, W, 3) uint8 BGR numpy array

        Returns:
            enhanced  : (H, W, 3) uint8 BGR numpy array (same size as input)
        """
        orig_h, orig_w = frame_bgr.shape[:2]

        # Optional downscale for real-time performance
        if self.scale != 1.0:
            sw = max(1, int(orig_w * self.scale))
            sh = max(1, int(orig_h * self.scale))
            inp = cv2.resize(frame_bgr, (sw, sh))
        else:
            inp = frame_bgr

        tensor = self._to_tensor(inp)

        with torch.no_grad():
            out_tensor = self.model(tensor)

        enhanced = self._to_bgr(out_tensor)

        # Restore original resolution
        if self.scale != 1.0:
            enhanced = cv2.resize(enhanced, (orig_w, orig_h))

        return enhanced

    # ── Private ───────────────────────────────────────────────────────────────

    def _to_tensor(self, bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 → (1, 3, H, W) float32 in [0,1] on self.device."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        return t.permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _to_bgr(self, tensor: torch.Tensor) -> np.ndarray:
        """(1, 3, H, W) float32 → BGR uint8 numpy array."""
        out = tensor.squeeze(0).cpu().clamp(0, 1)
        out = (out * 255).byte().permute(1, 2, 0).numpy()
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
