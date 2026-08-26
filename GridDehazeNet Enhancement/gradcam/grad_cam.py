"""
Grad-CAM for GridDehazeNet
===========================
Produces spatially rich heatmaps by fusing gradients from multiple
grid nodes across different scales.

Why multi-layer fusion matters here
-------------------------------------
GridDehazeNet's grid structure means:
  • Early nodes (col 0) capture local texture / fine edges
  • Mid nodes  (col 1) capture medium-scale fog density patterns
  • Late nodes (col 2) capture global haze distribution + semantic regions

Fusing all three gives a heatmap where:
  • Dense-fog zones (sky, far road) appear warm (orange/red)
  • High-contrast structures (lane markings, signs, cars) also activate
  • Clear regions remain blue/cool

Signal used for backpropagation
---------------------------------
We backprop through the FOG-REMOVAL SIGNAL:
    score = MSE(output, input)
This measures where the network changed the image most — exactly the
fog-dense regions. It produces much better spatial contrast than using
output.mean() which tends to give flat, uninformative heatmaps.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Core Grad-CAM engine
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Multi-layer Grad-CAM for any CNN-based image enhancement model.

    Usage
    -----
    cam = GradCAM(model, layer_names=['n00.block', 'n01.block', 'n02.block'])
    heatmap = cam.generate(input_tensor)          # (H, W) float32 in [0,1]
    overlay = cam.overlay(heatmap, image_bgr)     # (H, W, 3) uint8
    cam.remove_hooks()
    """

    # Default target layers — two nodes across scale and stage for memory efficiency.
    # n00 = early full-scale features (fine texture / local fog),
    # n02 = late full-scale features  (global fog / semantic regions).
    # Add 'n10', 'n11' if you have abundant RAM and want richer heatmaps.
    DEFAULT_LAYERS  = ['n00', 'n02']
    DEFAULT_WEIGHTS = [0.40,  0.60]

    def __init__(self,
                 model: torch.nn.Module,
                 layer_names: Optional[List[str]] = None,
                 layer_weights: Optional[List[float]] = None):
        self.model         = model
        self.layer_names   = layer_names   or self.DEFAULT_LAYERS
        self.layer_weights = layer_weights or self.DEFAULT_WEIGHTS

        assert len(self.layer_names) == len(self.layer_weights), \
            'layer_names and layer_weights must have the same length'

        self._activations: Dict[str, torch.Tensor] = {}
        self._gradients:   Dict[str, torch.Tensor] = {}
        self._hooks: list = []
        self._register_hooks()

    # ── Hook registration ─────────────────────────────────────────────────────

    def _register_hooks(self):
        for name in self.layer_names:
            module = self._get_module(name)
            if module is None:
                print(f'[GradCAM] Warning: layer "{name}" not found — skipping.')
                continue

            # Closures to capture the layer name correctly
            def _fwd(mod, inp, out, n=name):
                self._activations[n] = out

            def _bwd(mod, g_in, g_out, n=name):
                self._gradients[n] = g_out[0]

            self._hooks.append(module.register_forward_hook(_fwd))
            self._hooks.append(module.register_full_backward_hook(_bwd))

    def _get_module(self, name: str) -> Optional[torch.nn.Module]:
        """Resolve dotted name like 'n01.block.ca' → module."""
        parts  = name.split('.')
        module = self.model
        for part in parts:
            module = getattr(module, part, None)
            if module is None:
                return None
        return module

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── Core generation ───────────────────────────────────────────────────────

    def generate(self,
                 input_tensor: torch.Tensor,
                 signal: str = 'fog_removal') -> np.ndarray:
        """
        Compute the fused Grad-CAM heatmap.

        Args:
            input_tensor : (1, 3, H, W) float32 tensor in [0, 1]
            signal       : backprop signal —
                             'fog_removal' (recommended): MSE(output, input)
                             'output_mean': plain output mean

        Returns:
            heatmap : (H, W) float32 in [0, 1], percentile-normalised
        """
        self.model.eval()
        self._activations.clear()
        self._gradients.clear()

        x = input_tensor.detach().clone().requires_grad_(True)

        # Forward pass — hooks collect activations
        output = self.model(x)

        # Scalar signal: where did the network modify the image most?
        if signal == 'fog_removal':
            score = F.mse_loss(output, x.detach())
        else:
            score = output.mean()

        # Backward — hooks collect gradients
        self.model.zero_grad()
        score.backward()

        # ── Fuse per-layer CAMs ───────────────────────────────────────────────
        H, W  = input_tensor.shape[2], input_tensor.shape[3]
        fused = np.zeros((H, W), dtype=np.float32)

        for name, weight in zip(self.layer_names, self.layer_weights):
            act  = self._activations.get(name)
            grad = self._gradients.get(name)
            if act is None or grad is None:
                continue

            # Channel importance = mean absolute gradient across spatial dims
            # Use absolute value to capture magnitude of change (both + and -)
            alpha = torch.abs(grad.detach()).mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

            # Weighted activation sum → class activation map
            # Sum weighted channels, then take absolute to ensure positive output
            cam = torch.abs((alpha * act.detach()).sum(dim=1, keepdim=True))  # (1,1,h,w)

            # Resize to input resolution
            cam = F.interpolate(cam, size=(H, W),
                                mode='bilinear', align_corners=False)
            cam_np = cam.squeeze().cpu().numpy()

            # Normalise before fusing so all layers contribute equally
            cam_np = _percentile_norm(cam_np, lo=2, hi=98)
            fused += weight * cam_np

        # Final normalisation + smoothing
        fused = _percentile_norm(fused, lo=1, hi=99)
        fused = cv2.GaussianBlur(fused, (9, 9), sigmaX=2.5)
        return np.clip(fused, 0.0, 1.0).astype(np.float32)

    # ── Visualisation helpers ─────────────────────────────────────────────────

    @staticmethod
    def to_colormap(heatmap: np.ndarray) -> np.ndarray:
        """
        Convert [0,1] heatmap → BGR uint8 using TURBO colourmap.
        TURBO gives: blue (low) → cyan → green → yellow → orange → red (high)
        which produces the target appearance:
          • Road surface / uniform fog → blue/cyan
          • Lane markings / edges      → green/yellow
          • Dense fog zones / signs    → orange/red
        """
        u8 = (heatmap * 255).astype(np.uint8)
        try:
            return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
        except AttributeError:
            return cv2.applyColorMap(u8, cv2.COLORMAP_JET)

    @staticmethod
    def overlay(heatmap: np.ndarray,
                image_bgr: np.ndarray,
                alpha: float = 0.55) -> np.ndarray:
        """
        Blend Grad-CAM heatmap over an image.

        Args:
            heatmap   : (H, W) float32 in [0, 1]
            image_bgr : (H, W, 3) uint8 BGR image
            alpha     : heatmap opacity (0 = invisible, 1 = opaque)

        Returns:
            (H, W, 3) uint8 BGR
        """
        hm_color = GradCAM.to_colormap(heatmap)
        if hm_color.shape[:2] != image_bgr.shape[:2]:
            hm_color = cv2.resize(hm_color,
                                  (image_bgr.shape[1], image_bgr.shape[0]))
        return cv2.addWeighted(image_bgr, 1.0 - alpha, hm_color, alpha, 0)

    @staticmethod
    def add_colorbar(heatmap_bgr: np.ndarray,
                     width: int = 24) -> np.ndarray:
        """Append a vertical colour-scale bar to the right of a heatmap image."""
        h = heatmap_bgr.shape[0]
        bar_vals = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
        bar      = np.repeat(bar_vals, width, axis=1)
        try:
            bar_color = cv2.applyColorMap(bar, cv2.COLORMAP_TURBO)
        except AttributeError:
            bar_color = cv2.applyColorMap(bar, cv2.COLORMAP_JET)

        # Labels
        for pct, label in [(0.05, 'High'), (0.50, 'Mid'), (0.95, 'Low')]:
            y = int(h * pct)
            cv2.putText(bar_color, label, (1, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        (255, 255, 255), 1, cv2.LINE_AA)

        separator = np.full((h, 2, 3), 60, dtype=np.uint8)
        return np.hstack([heatmap_bgr, separator, bar_color])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _percentile_norm(arr: np.ndarray,
                     lo: float = 2, hi: float = 98) -> np.ndarray:
    """Clip to [lo, hi] percentile then scale to [0, 1]."""
    a = np.percentile(arr, lo)
    b = np.percentile(arr, hi)
    if b - a < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - a) / (b - a), 0.0, 1.0).astype(np.float32)
