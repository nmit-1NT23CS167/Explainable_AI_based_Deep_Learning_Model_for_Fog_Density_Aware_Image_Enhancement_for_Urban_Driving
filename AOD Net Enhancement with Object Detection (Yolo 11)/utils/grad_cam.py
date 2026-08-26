"""
Multi-Layer Grad-CAM for AOD-Net
=================================
Fuses conv3 + conv4 + conv5 gradients and uses the fog-removal signal
(output-input difference) as the backprop scalar. This produces the
spatially rich blue-road / orange-sign / red-hotspot heatmaps.
"""
import cv2, numpy as np
import torch, torch.nn.functional as F


class MultiLayerGradCAM:
    def __init__(self, model, layer_names=None, layer_weights=None):
        self.model         = model
        self.layer_names   = layer_names   or ["conv3", "conv4", "conv5"]
        self.layer_weights = layer_weights or [0.20,    0.30,    0.50]
        self._acts, self._grads = {}, {}
        self._hooks = []
        self._register()

    def _register(self):
        for name in self.layer_names:
            layer = getattr(self.model, name)

            def fwd(m, i, o, n=name):  self._acts[n]  = o
            def bwd(m, gi, go, n=name): self._grads[n] = go[0]

            self._hooks += [layer.register_forward_hook(fwd),
                            layer.register_full_backward_hook(bwd)]

    def remove_hooks(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        self._acts.clear(); self._grads.clear()

        x = input_tensor.detach().clone().requires_grad_(True)
        output = self.model(x)

        # Fog-removal signal: where did the model change the image most?
        score = ((output - x) ** 2).mean()
        self.model.zero_grad()
        score.backward()

        H, W   = input_tensor.shape[2], input_tensor.shape[3]
        fused  = np.zeros((H, W), dtype=np.float32)

        for name, w in zip(self.layer_names, self.layer_weights):
            act  = self._acts.get(name)
            grad = self._grads.get(name)
            if act is None or grad is None:
                continue
            # Channel-importance weights
            weights = grad.detach().mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * act.detach()).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, (H, W), mode="bilinear", align_corners=False)
            cam_np = _pct_norm(cam.squeeze().cpu().numpy())
            fused += w * cam_np

        fused = _pct_norm(fused, 1, 99)
        fused = cv2.GaussianBlur(fused, (7, 7), 2.0)
        return np.clip(fused, 0, 1).astype(np.float32)


# ── Backward-compat single-layer wrapper ─────────────────────────────────────
class GradCAM:
    """Drop-in replacement — internally uses MultiLayerGradCAM."""
    def __init__(self, model, target_layer=None):
        self._impl = MultiLayerGradCAM(model)

    def generate(self, inp):
        return self._impl.generate(inp)

    def remove_hooks(self):
        self._impl.remove_hooks()


# ── Colourmap helpers ─────────────────────────────────────────────────────────
def apply_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """[0,1] float → BGR uint8 using TURBO (falls back to JET)."""
    u8 = (heatmap * 255).astype(np.uint8)
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    return cv2.applyColorMap(u8, cmap)


def apply_colormap(heatmap: np.ndarray, image_bgr: np.ndarray,
                   alpha: float = 0.55) -> np.ndarray:
    hm = apply_heatmap(heatmap)
    if hm.shape[:2] != image_bgr.shape[:2]:
        hm = cv2.resize(hm, (image_bgr.shape[1], image_bgr.shape[0]))
    return cv2.addWeighted(image_bgr, 1.0 - alpha, hm, alpha, 0)


def _pct_norm(arr, lo=2, hi=98):
    a, b = np.percentile(arr, lo), np.percentile(arr, hi)
    if b - a < 1e-8: return np.zeros_like(arr)
    return np.clip((arr - a) / (b - a), 0, 1).astype(np.float32)
