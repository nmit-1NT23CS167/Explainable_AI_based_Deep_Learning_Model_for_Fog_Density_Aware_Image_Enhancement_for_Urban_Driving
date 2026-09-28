"""Multi-layer Grad-CAM for the spatial dehazing signal in AOD-Net."""
import cv2, numpy as np
import torch, torch.nn.functional as F


class MultiLayerGradCAM:
    def __init__(self, model, layer_names=None, layer_weights=None):
        self.model         = model
        self.layer_names   = layer_names   or ["relu3", "relu4", "relu5"]
        self.layer_weights = layer_weights or [0.20,    0.30,    0.50]
        if len(self.layer_names) != len(self.layer_weights):
            raise ValueError("layer_names and layer_weights must have equal length")
        if not self.layer_names:
            raise ValueError("At least one Grad-CAM layer is required")
        if any(weight < 0 for weight in self.layer_weights):
            raise ValueError("layer_weights must be non-negative")
        weight_sum = sum(self.layer_weights)
        if weight_sum <= 0:
            raise ValueError("At least one layer weight must be positive")
        self.layer_weights = [weight / weight_sum for weight in self.layer_weights]
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

    def generate(self, input_tensor: torch.Tensor,
                 target_mask: torch.Tensor = None) -> np.ndarray:
        self.model.eval()
        self._acts.clear(); self._grads.clear()

        x = input_tensor.detach().clone().requires_grad_(True)
        output = self.model(x)

        # Attribute the dehazing change, optionally restricted to a region.
        change = (output - x).abs().mean(dim=1, keepdim=True)
        if target_mask is not None:
            mask = torch.as_tensor(target_mask, device=change.device,
                                   dtype=change.dtype)
            if mask.ndim == 2:
                mask = mask[None, None]
            elif mask.ndim == 3:
                mask = mask[:, None]
            if mask.ndim != 4 or mask.shape[0] not in (1, change.shape[0]):
                raise ValueError("target_mask must have shape HxW, BxHxW, or Bx1xHxW")
            mask = F.interpolate(mask, size=change.shape[-2:], mode="nearest")
            mask = mask.clamp_min(0)
            if mask.shape[0] == 1 and change.shape[0] > 1:
                mask = mask.expand(change.shape[0], -1, -1, -1)
            mask_sum = mask.sum()
            if mask_sum.item() <= 0:
                raise ValueError("target_mask must contain at least one positive pixel")
            score = (change * mask).sum() / mask_sum
        else:
            score = change.mean()
        self.model.zero_grad()
        score.backward()

        H, W   = input_tensor.shape[2], input_tensor.shape[3]
        fused  = np.zeros((H, W), dtype=np.float32)
        total_weight = 0.0

        for name, w in zip(self.layer_names, self.layer_weights):
            act  = self._acts.get(name)
            grad = self._grads.get(name)
            if act is None or grad is None:
                continue
            weights = grad.detach().mean(dim=(2, 3), keepdim=True)
            response = (weights * act.detach()).sum(dim=1, keepdim=True)
            cam = F.relu(response)
            if not torch.any(cam):
                cam = response.abs()
            cam = F.interpolate(cam, (H, W), mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()
            cam = _pct_norm(cam, 1, 99)
            fused += w * cam
            total_weight += w

        if total_weight == 0:
            return fused
        fused = _pct_norm(fused / total_weight, 1, 99)
        fused = cv2.GaussianBlur(fused, (5, 5), 1.2)
        return np.clip(fused, 0, 1).astype(np.float32)


# ── Backward-compat single-layer wrapper ─────────────────────────────────────
class GradCAM:
    """Drop-in replacement — internally uses MultiLayerGradCAM."""
    def __init__(self, model, target_layer=None):
        self._impl = MultiLayerGradCAM(model)

    def generate(self, inp, target_mask=None):
        return self._impl.generate(inp, target_mask=target_mask)

    def remove_hooks(self):
        self._impl.remove_hooks()


# ── Colourmap helpers ─────────────────────────────────────────────────────────
def apply_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """[0,1] float -> a continuous blue-green-yellow-red BGR heatmap."""
    values = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    return cv2.applyColorMap(np.rint(values * 255).astype(np.uint8),
                             cv2.COLORMAP_JET)


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
