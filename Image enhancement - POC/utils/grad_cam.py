"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for AOD-Net
Visualizes which regions the model focuses on when removing fog.

For a regression/enhancement model (not classification), we use
the L1 reconstruction loss w.r.t. the target layer's activations.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM adapted for image enhancement models.
    Instead of a class score, we use the mean output activation
    as the scalar signal to backpropagate through.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given input.

        Args:
            input_tensor: (1, 3, H, W) float32 tensor in [0, 1]

        Returns:
            heatmap: (H, W) float32 numpy array in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        # Scalar signal: mean output (enhancement quality proxy)
        score = output.mean()

        # Backward pass
        self.model.zero_grad()
        score.backward()

        # Pool gradients across channels → importance weights
        weights = self.gradients.abs().mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Resize to input resolution
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
            cam = np.clip(cam ** 0.75, 0, 1)
        else:
            cam = np.zeros_like(cam)

        return cam


def apply_colormap(heatmap: np.ndarray, image_bgr: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original BGR image.

    Args:
        heatmap: (H, W) float32 in [0, 1]
        image_bgr: (H, W, 3) uint8 BGR image
        colormap: OpenCV colormap flag to use for visualization.
        alpha: Blend strength for the heatmap overlay.

    Returns:
        overlay: (H, W, 3) uint8 BGR overlay
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
    return overlay