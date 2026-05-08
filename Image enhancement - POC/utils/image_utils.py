"""
Image preprocessing and postprocessing utilities.
Handles BGR↔RGB conversion, tensor packing, and output normalization.
"""

import cv2
import numpy as np
import torch


def preprocess(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    BGR uint8 image → normalized float32 tensor for model input.

    Steps:
        1. BGR → RGB
        2. HWC → CHW
        3. [0,255] → [0.0, 1.0]
        4. Add batch dimension → (1, 3, H, W)
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor.to(device)


def postprocess(tensor: torch.Tensor) -> np.ndarray:
    """
    Model output tensor → BGR uint8 image for display/saving.

    Steps:
        1. Remove batch dim → (3, H, W)
        2. [0.0, 1.0] → [0, 255] uint8
        3. CHW → HWC
        4. RGB → BGR
    """
    out = tensor.squeeze(0).detach().cpu()
    out = (out.clamp(0.0, 1.0) * 255.0).byte()
    out = out.permute(1, 2, 0).numpy()  # HWC
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return bgr


def enhance_image(image_bgr: np.ndarray, apply_clahe: bool = True, gamma: float = 1.1) -> np.ndarray:
    """
    Apply aggressive defogging enhancement to the image.
    - Contrast stretching
    - CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast
    - Gamma correction for brightness adjustment
    """
    # Convert to LAB for better contrast enhancement
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Contrast stretching on L channel
    p2, p98 = np.percentile(l_channel, (2, 98))
    if p98 > p2:
        l_stretched = np.clip((l_channel - p2) * (255 / (p98 - p2)), 0, 255).astype(np.uint8)
    else:
        l_stretched = l_channel
    
    # Apply CLAHE for local contrast enhancement
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_stretched)
    else:
        l_enhanced = l_stretched
    
    # Gamma correction for brightness
    l_gamma = np.power(l_enhanced / 255.0, 1.0 / gamma) * 255.0
    l_gamma = np.clip(l_gamma, 0, 255).astype(np.uint8)
    
    # Merge back
    enhanced_lab = cv2.merge([l_gamma, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def sharpen_image(image_bgr: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Apply unsharp mask to restore perceived detail."""
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(image_bgr, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def add_text_label(image: np.ndarray, label: str, color=(255, 255, 255)) -> np.ndarray:
    """Overlay a text label on the top-left corner of an image."""
    img = image.copy()
    cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2, cv2.LINE_AA)
    return img


def make_comparison_grid(images: list, labels: list, target_width: int = 400) -> np.ndarray:
    """
    Stack multiple (image, label) pairs side by side for display.
    All images are resized to the same width while preserving aspect ratio.
    """
    resized = []
    for img, lbl in zip(images, labels):
        h, w = img.shape[:2]
        scale = target_width / w
        new_h = int(h * scale)
        r = cv2.resize(img, (target_width, new_h))
        r = add_text_label(r, lbl)
        resized.append(r)

    # Pad all to same height
    max_h = max(r.shape[0] for r in resized)
    padded = []
    for r in resized:
        pad = max_h - r.shape[0]
        padded.append(cv2.copyMakeBorder(r, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=0))

    return np.hstack(padded)