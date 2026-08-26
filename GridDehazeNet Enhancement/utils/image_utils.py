"""
Image I/O and preprocessing utilities for GridDehazeNet XAI pipeline.
"""

import cv2
import numpy as np
import torch
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str, max_side: int = 768) -> np.ndarray:
    """
    Load an image from disk as a uint8 BGR numpy array.
    Resizes so the longer side ≤ max_side (preserves aspect ratio).

    Args:
        path     : file path
        max_side : maximum pixel dimension

    Returns:
        (H, W, 3) uint8 BGR array
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {path!r}')

    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img   = cv2.resize(img,
                           (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Tensor conversions
# ─────────────────────────────────────────────────────────────────────────────

def bgr_to_tensor(image_bgr: np.ndarray,
                  device: torch.device | str = 'cpu') -> torch.Tensor:
    """
    uint8 BGR (H, W, 3) → float32 tensor (1, 3, H, W) in [0, 1].
    Converts BGR → RGB before packing.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t   = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    """
    float32 tensor (1, 3, H, W) in [0, 1] → uint8 BGR (H, W, 3).
    """
    out = tensor.squeeze(0).detach().cpu().clamp(0.0, 1.0)
    out = (out * 255.0).byte().permute(1, 2, 0).numpy()
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def post_process(enhanced_bgr: np.ndarray,
                 contrast_stretch: bool = True,
                 sharpen: bool = True,
                 sharpen_strength: float = 0.35) -> np.ndarray:
    """
    Mild post-processing to improve visual quality of the enhanced image.

    Steps:
        1. Percentile contrast stretching (2%–98%) — lifts shadow detail
        2. Unsharp mask sharpening — recovers edge crispness lost in fog

    Args:
        enhanced_bgr      : uint8 BGR dehazed image
        contrast_stretch  : apply percentile stretch
        sharpen           : apply unsharp mask
        sharpen_strength  : weight of the sharpening kernel

    Returns:
        processed uint8 BGR image
    """
    out = enhanced_bgr.copy().astype(np.float32)

    if contrast_stretch:
        lo, hi = np.percentile(out, 2), np.percentile(out, 98)
        if hi > lo:
            out = np.clip((out - lo) / (hi - lo) * 255.0, 0, 255)

    out = out.astype(np.uint8)

    if sharpen:
        blur    = cv2.GaussianBlur(out, (0, 0), sigmaX=2.0)
        out     = cv2.addWeighted(out, 1.0 + sharpen_strength,
                                  blur, -sharpen_strength, 0)

    return out.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_image(image: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    print(f'[IO] Saved → {path}')


def simple_fog_enhance(image_bgr: np.ndarray,
                       clip_limit: float = 3.0,
                       tile_grid_size: tuple[int, int] = (8, 8),
                       saturation_boost: float = 1.08) -> np.ndarray:
    """
    Lightweight fallback enhancement for foggy images when no trained model
    weights are available.

    This uses CLAHE on the L channel and a modest saturation boost to
    improve visibility and contrast without producing extreme colour shifts.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError('simple_fog_enhance requires a 3-channel BGR image')

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_boost, 0, 255)
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    enhanced = post_process(enhanced,
                            contrast_stretch=True,
                            sharpen=True,
                            sharpen_strength=0.35)
    return enhanced


def add_label(image: np.ndarray, text: str,
              bg: tuple = (20, 20, 30),
              fg: tuple = (230, 230, 230)) -> np.ndarray:
    """Stamp a header label bar on the top of an image panel."""
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), bg, -1)
    cv2.putText(out, text, (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, fg, 1, cv2.LINE_AA)
    return out


def make_panel_grid(panels: list[tuple[np.ndarray, str]],
                    cols: int = 3,
                    panel_w: int = 400) -> np.ndarray:
    """
    Arrange a list of (image, label) tuples into a grid.

    Args:
        panels  : list of (BGR image, label string) tuples
        cols    : number of columns in the grid
        panel_w : width of each panel in pixels

    Returns:
        Combined BGR image grid
    """
    # Resize and label each panel
    resized = []
    for img, label in panels:
        h, w = img.shape[:2]
        scale = panel_w / w
        new_h = max(1, int(h * scale))
        r = cv2.resize(img, (panel_w, new_h))
        r = add_label(r, label)
        resized.append(r)

    # Pad all panels to the same height
    max_h = max(p.shape[0] for p in resized)
    padded = []
    for p in resized:
        pad = max_h - p.shape[0]
        padded.append(cv2.copyMakeBorder(
            p, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(15, 15, 20)))

    # Arrange into rows
    rows = []
    for i in range(0, len(padded), cols):
        row_panels = padded[i:i + cols]
        # Pad row to full width with black
        while len(row_panels) < cols:
            blank = np.full_like(row_panels[0], 15)
            row_panels.append(blank)
        rows.append(np.hstack(row_panels))

    grid = np.vstack(rows)

    # Add thin dividers
    h, w = grid.shape[:2]
    for c in range(1, cols):
        x = c * panel_w
        if x < w:
            cv2.line(grid, (x, 0), (x, h), (60, 60, 70), 1)
    row_h = max_h
    for r in range(1, len(rows)):
        y = r * row_h
        if y < h:
            cv2.line(grid, (0, y), (w, y), (60, 60, 70), 1)

    return grid
