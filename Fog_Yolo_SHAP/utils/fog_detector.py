"""
Fog detector — estimates fog density from a single frame.

Uses two complementary signals:
  1. Laplacian variance  — foggy frames are blurry → low sharpness
  2. Mean brightness     — fog scatters light → uniformly bright frames

Returns a fog_score in [0.0, 1.0] where 1.0 = maximum fog.
"""

import cv2
import numpy as np
from config import FOG_LAPLACIAN_THRESH, FOG_BRIGHT_THRESH


def compute_fog_score(frame_bgr: np.ndarray) -> float:
    """
    Estimate fog density for a single BGR frame.

    Returns:
        float in [0.0, 1.0]  — 0 = clear, 1 = dense fog
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Signal 1: sharpness (low Laplacian variance → blurry → foggy)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Map: 0 var → fog=1.0, FOG_LAPLACIAN_THRESH var → fog=0.0
    sharpness_fog = max(0.0, 1.0 - lap_var / FOG_LAPLACIAN_THRESH)

    # Signal 2: brightness uniformity (high mean → washed-out → foggy)
    mean_bright = float(gray.mean())
    brightness_fog = max(0.0, (mean_bright - FOG_BRIGHT_THRESH) / (255 - FOG_BRIGHT_THRESH))

    # Combine: sharpness is the dominant signal
    fog_score = float(np.clip(0.7 * sharpness_fog + 0.3 * brightness_fog, 0.0, 1.0))
    return fog_score


def is_foggy(fog_score: float, threshold: float = 0.4) -> bool:
    return fog_score >= threshold


def fog_label(fog_score: float) -> str:
    if fog_score < 0.25:
        return "Clear"
    elif fog_score < 0.5:
        return "Light fog"
    elif fog_score < 0.75:
        return "Moderate fog"
    else:
        return "Dense fog"


def fog_color_bgr(fog_score: float):
    """Returns BGR color scaled from green (clear) to red (dense fog)."""
    r = int(fog_score * 255)
    g = int((1.0 - fog_score) * 200)
    return (0, g, r)
