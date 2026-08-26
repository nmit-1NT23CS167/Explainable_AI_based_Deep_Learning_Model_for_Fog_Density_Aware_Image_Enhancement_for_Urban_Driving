import cv2, numpy as np
from config import FOG_LAPLACIAN_THRESH, FOG_BRIGHT_THRESH


def compute_fog_score(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lap  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharp_fog = max(0.0, 1.0 - lap / FOG_LAPLACIAN_THRESH)
    bright_fog = max(0.0, (float(gray.mean()) - FOG_BRIGHT_THRESH) /
                    (255 - FOG_BRIGHT_THRESH))
    return float(np.clip(0.7 * sharp_fog + 0.3 * bright_fog, 0, 1))


def fog_label(s):
    if s < 0.25: return "Clear"
    if s < 0.50: return "Light fog"
    if s < 0.75: return "Moderate fog"
    return "Dense fog"


def fog_color_bgr(s):
    return (0, int((1-s)*200), int(s*255))
