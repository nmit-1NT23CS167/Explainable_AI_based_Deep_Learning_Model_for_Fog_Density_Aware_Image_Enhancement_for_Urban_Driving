"""Multi-cue distance estimation: width + height + area fusion with EMA smoothing."""
import numpy as np
from config import FOCAL_LENGTH_PX, KNOWN_WIDTHS_M, DEFAULT_WIDTH_M, \
                   ZONE_CRITICAL, ZONE_WARNING, ZONE_CAUTION

KNOWN_HEIGHTS_M = {
    "car": 1.5, "truck": 3.5, "bus": 3.2, "person": 1.75,
    "motorcycle": 1.2, "bicycle": 1.1, "traffic light": 0.9, "stop sign": 0.75,
}
FUSION_WEIGHTS = {
    "car":    (0.60, 0.25, 0.15), "truck":  (0.55, 0.30, 0.15),
    "bus":    (0.55, 0.30, 0.15), "person": (0.25, 0.60, 0.15),
    "motorcycle": (0.50, 0.35, 0.15), "bicycle": (0.40, 0.45, 0.15),
}
DEFAULT_WEIGHTS = (0.45, 0.40, 0.15)

_ema: dict[str, float] = {}
EMA_ALPHA = 0.35


def reset_ema(): _ema.clear()


def estimate_distance(box_xyxy, class_name: str,
                      focal_length_px: float = None,
                      smooth: bool = True) -> float:
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    bw = max(x2 - x1, 1.0);  bh = max(y2 - y1, 1.0)
    fl = focal_length_px or float(FOCAL_LENGTH_PX)
    cls = class_name.lower()
    rw = KNOWN_WIDTHS_M.get(cls, DEFAULT_WIDTH_M)
    rh = KNOWN_HEIGHTS_M.get(cls, 1.2)
    ww, wh, wa = FUSION_WEIGHTS.get(cls, DEFAULT_WEIGHTS)

    d_w = (rw * fl) / bw
    d_h = (rh * fl) / bh
    d_a = fl * (rw * rh) ** 0.5 / (bw * bh) ** 0.5
    fused = float(np.clip(ww*d_w + wh*d_h + wa*d_a, 0.5, 200.0))

    if smooth:
        _ema[cls] = EMA_ALPHA * fused + (1 - EMA_ALPHA) * _ema.get(cls, fused)
        fused = _ema[cls]
    return round(fused, 1)


def zone_label(d):
    if d <= ZONE_CRITICAL: return "CRITICAL"
    if d <= ZONE_WARNING:  return "WARNING"
    if d <= ZONE_CAUTION:  return "CAUTION"
    return "SAFE"


def zone_color_bgr(d):
    if d <= ZONE_CRITICAL: return (0, 0, 220)
    if d <= ZONE_WARNING:  return (0, 140, 255)
    if d <= ZONE_CAUTION:  return (0, 220, 220)
    return (0, 200, 80)


def format_distance(d):
    if d >= 200: return "far"
    return f"{d:.1f} m" if d >= 1 else f"{d*100:.0f} cm"
