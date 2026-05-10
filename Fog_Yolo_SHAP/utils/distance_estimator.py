"""
Distance estimation via the pinhole camera model.

    Distance (m) = (Real_Width_m × Focal_Length_px) / BBox_Width_px

Assumptions (PoC):
  - Camera focal length is calibrated or approximated.
  - Known real-world widths per object class (config.py).
  - No stereo depth; this is a monocular approximation suitable for a PoC.

Accuracy note: error is typically ±15–30% without proper calibration.
For a production system, use stereo cameras or LiDAR fusion.
"""

import numpy as np
from config import FOCAL_LENGTH_PX, KNOWN_WIDTHS_M, DEFAULT_WIDTH_M, \
    ZONE_CRITICAL, ZONE_WARNING, ZONE_CAUTION


def estimate_distance(bbox_width_px: float, class_name: str) -> float:
    """
    Estimate distance in metres from bounding-box pixel width.

    Args:
        bbox_width_px : pixel width of the detected bounding box
        class_name    : YOLO class label (e.g. "car", "person")

    Returns:
        distance_m : estimated distance in metres (>0)
    """
    if bbox_width_px < 1:
        return 9999.0

    real_width = KNOWN_WIDTHS_M.get(class_name.lower(), DEFAULT_WIDTH_M)
    distance_m = (real_width * FOCAL_LENGTH_PX) / bbox_width_px
    return max(0.1, round(distance_m, 1))


def zone_label(distance_m: float) -> str:
    if distance_m <= ZONE_CRITICAL:
        return "CRITICAL"
    elif distance_m <= ZONE_WARNING:
        return "WARNING"
    elif distance_m <= ZONE_CAUTION:
        return "CAUTION"
    else:
        return "SAFE"


def zone_color_bgr(distance_m: float):
    """BGR color per zone: red → amber → yellow → green."""
    if distance_m <= ZONE_CRITICAL:
        return (0, 0, 220)      # red
    elif distance_m <= ZONE_WARNING:
        return (0, 140, 255)    # amber
    elif distance_m <= ZONE_CAUTION:
        return (0, 220, 220)    # yellow
    else:
        return (0, 200, 80)     # green


def format_distance(distance_m: float) -> str:
    if distance_m > 999:
        return "unknown"
    if distance_m < 1.0:
        return f"{distance_m * 100:.0f} cm"
    return f"{distance_m:.1f} m"
