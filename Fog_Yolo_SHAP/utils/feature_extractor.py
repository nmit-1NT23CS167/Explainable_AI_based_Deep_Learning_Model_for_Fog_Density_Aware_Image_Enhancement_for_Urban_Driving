"""
Feature extractor: YOLO detection → flat numerical feature vector.

SHAP requires a fixed-size numerical input. This module converts the
raw bounding-box + metadata from each detection into a standardised
feature vector that the SHAP explainer can analyse.

Feature layout (matches config.FEATURE_NAMES):
    0  bbox_width_px
    1  bbox_height_px
    2  bbox_area_px
    3  center_x_norm    (0–1, left to right)
    4  center_y_norm    (0–1, top to bottom)
    5  confidence
    6  fog_score        (0–1)
    7  aspect_ratio     (width / height)
"""

import numpy as np
from config import FEATURE_NAMES


def detection_to_features(box_xyxy, confidence: float, fog_score: float,
                           frame_w: int, frame_h: int) -> np.ndarray:
    """
    Convert a single detection into a feature vector.

    Args:
        box_xyxy    : [x1, y1, x2, y2] bounding box in pixels
        confidence  : detection confidence in [0, 1]
        fog_score   : current frame fog score in [0, 1]
        frame_w/h   : frame dimensions for normalisation

    Returns:
        features : np.ndarray of shape (len(FEATURE_NAMES),) = (8,)
    """
    x1, y1, x2, y2 = float(box_xyxy[0]), float(box_xyxy[1]), \
                      float(box_xyxy[2]), float(box_xyxy[3])

    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    return np.array([
        w,                                          # bbox_width_px
        h,                                          # bbox_height_px
        w * h,                                      # bbox_area_px
        cx / max(frame_w, 1),                       # center_x_norm
        cy / max(frame_h, 1),                       # center_y_norm
        float(confidence),                          # confidence
        float(fog_score),                           # fog_score
        w / max(h, 1),                              # aspect_ratio
    ], dtype=np.float32)


def build_background_dataset(detections_log: list, n: int = 30) -> np.ndarray:
    """
    Build the SHAP background dataset from accumulated detection feature vectors.

    Args:
        detections_log : list of feature np.ndarrays collected across frames
        n              : target background size (randomly sampled)

    Returns:
        background : np.ndarray of shape (n, 8) or (len, 8) if fewer available
    """
    if len(detections_log) == 0:
        # Fallback: random uniform background
        return np.random.rand(n, len(FEATURE_NAMES)).astype(np.float32)

    data = np.stack(detections_log, axis=0)
    if len(data) >= n:
        idx = np.random.choice(len(data), n, replace=False)
        return data[idx]
    return data
