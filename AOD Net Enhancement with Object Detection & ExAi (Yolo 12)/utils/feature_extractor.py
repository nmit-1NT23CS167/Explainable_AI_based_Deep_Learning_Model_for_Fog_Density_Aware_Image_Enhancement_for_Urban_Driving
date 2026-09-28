import numpy as np
from config import FEATURE_NAMES


def detection_to_features(box_xyxy, confidence, fog_score, frame_w, frame_h):
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    w, h = x2-x1, y2-y1
    return np.array([
        w, h, w*h,
        (x1+x2)/2/max(frame_w,1),
        (y1+y2)/2/max(frame_h,1),
        float(confidence), float(fog_score), w/max(h,1),
    ], dtype=np.float32)


def build_background_dataset(log, n=30):
    if not log:
        return np.random.rand(n, len(FEATURE_NAMES)).astype(np.float32)
    data = np.stack(log)
    if len(data) >= n:
        return data[np.random.choice(len(data), n, replace=False)]
    return data
