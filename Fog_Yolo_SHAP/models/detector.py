"""
YOLOv11 detector wrapper (via Ultralytics).

Encapsulates model loading, inference, and result parsing so the
main pipeline doesn't need to touch the Ultralytics API directly.

YOLOv11 is the latest architecture from Ultralytics (2024).
It is loaded via: from ultralytics import YOLO
"""

from __future__ import annotations
import numpy as np
import torch
from ultralytics import YOLO
from config import YOLO_MODEL_SIZE, YOLO_CONF_THRESH, YOLO_IOU_THRESH, DEVICE


class YOLOv11Detector:
    """
    Thin wrapper around Ultralytics YOLOv11.

    Usage:
        detector = YOLOv11Detector()
        detections = detector.detect(frame_bgr)
        for det in detections:
            print(det["class_name"], det["confidence"], det["box_xyxy"])
    """

    def __init__(self, model_path: str = YOLO_MODEL_SIZE, device: str = DEVICE):
        print(f"[Detector] Loading {model_path} on {device}...")
        self.model  = YOLO(model_path)
        self.device = device
        # Warm-up pass to avoid first-frame latency spike
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        print("[Detector] Ready.")

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """
        Run YOLOv11 inference on a single BGR frame.

        Returns:
            List of detection dicts, each containing:
                class_id   : int
                class_name : str
                confidence : float  (0–1)
                box_xyxy   : np.ndarray [x1, y1, x2, y2] in pixels
        """
        results = self.model(
            frame_bgr,
            conf=YOLO_CONF_THRESH,
            iou=YOLO_IOU_THRESH,
            device=self.device,
            verbose=False,
        )

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id   = int(boxes.cls[i].item())
                cls_name = self.model.names[cls_id]
                conf     = float(boxes.conf[i].item())
                xyxy     = boxes.xyxy[i].cpu().numpy()   # [x1,y1,x2,y2]

                detections.append({
                    "class_id":   cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "box_xyxy":   xyxy,
                })

        return detections

    @property
    def class_names(self) -> dict:
        return self.model.names
