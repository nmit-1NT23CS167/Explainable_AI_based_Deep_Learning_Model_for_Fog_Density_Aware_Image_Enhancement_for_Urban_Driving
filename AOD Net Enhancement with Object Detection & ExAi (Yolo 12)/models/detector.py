"""YOLOv11 detector — thin Ultralytics wrapper."""
import numpy as np
from ultralytics import YOLO
from config import YOLO_MODEL_SIZE, YOLO_CONF_THRESH, YOLO_IOU_THRESH, \
                   DEVICE, YOLO_INFERENCE_SIZE


class YOLOv11Detector:
    def __init__(self, model_path: str = YOLO_MODEL_SIZE):
        print(f"[YOLO] Loading {model_path} on {DEVICE}...")
        self.model = YOLO(model_path)
        self.device = DEVICE
        # Warm-up
        self.model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
        print("[YOLO] Ready.")

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        results = self.model(
            frame_bgr,
            conf=YOLO_CONF_THRESH,
            iou=YOLO_IOU_THRESH,
            imgsz=YOLO_INFERENCE_SIZE,
            device=self.device,
            verbose=False,
        )
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                detections.append({
                    "class_id":   int(boxes.cls[i].item()),
                    "class_name": self.model.names[int(boxes.cls[i].item())],
                    "confidence": float(boxes.conf[i].item()),
                    "box_xyxy":   boxes.xyxy[i].cpu().numpy(),
                })
        return detections

    @property
    def class_names(self):
        return self.model.names
