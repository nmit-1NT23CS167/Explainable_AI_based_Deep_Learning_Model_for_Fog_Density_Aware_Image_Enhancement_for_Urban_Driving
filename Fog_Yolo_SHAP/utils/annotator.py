"""
Frame annotation + display layout utilities.

Draws YOLO bounding boxes, distance labels, zone colours, fog HUD, and the
four-panel comparison grid:

    ┌─────────────────────┬─────────────────────┐
    │  Original foggy     │  AOD-Net enhanced   │
    │  (raw input)        │  (dehazed)          │
    ├─────────────────────┼─────────────────────┤
    │  YOLO detections    │  Status dashboard   │
    │  on enhanced frame  │  Fog / Alerts / XAI │
    └─────────────────────┴─────────────────────┘

Window is created with WINDOW_NORMAL so the user can freely resize it.
"""

import cv2
import numpy as np
from utils.distance_estimator import zone_color_bgr, zone_label, format_distance
from utils.fog_detector import fog_color_bgr, fog_label
from config import FONT_SCALE, FONT_THICKNESS, ZONE_CRITICAL, ZONE_WARNING

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN_NAME = "Fog Enhancement + YOLOv11 + SHAP  |  Q=quit  S=save  E=enhance  X=SHAP"


# ── Window setup ──────────────────────────────────────────────────────────────

def init_window(frame_w: int, frame_h: int):
    """Create a resizable OpenCV window sized to show the full 4-panel grid."""
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WIN_NAME, frame_w, frame_h)


# ── Per-detection drawing ─────────────────────────────────────────────────────

def draw_detection(frame: np.ndarray, box_xyxy,
                   class_name: str, confidence: float,
                   distance_m: float) -> np.ndarray:
    """Zone-coloured bounding box + label."""
    x1, y1, x2, y2 = (int(v) for v in box_xyxy)
    color = zone_color_bgr(distance_m)
    zone  = zone_label(distance_m)
    dist  = format_distance(distance_m)
    thick = 3 if distance_m <= ZONE_CRITICAL else 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

    label = f"{class_name} [{zone}] {dist}"
    (lw, lh), base = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    ly = max(y1 - 6, lh + 4)
    cv2.rectangle(frame, (x1, ly - lh - 4), (x1 + lw + 4, ly + base), color, -1)
    cv2.putText(frame, label, (x1 + 2, ly - 2), FONT,
                FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)
    return frame


def draw_fog_hud(frame: np.ndarray, fog_score: float) -> np.ndarray:
    """Top-right fog indicator."""
    h, w = frame.shape[:2]
    label = f"Fog: {fog_label(fog_score)} ({fog_score:.2f})"
    color = fog_color_bgr(fog_score)
    cv2.putText(frame, label, (w - 280, 24), FONT, 0.55, color, 2, cv2.LINE_AA)
    bx, by, bw, bh = w - 280, 32, 240, 6
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
    filled = int(bw * fog_score)
    if filled > 0:
        cv2.rectangle(frame, (bx, by), (bx + filled, by + bh), color, -1)
    return frame


def draw_alert_banner(frame: np.ndarray, closest_m: float,
                      class_name: str) -> np.ndarray:
    """Full-width red banner when object is in critical zone."""
    if closest_m > ZONE_CRITICAL:
        return frame
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 54), (w, h), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    msg = f"  !! ALERT: {class_name} at {format_distance(closest_m)} — BRAKE NOW!"
    cv2.putText(frame, msg, (8, h - 16), FONT, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_shap_badge(frame: np.ndarray, top_feature: str,
                    shap_val: float) -> np.ndarray:
    sign  = "+" if shap_val >= 0 else ""
    badge = f"SHAP: {top_feature}  {sign}{shap_val:.3f}"
    cv2.putText(frame, badge, (10, frame.shape[0] - 62),
                FONT, 0.50, (220, 180, 80), 1, cv2.LINE_AA)
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 24),
                FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


# ── Panel label helper ────────────────────────────────────────────────────────

def _label_panel(img: np.ndarray, text: str,
                 bg=(28, 28, 36), fg=(230, 230, 230)) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (img.shape[1], 24), bg, -1)
    cv2.putText(out, text, (8, 16), FONT, 0.50, fg, 1, cv2.LINE_AA)
    return out


# ── Status panel (bottom-right) ───────────────────────────────────────────────

def make_status_panel(pw: int, ph: int,
                      fog_score: float,
                      enriched: list,
                      enhancement_on: bool,
                      shap_result: dict = None,
                      frame_count: int = 0,
                      fps: float = 0.0) -> np.ndarray:
    """Dark-background status panel showing pipeline state, alerts, and SHAP."""
    panel = np.zeros((ph, pw, 3), dtype=np.uint8)
    panel[:] = (18, 18, 24)

    y = 0
    dy = 20

    def row(s, x=10, color=(210, 210, 210), scale=0.46, thick=1):
        nonlocal y
        y += dy
        cv2.putText(panel, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)

    def divider():
        nonlocal y
        y += 6
        cv2.line(panel, (8, y), (pw - 8, y), (55, 55, 68), 1)
        y += 4

    # Header bar
    cv2.rectangle(panel, (0, 0), (pw, 22), (38, 38, 52), -1)
    cv2.putText(panel, "SYSTEM STATUS", (8, 15), FONT, 0.50,
                (170, 170, 240), 1, cv2.LINE_AA)
    y = 22

    enh_col = (80, 220, 120) if enhancement_on else (110, 110, 110)
    row(f"AOD-Net: {'ON  — dehazing active' if enhancement_on else 'OFF — raw video'}", color=enh_col)
    row(f"Fog: {fog_label(fog_score)}  ({fog_score:.2f})",
        color=fog_color_bgr(fog_score))
    row(f"Frame: {frame_count}    FPS: {fps:.1f}", color=(150, 150, 150), scale=0.43)

    divider()
    row("Detections:", color=(200, 200, 200))
    if not enriched:
        row("  (none in frame)", color=(90, 90, 90), scale=0.43)
    else:
        for det in sorted(enriched, key=lambda d: d["distance_m"])[:5]:
            zcol = zone_color_bgr(det["distance_m"])
            row(f"  {det['class_name']:<12} {format_distance(det['distance_m']):>8}"
                f"  [{zone_label(det['distance_m'])}]", color=zcol, scale=0.43)

    divider()
    row("SHAP explanation:", color=(200, 200, 200))
    if shap_result:
        row(f"  Risk score: {shap_result['risk_score']:.3f}"
            f"  (base {shap_result['base_value']:.3f})",
            color=(200, 175, 80), scale=0.43)
        for feat, val in shap_result["top_features"][:4]:
            sign = "+" if val >= 0 else ""
            col  = (100, 100, 230) if val >= 0 else (80, 200, 100)
            row(f"  {feat:<18} {sign}{val:.3f}", color=col, scale=0.42)
    else:
        row("  Warming up (need 30+ detections)", color=(90, 90, 90), scale=0.42)

    # Bottom controls hint
    hint_y = ph - 8
    for line in ["S=save  Q/ESC=quit", "E=toggle enhance  X=SHAP"]:
        cv2.putText(panel, line, (10, hint_y), FONT, 0.40,
                    (110, 110, 110), 1, cv2.LINE_AA)
        hint_y -= 16

    return panel


# ── 4-panel compositor ────────────────────────────────────────────────────────

def build_4panel(raw_frame: np.ndarray,
                 enhanced_frame: np.ndarray,
                 detection_frame: np.ndarray,
                 fog_score: float,
                 enriched: list,
                 enhancement_on: bool,
                 shap_result: dict = None,
                 frame_count: int = 0,
                 fps: float = 0.0) -> np.ndarray:
    """
    Build the full 4-panel display grid from the four data sources.

    Each panel is the frame at half resolution so the total grid equals
    the original frame size (same as the video), keeping the window tidy.
    """
    h, w = raw_frame.shape[:2]
    ph, pw = h // 2, w // 2

    def fit(img):
        return cv2.resize(img, (pw, ph))

    tl = _label_panel(fit(raw_frame),       "[ 1 ] RAW FOGGY INPUT")
    tr = _label_panel(fit(enhanced_frame),  "[ 2 ] AOD-NET DEHAZED")
    bl = _label_panel(fit(detection_frame), "[ 3 ] YOLOv11 DETECTIONS + ALERTS")
    br = _label_panel(
            make_status_panel(pw, ph, fog_score, enriched,
                              enhancement_on, shap_result, frame_count, fps),
            "[ 4 ] STATUS / SHAP XAI")

    top    = np.hstack([tl, tr])
    bottom = np.hstack([bl, br])
    grid   = np.vstack([top, bottom])

    # Divider lines
    cv2.line(grid, (pw, 0), (pw, h), (70, 70, 85), 1)
    cv2.line(grid, (0, ph), (w, ph), (70, 70, 85), 1)

    return grid