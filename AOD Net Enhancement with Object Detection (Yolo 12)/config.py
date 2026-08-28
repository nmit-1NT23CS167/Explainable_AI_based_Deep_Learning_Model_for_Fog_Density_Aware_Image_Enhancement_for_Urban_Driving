"""
Unified Configuration — AOD-Net + GradCAM + YOLOv12 + SHAP
All latency-reduction switches are in the PERFORMANCE section.
"""

# ── Models ────────────────────────────────────────────────────────────────────
AOD_WEIGHTS       = "AOD Net Enhancement with Object Detection (Yolo 12)/models/AOD_net_epoch_relu_10.pth"
YOLO_MODEL_SIZE   = "yolo12n.pt"        # nano=fastest; s/m/l/x for accuracy
YOLO_CONF_THRESH  = 0.35
YOLO_IOU_THRESH   = 0.45
DEVICE            = "cpu"               # "cuda" if GPU available

# ── Distance estimation ───────────────────────────────────────────────────────
FOCAL_LENGTH_PX   = 800
KNOWN_WIDTHS_M    = {
    "car": 1.8, "truck": 2.5, "bus": 2.5, "person": 0.5,
    "motorcycle": 0.8, "bicycle": 0.6, "traffic light": 0.3, "stop sign": 0.6,
}
DEFAULT_WIDTH_M   = 1.0
ZONE_CRITICAL     = 10.0
ZONE_WARNING      = 25.0
ZONE_CAUTION      = 50.0

# ── Fog detection ─────────────────────────────────────────────────────────────
FOG_LAPLACIAN_THRESH  = 80.0
FOG_BRIGHT_THRESH     = 170
FOG_CHECK_INTERVAL    = 15      # re-measure fog every N frames

# ── SHAP ──────────────────────────────────────────────────────────────────────
SHAP_BACKGROUND_SAMPLES = 30
SHAP_EXPLAIN_EVERY_N    = 60    # run SHAP every N frames (expensive)
SHAP_MAX_EVALS          = 300
SHAP_TOP_K_FEATURES     = 8
SHAP_SAVE_PLOTS         = True
FEATURE_NAMES = [
    "bbox_width_px", "bbox_height_px", "bbox_area_px",
    "center_x_norm", "center_y_norm", "confidence",
    "fog_score", "aspect_ratio",
]

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
GRADCAM_LAYERS        = ["conv3", "conv4", "conv5"]
GRADCAM_WEIGHTS       = [0.20, 0.30, 0.50]
GRADCAM_EVERY_N       = 5       # recompute heatmap every N frames
GRADCAM_OVERLAY_ALPHA = 0.50    # blend strength on overlay panel

# ── PERFORMANCE / LATENCY ─────────────────────────────────────────────────────
# These are the primary knobs for reducing end-to-end latency.
AOD_INFERENCE_SCALE   = 0.5    # run AOD-Net at 50% resolution, resize back
                                # 0.5 gives ~4x speed-up with minor quality loss
YOLO_INFERENCE_SIZE   = 416    # YOLO input resolution (320/416/640)
                                # 320 fastest, 640 most accurate
DISPLAY_SCALE         = 1.0    # scale the 4-panel window (0.75 saves bandwidth)
SKIP_ENHANCEMENT_WHEN_CLEAR = True   # skip AOD-Net if fog_score < threshold
FOG_SKIP_THRESH       = 0.15   # fog_score below this → skip enhancement
USE_THREADING         = True   # run SHAP + GradCAM on background thread
FRAME_SKIP            = 0      # process every Nth frame (0=all, 1=every other)

# ── Voice alerts ─────────────────────────────────────────────────────────────
VOICE_ALERTS          = True   # speak cooldown-gated alerts asynchronously
VOICE_RATE            = 165    # words per minute
VOICE_VOLUME          = 1.0    # range 0.0–1.0

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR      = "AOD Net Enhancement with Object Detection (Yolo 12)/output"
VIDEO_FPS       = 20
FONT_SCALE      = 0.55
FONT_THICKNESS  = 2
SAVE_ANNOTATED  = True
