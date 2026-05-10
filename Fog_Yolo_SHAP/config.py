"""
Central configuration for Fog + YOLO v11 + SHAP POC.
Edit this file to tune detection thresholds, alert zones, and SHAP settings.
"""

# ── Model ──────────────────────────────────────────────────────────────────────
YOLO_MODEL_SIZE   = "yolo11n.pt"   # nano=fastest; also: yolo11s/m/l/x.pt
YOLO_CONF_THRESH  = 0.35           # detection confidence threshold
YOLO_IOU_THRESH   = 0.45           # NMS IoU threshold
DEVICE            = "cpu"          # "cuda" if GPU available

# ── Camera / focal-length distance estimation ─────────────────────────────────
# Formula: Distance = (Known_Width_m × Focal_Length_px) / BBox_Width_px
# Focal length is approximated from a calibration image or set manually.
FOCAL_LENGTH_PX   = 800            # approximate for a standard dashcam
KNOWN_WIDTHS_M    = {              # average real-world widths per class
    "car":          1.8,
    "truck":        2.5,
    "bus":          2.5,
    "person":       0.5,
    "motorcycle":   0.8,
    "bicycle":      0.6,
    "traffic light":0.3,
    "stop sign":    0.6,
}
DEFAULT_WIDTH_M   = 1.0            # fallback for unknown classes

# ── Alert zones (meters) ───────────────────────────────────────────────────────
ZONE_CRITICAL     = 10.0           # RED   — immediate brake
ZONE_WARNING      = 25.0           # AMBER — slow down
ZONE_CAUTION      = 50.0           # YELLOW — be aware

# ── Fog detection ─────────────────────────────────────────────────────────────
FOG_LAPLACIAN_THRESH  = 80.0       # below this → frame considered foggy
FOG_BRIGHT_THRESH     = 170        # mean brightness above this also flags fog
FOG_CHECK_INTERVAL    = 15         # check every N frames (performance)

# ── SHAP ──────────────────────────────────────────────────────────────────────
SHAP_BACKGROUND_SAMPLES = 30       # KernelSHAP background dataset size
SHAP_EXPLAIN_EVERY_N    = 30       # run SHAP explanation every N frames
SHAP_MAX_EVALS          = 500      # max function evaluations (speed vs accuracy)
SHAP_TOP_K_FEATURES     = 8        # features shown in waterfall plot
SHAP_SAVE_PLOTS         = True     # save PNG plots to output/shap/

# ── Feature names fed to SHAP ─────────────────────────────────────────────────
# These are the per-detection scalar features extracted for SHAP analysis.
FEATURE_NAMES = [
    "bbox_width_px",
    "bbox_height_px",
    "bbox_area_px",
    "center_x_norm",     # 0–1 (left→right)
    "center_y_norm",     # 0–1 (top→bottom)
    "confidence",
    "fog_score",         # 0=clear, 1=dense fog
    "aspect_ratio",      # w/h of bounding box
]

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR        = "output"
SAVE_ANNOTATED    = True           # save annotated frames
VIDEO_FPS         = 20
FONT_SCALE        = 0.6
FONT_THICKNESS    = 2
