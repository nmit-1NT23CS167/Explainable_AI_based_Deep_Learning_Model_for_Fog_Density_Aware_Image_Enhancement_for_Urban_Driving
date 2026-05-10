"""
Streamlit Dashboard — Fog + YOLOv11 + SHAP Alert System
=========================================================
Usage:
    streamlit run dashboard.py

Features:
  • Upload a video or image
  • Run YOLOv11 detection with distance overlay
  • Show fog score and alert status
  • Display SHAP waterfall for most-at-risk detection
  • Download annotated output
"""

import os, sys, io, time, tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(page_title="Fog Safety AI", page_icon="🚗", layout="wide")

st.title("🚗 Fog Obstacle Detection + SHAP Explainable AI")
st.caption("YOLOv11 · Distance estimation · SHAP risk explanation · Foggy driving safety")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    model_size = st.selectbox("YOLO model", ["yolo11n.pt","yolo11s.pt","yolo11m.pt"],
                              help="n=fastest, m=most accurate")
    conf_thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
    run_shap    = st.checkbox("Run SHAP explanation", value=True)
    max_frames  = st.slider("Max frames to process", 5, 100, 20)
    st.divider()
    st.caption("SHAP initialises after 30 detections are collected.")


@st.cache_resource
def load_detector(model_size):
    from models.detector import YOLOv11Detector
    return YOLOv11Detector(model_path=model_size)


detector = load_detector(model_size)

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload foggy driving video or image",
    type=["mp4", "avi", "mov", "jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info("Upload a foggy road video or image to begin.")
    st.stop()

# ── Process ────────────────────────────────────────────────────────────────────
is_video = uploaded.name.lower().endswith(("mp4","avi","mov"))

if is_video:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
else:
    pil_img = Image.open(uploaded).convert("RGB")
    frame_bgr = np.array(pil_img)[:,:,::-1].copy()
    cap = None

from utils.fog_detector import compute_fog_score, fog_label
from utils.distance_estimator import estimate_distance, zone_label, zone_color_bgr
from utils.feature_extractor import detection_to_features, build_background_dataset
from utils.annotator import draw_detection, draw_fog_hud, draw_alert_banner

# ── Process frames ─────────────────────────────────────────────────────────────
frames_out = []
detection_log = []
all_detections_info = []

progress = st.progress(0, text="Processing frames...")
frame_count = 0

def process_frame(frame_bgr):
    fog_score = compute_fog_score(frame_bgr)
    h, w = frame_bgr.shape[:2]
    detections = detector.detect(frame_bgr)
    enriched = []
    for det in detections:
        box  = det["box_xyxy"]
        dist = estimate_distance(float(box[2]-box[0]), det["class_name"])
        feat = detection_to_features(box, det["confidence"], fog_score, w, h)
        enriched.append({**det, "distance_m": dist, "features": feat})
        detection_log.append(feat)
        all_detections_info.append({
            "class": det["class_name"], "conf": det["confidence"],
            "dist_m": dist, "zone": zone_label(dist), "fog": fog_score
        })
        draw_detection(frame_bgr, box, det["class_name"], det["confidence"], dist)
    draw_fog_hud(frame_bgr, fog_score)
    if enriched:
        closest = min(enriched, key=lambda d: d["distance_m"])
        draw_alert_banner(frame_bgr, closest["distance_m"], closest["class_name"])
    return frame_bgr, fog_score, enriched

if cap:
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        out_frame, fog_score, enriched = process_frame(frame)
        frames_out.append(out_frame)
        frame_count += 1
        progress.progress(frame_count / max(total_frames, 1),
                          text=f"Frame {frame_count}/{total_frames}")
    cap.release()
else:
    out_frame, fog_score, enriched = process_frame(frame_bgr)
    frames_out.append(out_frame)
    frame_count = 1
    progress.progress(1.0, text="Done")

progress.empty()

# ── Display results ────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Annotated output")
    if frames_out:
        display_frame = cv2.cvtColor(frames_out[-1], cv2.COLOR_BGR2RGB)
        st.image(display_frame, use_container_width=True,
                 caption=f"Frame {frame_count} — last processed")

with col2:
    st.subheader("Summary metrics")
    n_det = len(all_detections_info)
    n_crit = sum(1 for d in all_detections_info if d["zone"] == "CRITICAL")
    n_warn = sum(1 for d in all_detections_info if d["zone"] == "WARNING")
    avg_fog = np.mean([d["fog"] for d in all_detections_info]) if all_detections_info else 0

    st.metric("Total detections", n_det)
    st.metric("Critical alerts", n_crit, delta=None)
    st.metric("Warning alerts", n_warn)
    st.metric("Avg fog score", f"{avg_fog:.2f}")

    if all_detections_info:
        import pandas as pd
        df = pd.DataFrame(all_detections_info)
        st.dataframe(df.style.format({"conf":"{:.2f}","dist_m":"{:.1f}","fog":"{:.2f}"}),
                     height=220)

# ── SHAP section ───────────────────────────────────────────────────────────────
if run_shap and len(detection_log) >= 5:
    st.subheader("SHAP Explainability")

    with st.spinner("Running SHAP KernelExplainer..."):
        from xai.shap_explainer import FogAlertSHAPExplainer
        bg = build_background_dataset(detection_log, min(len(detection_log), 30))
        explainer = FogAlertSHAPExplainer(bg)

        # Explain the most-at-risk detection across all frames
        all_feats = np.stack(detection_log)
        from xai.shap_explainer import compute_risk_score
        risks = compute_risk_score(all_feats)
        top_idx = int(np.argmax(risks))
        result = explainer.explain(all_feats[top_idx])

    st.caption(
        f"SHAP analysis on the highest-risk detection "
        f"(risk score: {result['risk_score']:.3f}, base: {result['base_value']:.3f})"
    )

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.5))
    names  = [f[0] for f in result["top_features"]]
    values = [f[1] for f in result["top_features"]]
    colors = ["#d73027" if v > 0 else "#4575b4" for v in values]
    ax.barh(names, values, color=colors, height=0.55)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (contribution to risk score)")
    ax.set_title(f"SHAP waterfall — highest-risk detection\n"
                 f"Base: {result['base_value']:.3f} → Risk: {result['risk_score']:.3f}")
    for i, (v, name) in enumerate(zip(values, names)):
        sign = "+" if v >= 0 else ""
        ax.text(v + (0.005 if v >= 0 else -0.005), i,
                f"{sign}{v:.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_xlim(-0.6, 0.6)
    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("How to read this chart"):
        st.markdown("""
        Each bar shows how much a feature **pushes the risk score** up (red) or down (blue)
        from the baseline.

        - **fog_score +0.28** → fog is the biggest danger amplifier for this detection
        - **bbox_width_px +0.22** → large bounding box = object is close = high risk
        - **center_x_norm −0.10** → object is off-centre (beside lane), slightly safer
        - **confidence +0.05** → YOLO is sure → detection is real, not a ghost

        The sum of all bars + base_value ≈ the final predicted risk score.
        """)

elif run_shap:
    st.info("Not enough detections yet for SHAP. Process a longer video clip.")

st.success(f"Pipeline complete — {frame_count} frames, {len(all_detections_info)} detections.")
