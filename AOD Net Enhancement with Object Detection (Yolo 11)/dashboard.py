"""
Unified Streamlit Dashboard
============================
AOD-Net Enhancement + Grad-CAM XAI + YOLOv11 Detection + SHAP Risk Explanation

Usage:
    streamlit run dashboard.py
"""
import os, sys, tempfile
import cv2, numpy as np
import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="Fog Safety AI — Unified",
    page_icon="🚗", layout="wide"
)
st.title("🚗🌫️ Fog Safety AI — Enhancement + Detection + XAI")
st.caption("AOD-Net · Grad-CAM · YOLOv11 · SHAP · Autonomous Driving Safety")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    aod_weights  = st.text_input("AOD-Net weights", "models/AOD_net_epoch_relu_10.pth")
    yolo_model   = st.selectbox("YOLO model", ["yolo11n.pt","yolo11s.pt","yolo11m.pt"])
    conf         = st.slider("YOLO confidence", 0.1, 0.9, 0.35, 0.05)
    run_gradcam  = st.checkbox("Grad-CAM heatmap", True)
    run_shap     = st.checkbox("SHAP explanation", True)
    max_frames   = st.slider("Max video frames", 5, 60, 15)
    enhance_scale = st.slider("AOD-Net speed (scale)", 0.25, 1.0, 0.5, 0.25,
                              help="0.5 = 2× faster, slight quality loss")
    st.divider()
    st.caption("SHAP initialises after 30+ detections.")


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_enhancer(weights, scale):
    from models.aod_net import AODNetEnhancer
    return AODNetEnhancer(weights_path=weights, device="cpu",
                          inference_scale=scale)

@st.cache_resource
def load_detector(model_name):
    from models.detector import YOLOv11Detector
    return YOLOv11Detector(model_path=model_name)


enhancer = load_enhancer(aod_weights, enhance_scale)
detector = load_detector(yolo_model)

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload foggy driving video or image",
    type=["mp4","avi","mov","jpg","jpeg","png"]
)
if not uploaded:
    st.info("Upload a foggy road video or image to begin.")
    st.stop()

is_video = uploaded.name.lower().endswith(("mp4","avi","mov"))

# ── Process ────────────────────────────────────────────────────────────────────
from utils.fog_detector      import compute_fog_score, fog_label
from utils.distance_estimator import estimate_distance, zone_label, zone_color_bgr, format_distance
from utils.feature_extractor  import detection_to_features, build_background_dataset
from utils.grad_cam           import MultiLayerGradCAM, apply_colormap, apply_heatmap
from config import ZONE_CRITICAL, GRADCAM_LAYERS, GRADCAM_WEIGHTS

def annotate_frame(frame_bgr, enriched, fog_score):
    out = frame_bgr.copy()
    for det in enriched:
        x1,y1,x2,y2 = (int(v) for v in det["box_xyxy"])
        color = zone_color_bgr(det["distance_m"])
        thick = 3 if det["distance_m"] <= ZONE_CRITICAL else 2
        cv2.rectangle(out,(x1,y1),(x2,y2),color,thick)
        label = f"{det['class_name']} [{zone_label(det['distance_m'])}] {format_distance(det['distance_m'])}"
        (lw,lh),base = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.55,2)
        ly = max(y1-6,lh+4)
        cv2.rectangle(out,(x1,ly-lh-4),(x1+lw+4,ly+base),color,-1)
        cv2.putText(out,label,(x1+2,ly-2),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),2,cv2.LINE_AA)
    # fog HUD
    h,w = out.shape[:2]
    from utils.fog_detector import fog_color_bgr
    col = fog_color_bgr(fog_score)
    cv2.putText(out,f"Fog: {fog_label(fog_score)} ({fog_score:.2f})",
                (w-280,24),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,2,cv2.LINE_AA)
    if enriched:
        closest = min(enriched,key=lambda d:d["distance_m"])
        if closest["distance_m"] <= ZONE_CRITICAL:
            ov=out.copy()
            cv2.rectangle(ov,(0,h-50),(w,h),(0,0,200),-1)
            cv2.addWeighted(ov,0.6,out,0.4,0,out)
            cv2.putText(out,f"!! ALERT: {closest['class_name']} @ {format_distance(closest['distance_m'])} — BRAKE!",
                        (8,h-14),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2,cv2.LINE_AA)
    return out


if is_video:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(uploaded.read()); tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
else:
    pil = Image.open(uploaded).convert("RGB")
    frame_bgr0 = np.array(pil)[:,:,::-1].copy()
    cap = None

frames_out, raw_frames, detection_log, all_info = [], [], [], []
last_enh, last_overlay = None, None
progress = st.progress(0, "Processing...")
frame_count = 0

def process_one(frame_bgr):
    h, w = frame_bgr.shape[:2]
    fog  = compute_fog_score(frame_bgr)
    enh  = enhancer.enhance(frame_bgr)
    dets = detector.detect(enh)
    enriched = []
    for det in dets:
        box  = det["box_xyxy"]
        dist = estimate_distance(box, det["class_name"], smooth=False)
        feat = detection_to_features(box, det["confidence"], fog, w, h)
        enriched.append({**det, "distance_m": dist, "features": feat})
        detection_log.append(feat)
        all_info.append({"class": det["class_name"], "conf": det["confidence"],
                         "dist_m": dist, "zone": zone_label(dist), "fog": fog})
    ann = annotate_frame(enh, enriched, fog)
    return frame_bgr, enh, ann, fog, enriched

if cap:
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        raw, enh, ann, fog, enriched = process_one(frame)
        raw_frames.append(raw); frames_out.append(ann)
        last_enh = enh; last_fog = fog; last_enriched = enriched
        frame_count += 1
        progress.progress(frame_count/max(total,1), f"Frame {frame_count}/{total}")
    cap.release()
else:
    raw, enh, ann, fog, enriched = process_one(frame_bgr0)
    raw_frames.append(raw); frames_out.append(ann)
    last_enh = enh; last_fog = fog; last_enriched = enriched
    frame_count = 1
    progress.progress(1.0, "Done")

progress.empty()

# ── Results grid ──────────────────────────────────────────────────────────────
st.subheader("Results")
cols = st.columns(3)
cols[0].image(cv2.cvtColor(raw_frames[-1],    cv2.COLOR_BGR2RGB),
              caption="1. Raw foggy input", use_container_width=True)
cols[1].image(cv2.cvtColor(last_enh,          cv2.COLOR_BGR2RGB),
              caption="2. AOD-Net enhanced", use_container_width=True)
cols[2].image(cv2.cvtColor(frames_out[-1],    cv2.COLOR_BGR2RGB),
              caption="3. YOLO detections + alerts", use_container_width=True)

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
if run_gradcam and last_enh is not None:
    with st.spinner("Running Grad-CAM (multi-layer)..."):
        gc = MultiLayerGradCAM(enhancer.model, GRADCAM_LAYERS, GRADCAM_WEIGHTS)
        inp = (torch.from_numpy(cv2.cvtColor(last_enh, cv2.COLOR_BGR2RGB)
                                .astype(np.float32)/255.)
               .permute(2,0,1).unsqueeze(0))
        heatmap = gc.generate(inp)
        hm_bgr  = apply_heatmap(heatmap)
        overlay = apply_colormap(heatmap, last_enh, alpha=0.55)
        gc.remove_hooks()

    gc_cols = st.columns(2)
    gc_cols[0].image(cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2RGB),
                     caption="4. Grad-CAM heatmap (TURBO)", use_container_width=True)
    gc_cols[1].image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                     caption="5. XAI overlay on enhanced", use_container_width=True)

    with st.expander("What does the Grad-CAM show?"):
        st.markdown("""
        - 🔵 **Blue** = road surface — large uniform fog region the model worked to clear
        - 🟢 **Green/yellow** = lane markings, road edges — high-contrast boundaries
        - 🟠 **Orange/red** = traffic signs, vehicles — salient objects the model focused on most
        - The heatmap uses **three convolutional layers fused** (conv3+conv4+conv5)
          so it captures both fine edges (early) and semantic regions (late).
        """)

# ── Metrics ───────────────────────────────────────────────────────────────────
st.subheader("Metrics")
m1,m2,m3,m4 = st.columns(4)
n_det  = len(all_info)
n_crit = sum(1 for d in all_info if d["zone"]=="CRITICAL")
n_warn = sum(1 for d in all_info if d["zone"]=="WARNING")
avg_fog = float(np.mean([d["fog"] for d in all_info])) if all_info else 0
m1.metric("Detections", n_det)
m2.metric("Critical alerts", n_crit)
m3.metric("Warning alerts", n_warn)
m4.metric("Avg fog score", f"{avg_fog:.2f}")

if all_info:
    import pandas as pd
    df = pd.DataFrame(all_info)
    st.dataframe(df.style.format({"conf":"{:.2f}","dist_m":"{:.1f}","fog":"{:.2f}"}),
                 height=200)

# ── SHAP ──────────────────────────────────────────────────────────────────────
if run_shap and len(detection_log) >= 5:
    st.subheader("SHAP Explainability")
    with st.spinner("Running SHAP KernelExplainer..."):
        from xai.shap_explainer import FogAlertSHAPExplainer, compute_risk_score
        bg     = build_background_dataset(detection_log, min(len(detection_log), 30))
        exp    = FogAlertSHAPExplainer(bg)
        all_f  = np.stack(detection_log)
        top_i  = int(np.argmax(compute_risk_score(all_f)))
        result = exp.explain(all_f[top_i])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.5))
    names  = [f[0] for f in result["top_features"]]
    values = [f[1] for f in result["top_features"]]
    colors = ["#d73027" if v>0 else "#4575b4" for v in values]
    ax.barh(names, values, color=colors, height=0.55)
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    for i,(v,_) in enumerate(zip(values, names)):
        sign="+"; sign="" if v<0 else "+"
        ax.text(v+(0.005 if v>=0 else -0.005),i,f"{sign}{v:.3f}",
                va="center",ha="left" if v>=0 else "right",fontsize=9)
    ax.set_xlabel("SHAP value (contribution to risk score)")
    ax.set_title(f"SHAP waterfall — highest-risk detection\n"
                 f"Base: {result['base_value']:.3f} → Risk: {result['risk_score']:.3f}")
    ax.set_xlim(-0.6, 0.6)
    plt.tight_layout()
    st.pyplot(fig)

elif run_shap:
    st.info("Process a longer video for SHAP (needs 30+ detections).")

st.success(f"✅ Unified pipeline complete — {frame_count} frames, "
           f"{len(all_info)} detections.")
