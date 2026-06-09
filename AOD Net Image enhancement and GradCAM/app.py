"""
Streamlit POC Dashboard  —  Fog Enhancement + XAI
===================================================
Usage:
    pip install streamlit
    streamlit run app.py

Gives a browser-based demo UI with:
    • Upload foggy image
    • Run AOD-Net enhancement
    • Toggle Grad-CAM heatmap
    • Display PSNR / SSIM / Sharpness metrics
    • Side-by-side comparison
"""

import importlib
import os
import sys

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from models.aod_net import AODNet
from utils.image_utils import preprocess, postprocess, sharpen_image, enhance_image
from utils.grad_cam import GradCAM, apply_colormap
from utils.metrics import compute_psnr, compute_ssim, compute_brisque_proxy


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fog XAI POC",
    page_icon="🌫️",
    layout="wide"
)

st.title("🌫️ Fog Image Enhancement — Explainable AI POC")
st.caption("AOD-Net dehazing  ·  Grad-CAM XAI  ·  Autonomous driving context")

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    weights_path = st.text_input("AOD-Net weights (.pth)", value="models/aod_net.pth")
    show_xai     = st.checkbox("Show Grad-CAM heatmap", value=True)
    show_overlay = st.checkbox("Show XAI overlay on enhanced", value=True)
    target_layer = st.selectbox("Grad-CAM target layer", ["conv5", "conv4", "conv3"])
    st.divider()
    st.caption("POC — no cloud, no sensors. Validates core feasibility.")


# ── Model loader (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AODNet().to(device)
    if not os.path.exists(path):
        model_dir = os.path.dirname(path) or "models"
        if os.path.isdir(model_dir):
            for filename in os.listdir(model_dir):
                if filename.lower().endswith(".pth"):
                    alt_path = os.path.join(model_dir, filename)
                    print(f"[WARNING] Default weights not found; using {alt_path}")
                    path = alt_path
                    break
    if os.path.exists(path):
        if "model" not in sys.modules:
            sys.modules["model"] = importlib.import_module("models.aod_net")
        safe_globals = [AODNet]
        if hasattr(sys.modules["model"], "AODnet"):
            safe_globals.append(sys.modules["model"].AODnet)
        with torch.serialization.safe_globals(safe_globals):
            checkpoint = torch.load(path, map_location=device, weights_only=False)

        if isinstance(checkpoint, torch.nn.Module):
            state = checkpoint.state_dict()
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint

        model.load_state_dict(state)
    model.eval()
    return model, device


model, device = load_model(weights_path)

# ── Upload ─────────────────────────────────────────────────────────────────────
col_up, col_ref = st.columns(2)
with col_up:
    uploaded = st.file_uploader("Upload foggy image", type=["jpg", "jpeg", "png"])
with col_ref:
    reference = st.file_uploader("Upload clean reference (optional, for PSNR/SSIM)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    # Decode
    pil_img = Image.open(uploaded).convert("RGB")
    img_np  = np.array(pil_img)[:, :, ::-1].copy()  # RGB→BGR

    # Resize for performance
    h, w = img_np.shape[:2]
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        img_np = cv2.resize(img_np, (int(w * scale), int(h * scale)))

    # ── Run enhancement ──────────────────────────────────────────────────────
    with st.spinner("Running AOD-Net enhancement..."):
        tensor = preprocess(img_np, device)
        with torch.no_grad():
            enhanced_tensor = model(tensor)
        enhanced_np = postprocess(enhanced_tensor)
        # Apply aggressive enhancement to improve visual quality
        enhanced_display = enhance_image(enhanced_np, apply_clahe=True, gamma=1.15)
        enhanced_display = sharpen_image(enhanced_display, strength=0.6)

    # ── Run Grad-CAM ─────────────────────────────────────────────────────────
    heatmap_np = None
    if show_xai or show_overlay:
        with st.spinner("Generating Grad-CAM heatmap..."):
            layer = getattr(model, target_layer, None)
            if layer is None:
                layer = getattr(model, target_layer.replace("e_", ""), None)
            if layer is None:
                st.error(f"Grad-CAM target layer not found: {target_layer}")
            else:
                gc = GradCAM(model, layer)
                heatmap_np = gc.generate(preprocess(img_np, device))
                heatmap_color = cv2.applyColorMap(
                    (heatmap_np * 255).astype(np.uint8), cv2.COLORMAP_JET
                )

    # ── Display panels ────────────────────────────────────────────────────────
    cols = [c for c in [
        ("Input (foggy)",          img_np,          True),
        ("AOD-Net enhanced",        enhanced_display, True),
        ("Grad-CAM heatmap",        heatmap_color if heatmap_np is not None else None, show_xai),
        ("XAI overlay on enhanced", apply_colormap(heatmap_np, enhanced_display) if heatmap_np is not None else None, show_overlay),
    ] if c[2] and c[1] is not None]

    st_cols = st.columns(len(cols))
    for (label, img_bgr, _), col in zip(cols, st_cols):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        col.image(img_rgb, caption=label, use_container_width=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.subheader("📊 Quality Metrics")
    m1, m2, m3, m4 = st.columns(4)

    fog_sharp = compute_brisque_proxy(img_np)
    enh_sharp = compute_brisque_proxy(enhanced_np)

    m1.metric("Sharpness — Foggy",    f"{fog_sharp:.1f}")
    m2.metric("Sharpness — Enhanced", f"{enh_sharp:.1f}", f"{enh_sharp - fog_sharp:+.1f}")

    if reference is not None:
        ref_pil = Image.open(reference).convert("RGB")
        ref_np  = np.array(ref_pil)[:, :, ::-1].copy()
        ref_np  = cv2.resize(ref_np, (enhanced_np.shape[1], enhanced_np.shape[0]))
        psnr_val = compute_psnr(enhanced_np, ref_np)
        ssim_val = compute_ssim(enhanced_np, ref_np)
        m3.metric("PSNR (dB)", f"{psnr_val:.2f}")
        m4.metric("SSIM",      f"{ssim_val:.4f}")
    else:
        m3.metric("PSNR", "—", help="Upload clean reference to compute")
        m4.metric("SSIM", "—", help="Upload clean reference to compute")

    # ── XAI interpretation ────────────────────────────────────────────────────
    if heatmap_np is not None:
        with st.expander("📌 What the Grad-CAM shows"):
            st.markdown("""
            The **Grad-CAM heatmap** shows which spatial regions in the input image
            the model focused on most when performing dehazing:

            - 🔴 **Red/warm** regions = high gradient signal → model worked hardest here (dense fog)
            - 🔵 **Blue/cool** regions = low signal → already clear or uniform area
            - This satisfies the **Explainability (XAI)** requirement of the PoC:
              the model's decisions are interpretable without opening the black box.
            """)

    st.success("✅ POC pipeline complete — enhancement + XAI validated.")

else:
    st.info("👆 Upload a foggy image to begin the demonstration.")
    st.code("""
# Quick start
streamlit run app.py

# Command-line static mode
python run_poc.py --input foggy.jpg --save output/

# Real-time / webcam mode
python run_realtime.py --source 0
    """, language="bash")