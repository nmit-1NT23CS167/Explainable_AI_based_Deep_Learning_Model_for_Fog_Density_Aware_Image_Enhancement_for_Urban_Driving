"""
POC — Simulated Real-Time Mode
================================
Usage:
    python run_realtime.py                        # uses webcam (index 0)
    python run_realtime.py --source video.mp4     # uses a video file
    python run_realtime.py --source 1             # webcam index 1

Controls:
    Q / ESC  → quit
    X        → toggle Grad-CAM XAI overlay on/off
    S        → save current frame to output/

Each frame goes through:
    Frame → preprocess → AOD-Net → postprocess → [Grad-CAM] → display
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from models.aod_net import AODNet
from utils.image_utils import preprocess, postprocess, add_text_label, enhance_image, sharpen_image
from utils.grad_cam import GradCAM, apply_colormap


RESIZE_W = 480  # resize input for real-time performance


def _try_open_capture(source, backend=None):
    if backend is None:
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(source, backend)


def open_video_source(source):
    backends = [None]
    if hasattr(cv2, "CAP_FFMPEG"):
        backends.append(cv2.CAP_FFMPEG)
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_ANY"):
        backends.append(cv2.CAP_ANY)

    for backend in backends:
        cap = _try_open_capture(source, backend)
        if not cap.isOpened():
            continue

        ret, _ = cap.read()
        if not ret:
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        backend_name = "default"
        if backend == cv2.CAP_FFMPEG:
            backend_name = "FFMPEG"
        elif backend == cv2.CAP_DSHOW:
            backend_name = "DSHOW"
        elif backend == cv2.CAP_ANY:
            backend_name = "ANY"
        print(f"[RT] Opened source with backend: {backend_name}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if frame_count > 0:
            print(f"[RT] Source info: frames={frame_count} fps={fps:.2f}")
        return cap

    return None


def load_model(weights_path: str, device: torch.device) -> AODNet:
    model = AODNet().to(device)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    if weights_path:
        weights_path = os.path.expanduser(weights_path)
        if not os.path.isabs(weights_path):
            weights_path = os.path.join(script_dir, weights_path)

    model_dir = os.path.join(script_dir, "models")
    fallback_path = None
    if os.path.isdir(model_dir):
        pths = [f for f in os.listdir(model_dir) if f.lower().endswith(".pth")]
        pths.sort()
        if pths:
            fallback_path = os.path.join(model_dir, pths[0])

    if weights_path and not os.path.exists(weights_path):
        if fallback_path and os.path.exists(fallback_path):
            print(f"[RT] Requested weights '{weights_path}' not found; using fallback {fallback_path}")
            weights_path = fallback_path
        else:
            weights_path = None

    if not weights_path and fallback_path and os.path.exists(fallback_path):
        weights_path = fallback_path
        print(f"[RT] Using fallback weights {weights_path}")

    if weights_path and os.path.exists(weights_path):
        with torch.serialization.safe_globals([AODNet]):
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, torch.nn.Module):
            state = checkpoint.state_dict()
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint
        model.load_state_dict(state)
        print(f"[RT] Loaded weights from {weights_path}")
    else:
        print("[RT] ⚠  No weights — running with untrained model (demo only).")
    model.eval()
    return model


def process_frame(frame_bgr, model, grad_cam, device, show_xai: bool):
    """Run one frame through the full pipeline."""
    # Resize for speed
    h, w = frame_bgr.shape[:2]
    scale = RESIZE_W / w
    small = cv2.resize(frame_bgr, (RESIZE_W, int(h * scale)))

    tensor = preprocess(small, device)

    with torch.no_grad():
        enhanced_tensor = model(tensor)
    base_enhanced = postprocess(enhanced_tensor)

    # Apply the stronger enhancement path used in the static POC for better clarity.
    enhanced = enhance_image(base_enhanced, apply_clahe=True, gamma=1.15)
    enhanced = sharpen_image(enhanced, strength=0.6)

    if show_xai:
        heatmap = grad_cam.generate(tensor)
        xai_overlay = apply_colormap(heatmap, enhanced)
        return small, enhanced, xai_overlay
    else:
        return small, enhanced, None


def run_realtime(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RT] Device: {device}")

    model = load_model(args.weights, device)
    grad_target = getattr(model, "e_conv5", None) or getattr(model, "conv5", None)
    if grad_target is None:
        print("[RT] ERROR: Could not find a compatible Grad-CAM target layer on the model.")
        sys.exit(1)
    grad_cam = GradCAM(model, grad_target)

    # Open source
    src = int(args.source) if args.source.isdigit() else args.source
    cap = open_video_source(src)
    if cap is None or not cap.isOpened():
        print(f"[RT] ERROR: Cannot open source '{args.source}'")
        sys.exit(1)

    print("[RT] Running... Press Q/ESC to quit, X to toggle XAI, S to save frame.")

    show_xai = False
    frame_count = 0
    fps_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[RT] Stream ended.")
            break

        foggy, enhanced, xai = process_frame(frame, model, grad_cam, device, show_xai)

        # FPS counter
        frame_count += 1
        elapsed = time.time() - fps_start
        fps = frame_count / elapsed if elapsed > 0 else 0

        # Build side-by-side display
        foggy_labeled   = add_text_label(foggy,    f"Foggy input  {fps:.1f} fps")
        enhanced_labeled = add_text_label(enhanced, "AOD-Net enhanced")

        if show_xai and xai is not None:
            xai_labeled = add_text_label(xai, "Grad-CAM XAI [X to hide]")
            row = np.hstack([foggy_labeled, enhanced_labeled, xai_labeled])
        else:
            hint = add_text_label(enhanced.copy(), "Press X for XAI")
            row = np.hstack([foggy_labeled, hint])

        cv2.imshow("Fog Enhancement + XAI  |  Real-Time POC", row)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):    # Q or ESC
            break
        elif key == ord('x'):
            show_xai = not show_xai
            print(f"[RT] XAI {'ON' if show_xai else 'OFF'}")
        elif key == ord('s'):
            os.makedirs("output", exist_ok=True)
            ts = int(time.time())
            cv2.imwrite(f"output/frame_{ts}_foggy.png",    foggy)
            cv2.imwrite(f"output/frame_{ts}_enhanced.png", enhanced)
            if xai is not None:
                cv2.imwrite(f"output/frame_{ts}_xai.png", xai)
            print(f"[RT] Saved frame {ts} to output/")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fog XAI POC — Real-Time Mode")
    parser.add_argument("--source",  default="0",  help="Webcam index (0,1,...) or path to video file")
    parser.add_argument("--weights", default="models/AOD_net_epoch_relu_10.pth", help="AOD-Net weights (.pth)")
    args = parser.parse_args()
    run_realtime(args)