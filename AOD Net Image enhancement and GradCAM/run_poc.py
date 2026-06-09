"""
POC Entry Point — Static Image Mode
====================================
Usage:
    python run_poc.py --input foggy.jpg [--reference clear.jpg] [--save output/]

What it does:
    1. Loads a foggy image
    2. Runs AOD-Net enhancement (pre-trained weights)
    3. Applies Grad-CAM to show which regions the model "worked hardest" on
    4. Computes PSNR / SSIM (if reference provided) + sharpness
    5. Displays a 4-panel grid: Foggy | Enhanced | Heatmap | Overlay

Requirements (install once):
    pip install torch torchvision opencv-python scikit-image matplotlib
"""

import argparse
import importlib
import os
import sys

import cv2
import numpy as np
import torch


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[POC] Using device: {device}")
    return device


def load_model(weights_path: str, device: torch.device):
    """Load AOD-Net. Downloads pre-trained weights if not found."""
    # Local import so file is standalone-runnable
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, script_dir)
    from models.aod_net import AODNet, AODnet

    # Allow loading legacy checkpoints that were saved from `model.AODnet`.
    if "model" not in sys.modules:
        sys.modules["model"] = importlib.import_module("models.aod_net")

    model = AODNet().to(device)

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
            print(f"[POC] Requested weights '{weights_path}' not found; using fallback {fallback_path}")
            weights_path = fallback_path
        else:
            weights_path = None

    if not weights_path and fallback_path and os.path.exists(fallback_path):
        weights_path = fallback_path
        print(f"[POC] Using fallback weights {weights_path}")

    if weights_path and os.path.exists(weights_path):
        with torch.serialization.safe_globals([AODNet, AODnet]):
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, torch.nn.Module):
            state = checkpoint.state_dict()
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint

        # Remap legacy layer keys from e_conv1..e_conv5 to conv1..conv5 if needed.
        state_keys = list(state.keys())
        if any(k.startswith("e_conv") for k in state_keys) and not any(k.startswith("conv") for k in state_keys):
            mapped_state = {}
            for key, value in state.items():
                if key.startswith("e_conv"):
                    mapped_state[key.replace("e_conv", "conv", 1)] = value
                else:
                    mapped_state[key] = value
            state = mapped_state
            print("[POC] Legacy checkpoint detected: remapped e_conv1..e_conv5 -> conv1..conv5")

        model.load_state_dict(state)
        print(f"[POC] Loaded weights from {weights_path}")
    else:
        print("[POC] ⚠  No weights file found — using random (untrained) weights.")
        print("[POC]    Download pre-trained AOD-Net weights and pass --weights path/to/aod_net.pth")
        print("[POC]    Ref: https://github.com/TheFairBandit/AOD-Net")

    model.eval()
    return model


def run_single_image(args):
    from utils.image_utils import preprocess, postprocess, make_comparison_grid, add_text_label, sharpen_image, enhance_image
    from utils.grad_cam import GradCAM, apply_colormap
    from utils.metrics import print_metrics

    device = get_device()
    model = load_model(args.weights, device)

    # ── 1. Load input ──────────────────────────────────────────────────────────
    foggy_bgr = cv2.imread(args.input)
    if foggy_bgr is None:
        print(f"[ERROR] Cannot read image: {args.input}")
        sys.exit(1)

    # Resize to manageable resolution for demo
    h, w = foggy_bgr.shape[:2]
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        foggy_bgr = cv2.resize(foggy_bgr, (int(w * scale), int(h * scale)))

    print(f"[POC] Input: {args.input}  ({foggy_bgr.shape[1]}×{foggy_bgr.shape[0]})")

    # ── 2. Enhancement ────────────────────────────────────────────────────────
    input_tensor = preprocess(foggy_bgr, device)
    with torch.no_grad():
        enhanced_tensor = model(input_tensor)
    enhanced_bgr = postprocess(enhanced_tensor)
    enhanced_display = enhance_image(enhanced_bgr, apply_clahe=False, gamma=1.0, saturation_scale=1.0)
    enhanced_display = sharpen_image(enhanced_display, strength=0.3)
    print("[POC] Enhancement complete.")

    # ── 3. Grad-CAM ───────────────────────────────────────────────────────────
    # Target the last convolutional layer (conv5) for most informative maps
    grad_cam = GradCAM(model, model.conv5)
    input_grad = preprocess(foggy_bgr, device)
    heatmap = grad_cam.generate(input_grad)

    heatmap_bgr = (heatmap * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_bgr, cv2.COLORMAP_JET)
    overlay_bgr = apply_colormap(heatmap, enhanced_display)
    print("[POC] Grad-CAM heatmap generated.")

    # ── 4. Metrics ────────────────────────────────────────────────────────────
    reference = None
    if args.reference and os.path.exists(args.reference):
        reference = cv2.imread(args.reference)

    print_metrics(foggy_bgr, enhanced_bgr, reference)

    # ── 5. Compose 4-panel display ────────────────────────────────────────────
    grid = make_comparison_grid(
        [foggy_bgr, enhanced_display, heatmap_bgr, overlay_bgr],
        ["Input (foggy)", "AOD-Net enhanced", "Grad-CAM heatmap", "XAI overlay"],
        target_width=360
    )

    cv2.imshow("Fog Enhancement + XAI  —  POC", grid)
    print("[POC] Showing result. Press any key to close.")

    # ── 6. Save output ────────────────────────────────────────────────────────
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        cv2.imwrite(os.path.join(args.save, "enhanced.png"), enhanced_bgr)
        cv2.imwrite(os.path.join(args.save, "heatmap.png"), heatmap_bgr)
        cv2.imwrite(os.path.join(args.save, "overlay.png"), overlay_bgr)
        cv2.imwrite(os.path.join(args.save, "comparison_grid.png"), grid)
        print(f"[POC] Saved outputs to {args.save}/")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fog Image Enhancement with XAI — POC Demo"
    )
    parser.add_argument("--input",     required=True,  help="Path to foggy input image")
    parser.add_argument("--reference", default=None,   help="Path to clean reference image (for PSNR/SSIM)")
    parser.add_argument("--weights",   default="models/AOD_net_epoch_relu_10.pth", help="Path to AOD-Net weights (.pth)")
    parser.add_argument("--save",      default="output", help="Directory to save outputs")
    args = parser.parse_args()
    run_single_image(args)