"""
GridDehazeNet XAI Pipeline — Single Image Mode
================================================

Usage
-----
    python infer.py --input foggy.jpg
    python infer.py --input foggy.jpg --reference clear.jpg
    python infer.py --input foggy.jpg --weights models/GridDehazeNet.pth
    python infer.py --input foggy.jpg --no-display --save

What it does (in order)
------------------------
    1.  Load foggy image
    2.  Enhance with GridDehazeNet
    3.  Post-process (contrast stretch + unsharp mask)
    4.  Run multi-layer Grad-CAM (fog-removal signal)
    5.  Compute PSNR / SSIM / MSE / Sharpness (print to terminal)
    6.  Build 3-panel display: Original | Enhanced | Grad-CAM
    7.  Save outputs to --save-dir

Execution
---------
    pip install -r requirements.txt
    python infer.py --input assets/sample_foggy.png
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

# ── project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.grid_dehaze_net import build_model
from gradcam.grad_cam        import GradCAM
from utils.image_utils       import (load_image, bgr_to_tensor, tensor_to_bgr,
                                     post_process, simple_fog_enhance,
                                     save_image, make_panel_grid, add_label)
from utils.metrics           import compare_metrics, compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:

    device = torch.device(
        'cuda' if args.device == 'cuda' and torch.cuda.is_available()
        else 'cpu'
    )
    print(f'\n[Pipeline] Device : {device}')
    print(f'[Pipeline] Input  : {args.input}')
    if args.weights:
        print(f'[Pipeline] Weights: {args.weights}')

    # ── 1. Load image ─────────────────────────────────────────────────────────
    foggy_bgr = load_image(args.input, max_side=args.max_side)
    H, W      = foggy_bgr.shape[:2]
    print(f'[Pipeline] Image  : {W}×{H} px')

    reference_bgr = None
    if args.reference:
        reference_bgr = load_image(args.reference, max_side=args.max_side)
        print(f'[Pipeline] Ref    : {args.reference}')

    # ── 2. Load GridDehazeNet ─────────────────────────────────────────────────
    print('\n[Pipeline] Loading GridDehazeNet …')
    model  = build_model(weights_path=args.weights, device=str(device))
    tensor = bgr_to_tensor(foggy_bgr, device=device)

    # ── 3. Enhancement ────────────────────────────────────────────────────────
    weights_ok = getattr(model, 'weights_loaded', False)
    if weights_ok:
        print('[Pipeline] Running enhancement …')
        t0 = time.perf_counter()
        with torch.no_grad():
            enhanced_tensor = model(tensor)
        t_enhance = (time.perf_counter() - t0) * 1000
        print(f'[Pipeline] Enhancement done in {t_enhance:.1f} ms')

        enhanced_raw = tensor_to_bgr(enhanced_tensor)
        enhanced_bgr = post_process(enhanced_raw,
                                    contrast_stretch=not args.no_post,
                                    sharpen=not args.no_post,
                                    sharpen_strength=0.35)
    else:
        print('[Pipeline] No valid model weights loaded — using fallback enhancement.')
        enhanced_bgr = simple_fog_enhance(foggy_bgr)
        enhanced_raw = enhanced_bgr.copy()

    # ── 4. Grad-CAM ───────────────────────────────────────────────────────────
    if weights_ok:
        print('[Pipeline] Computing Grad-CAM …')
        cam = GradCAM(model)
        t0  = time.perf_counter()
        heatmap  = cam.generate(tensor, signal='fog_removal')
        t_cam    = (time.perf_counter() - t0) * 1000
        print(f'[Pipeline] Grad-CAM done in {t_cam:.1f} ms')

        heatmap_bgr  = GradCAM.to_colormap(heatmap)
        heatmap_bgr  = GradCAM.add_colorbar(heatmap_bgr)
        overlay_bgr  = GradCAM.overlay(heatmap, enhanced_bgr, alpha=0.55)
        cam.remove_hooks()
    else:
        print('[Pipeline] Skipping Grad-CAM because no trained weights were loaded.')
        heatmap_bgr = np.full((H, W, 3), (18, 18, 28), dtype=np.uint8)
        cv2.putText(heatmap_bgr,
                    'NO MODEL WEIGHTS: fallback enhancement',
                    (10, min(30, H-10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (220, 220, 220), 1, cv2.LINE_AA)
        overlay_bgr = enhanced_bgr.copy()

    # ── 5. Metrics (printed to terminal) ──────────────────────────────────────
    print()
    compare_metrics(foggy_bgr, enhanced_bgr, reference=reference_bgr)
    m = compute_metrics(enhanced_bgr, reference=reference_bgr)
    fog_m = compute_metrics(foggy_bgr, reference=reference_bgr)

    # ── 6. Build 3-panel display ──────────────────────────────────────────────
    panels = [
        (foggy_bgr,    'ORIGINAL FOGGY INPUT'),
        (enhanced_bgr, 'GridDehazeNet ENHANCED'),
        (heatmap_bgr,  'GRAD-CAM HEATMAP  (fog focus regions)'),
    ]

    # Annotate metrics onto the enhanced panel
    info_panel = _make_info_panel(fog_m, m, W=400)

    panels_display = [
        (foggy_bgr,    'ORIGINAL FOGGY INPUT'),
        (enhanced_bgr, 'GridDehazeNet ENHANCED'),
        (heatmap_bgr,  'GRAD-CAM HEATMAP'),
        (overlay_bgr,  'XAI OVERLAY (heatmap on enhanced)'),
        (info_panel,   'METRICS SUMMARY'),
    ]

    grid = make_panel_grid(panels_display, cols=3, panel_w=400)

    # Title bar
    title_bar = np.full((36, grid.shape[1], 3), (18, 18, 28), dtype=np.uint8)
    cv2.putText(title_bar,
                'GridDehazeNet XAI  —  Explainable Fog Image Enhancement',
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (160, 200, 255), 1, cv2.LINE_AA)
    grid = np.vstack([title_bar, grid])

    # ── 7. Save outputs ───────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    save_image(enhanced_bgr, os.path.join(args.save_dir, 'enhanced.png'))
    save_image(heatmap_bgr,  os.path.join(args.save_dir, 'gradcam_heatmap.png'))
    save_image(overlay_bgr,  os.path.join(args.save_dir, 'gradcam_overlay.png'))
    save_image(grid,         os.path.join(args.save_dir, 'result_grid.png'))
    print(f'[Pipeline] All outputs saved to → {args.save_dir}/')

    # ── 8. Display ────────────────────────────────────────────────────────────
    if not args.no_display:
        # Scale down for screen if very large
        max_w = 1280
        if grid.shape[1] > max_w:
            scale = max_w / grid.shape[1]
            disp  = cv2.resize(grid, (max_w, int(grid.shape[0] * scale)))
        else:
            disp = grid

        win = 'GridDehazeNet XAI  |  Press any key to close'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(win, disp)
        print('\n[Display] Showing result window — press any key to close.')
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics info panel
# ─────────────────────────────────────────────────────────────────────────────

def _make_info_panel(fog_metrics, enh_metrics,
                     W: int = 400, H: int = 300) -> np.ndarray:
    """Render a dark-background metrics comparison panel."""
    panel = np.full((H, W, 3), (18, 18, 28), dtype=np.uint8)
    FONT  = cv2.FONT_HERSHEY_SIMPLEX
    y     = 30

    def row(label, fog_val, enh_val, higher_better=True, color=None):
        nonlocal y
        cv2.putText(panel, label, (10, y), FONT, 0.42,
                    (160, 160, 180), 1, cv2.LINE_AA)

        def _fmt(v): return f'{v:.3f}' if v is not None else 'N/A'
        fv = _fmt(fog_val)
        ev = _fmt(enh_val)

        if fog_val is not None and enh_val is not None:
            d = enh_val - fog_val
            if d > 0:
                arrow_col = (80, 220, 80) if higher_better else (80, 80, 220)
                arrow     = '↑'
            else:
                arrow_col = (80, 80, 220) if higher_better else (80, 220, 80)
                arrow     = '↓'
            delta_txt = f'{arrow} {abs(d):.3f}'
        else:
            arrow_col, delta_txt = (150, 150, 150), '—'

        cv2.putText(panel, fv,        (170, y), FONT, 0.40, (180,180,180), 1, cv2.LINE_AA)
        cv2.putText(panel, ev,        (260, y), FONT, 0.40, (120,220,120), 1, cv2.LINE_AA)
        cv2.putText(panel, delta_txt, (340, y), FONT, 0.40, arrow_col,     1, cv2.LINE_AA)
        y += 22

    # Header
    cv2.putText(panel, 'Metric',    (10,  y-8), FONT, 0.44, (200,200,255), 1, cv2.LINE_AA)
    cv2.putText(panel, 'Foggy',     (170, y-8), FONT, 0.44, (200,200,255), 1, cv2.LINE_AA)
    cv2.putText(panel, 'Enhanced',  (260, y-8), FONT, 0.44, (200,200,255), 1, cv2.LINE_AA)
    cv2.putText(panel, 'Δ',         (350, y-8), FONT, 0.44, (200,200,255), 1, cv2.LINE_AA)
    cv2.line(panel, (8, y+2), (W-8, y+2), (55,55,70), 1)
    y += 14

    row('PSNR (dB)',     fog_metrics.psnr, enh_metrics.psnr,  higher_better=True)
    row('SSIM',         fog_metrics.ssim, enh_metrics.ssim,  higher_better=True)
    row('MSE (px²)',    fog_metrics.mse,  enh_metrics.mse,   higher_better=False)
    row('Sharpness',    fog_metrics.sharpness, enh_metrics.sharpness, higher_better=True)

    cv2.line(panel, (8, y+4), (W-8, y+4), (55,55,70), 1)
    y += 18
    note = '* PSNR/SSIM/MSE need --reference' if fog_metrics.psnr is None else ''
    cv2.putText(panel, note, (10, y), FONT, 0.36, (110,110,130), 1, cv2.LINE_AA)

    return panel


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='GridDehazeNet XAI — Fog Image Enhancement with Explainability')
    p.add_argument('--input',      required=True,
                   help='Path to foggy input image')
    p.add_argument('--reference',  default=None,
                   help='Path to clean reference image (enables PSNR/SSIM/MSE)')
    p.add_argument('--weights',    default=None,
                   help='Path to GridDehazeNet .pth weights file')
    p.add_argument('--device',     default='cpu',
                   choices=['cpu', 'cuda'], help='Inference device')
    p.add_argument('--max-side',   type=int, default=512,
                   help='Resize so longer side ≤ this (default 768)')
    p.add_argument('--no-post',    action='store_true',
                   help='Skip post-processing (raw model output only)')
    p.add_argument('--no-display', action='store_true',
                   help='Skip GUI window (useful in headless environments)')
    p.add_argument('--save-dir',   default='output',
                   help='Directory to save result images (default: output/)')
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
