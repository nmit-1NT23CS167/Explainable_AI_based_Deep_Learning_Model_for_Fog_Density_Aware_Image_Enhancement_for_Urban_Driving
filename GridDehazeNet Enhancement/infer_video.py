"""
GridDehazeNet XAI Pipeline — Video / Real-Time Mode
=====================================================

Usage
-----
    python infer_video.py --source video.mp4
    python infer_video.py --source 0               # webcam index 0
    python infer_video.py --source video.mp4 --save

Controls
--------
    Q / ESC   quit
    G         toggle Grad-CAM overlay on/off
    S         save current frame
    SPACE     pause / resume

Performance notes
-----------------
    Grad-CAM is expensive — recomputed every GRADCAM_EVERY_N frames (default 5).
    Between recomputations the previous heatmap is reused.
    Set --gradcam-interval 1 for max quality, 10+ for max speed.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.grid_dehaze_net import build_model
from gradcam.grad_cam        import GradCAM
from utils.image_utils       import (bgr_to_tensor, tensor_to_bgr, post_process,
                                     simple_fog_enhance)
from utils.metrics           import compute_metrics

FONT = cv2.FONT_HERSHEY_SIMPLEX


def process_frame(frame_bgr, model, device,
                  cam: GradCAM | None, run_gradcam: bool,
                  sharpen_strength: float = 0.6):
    """Run one frame through enhancement + optional Grad-CAM.

    If the model was created without valid pretrained weights we use a
    lightweight fallback enhancer (`simple_fog_enhance`) to avoid random
    model outputs that can cause extreme colour shifts.
    """
    tensor = bgr_to_tensor(frame_bgr, device=device)

    heatmap = overlay = None
    weights_ok = getattr(model, 'weights_loaded', False)

    if weights_ok:
        with torch.no_grad():
            enhanced_tensor = model(tensor)
        enhanced = post_process(tensor_to_bgr(enhanced_tensor),
                                contrast_stretch=True,
                                sharpen=True,
                                sharpen_strength=sharpen_strength)

        if run_gradcam and cam is not None:
            try:
                heatmap = cam.generate(tensor, signal='fog_removal')
                if heatmap is None or heatmap.size == 0:
                    print('[Video] Grad-CAM returned empty heatmap')
                    heatmap = None
                else:
                    print(f'[Video] Grad-CAM heatmap shape {heatmap.shape} min/max {heatmap.min():.4f}/{heatmap.max():.4f}')
                    overlay = GradCAM.overlay(heatmap, enhanced, alpha=0.55)
            except Exception as e:
                print(f'[Video] Grad-CAM generation failed: {e}')
                heatmap = None
                overlay = None
    else:
        # Fallback: CLAHE + gentle saturation + post-processing
        enhanced = simple_fog_enhance(frame_bgr)
        # If user requests Grad-CAM (presses G) but no trained weights are
        # available, compute a simple per-pixel difference heatmap as a
        # fallback to highlight where the enhancement acted most.
        if run_gradcam:
            diff = np.abs(enhanced.astype(np.float32) - frame_bgr.astype(np.float32))
            diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            lo, hi = np.percentile(diff_gray, (2, 98))
            if hi - lo < 1e-6:
                heatmap = np.zeros_like(diff_gray, dtype=np.float32)
            else:
                heatmap = np.clip((diff_gray - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
            overlay = GradCAM.overlay(heatmap, enhanced, alpha=0.55)

    return enhanced, heatmap, overlay


def build_display(raw, enhanced, heatmap, overlay, fog_m, enh_m,
                  fps: float, frame_n: int, show_cam: bool):
    """Compose a side-by-side display frame."""
    h, w = raw.shape[:2]
    half_h = h // 2
    half_w = w // 2

    def fit(img, tw=half_w, th=half_h):
        h0, w0 = img.shape[:2]
        if w0 > tw or h0 > th:
            interp = cv2.INTER_AREA
        else:
            # Use bicubic when upscaling to preserve detail and reduce blockiness
            interp = cv2.INTER_CUBIC
        return cv2.resize(img, (tw, th), interpolation=interp)

    def lbl(img, text):
        out = img.copy()
        cv2.rectangle(out, (0,0), (img.shape[1],22), (20,20,30), -1)
        cv2.putText(out, text, (6,15), FONT, 0.46, (220,220,220), 1, cv2.LINE_AA)
        return out

    tl = lbl(fit(raw),      '[ 1 ] FOGGY INPUT')
    tr = lbl(fit(enhanced), '[ 2 ] ENHANCED')

    if show_cam and overlay is not None:
        # Reserve space for a narrow colorbar (bar_w) and separator (sep_w)
        bar_w = 24
        sep_w = 2
        ov_w = max(1, half_w - (bar_w + sep_w))
        ov = fit(overlay, tw=ov_w, th=half_h)

        if heatmap is not None:
            bar_h = ov.shape[0]
            vals = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
            try:
                bar_color = cv2.applyColorMap(vals, cv2.COLORMAP_TURBO)
            except Exception:
                bar_color = cv2.applyColorMap(vals, cv2.COLORMAP_JET)
            # Ensure bar_color has exactly bar_w width
            bar_color = cv2.resize(bar_color, (bar_w, bar_h), interpolation=cv2.INTER_AREA)
            sep = np.full((bar_h, sep_w, 3), 60, dtype=np.uint8)
            bl_img = np.hstack([ov, sep, bar_color])
        else:
            # Pad overlay to full half_w if no heatmap
            pad_w = half_w - ov.shape[1]
            if pad_w > 0:
                bl_img = cv2.copyMakeBorder(ov, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(15,15,20))
            else:
                bl_img = ov
        bl = lbl(bl_img, '[ 3 ] GRAD-CAM OVERLAY')
    else:
        blank = np.full((half_h, half_w, 3), 18, dtype=np.uint8)
        cv2.putText(blank, 'Press G to enable GradCAM',
                    (20, half_h//2), FONT, 0.5, (80,80,100), 1)
        bl = lbl(blank, '[ 3 ] GRAD-CAM')

    # Metrics panel
    mp = np.full((half_h, half_w, 3), (18,18,28), dtype=np.uint8)
    y  = 30
    for label, v_fog, v_enh, hi_good in [
        ('Sharpness', fog_m.sharpness, enh_m.sharpness, True),
        ('PSNR(dB)',  fog_m.psnr,      enh_m.psnr,      True),
        ('SSIM',      fog_m.ssim,      enh_m.ssim,      True),
        ('MSE(px²)',  fog_m.mse,       enh_m.mse,       False),
    ]:
        def _f(v): return f'{v:.3f}' if v is not None else 'N/A'
        fog_s = _f(v_fog); enh_s = _f(v_enh)
        if v_fog is not None and v_enh is not None:
            d = v_enh - v_fog
            col = (80,220,80) if (d > 0) == hi_good else (80,80,220)
            delta = f'{"+":s}{d:.3f}' if d >= 0 else f'{d:.3f}'
        else:
            col, delta = (120,120,120), ''
        cv2.putText(mp, f'{label:<12}: {fog_s} → {enh_s}  {delta}',
                    (10, y), FONT, 0.40, col, 1, cv2.LINE_AA)
        y += 20

    y += 4
    cv2.putText(mp, f'Frame: {frame_n}   FPS: {fps:.1f}',
                (10, y), FONT, 0.42, (140,140,160), 1, cv2.LINE_AA)
    y += 20
    cv2.putText(mp, 'G=gradcam  S=save  SPACE=pause  Q=quit',
                (10, y), FONT, 0.38, (90,90,110), 1, cv2.LINE_AA)

    br = lbl(mp, '[ 4 ] METRICS')

    top    = np.hstack([tl, tr])
    bottom = np.hstack([bl, br])
    grid   = np.vstack([top, bottom])

    # dividers
    cv2.line(grid, (half_w,0), (half_w, h), (60,60,70), 1)
    cv2.line(grid, (0,half_h), (w,half_h), (60,60,70), 1)
    return grid


def run_video(args: argparse.Namespace) -> None:
    device = torch.device(
        'cuda' if args.device == 'cuda' and torch.cuda.is_available()
        else 'cpu')
    print(f'[Video] Device : {device}')

    model = build_model(weights_path=args.weights, device=str(device))
    weights_ok = getattr(model, 'weights_loaded', False)
    cam   = GradCAM(model) if weights_ok else None

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open source: {args.source!r}')
        sys.exit(1)

    fw   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'[Video] Source : {args.source}  ({fw}×{fh})')

    writer = None
    if args.save:
        os.makedirs(args.save_dir, exist_ok=True)
        out_path = os.path.join(args.save_dir, 'video_output.mp4')
        writer   = cv2.VideoWriter(out_path,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   20, (fw, fh))
        print(f'[Video] Saving → {out_path}')

    WIN = 'GridDehazeNet XAI — Video  |  G=GradCAM  Q=quit'
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WIN, fw, fh)

    show_cam  = True
    paused    = False
    n         = 0
    fps_start = time.perf_counter()
    last_hm   = None
    last_ov   = None
    force_recompute_gc = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            n += 1

            # Resize for performance
            if max(frame.shape[:2]) > args.max_side:
                scale = args.max_side / max(frame.shape[:2])
                frame = cv2.resize(frame,
                                   (int(frame.shape[1]*scale),
                                    int(frame.shape[0]*scale)))

            run_gc = ((n % args.gradcam_interval == 0) and show_cam) or force_recompute_gc
            enhanced, hm, ov = process_frame(
                frame, model, device, cam, run_gc, sharpen_strength=args.sharpen_strength)

            if hm is not None:
                last_hm, last_ov = hm, ov
            # reset single-shot force flag after use
            if force_recompute_gc:
                force_recompute_gc = False

            fog_m = compute_metrics(frame)
            enh_m = compute_metrics(enhanced)
            fps   = n / max(time.perf_counter() - fps_start, 1e-5)

            grid = build_display(frame, enhanced,
                                 last_hm if show_cam else None,
                                 last_ov if show_cam else None,
                                 fog_m, enh_m, fps, n, show_cam)
            cv2.imshow(WIN, grid)
            if writer:
                writer.write(grid)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('g'):
            # Toggle display of Grad-CAM. When turning it on, request an
            # immediate recompute on the next frame (single-shot) so the
            # user sees the heatmap without waiting for the interval.
            show_cam = not show_cam
            if show_cam:
                force_recompute_gc = True
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            os.makedirs(args.save_dir, exist_ok=True)
            fp = os.path.join(args.save_dir, f'frame_{n:05d}.png')
            cv2.imwrite(fp, grid)
            print(f'[Video] Saved frame → {fp}')

    cap.release()
    if writer:
        writer.release()
    if cam is not None:
        cam.remove_hooks()
    cv2.destroyAllWindows()
    print(f'[Video] Done. {n} frames processed.')


def parse_args():
    p = argparse.ArgumentParser(
        description='GridDehazeNet XAI — Video / Real-Time Mode')
    p.add_argument('--source',   default='0',
                   help='Video file path or webcam index (default: 0)')
    p.add_argument('--weights',  default=None)
    p.add_argument('--device',   default='cpu', choices=['cpu','cuda'])
    p.add_argument('--max-side', type=int, default=640)
    p.add_argument('--gradcam-interval', type=int, default=5,
                   help='Recompute Grad-CAM every N frames (default 5)')
    p.add_argument('--sharpen-strength', type=float, default=0.6,
                   help='Strength of unsharp mask sharpening for enhanced frames')
    p.add_argument('--save',     action='store_true')
    p.add_argument('--save-dir', default='output')
    return p.parse_args()


if __name__ == '__main__':
    run_video(parse_args())
