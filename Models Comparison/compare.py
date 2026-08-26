"""
compare.py — AOD-Net vs GridDehazeNet Model Comparison
=======================================================

Single entry point. Runs the full evaluation pipeline:
  1. Load both models
  2. Run inference on every foggy image
  3. Compute all metrics (PSNR, SSIM, MSE, Sharpness, Fog Density, etc.)
  4. Print colour-coded terminal table
  5. Save CSV / LaTeX / HTML result tables
  6. Generate 12 comparison plots

Usage
-----
    # Quickest — demo mode, no weights needed
    python compare.py

    # With your weights files
    python compare.py --aod-weights  models/AOD_net_epoch_relu_10.pth \\
                      --grid-weights models/GridDehazeNet.pth

    # Provide clean reference images for PSNR/SSIM/MSE
    python compare.py --foggy-dir data/foggy --clear-dir data/clear

    # Full options
    python compare.py \\
        --foggy-dir  data/foggy \\
        --clear-dir  data/clear \\
        --aod-weights  models/AOD_net_epoch_relu_10.pth \\
        --grid-weights models/GridDehazeNet.pth \\
        --device cpu \\
        --max-side 512 \\
        --out-dir  results
"""

import argparse
import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import cv2
import numpy as np
import torch

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.aod_net         import build_aodnet
from models.grid_dehaze_net import build_griddehazenet
from metrics.all_metrics    import (compute_all, measure_inference_time,
                                     count_parameters)
from utils.table_writer     import (print_terminal_table, save_csv,
                                     save_latex, save_html)
from utils.plot_generator   import generate_all_plots


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers (kept here so compare.py is fully self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def _to_tensor(bgr: np.ndarray, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)


def _to_bgr(t: torch.Tensor) -> np.ndarray:
    out = t.squeeze(0).detach().cpu().clamp(0, 1)
    out = (out * 255).byte().permute(1, 2, 0).numpy()
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _post_process(bgr: np.ndarray) -> np.ndarray:
    """Minimal, color-safe sharpening: clear haze while keeping natural colors."""
    blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=0.8)
    return cv2.addWeighted(bgr, 1.08, blur, -0.08, 0)


def _find_reference(foggy_path: str, clear_dir: str):
    """
    Match:
        foggy_road12.png -> no_fog12.png
        foggy_road13.png -> no_fog13.png
    """

    # Get filename without extension
    foggy_name = os.path.splitext(os.path.basename(foggy_path))[0]

    # Extract the number at the end
    number = ""

    for char in reversed(foggy_name):
        if char.isdigit():
            number = char + number
        else:
            break

    if not number:
        print(f"[WARNING] Cannot find image number: {foggy_name}")
        return None

    # Search for corresponding clear image
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:

        clear_path = os.path.join(
            clear_dir,
            "no_fog" + number + ext
        )

        if os.path.exists(clear_path):
            print(
                f"  Matched: {os.path.basename(foggy_path)} "
                f"-> {os.path.basename(clear_path)}"
            )

            image = cv2.imread(clear_path)

            if image is not None:
                return image

    print(
        f"[WARNING] No clear image found for "
        f"{os.path.basename(foggy_path)}"
    )

    return None
# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(args: argparse.Namespace) -> None:

    t_start = time.perf_counter()

    print()
    print('╔' + '═'*62 + '╗')
    print('║   AOD-Net  vs  GridDehazeNet  —  Comparison Pipeline     ║')
    print('╚' + '═'*62 + '╝')
    print()

    device = (args.device if (args.device == 'cuda' and
              torch.cuda.is_available()) else 'cpu')
    print(f'  Device   : {device}')
    print(f'  Foggy dir: {args.foggy_dir}')
    print(f'  Clear dir: {args.clear_dir or "None (no-reference mode)"}')
    print(f'  Max side : {args.max_side} px')
    print()

    # ── 1. Load models ────────────────────────────────────────────────────────
    print('─'*64)
    print('  [1/5] Loading models …')
    print('─'*64)

    models = {
        'AOD-Net':       build_aodnet(args.aod_weights, device),
        'GridDehazeNet': build_griddehazenet(args.grid_weights, device),
    }

    model_stats = {}
    for name, model in models.items():
        p = count_parameters(model)
        model_stats[name] = {'parameters': p, 'params_M': round(p / 1e6, 3)}
        print(f'  {name:<20} {p:>12,} params  ({p/1e6:.3f} M)')
    print()

    # ── 2. Collect image paths ────────────────────────────────────────────────
    import glob
    exts  = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(args.foggy_dir, e))
    paths.sort()

    if not paths:
        print(f'  [ERROR] No images found in {args.foggy_dir!r}')
        print('  Run:  python compare.py  (uses bundled sample data)')
        sys.exit(1)

    print('─'*64)
    print(f'  [2/5] Running inference on {len(paths)} images …')
    print('─'*64)

    results       = {n: {'per_image': []} for n in models}
    enhanced_cache = {}   # {(model_name, img_name): enhanced_bgr}

    for idx, fp in enumerate(paths):
        from pathlib import Path
        img_name = Path(fp).stem
        foggy    = cv2.imread(fp)
        if foggy is None:
            print(f'  [skip] {fp}')
            continue

        # Resize for performance
        h, w = foggy.shape[:2]
        if max(h, w) > args.max_side:
            s     = args.max_side / max(h, w)
            foggy = cv2.resize(foggy, (int(w*s), int(h*s)))

        ref = _find_reference(fp, args.clear_dir) if args.clear_dir else None
        if ref is not None:
            ref = cv2.resize(ref, (foggy.shape[1], foggy.shape[0]))

        tensor   = _to_tensor(foggy, device)
        ref_mark = '✓ ref' if ref is not None else 'no ref'
        print(f'  [{idx+1:2d}/{len(paths)}] {img_name:<32} ({ref_mark})')

        for model_name, model in models.items():
            with torch.no_grad():
                t0    = time.perf_counter()
                out_t = model(tensor)
                ms    = (time.perf_counter() - t0) * 1000

            enhanced = _post_process(_to_bgr(out_t))
            enhanced_cache[(model_name, img_name)] = enhanced.copy()

            m           = compute_all(enhanced, foggy, ref)
            m['name']   = img_name
            m['infer_ms'] = ms
            results[model_name]['per_image'].append(m)

    # ── 3. Aggregate ──────────────────────────────────────────────────────────
    print()
    print('─'*64)
    print('  [3/5] Aggregating statistics …')
    print('─'*64)

    for model_name in models:
        rows     = results[model_name]['per_image']
        all_keys = [k for k in rows[0] if k != 'name' and
                    rows[0][k] is not None]
        mean_d, std_d = {}, {}
        for k in all_keys:
            vals = [r[k] for r in rows if r.get(k) is not None]
            if vals:
                mean_d[k] = float(np.mean(vals))
                std_d[k]  = float(np.std(vals))

        results[model_name]['mean']        = mean_d
        results[model_name]['std']         = std_d
        results[model_name]['model_stats'] = model_stats[model_name]

        # Speed on fixed 256×256 tensor (warm+measured)
        dummy = torch.zeros(1, 3, 256, 256, device=device)
        speed = measure_inference_time(models[model_name], dummy, runs=8)
        results[model_name]['speed'] = speed
        print(f'  {model_name:<20} '
              f'mean_time={speed["mean_ms"]:.1f} ms  '
              f'fps={speed["fps"]:.1f}')

    # ── 4. Print terminal table ───────────────────────────────────────────────
    print()
    print('─'*64)
    print('  [4/5] Results …')
    print('─'*64)
    print_terminal_table(results)

    # ── 5. Save outputs ───────────────────────────────────────────────────────
    print('─'*64)
    print('  [5/5] Saving tables and plots …')
    print('─'*64)

    table_dir = os.path.join(args.out_dir, 'tables')
    plot_dir  = os.path.join(args.out_dir, 'plots')

    save_csv(results, table_dir)
    save_latex(results, table_dir)
    save_html(results, table_dir)

    generate_all_plots(
        results,
        out_dir      = plot_dir,
        foggy_dir    = args.foggy_dir,
        enhanced_cache = enhanced_cache,
    )

    # ── Save per-image visual strips ──────────────────────────────────────────
    strip_dir = os.path.join(args.out_dir, 'per_image')
    os.makedirs(strip_dir, exist_ok=True)
    for fp in paths[:8]:   # first 8 images
        from pathlib import Path
        img_name = Path(fp).stem
        foggy    = cv2.imread(fp)
        if foggy is None:
            continue
        h, w = foggy.shape[:2]
        if max(h,w) > args.max_side:
            s=args.max_side/max(h,w); foggy=cv2.resize(foggy,(int(w*s),int(h*s)))

        panels = [foggy]
        labels = ['FOGGY INPUT']
        for mn in ['AOD-Net','GridDehazeNet']:
            enh = enhanced_cache.get((mn, img_name))
            if enh is not None:
                panels.append(enh); labels.append(mn)

        ph, pw = panels[0].shape[:2]
        strip  = np.zeros((ph + 28, pw * len(panels), 3), dtype=np.uint8)
        for i, (panel, lbl) in enumerate(zip(panels, labels)):
            x = i * pw
            strip[28:, x:x+pw] = panel
            cv2.rectangle(strip, (x,0), (x+pw,28), (18,18,28), -1)
            cv2.putText(strip, lbl, (x+6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220,220,220), 1)
            if i > 0:
                cv2.line(strip, (x,0), (x,strip.shape[0]), (50,50,70), 1)

        cv2.imwrite(os.path.join(strip_dir, f'{img_name}_compare.png'), strip)

    elapsed = time.perf_counter() - t_start
    print()
    t_dir = os.path.join(args.out_dir, 'tables')
    p_dir = os.path.join(args.out_dir, 'plots')
    s_dir = os.path.join(args.out_dir, 'per_image')
    print('╔' + '═'*62 + '╗')
    print(f'║   DONE in {elapsed:.1f}s'.ljust(63) + '║')
    print(f'║   Tables  → {t_dir}'.ljust(63) + '║')
    print(f'║   Plots   → {p_dir}'.ljust(63) + '║')
    print(f'║   Strips  → {s_dir}'.ljust(63) + '║')
    print('╚' + '═'*62 + '╝')
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='AOD-Net vs GridDehazeNet — Full Comparison Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Quickest run (demo mode, sample data included):
  python compare.py

  # With your weights:
  python compare.py --aod-weights models/AOD_net.pth \\
                    --grid-weights models/GridDehazeNet.pth

  # Full evaluation with reference images:
  python compare.py --foggy-dir data/foggy \\
                    --clear-dir data/clear \\
                    --aod-weights  models/AOD_net.pth \\
                    --grid-weights models/GridDehazeNet.pth

  # GPU inference:
  python compare.py --device cuda
        """
    )
    p.add_argument('--foggy-dir',    default='data/foggy',
                   help='Directory of foggy input images  [default: data/foggy]')
    p.add_argument('--clear-dir',    default='data/clear',
                   help='Directory of clean reference images [default: data/clear]')
    p.add_argument('--aod-weights',  default=None,
                   help='AOD-Net weights (.pth)  — None = demo mode')
    p.add_argument('--grid-weights', default=None,
                   help='GridDehazeNet weights (.pth) — None = demo mode')
    p.add_argument('--device',       default='cpu',
                   choices=['cpu', 'cuda'])
    p.add_argument('--max-side',     type=int, default=512,
                   help='Resize images: max(H,W) ≤ this  [default: 512]')
    p.add_argument('--out-dir',      default='results',
                   help='Output directory  [default: results]')
    return p.parse_args()


if __name__ == '__main__':
    run_comparison(parse_args())
