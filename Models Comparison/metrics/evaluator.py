"""
Model Evaluator — runs AOD-Net and GridDehazeNet on a dataset
and collects all metrics into a structured results dictionary.

Usage
-----
    evaluator = Evaluator(aod_weights=None, grid_weights=None, device='cpu')
    results   = evaluator.run(foggy_dir='data/foggy', clear_dir='data/clear')
    # results: dict with keys 'AOD-Net' and 'GridDehazeNet',
    #          each containing per-image metrics and aggregated stats
"""

import os
import sys
import time
import glob
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.aod_net         import build_aodnet
from models.grid_dehaze_net import build_griddehazenet
from metrics.all_metrics    import compute_all, measure_inference_time, count_parameters


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_tensor(bgr: np.ndarray, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(device)


def _to_bgr(t: torch.Tensor) -> np.ndarray:
    out = t.squeeze(0).detach().cpu().clamp(0,1)
    out = (out*255).byte().permute(1,2,0).numpy()
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _post_process(bgr: np.ndarray) -> np.ndarray:
    """Mild percentile stretch + unsharp mask — applied identically to both models."""
    lo,hi = np.percentile(bgr, 2), np.percentile(bgr, 98)
    if hi > lo:
        bgr = np.clip((bgr.astype(np.float32)-lo)/(hi-lo)*255, 0,255).astype(np.uint8)
    blur = cv2.GaussianBlur(bgr, (0,0), sigmaX=1.5)
    return cv2.addWeighted(bgr, 1.25, blur, -0.25, 0)


def _find_reference(foggy_path: str, clear_dir: str) -> np.ndarray | None:
    """
    Match a foggy image to its clear reference.
    Naming convention: foggy/scene_XX_fogY.png → clear/scene_XX.png
    """
    name = Path(foggy_path).stem           # e.g. 'scene_03_fog1'
    # Strip fog suffix
    base = name.split('_fog')[0]           # 'scene_03'
    for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
        ref_path = os.path.join(clear_dir, base + ext)
        if os.path.exists(ref_path):
            return cv2.imread(ref_path)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Evaluates AOD-Net and GridDehazeNet on a directory of images.

    Parameters
    ----------
    aod_weights   : path to AOD-Net .pth file (None → demo mode)
    grid_weights  : path to GridDehazeNet .pth file (None → demo mode)
    device        : 'cpu' or 'cuda'
    max_side      : resize images so max dimension ≤ this (speed vs quality)
    post_process  : apply identical mild post-processing to both outputs
    speed_runs    : number of inference runs for timing measurement
    """

    def __init__(self,
                 aod_weights:   str | None = None,
                 grid_weights:  str | None = None,
                 device:        str = 'cpu',
                 max_side:      int = 512,
                 post_process:  bool = True,
                 speed_runs:    int = 8):

        self.device       = device
        self.max_side     = max_side
        self.do_post      = post_process
        self.speed_runs   = speed_runs

        print('\n' + '='*60)
        print('  Loading models …')
        print('='*60)
        self.models = {
            'AOD-Net':       build_aodnet(aod_weights, device),
            'GridDehazeNet': build_griddehazenet(grid_weights, device),
        }

        # Model statistics
        self.model_stats = {}
        for name, model in self.models.items():
            params = count_parameters(model)
            self.model_stats[name] = {
                'parameters': params,
                'params_M':   round(params / 1e6, 3),
            }
            print(f'  {name:<18} params: {params:,}  ({params/1e6:.3f} M)')
        print()

    # ── Main evaluation loop ──────────────────────────────────────────────────

    def run(self,
            foggy_dir: str,
            clear_dir: str | None = None,
            verbose:   bool = True) -> dict:
        """
        Run both models on all images in foggy_dir.

        Args:
            foggy_dir : directory containing foggy images
            clear_dir : directory containing clean references (optional)
            verbose   : print per-image progress

        Returns:
            results dict — structure:
            {
              'AOD-Net': {
                  'per_image': [{'name':…, 'psnr':…, 'ssim':…, …}, …],
                  'mean': {metric: value, …},
                  'std':  {metric: value, …},
                  'speed': {'mean_ms':…, 'fps':…},
                  'model_stats': {…},
              },
              'GridDehazeNet': { … }
            }
        """
        # Collect image paths
        exts = ('*.png','*.jpg','*.jpeg','*.bmp','*.tif')
        paths = []
        for e in exts:
            paths += glob.glob(os.path.join(foggy_dir, e))
        paths.sort()

        if not paths:
            raise FileNotFoundError(f'No images found in {foggy_dir!r}')

        print(f'Found {len(paths)} foggy images in {foggy_dir!r}')
        if clear_dir:
            print(f'Reference dir: {clear_dir!r}')
        else:
            print('No reference dir — running no-reference metrics only.')
        print()

        results = {name: {'per_image': []} for name in self.models}

        for img_idx, fp in enumerate(paths):
            img_name = Path(fp).stem
            foggy    = cv2.imread(fp)
            if foggy is None:
                print(f'  [skip] Cannot read {fp}')
                continue

            # Resize for performance
            h,w = foggy.shape[:2]
            if max(h,w) > self.max_side:
                s = self.max_side / max(h,w)
                foggy = cv2.resize(foggy, (int(w*s), int(h*s)))

            ref = _find_reference(fp, clear_dir) if clear_dir else None
            if ref is not None:
                ref = cv2.resize(ref, (foggy.shape[1], foggy.shape[0]))

            tensor = _to_tensor(foggy, self.device)

            if verbose:
                ref_info = '(ref ✓)' if ref is not None else '(no ref)'
                print(f'  [{img_idx+1:2d}/{len(paths)}] {img_name:<28} {ref_info}')

            for model_name, model in self.models.items():
                with torch.no_grad():
                    t0 = time.perf_counter()
                    out_t = model(tensor)
                    infer_ms = (time.perf_counter() - t0) * 1000

                enhanced = _to_bgr(out_t)
                if self.do_post:
                    enhanced = _post_process(enhanced)

                m = compute_all(enhanced, foggy, ref)
                m['name']     = img_name
                m['infer_ms'] = infer_ms
                results[model_name]['per_image'].append(m)

        # ── Aggregate statistics ──────────────────────────────────────────────
        print('\n  Aggregating results …')
        for model_name in self.models:
            rows      = results[model_name]['per_image']
            all_keys  = [k for k in rows[0] if k != 'name'
                         and rows[0][k] is not None]

            mean_d, std_d = {}, {}
            for k in all_keys:
                vals = [r[k] for r in rows if r.get(k) is not None]
                if vals:
                    mean_d[k] = float(np.mean(vals))
                    std_d[k]  = float(np.std(vals))

            results[model_name]['mean']        = mean_d
            results[model_name]['std']         = std_d
            results[model_name]['model_stats'] = self.model_stats[model_name]

            # Dedicated speed measurement on a fixed tensor
            dummy = torch.zeros(1, 3, 256, 256, device=self.device)
            speed = measure_inference_time(
                self.models[model_name], dummy, runs=self.speed_runs)
            results[model_name]['speed'] = speed

        return results
