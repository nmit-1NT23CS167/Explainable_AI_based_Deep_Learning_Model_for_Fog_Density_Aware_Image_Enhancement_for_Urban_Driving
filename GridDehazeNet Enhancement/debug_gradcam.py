"""Small debug helper to verify Grad-CAM generation with provided weights.

Usage:
    python debug_gradcam.py --image assets/sample_foggy.png --weights models/GridDehazeNet.pth

Saves: output/debug_heatmap.png and output/debug_overlay.png
"""
import argparse
import os

import cv2
import numpy as np

from models.grid_dehaze_net import build_model
from gradcam.grad_cam import GradCAM
from utils.image_utils import load_image, bgr_to_tensor, tensor_to_bgr, post_process, save_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--weights', default=None)
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def main():
    args = parse_args()
    img = load_image(args.image)
    model = build_model(weights_path=args.weights, device=args.device)
    print('weights_loaded=', getattr(model, 'weights_loaded', False))

    tensor = bgr_to_tensor(img, device=args.device)
    with __import__('torch').no_grad():
        out_t = model(tensor)

    enhanced = post_process(tensor_to_bgr(out_t))

    # Always compute Grad-CAM (even for testing with untrained network)
    try:
        cam = GradCAM(model)
        heatmap = cam.generate(tensor, signal='fog_removal')
        heatmap_bgr = GradCAM.to_colormap(heatmap)
        heatmap_bgr = GradCAM.add_colorbar(heatmap_bgr)
        overlay = GradCAM.overlay(heatmap, enhanced, alpha=0.55)
        os.makedirs('output', exist_ok=True)
        save_image(heatmap_bgr, 'output/debug_heatmap.png')
        save_image(overlay, 'output/debug_overlay.png')
        print('Saved output/debug_heatmap.png and output/debug_overlay.png')
        cam.remove_hooks()
    except Exception as e:
        print(f'Grad-CAM computation failed: {e}')
        os.makedirs('output', exist_ok=True)
        save_image(enhanced, 'output/debug_enhanced.png')
        print('Saved fallback enhanced image to output/debug_enhanced.png')


if __name__ == '__main__':
    main()
