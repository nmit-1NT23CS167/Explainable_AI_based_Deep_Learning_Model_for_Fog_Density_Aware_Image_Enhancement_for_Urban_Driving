# GridDehazeNet XAI — Explainable Fog Image Enhancement for Autonomous Driving

> **Grad-CAM explainability + GridDehazeNet dehazing + image quality metrics**
> in a single, clean pipeline — images and videos both supported.

---

## Overview

This project implements an end-to-end fog image enhancement pipeline using
**GridDehazeNet** (Liu et al., ICCV 2019) with **Explainable AI** (Grad-CAM)
visualisation. It is designed as a proof-of-concept for autonomous driving
perception systems operating in foggy conditions.

```
Foggy Image / Video
       │
       ▼
  GridDehazeNet          ← multi-scale grid attention dehazing
       │
       ├──► Enhanced image
       │
       ├──► Grad-CAM heatmap  ← multi-layer gradient visualisation
       │         (fog removal signal: where the model worked hardest)
       │
       └──► Metrics to terminal
                PSNR  /  SSIM  /  MSE  /  Sharpness
```

---

## Project Structure

```
griddehazenet_xai/
│
├── models/
│   └── grid_dehaze_net.py     GridDehazeNet architecture (3×3 grid)
│
├── gradcam/
│   └── grad_cam.py            Multi-layer Grad-CAM engine
│
├── utils/
│   ├── image_utils.py         Load / save / tensor conversions / panel grid
│   └── metrics.py             PSNR, SSIM, MSE, Sharpness (Laplacian var)
│
├── assets/
│   └── sample_foggy.png       Synthetic foggy road image (auto-generated)
│
├── output/                    All results saved here
│
├── infer.py                   ← Single image mode
├── infer_video.py             ← Video / webcam mode
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone / download the project
cd griddehazenet_xai

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Single Image

```bash
# Basic (demo mode — random weights, for architecture/XAI validation)
python infer.py --input assets/sample_foggy.png

# With pre-trained weights
python infer.py --input foggy.jpg --weights models/GridDehazeNet.pth

# With a clean reference (enables PSNR / SSIM / MSE)
python infer.py --input foggy.jpg --reference clear.jpg

# CUDA GPU
python infer.py --input foggy.jpg --device cuda

# Headless (no display window, just save outputs)
python infer.py --input foggy.jpg --no-display --save-dir results/
```

### Video / Webcam

```bash
# Video file
python infer_video.py --source dashcam.mp4

# Webcam (index 0)
python infer_video.py --source 0

# With pre-trained weights, save output
python infer_video.py --source dashcam.mp4 --weights models/GridDehazeNet.pth --save

# Faster (less frequent Grad-CAM)
python infer_video.py --source 0 --gradcam-interval 10
```

### Video Controls

| Key | Action |
|-----|--------|
| `G` | Toggle Grad-CAM overlay |
| `S` | Save current frame |
| `SPACE` | Pause / resume |
| `Q` / `ESC` | Quit |

---

## Pre-trained Weights

The architecture is fully implemented and runnable without weights (demo mode).
For real dehazing quality, download pre-trained weights from the official repo:

```
https://github.com/proteus1991/GridDehazeNet
```

Place the downloaded `.pth` file in the `models/` folder and pass it via
`--weights models/GridDehazeNet.pth`.

---

## Output Files (saved to `output/`)

| File | Description |
|------|-------------|
| `enhanced.png` | Dehazed image |
| `gradcam_heatmap.png` | Grad-CAM colour map + scale bar |
| `gradcam_overlay.png` | Heatmap blended onto enhanced image |
| `result_grid.png` | 5-panel comparison grid |
| `video_output.mp4` | (video mode) Annotated output video |

---

## How Grad-CAM Works Here

Standard Grad-CAM was designed for classification. For **image enhancement**
we adapt it using the **fog-removal signal**:

```
score = MSE(output, input)
```

This measures where the network *changed the image most* — i.e. the fog-dense
regions. Backpropagating through this score and weighting the activations in
grid nodes `n00 → n02` (full scale) and `n10, n11` (half scale) gives a
spatially rich heatmap:

- 🔵 **Blue / cool** — clear or uniform regions (road surface far from camera)
- 🟢 **Green / yellow** — medium fog, lane markings, structural edges
- 🟠 **Orange / red** — dense fog zones, signs, vehicles the model focused on

Three layers are fused with weights `[0.30, 0.25, 0.20, 0.15, 0.10]`
and percentile-normalised to ensure full colour-scale usage.

---

## Metrics Explained

| Metric | Full name | What it measures | Range |
|--------|-----------|-----------------|-------|
| PSNR | Peak Signal-to-Noise Ratio | Pixel-level fidelity vs reference | Higher → better (>28 dB = good) |
| SSIM | Structural Similarity Index | Perceptual structure similarity | 0–1, closer to 1 = better |
| MSE | Mean Squared Error | Average pixel error | Lower → better |
| Sharpness | Laplacian variance | Edge sharpness (no reference needed) | Higher → sharper |

PSNR / SSIM / MSE require a clean reference image (`--reference`).
Sharpness is computed from the output alone.

---

## Reference

```bibtex
@inproceedings{liu2019griddehazenet,
  title     = {GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing},
  author    = {Liu, Xiaohong and Ma, Yongrui and Shi, Zhihao and Chen, Jun},
  booktitle = {ICCV},
  year      = {2019}
}
```
