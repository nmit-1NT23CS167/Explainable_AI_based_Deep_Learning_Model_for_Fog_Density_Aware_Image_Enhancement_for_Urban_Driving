# Fog Safety AI — Unified POC

**AOD-Net Enhancement · Grad-CAM XAI · YOLOv11 Detection · SHAP Risk Explanation**

## Quick start
```bash
pip install -r requirements.txt

# CLI — video file
python run_pipeline.py --source dashcam.mp4

# CLI — webcam
python run_pipeline.py --source 0

# Browser dashboard
streamlit run dashboard.py
```

## Controls (OpenCV window)
| Key | Action |
|-----|--------|
| E | Toggle AOD-Net enhancement |
| G | Toggle Grad-CAM panel |
| X | Toggle SHAP badge |
| S | Save current frame |
| F | Toggle fullscreen |
| Q/ESC | Quit |

## File structure
```
unified_poc/
├── config.py              All parameters + latency knobs
├── run_pipeline.py        CLI entry point (threaded)
├── dashboard.py           Streamlit UI
├── requirements.txt
│
├── models/
│   ├── aod_net.py         AOD-Net enhancer (conv1–conv5)
│   └── detector.py        YOLOv11 wrapper
│
├── utils/
│   ├── fog_detector.py    Laplacian fog score
│   ├── distance_estimator.py  Multi-cue + EMA
│   ├── feature_extractor.py   YOLO det → SHAP vector
│   └── grad_cam.py        Multi-layer GradCAM + TURBO
│
├── xai/
│   └── shap_explainer.py  SHAP KernelExplainer
│
├── alerts/
│   └── alert_manager.py   Cooldown-gated alerts
│
└── output/
    ├── unified_output.mp4
    └── shap/
```

## Latency knobs (config.py)
| Parameter | Default | Effect |
|-----------|---------|--------|
| AOD_INFERENCE_SCALE | 0.5 | AOD-Net at 50% res → ~4× faster |
| YOLO_INFERENCE_SIZE | 416 | Smaller = faster (320 for max speed) |
| GRADCAM_EVERY_N | 5 | Reuse heatmap for 5 frames |
| SHAP_EXPLAIN_EVERY_N | 60 | SHAP once per ~3 s at 20 fps |
| FOG_CHECK_INTERVAL | 15 | Skip fog measurement 14/15 frames |
| SKIP_ENHANCEMENT_WHEN_CLEAR | True | Bypass AOD-Net on clear frames |
| FRAME_SKIP | 0 | 1 = process every other frame |
| USE_THREADING | True | GradCAM + SHAP run in background |
