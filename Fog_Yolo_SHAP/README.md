# Fog Obstacle Detection + SHAP XAI — POC

**YOLOv11 · Distance Estimation · SHAP Explainability · Autonomous Driving Safety**

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run on a video file
python run_pipeline.py --source dashcam.mp4

# 3. Run on webcam
python run_pipeline.py --source 0

# 4. Streamlit dashboard (browser UI)
streamlit run dashboard.py

# 5. Skip SHAP for faster processing
python run_pipeline.py --source video.mp4 --no-shap

# 6. Save annotated output
python run_pipeline.py --source video.mp4 --save
```

## Controls (OpenCV window)
| Key | Action |
|-----|--------|
| Q / ESC | Quit |
| S | Save current frame |
| X | Toggle SHAP badge overlay |

## How distance estimation works
Uses the **pinhole camera model**:

```
Distance (m) = (Real_Width_m × Focal_Length_px) / BBox_Width_px
```

Real-world widths per class are defined in `config.py → KNOWN_WIDTHS_M`.
Focal length defaults to 800 px (typical dashcam). Calibrate for your camera.

## Alert zones
| Zone | Distance | Action |
|------|----------|--------|
| CRITICAL | ≤ 10 m | Red banner — brake immediately |
| WARNING | ≤ 25 m | Amber — slow down |
| CAUTION | ≤ 50 m | Yellow — be aware |
| SAFE | > 50 m | Green — normal driving |

## SHAP explanation
SHAP (SHapley Additive exPlanations) explains **why a detection is rated as high risk**.

Features analysed per detection:
- `bbox_width_px` — larger box = closer object
- `fog_score` — fog amplifies danger
- `center_x_norm` — central lane objects are more dangerous
- `confidence` — detection certainty
- `aspect_ratio`, `bbox_area_px`, `center_y_norm`, `bbox_height_px`

SHAP waterfall plots saved to `output/shap/frame_NNNNN.png`.
Summary beeswarm plot saved to `output/shap/summary.png` at end of run.

## File structure
```
fog_yolo_shap/
├── config.py               All tunable parameters
├── run_pipeline.py         Main CLI entry point
├── dashboard.py            Streamlit web dashboard
├── requirements.txt
├── models/
│   └── detector.py         YOLOv11 wrapper
├── utils/
│   ├── fog_detector.py     Frame-level fog estimation
│   ├── distance_estimator.py  Pinhole camera distance
│   ├── feature_extractor.py   YOLO det → SHAP feature vector
│   └── annotator.py        CV2 frame annotation
├── xai/
│   └── shap_explainer.py   SHAP KernelExplainer + plots
├── alerts/
│   └── alert_manager.py    Cooldown-gated alert dispatch
└── output/                 Annotated frames + SHAP plots
```
