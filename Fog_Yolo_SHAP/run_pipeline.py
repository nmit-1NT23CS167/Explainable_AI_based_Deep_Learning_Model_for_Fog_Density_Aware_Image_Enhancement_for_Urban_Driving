"""
Main Pipeline — Fog + YOLOv11 + Distance + SHAP Alert System
=============================================================

Usage:
    python run_pipeline.py --source video.mp4
    python run_pipeline.py --source 0              # webcam
    python run_pipeline.py --source video.mp4 --no-shap   # skip SHAP (faster)
    python run_pipeline.py --source video.mp4 --save

Controls (OpenCV window):
    Q / ESC → quit
    S       → save current annotated frame
    X       → toggle SHAP badge overlay

Pipeline per frame:
    1. Read frame from video/webcam
    2. Estimate fog score (Laplacian + brightness)
    3. Run YOLOv11 detection
    4. Estimate distance per detection (pinhole camera model)
    5. Evaluate alerts (cooldown-gated, priority-sorted)
    6. Every N frames: run SHAP on most-at-risk detection
    7. Annotate frame (boxes, distances, fog HUD, SHAP badge)
    8. Display and optionally save
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

from config import (FOG_CHECK_INTERVAL, SHAP_EXPLAIN_EVERY_N,
                    SHAP_BACKGROUND_SAMPLES, SHAP_SAVE_PLOTS,
                    OUTPUT_DIR, VIDEO_FPS, ZONE_CRITICAL)
from models.detector import YOLOv11Detector
from utils.fog_detector import compute_fog_score, is_foggy
from utils.distance_estimator import estimate_distance
from utils.feature_extractor import detection_to_features, build_background_dataset
from utils.annotator import (draw_detection, draw_fog_hud,
                              draw_alert_banner, draw_shap_badge, draw_fps)
from alerts.alert_manager import AlertManager


def parse_args():
    p = argparse.ArgumentParser(description="Fog YOLO SHAP Alert Pipeline")
    p.add_argument("--source",   default="0", help="Video file path or webcam index")
    p.add_argument("--save",     action="store_true", help="Save annotated output video")
    p.add_argument("--no-shap", action="store_true",  help="Disable SHAP (faster)")
    p.add_argument("--no-display", action="store_true", help="Headless mode")
    return p.parse_args()


def main():
    args = parse_args()
    use_shap = not args.no_shap

    # ── Init ──────────────────────────────────────────────────────────────────
    detector = YOLOv11Detector()
    alert_mgr = AlertManager(audio=False)

    shap_explainer = None
    detection_log: list[np.ndarray] = []   # accumulate feature vectors for SHAP bg
    last_shap_result = None
    show_shap_badge = True
    frame_count = 0
    fog_score = 0.0
    fps_start = time.time()

    # ── Video source ──────────────────────────────────────────────────────────
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {args.source}")
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Output video writer ───────────────────────────────────────────────────
    writer = None
    if args.save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "annotated_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, VIDEO_FPS, (frame_w, frame_h))
        print(f"[Pipeline] Saving to {out_path}")

    print("[Pipeline] Running. Press Q to quit.")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Pipeline] Stream ended.")
            break

        frame_count += 1

        # 1. Fog estimation (every N frames for performance)
        if frame_count % FOG_CHECK_INTERVAL == 0:
            fog_score = compute_fog_score(frame)

        # 2. YOLOv11 detection
        detections = detector.detect(frame)

        # 3. Distance estimation + feature extraction
        enriched = []
        for det in detections:
            box   = det["box_xyxy"]
            w_px  = float(box[2] - box[0])
            dist  = estimate_distance(w_px, det["class_name"])
            feat  = detection_to_features(box, det["confidence"], fog_score, frame_w, frame_h)

            enriched.append({**det, "distance_m": dist, "features": feat})
            detection_log.append(feat)

        # 4. Alert evaluation
        alert_inputs = [{"class_name": d["class_name"], "distance_m": d["distance_m"]}
                        for d in enriched]
        active_alerts = alert_mgr.evaluate(alert_inputs, fog_score)

        # 5. SHAP explanation (lazy init after enough background data)
        if use_shap and frame_count % SHAP_EXPLAIN_EVERY_N == 0 and enriched:
            if (shap_explainer is None and
                    len(detection_log) >= SHAP_BACKGROUND_SAMPLES):
                from xai.shap_explainer import FogAlertSHAPExplainer
                bg = build_background_dataset(detection_log, SHAP_BACKGROUND_SAMPLES)
                shap_explainer = FogAlertSHAPExplainer(bg)
                print("[SHAP] Explainer initialised.")

            if shap_explainer is not None:
                # Explain the closest (highest-risk) detection
                closest = min(enriched, key=lambda d: d["distance_m"])
                last_shap_result = shap_explainer.explain(closest["features"])

                if SHAP_SAVE_PLOTS:
                    save_path = os.path.join(
                        OUTPUT_DIR, "shap", f"frame_{frame_count:05d}.png")
                    shap_explainer.plot_waterfall(
                        last_shap_result,
                        title=f"Frame {frame_count} | {closest['class_name']} | "
                              f"{closest['distance_m']:.1f} m",
                        save_path=save_path
                    )

        # 6. Annotate frame
        for det in enriched:
            draw_detection(frame, det["box_xyxy"], det["class_name"],
                           det["confidence"], det["distance_m"])

        draw_fog_hud(frame, fog_score)

        if enriched:
            closest = min(enriched, key=lambda d: d["distance_m"])
            draw_alert_banner(frame, closest["distance_m"], closest["class_name"])

        if show_shap_badge and last_shap_result:
            top_name, top_val = last_shap_result["top_features"][0]
            draw_shap_badge(frame, top_name, top_val)

        elapsed = time.time() - fps_start
        fps = frame_count / max(elapsed, 1e-5)
        draw_fps(frame, fps)

        # 7. Display
        if not args.no_display:
            cv2.imshow("Fog + YOLO v11 + SHAP Alert System", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                save_f = os.path.join(OUTPUT_DIR, f"frame_{frame_count:05d}.png")
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                cv2.imwrite(save_f, frame)
                print(f"[Pipeline] Saved {save_f}")
            elif key == ord('x'):
                show_shap_badge = not show_shap_badge

        # 8. Write output video
        if writer:
            writer.write(frame)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Final SHAP summary plot
    if use_shap and shap_explainer and len(detection_log) >= 5:
        print("[SHAP] Generating summary plot...")
        summary_data = build_background_dataset(detection_log, 50)
        shap_explainer.plot_summary(
            summary_data,
            save_path=os.path.join(OUTPUT_DIR, "shap", "summary.png")
        )
        print(f"[SHAP] Summary saved to {OUTPUT_DIR}/shap/summary.png")

    print(f"[Pipeline] Done. Processed {frame_count} frames.")


if __name__ == "__main__":
    main()