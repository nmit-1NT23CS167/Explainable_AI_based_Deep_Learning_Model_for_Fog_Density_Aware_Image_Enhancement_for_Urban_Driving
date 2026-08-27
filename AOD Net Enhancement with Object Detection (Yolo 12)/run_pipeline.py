"""
Unified Pipeline — AOD-Net + GradCAM + YOLOv11 + SHAP
======================================================
Full flow per frame:
    Raw frame
      │
      ├─ [Thread A — Main, real-time]
      │     AOD-Net enhance (at 0.5× scale for speed)
      │       → YOLOv12 detect on enhanced frame
      │         → Distance estimate
      │           → Alert evaluate
      │             → Annotate + display  (target: <40 ms/frame)
      │
      └─ [Thread B — Background, non-blocking]
            GradCAM  (every GRADCAM_EVERY_N frames)
            SHAP     (every SHAP_EXPLAIN_EVERY_N frames)
              → Results stored in shared state
              → Picked up by Thread A on next render

LATENCY REDUCTION TECHNIQUES USED
───────────────────────────────────
1. AOD_INFERENCE_SCALE = 0.5   → AOD-Net runs at 50% resolution (~4× faster)
2. YOLO_INFERENCE_SIZE = 416   → smaller than default 640 (~2× faster)
3. Threading              → GradCAM + SHAP never block the display loop
4. GRADCAM_EVERY_N = 5   → heatmap reused across 5 frames, not recomputed
5. SHAP_EXPLAIN_EVERY_N = 60  → SHAP runs ~once per 3 seconds at 20 fps
6. FOG_CHECK_INTERVAL = 15    → fog measurement skipped most frames
7. SKIP_ENHANCEMENT_WHEN_CLEAR → AOD-Net bypassed entirely on clear frames
8. torch.no_grad()         → no gradient tape during YOLO or AOD inference
9. model.eval()            → disables batchnorm/dropout overhead

Usage:
    python run_pipeline.py --source video.mp4
    python run_pipeline.py --source 0                   # webcam
    python run_pipeline.py --source video.mp4 --save    # save output

Controls:
    Q / ESC   quit
    E         toggle AOD-Net enhancement on/off
    G         toggle Grad-CAM panel on/off
    X         toggle SHAP badge
    S         save current frame
    F         toggle fullscreen
"""

import argparse, os, sys, time, threading, copy, queue
import cv2, numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    FOG_CHECK_INTERVAL, GRADCAM_EVERY_N, SHAP_EXPLAIN_EVERY_N,
    SHAP_BACKGROUND_SAMPLES, SHAP_SAVE_PLOTS, OUTPUT_DIR, VIDEO_FPS,
    DEVICE, AOD_INFERENCE_SCALE, ZONE_CRITICAL, ZONE_WARNING,
    SKIP_ENHANCEMENT_WHEN_CLEAR, FOG_SKIP_THRESH, FRAME_SKIP,
    GRADCAM_LAYERS, GRADCAM_WEIGHTS, GRADCAM_OVERLAY_ALPHA,
    FONT_SCALE, FONT_THICKNESS,
)
from models.aod_net      import AODNetEnhancer
from models.detector     import YOLOv11Detector
from utils.fog_detector  import compute_fog_score, fog_label, fog_color_bgr
from utils.distance_estimator import (estimate_distance, zone_label,
                                       zone_color_bgr, format_distance, reset_ema)
from utils.feature_extractor  import detection_to_features, build_background_dataset
from alerts.alert_manager     import AlertManager

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── Shared async state ────────────────────────────────────────────────────────

class AsyncState:
    """Thread-safe container for GradCAM + SHAP results."""
    def __init__(self):
        self._lock        = threading.Lock()
        self.heatmap      = None   # (H, W) float32
        self.heatmap_bgr  = None   # colourised (H, W, 3) uint8
        self.shap_result  = None   # dict from FogAlertSHAPExplainer.explain()
        self.gradcam_busy = False
        self.shap_busy    = False


# ── Frame annotation helpers ──────────────────────────────────────────────────

def _draw_detection(frame, box, cls, conf, dist):
    x1, y1, x2, y2 = (int(v) for v in box)
    color = zone_color_bgr(dist)
    thick = 3 if dist <= ZONE_CRITICAL else 2
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, thick)
    label = f"{cls} [{zone_label(dist)}] {format_distance(dist)}"
    (lw, lh), base = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    ly = max(y1-6, lh+4)
    cv2.rectangle(frame, (x1, ly-lh-4), (x1+lw+4, ly+base), color, -1)
    cv2.putText(frame, label, (x1+2, ly-2), FONT,
                FONT_SCALE, (0,0,0), FONT_THICKNESS, cv2.LINE_AA)


def _draw_fog_hud(frame, fog_score):
    h, w = frame.shape[:2]
    label = f"Fog: {fog_label(fog_score)} ({fog_score:.2f})"
    color = fog_color_bgr(fog_score)
    cv2.putText(frame, label, (w-280, 24), FONT, 0.55, color, 2, cv2.LINE_AA)
    bx, by, bw, bh = w-280, 32, 240, 6
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (50,50,50), -1)
    filled = int(bw * fog_score)
    if filled > 0:
        cv2.rectangle(frame, (bx, by), (bx+filled, by+bh), color, -1)


def _draw_alert_banner(frame, dist, cls):
    if dist > ZONE_CRITICAL: return
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, h-54), (w, h), (0, 0, 200), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, f"  !! ALERT: {cls} @ {format_distance(dist)} — BRAKE!",
                (8, h-16), FONT, 0.75, (255,255,255), 2, cv2.LINE_AA)


def _label_panel(img, text, bg=(22,22,32), fg=(230,230,230)):
    out = img.copy()
    cv2.rectangle(out, (0,0), (img.shape[1], 22), bg, -1)
    cv2.putText(out, text, (8,15), FONT, 0.48, fg, 1, cv2.LINE_AA)
    return out


def _make_status_panel(pw, ph, fog, enriched, enh_on, shap_result,
                        frame_count, fps, gradcam_on):
    panel = np.full((ph, pw, 3), (18, 18, 24), dtype=np.uint8)
    cv2.rectangle(panel, (0,0), (pw, 22), (38,38,52), -1)
    cv2.putText(panel, "SYSTEM STATUS", (8,15), FONT, 0.50,
                (170,170,240), 1, cv2.LINE_AA)
    y, dy = 38, 19

    def row(s, color=(200,200,200), scale=0.44):
        nonlocal y
        cv2.putText(panel, s, (10, y), FONT, scale, color, 1, cv2.LINE_AA)
        y += dy

    enh_col = (80,220,120) if enh_on else (110,110,110)
    row(f"AOD-Net: {'ON — dehazing' if enh_on else 'OFF — raw'}", enh_col)
    row(f"Fog: {fog_label(fog)} ({fog:.2f})", fog_color_bgr(fog))
    row(f"Frame: {frame_count}   FPS: {fps:.1f}", (150,150,150), 0.42)
    row(f"GradCAM: {'ON' if gradcam_on else 'OFF'}", (120,200,120), 0.42)

    y += 4
    cv2.line(panel, (8,y), (pw-8,y), (55,55,68), 1); y += 8
    row("Detections:")
    if not enriched:
        row("  (none)", (90,90,90), 0.41)
    else:
        for det in sorted(enriched, key=lambda d: d["distance_m"])[:5]:
            c = zone_color_bgr(det["distance_m"])
            row(f"  {det['class_name']:<12} {format_distance(det['distance_m']):>8}"
                f"  [{zone_label(det['distance_m'])}]", c, 0.41)

    y += 4
    cv2.line(panel, (8,y), (pw-8,y), (55,55,68), 1); y += 8
    row("SHAP explanation:")
    if shap_result:
        row(f"  Risk: {shap_result['risk_score']:.3f}  "
            f"(base {shap_result['base_value']:.3f})",
            (200,175,80), 0.41)
        for feat, val in shap_result["top_features"][:4]:
            sign = "+" if val >= 0 else ""
            col  = (100,100,230) if val >= 0 else (80,200,100)
            row(f"  {feat:<18} {sign}{val:.3f}", col, 0.40)
    else:
        row("  Warming up...", (90,90,90), 0.40)

    hint_y = ph - 8
    for line in ["S=save  Q=quit", "E=enhance  G=gradcam  X=shap  F=fullscreen"]:
        cv2.putText(panel, line, (10, hint_y), FONT, 0.38, (110,110,110), 1, cv2.LINE_AA)
        hint_y -= 14
    return panel


def _build_grid(raw, enhanced, det_frame, heatmap_bgr, overlay,
                fog, enriched, enh_on, shap_result, frame_count, fps,
                gradcam_on, show_gradcam):
    h, w = raw.shape[:2]
    ph, pw = h // 2, w // 2

    def fit(img): return cv2.resize(img, (pw, ph))

    tl = _label_panel(fit(raw),          "[ 1 ] RAW FOGGY INPUT")
    tr = _label_panel(fit(enhanced),     "[ 2 ] AOD-Net DEHAZED")
    bl = _label_panel(fit(det_frame),    "[ 3 ] YOLO DETECTIONS + ALERTS")

    if show_gradcam and overlay is not None:
        br_img = fit(overlay)
        br = _label_panel(br_img, "[ 4 ] GRAD-CAM + XAI OVERLAY")
    elif show_gradcam and heatmap_bgr is not None:
        br_img = fit(heatmap_bgr)
        br = _label_panel(br_img, "[ 4 ] GRAD-CAM HEATMAP")
    else:
        status = _make_status_panel(pw, ph, fog, enriched, enh_on,
                                     shap_result, frame_count, fps, gradcam_on)
        br = _label_panel(status, "[ 4 ] STATUS / SHAP")

    grid = np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])
    cv2.line(grid, (pw,0), (pw,h), (70,70,85), 1)
    cv2.line(grid, (0,ph), (w,ph), (70,70,85), 1)
    return grid


WIN = "Fog+Enhancement+YOLO+SHAP  |  Unified POC"


# ── Background XAI worker ─────────────────────────────────────────────────────

def _xai_worker(task_queue: queue.Queue, state: AsyncState,
                aod_model, detection_log: list, frame_count_ref: list):
    """
    Background thread: picks tasks from queue and updates AsyncState.
    Never touches the display or OpenCV — fully thread-safe.
    """
    gradcam_obj = None
    shap_obj    = None

    while True:
        try:
            task = task_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if task is None:   # poison pill → shut down
            break

        task_type = task["type"]

        if task_type == "gradcam":
            if gradcam_obj is None:
                from utils.grad_cam import MultiLayerGradCAM
                gradcam_obj = MultiLayerGradCAM(
                    aod_model, GRADCAM_LAYERS, GRADCAM_WEIGHTS)
            inp = task["input_tensor"]
            try:
                # Ensure the tensor matches the model device for GPU support
                device = next(aod_model.parameters()).device
                inp = inp.to(device)
                heatmap = gradcam_obj.generate(inp)
                from utils.grad_cam import apply_heatmap, apply_colormap
                hm_bgr   = apply_heatmap(heatmap)
                overlay  = apply_colormap(heatmap, task["enhanced_bgr"],
                                          GRADCAM_OVERLAY_ALPHA)
                with state._lock:
                    state.heatmap     = heatmap
                    state.heatmap_bgr = hm_bgr
                    state.overlay_bgr = overlay
            except Exception as e:
                print(f"[GradCAM thread] {e}")
            finally:
                with state._lock: state.gradcam_busy = False

        elif task_type == "shap":
            if shap_obj is None and len(detection_log) >= SHAP_BACKGROUND_SAMPLES:
                from xai.shap_explainer import FogAlertSHAPExplainer
                bg = build_background_dataset(detection_log, SHAP_BACKGROUND_SAMPLES)
                shap_obj = FogAlertSHAPExplainer(bg)
                print("[SHAP thread] Explainer initialised.")
            if shap_obj and task.get("features") is not None:
                try:
                    result = shap_obj.explain(task["features"])
                    with state._lock:
                        state.shap_result = result
                    if SHAP_SAVE_PLOTS:
                        save_path = os.path.join(
                            OUTPUT_DIR, "shap",
                            f"frame_{frame_count_ref[0]:05d}.png")
                        shap_obj.plot_waterfall(result,
                            title=task.get("title",""), save_path=save_path)
                except Exception as e:
                    print(f"[SHAP thread] {e}")
            with state._lock: state.shap_busy = False

        task_queue.task_done()


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",  default="0")
    p.add_argument("--weights", default=None, help="AOD-Net .pth path")
    p.add_argument("--save",    action="store_true")
    p.add_argument("--no-shap", action="store_true")
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    use_shap = not args.no_shap

    # ── Load models ───────────────────────────────────────────────────────────
    print("[Pipeline] Loading AOD-Net...")
    enhancer = AODNetEnhancer(
        weights_path=args.weights,
        device=DEVICE,
        inference_scale=AOD_INFERENCE_SCALE,
    )

    print("[Pipeline] Loading YOLOv11...")
    detector  = YOLOv11Detector()
    alert_mgr = AlertManager()

    # ── Shared state ──────────────────────────────────────────────────────────
    async_state     = AsyncState()
    async_state.overlay_bgr = None
    detection_log   = []
    frame_count_ref = [0]   # mutable ref for thread
    task_queue      = queue.Queue(maxsize=4)

    # ── Start background XAI thread ───────────────────────────────────────────
    xai_thread = threading.Thread(
        target=_xai_worker,
        args=(task_queue, async_state, enhancer.model,
              detection_log, frame_count_ref),
        daemon=True,
    )
    xai_thread.start()

    # ── Video source ──────────────────────────────────────────────────────────
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {args.source}"); sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Pipeline] Source: {args.source} ({frame_w}×{frame_h})")
    reset_ema()

    # ── Window ────────────────────────────────────────────────────────────────
    if not args.no_display:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WIN, frame_w, frame_h)

    # ── Output writer ─────────────────────────────────────────────────────────
    writer = None
    if args.save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "unified_output.mp4")
        writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                   VIDEO_FPS, (frame_w, frame_h))
        print(f"[Pipeline] Saving → {out_path}")

    # ── Loop state ────────────────────────────────────────────────────────────
    enhancement_on = True
    show_gradcam   = True
    show_shap_badge = True
    fullscreen      = False
    fog_score       = 0.0
    frame_count     = 0
    fps_start       = time.time()

    print("[Pipeline] Running.  E=enhance  G=gradcam  X=shap  S=save  Q=quit")

    while True:
        ret, raw_frame = cap.read()
        if not ret: break

        frame_count += 1
        frame_count_ref[0] = frame_count

        # ── FRAME SKIP (latency knob) ─────────────────────────────────────────
        if FRAME_SKIP > 0 and frame_count % (FRAME_SKIP + 1) != 0:
            continue

        t0 = time.perf_counter()

        # ── STEP 1: Fog estimation ────────────────────────────────────────────
        if frame_count % FOG_CHECK_INTERVAL == 0 or frame_count == 1:
            fog_score = compute_fog_score(raw_frame)

        # ── STEP 2: AOD-Net enhancement ───────────────────────────────────────
        if enhancement_on:
            if SKIP_ENHANCEMENT_WHEN_CLEAR and fog_score < FOG_SKIP_THRESH:
                enhanced_frame = raw_frame.copy()  # skip — frame is clear
            else:
                enhanced_frame = enhancer.enhance(raw_frame)
        else:
            enhanced_frame = raw_frame.copy()

        # ── STEP 3: YOLOv11 on enhanced frame ────────────────────────────────
        detections = detector.detect(enhanced_frame)

        # ── STEP 4: Distance + feature extraction ────────────────────────────
        enriched = []
        for det in detections:
            box  = det["box_xyxy"]
            dist = estimate_distance(box, det["class_name"], smooth=True)
            feat = detection_to_features(
                box, det["confidence"], fog_score, frame_w, frame_h)
            enriched.append({**det, "distance_m": dist, "features": feat})
            detection_log.append(feat)

        # ── STEP 5: Alerts ────────────────────────────────────────────────────
        alert_mgr.evaluate(
            [{"class_name": d["class_name"], "distance_m": d["distance_m"]}
             for d in enriched], fog_score)

        # ── STEP 6: Queue GradCAM task (non-blocking) ─────────────────────────
        if (frame_count % GRADCAM_EVERY_N == 0
                and not async_state.gradcam_busy
                and not task_queue.full()):
            inp_t = (torch.from_numpy(
                         cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
                         .astype(np.float32) / 255.)
                     .permute(2, 0, 1).unsqueeze(0))
            with async_state._lock:
                async_state.gradcam_busy = True
            task_queue.put_nowait({
                "type": "gradcam",
                "input_tensor": inp_t,
                "enhanced_bgr": enhanced_frame.copy(),
            })

        # ── STEP 7: Queue SHAP task (non-blocking) ────────────────────────────
        if (use_shap
                and frame_count % SHAP_EXPLAIN_EVERY_N == 0
                and enriched
                and not async_state.shap_busy
                and not task_queue.full()
                and len(detection_log) >= SHAP_BACKGROUND_SAMPLES):
            closest = min(enriched, key=lambda d: d["distance_m"])
            with async_state._lock:
                async_state.shap_busy = True
            task_queue.put_nowait({
                "type":     "shap",
                "features": closest["features"].copy(),
                "title":    (f"Frame {frame_count} | "
                             f"{closest['class_name']} | "
                             f"{closest['distance_m']:.1f} m"),
            })

        # ── STEP 8: Annotate detection panel ─────────────────────────────────
        det_frame = enhanced_frame.copy()
        for det in enriched:
            _draw_detection(det_frame, det["box_xyxy"],
                            det["class_name"], det["confidence"],
                            det["distance_m"])
        _draw_fog_hud(det_frame, fog_score)
        if enriched:
            closest = min(enriched, key=lambda d: d["distance_m"])
            _draw_alert_banner(det_frame, closest["distance_m"],
                               closest["class_name"])

        # ── STEP 9: Read latest async results (thread-safe) ───────────────────
        with async_state._lock:
            overlay_bgr  = getattr(async_state, "overlay_bgr", None)
            shap_result  = async_state.shap_result

        # FPS
        elapsed = time.perf_counter() - fps_start if fps_start else 1
        fps     = frame_count / max(elapsed, 1e-5)
        cv2.putText(det_frame, f"FPS:{fps:.1f}",
                    (10, 22), FONT, 0.55, (200,200,200), 1, cv2.LINE_AA)

        # ── STEP 10: Build 4-panel grid ───────────────────────────────────────
        grid = _build_grid(
            raw_frame, enhanced_frame, det_frame,
            getattr(async_state, "heatmap_bgr", None),
            overlay_bgr, fog_score, enriched, enhancement_on,
            shap_result, frame_count, fps, show_gradcam, show_gradcam,
        )

        # ── STEP 11: Display ──────────────────────────────────────────────────
        if not args.no_display:
            cv2.imshow(WIN, grid)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('e'), ord('E')):
                enhancement_on = not enhancement_on
                print(f"[Pipeline] Enhancement: {'ON' if enhancement_on else 'OFF'}")
            elif key in (ord('g'), ord('G')):
                show_gradcam = not show_gradcam
                print(f"[Pipeline] GradCAM: {'ON' if show_gradcam else 'OFF'}")
            elif key in (ord('x'), ord('X')):
                show_shap_badge = not show_shap_badge
            elif key in (ord('f'), ord('F')):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    WIN, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key in (ord('s'), ord('S')):
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                fp = os.path.join(OUTPUT_DIR, f"frame_{frame_count:05d}.png")
                cv2.imwrite(fp, grid); print(f"[Pipeline] Saved: {fp}")

        if writer:
            writer.write(grid)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    task_queue.put(None)   # stop background thread
    xai_thread.join(timeout=3)
    alert_mgr.close()
    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print(f"[Pipeline] Done. {frame_count} frames processed.")


if __name__ == "__main__":
    main()
