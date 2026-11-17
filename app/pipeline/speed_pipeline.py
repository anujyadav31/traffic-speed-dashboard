"""Detection + tracking + centreline-calibrated speed estimation.

The per-frame logic (YOLO + BoT-SORT tracking, pedestrian suppression, class
smoothing, re-ID gap handling, time-window speed, trajectory drawing) is the
same as the main loop of `Vehicle detection and speed estimation/8+8+++_fixed.py`.

Differences, all needed to run as a background job:
* no cv2.imshow / keyboard handling - the latest annotated frame is written to
  `preview.jpg` instead, so the dashboard can show processing live;
* a progress callback, optional `max_seconds` limit and cancellation;
* the tracker YAML and all outputs go into the job's own output folder;
* per-sample lateral offset and exact frame index are kept for the dashboard;
* device "auto" picks an NVIDIA GPU, else the Apple GPU (mps), else the CPU
  (8+8+++_fixed.py leaves the choice to Ultralytics, which never picks mps).
"""

import math
import os
import subprocess
import time
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from app.pipeline import drawing
from app.pipeline.analysis import Sample
from app.pipeline.config import (
    BOTSORT_CONFIG,
    PERSON_CLASS_ID,
    TRACK_CLASSES,
    VEHICLE_PERSON_OVERLAP_CLASSES,
    PipelineConfig,
)
from app.pipeline.device import mps_cpu_fallback_enabled, resolve_device
from app.pipeline.geometry import RoadMapper

ProgressCallback = Callable[[int, int, dict], bool | None]
"""(frames_done, total_frames, live_stats) -> return False to cancel."""


class CancelledError(Exception):
    pass


def run_speed_pipeline(
    video_path: Path,
    calibration: dict,
    model_path: Path,
    output_dir: Path,
    cfg: PipelineConfig,
    progress: ProgressCallback | None = None,
) -> dict:
    """Process one video. Returns histories, classes, fps and frame counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    det = cfg.detection
    sp = cfg.speed
    device = resolve_device(det.device)

    tracker_yaml = output_dir / "botsort.yaml"
    with open(tracker_yaml, "w") as f:
        yaml.dump(BOTSORT_CONFIG, f)

    # A fresh YOLO object per job: model.track(persist=True) keeps tracker state on
    # the model, so sharing one between videos would leak track ids across jobs.
    model = YOLO(str(model_path))
    mapper = RoadMapper(calibration)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    video_frame_count = total_frames
    if cfg.max_seconds:
        limit = int(math.ceil(cfg.max_seconds * fps))
        total_frames = min(total_frames, limit) if total_frames else limit
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_video = output_dir / "annotated_raw.mp4"
    writer = cv2.VideoWriter(
        str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h)
    )
    preview_path = output_dir / "preview.jpg"

    track_history = defaultdict(lambda: deque(maxlen=500))
    track_pixel_history = defaultdict(lambda: deque(maxlen=drawing.TRAJECTORY_MAX_POINTS))
    track_colors: dict[int, tuple] = {}
    track_classes: dict[int, int] = {}
    track_class_history = defaultdict(lambda: deque(maxlen=10))
    histories: dict[str, list[Sample]] = defaultdict(list)
    track_last_seen: dict[int, float] = {}
    track_export_id: dict[int, str] = {}
    track_reuse_count: dict[int, int] = defaultdict(int)
    export_class: dict[str, int] = {}

    frame_counter = 0
    started = time.time()
    try:
        while True:
            if cfg.max_seconds and frame_counter >= total_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame_time = frame_counter / fps if fps > 0 else 0
            if frame_counter == 0:
                # Clean (unannotated) full-resolution frame: background for drawing the
                # saved pixel trajectories in the dashboard.
                cv2.imwrite(str(output_dir / "frame.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            track_kwargs = dict(
                persist=True,
                tracker=str(tracker_yaml),
                conf=det.confidence_threshold,
                iou=det.iou_threshold,
                classes=list(TRACK_CLASSES.keys()),
                imgsz=det.imgsz,
                verbose=False,
                device=device,
            )
            results = model.track(frame, **track_kwargs)

            live_speeds: list[float] = []
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()  # stored only, not used

                detections = []
                for box, tid, cls, conf in zip(boxes, track_ids, cls_ids, confs, strict=False):
                    x1, y1, x2, y2 = box
                    detections.append(
                        {
                            "box": (x1, y1, x2, y2),
                            "centroid": (int((x1 + x2) / 2), y2),
                            "track_id": int(tid),
                            "class": int(cls),
                            "conf": float(conf),
                        }
                    )

                # Suppress pedestrians inside/behind vehicles
                person_dets = [d for d in detections if d["class"] == PERSON_CLASS_ID]
                vehicle_dets = [
                    d for d in detections if d["class"] in VEHICLE_PERSON_OVERLAP_CLASSES
                ]
                suppressed_person_ids = set()
                for p in person_dets:
                    for v in vehicle_dets:
                        if drawing.compute_iou(p["box"], v["box"]) >= 0.5:
                            suppressed_person_ids.add(p["track_id"])
                            break

                for d in detections:
                    tid, cls, box, centroid = d["track_id"], d["class"], d["box"], d["centroid"]
                    if cls == PERSON_CLASS_ID and tid in suppressed_person_ids:
                        continue

                    if tid not in track_colors:
                        track_colors[tid] = drawing.generate_random_color()
                        track_classes[tid] = cls
                        track_class_history[tid].clear()
                    track_class_history[tid].append(cls)
                    smoothed_cls = drawing.most_frequent_class(track_class_history[tid])
                    if smoothed_cls != -1:
                        track_classes[tid] = smoothed_cls

                    # Re-ID guard: a recycled tracker id after a long gap is a new vehicle.
                    last_seen = track_last_seen.get(tid)
                    if tid not in track_export_id or (
                        last_seen is not None and (frame_time - last_seen) > sp.reid_gap_s
                    ):
                        if last_seen is not None:
                            track_reuse_count[tid] += 1
                        suffix = track_reuse_count[tid]
                        track_export_id[tid] = str(tid) if suffix == 0 else f"{tid}_{suffix}"
                        track_pixel_history[tid].clear()
                    track_last_seen[tid] = frame_time
                    export_id = track_export_id[tid]
                    export_class[export_id] = track_classes[tid]

                    s, l = mapper.image_to_world(centroid)
                    track_history[tid].append((s, l, frame_time))

                    pix_hist = track_pixel_history[tid]
                    if (
                        not pix_hist
                        or math.hypot(centroid[0] - pix_hist[-1][0], centroid[1] - pix_hist[-1][1])
                        >= drawing.TRAJECTORY_MIN_DRAW_PX
                    ):
                        pix_hist.append((centroid[0], centroid[1], frame_time))

                    # ---- Speed over the last TIME_WINDOW seconds ----
                    hist = track_history[tid]
                    recent = [p for p in hist if p[2] >= frame_time - sp.time_window_s]
                    speed = 0.0
                    if len(recent) >= 2:
                        total_dist = 0.0
                        total_time = recent[-1][2] - recent[0][2]
                        for i in range(len(recent) - 1):
                            s1, l1, _ = recent[i]
                            s2, l2, _ = recent[i + 1]
                            total_dist += mapper.physical_distance((s1, l1), (s2, l2))
                        if total_time > 0 and total_dist >= sp.min_movement_m:
                            speed = (total_dist / total_time) * 3.6

                    histories[export_id].append(
                        Sample(
                            frame_counter,
                            frame_time,
                            float(hist[-1][0]),
                            float(hist[-1][1]),
                            speed,
                            x_px=float(centroid[0]),
                            y_px=float(centroid[1]),
                            bbox=tuple(int(v) for v in box),
                            confidence=d["conf"],
                            class_raw=TRACK_CLASSES.get(cls, "unknown"),
                        )
                    )
                    if speed > 0:
                        live_speeds.append(speed)

                    # Draw
                    x1, y1, x2, y2 = box
                    color = track_colors[tid]
                    if cfg.show_trajectories:
                        trail = [
                            (px, py)
                            for px, py, pt in pix_hist
                            if pt >= frame_time - drawing.TRAJECTORY_SECONDS
                        ]
                        drawing.draw_trajectory(frame, trail, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    class_name = TRACK_CLASSES.get(track_classes[tid], "unknown")
                    label = f"{class_name[:3]} {tid} {speed:.1f} km/h"
                    cv2.putText(
                        frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

            writer.write(frame)
            frame_counter += 1

            if frame_counter % cfg.preview_every_n_frames == 0 or frame_counter == 1:
                _write_preview(frame, preview_path)
            if progress is not None and (frame_counter % 5 == 0 or frame_counter == 1):
                elapsed = time.time() - started
                stats = {
                    "tracks_so_far": len(histories),
                    "vehicles_in_frame": len(live_speeds),
                    "mean_speed_in_frame": sum(live_speeds) / len(live_speeds)
                    if live_speeds
                    else None,
                    "video_time_s": frame_time,
                    "processing_fps": frame_counter / elapsed if elapsed > 0 else None,
                }
                if progress(frame_counter, total_frames, stats) is False:
                    raise CancelledError("Job cancelled")
    finally:
        cap.release()
        writer.release()
        tracker_yaml.unlink(missing_ok=True)

    video_ok = reencode_for_browser(raw_video, output_dir / "annotated.mp4")
    if video_ok:
        raw_video.unlink(missing_ok=True)
    else:
        raw_video.replace(output_dir / "annotated.mp4")

    classes = {eid: TRACK_CLASSES.get(c, "unknown") for eid, c in export_class.items()}
    return {
        "histories": dict(histories),
        "classes": classes,
        "fps": fps,
        "frames_processed": frame_counter,
        "frame_size": (frame_w, frame_h),
        "browser_video": video_ok,
        "video_frame_count": video_frame_count,
        "device": device,
        "mps_cpu_fallback": device == "mps" and mps_cpu_fallback_enabled(),
        "precision": _model_precision(model),
    }


def _model_precision(model: YOLO) -> str | None:
    """Number format the detector actually ran in (float32 unless half precision is on)."""
    try:
        return str(next(model.predictor.model.model.parameters()).dtype).removeprefix("torch.")
    except (AttributeError, StopIteration):
        return None  # no frame was processed


def _write_preview(frame, path: Path) -> None:
    """Write atomically so the API never serves a half-written JPEG."""
    h, w = frame.shape[:2]
    scale = 960 / w if w > 960 else 1.0
    small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(buf.tobytes())
        os.replace(tmp, path)


def reencode_for_browser(src: Path, dst: Path) -> bool:
    """Re-encode OpenCV's mp4v output to H.264 so browsers can play it."""
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(dst),
            ],
            check=True,
            timeout=3600,
        )
        return dst.exists() and dst.stat().st_size > 0
    except (OSError, subprocess.SubprocessError):
        dst.unlink(missing_ok=True)
        return False
