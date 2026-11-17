"""Job execution: shared by the Celery worker and the in-process thread executor."""

import hashlib
import json
import logging
import platform
import time
import traceback
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from sqlalchemy import delete, insert, select

from app.core.config import settings
from app.core.database import SessionLocal
from app.models import Job, QueueEvent, Sample, StopEvent, Vehicle
from app.pipeline import exports
from app.pipeline.analysis import Sample as AnalysisSample
from app.pipeline.analysis import analyse
from app.pipeline.config import BOTSORT_CONFIG, TRACK_CLASSES, PipelineConfig

log = logging.getLogger(__name__)

VEHICLE_COLUMNS = {c.name for c in Vehicle.__table__.columns} - {"id", "job_id", "stats"}
_HASH_CACHE: dict[tuple, str] = {}


def load_calibration(name: str) -> dict:
    path = settings.calibrations_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Calibration '{name}' not found in {settings.calibrations_dir}")
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, cached by (path, size, modified time)."""
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime)
    if key not in _HASH_CACHE:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _HASH_CACHE[key] = h.hexdigest()
    return _HASH_CACHE[key]


def _file_info(path: Path) -> dict:
    st = path.stat()
    return {
        "file": path.name,
        "path": str(path),
        "size_bytes": st.st_size,
        "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    }


def software_versions() -> dict:
    out = {"pipeline_version": settings.pipeline_version, "python": platform.python_version()}
    for pkg in ("ultralytics", "torch", "opencv-python", "numpy", "lap"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def _run_header(job: Job) -> dict:
    return {
        "run_id": job.id,
        "name": job.name,
        "source": job.source,
        "location": job.location,
        "recorded_at": job.recorded_at.isoformat() if job.recorded_at else None,
        "processed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def run_job(job_id: str) -> None:
    """Entry point for both executors. Never raises; failures are stored on the job."""
    try:
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            if job is None:
                log.error("Job %s not found", job_id)
                return
            job.status = "running"
            job.started_at = datetime.now(UTC)
            db.commit()
            source = job.source

        if source == "import":
            _run_import(job_id)
        else:
            _run_pipeline(job_id)
    except Exception as exc:  # noqa: BLE001 - report any failure on the job
        log.exception("Job %s failed", job_id)
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            if job:
                cancelled = type(exc).__name__ == "CancelledError"
                job.status = "cancelled" if cancelled else "failed"
                job.error = str(exc) if cancelled else f"{exc}\n\n{traceback.format_exc()}"
                job.finished_at = datetime.now(UTC)
                db.commit()


def _run_pipeline(job_id: str) -> None:
    # Imported here so the API process does not load PyTorch unless it runs jobs itself.
    from app.pipeline.speed_pipeline import run_speed_pipeline

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        cfg = PipelineConfig.model_validate(job.config)
        video_path = settings.videos_dir / job.video
        calibration = load_calibration(job.calibration)
        output_dir = Path(job.output_dir)
        run_info = _run_header(job)
        calibration_name = job.calibration

    last_write = 0.0

    def on_progress(done: int, total: int, stats: dict) -> bool:
        nonlocal last_write
        now = time.time()
        if now - last_write < 1.0 and done != total:
            return True
        last_write = now
        with SessionLocal() as s:
            j = s.get(Job, job_id)
            j.frames_done = done
            j.frames_total = total
            j.progress = min(done / total, 1.0) if total else 0.0
            j.live_stats = stats
            s.commit()
            return not j.cancel_requested

    result = run_speed_pipeline(
        video_path, calibration, settings.model_path, output_dir, cfg, on_progress
    )
    width, height = result["frame_size"]
    run_info |= {
        "video": _file_info(video_path)
        | {
            "fps": result["fps"],
            "frame_count": result["video_frame_count"],
            "width": width,
            "height": height,
            "frames_processed": result["frames_processed"],
            "duration_processed_s": result["frames_processed"] / result["fps"],
        },
        # A full copy, so the run stays reproducible even if the calibration file changes.
        "calibration": {"name": calibration_name, "data": calibration},
        "model": _file_info(settings.model_path) | {"sha256": file_sha256(settings.model_path)},
        "detection": cfg.detection.model_dump()
        | {
            "device_requested": cfg.detection.device or "auto",
            "device_used": result["device"],
            "precision": result["precision"],
            "mps_cpu_fallback": result["mps_cpu_fallback"],
        },
        "speed": cfg.speed.model_dump(),
        "tracker": BOTSORT_CONFIG,
        "classes": TRACK_CLASSES,
        "software": software_versions(),
    }
    _finish(job_id, result, cfg, output_dir, has_video=True, run_info=run_info)


def _run_import(job_id: str) -> None:
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        cfg = PipelineConfig.model_validate(job.config)
        excel_path = Path(job.video)
        output_dir = Path(job.output_dir)
        run_info = _run_header(job)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = exports.read_excel_samples(excel_path)
    run_info |= {
        "source_excel": _file_info(excel_path) | {"sha256": file_sha256(excel_path)},
        "video": {"fps_estimated_from_excel": result["fps"]},
        "note": "Imported from Excel: no video, calibration or model details are available.",
        "software": software_versions(),
    }
    _finish(job_id, result, cfg, output_dir, has_video=False, run_info=run_info)


def reanalyse(job_id: str, cfg: PipelineConfig) -> None:
    """Recompute statuses, events, flow and LOS for a finished job with new thresholds.

    Uses the stored per-frame samples, so the video is not processed again. Detection,
    tracking and speed settings cannot change this way - those need a new run.
    """
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        old = PipelineConfig.model_validate(job.config)
        # Speed samples were measured with the old detection/speed settings; keep them.
        cfg = cfg.model_copy(
            update={"detection": old.detection, "speed": old.speed, "max_seconds": old.max_seconds}
        )
        histories: dict[str, list[AnalysisSample]] = {}
        rows = db.scalars(
            select(Sample)
            .where(Sample.job_id == job_id)
            .order_by(Sample.vehicle_id, Sample.frame_idx)
        )
        for r in rows:
            box = (r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2)
            histories.setdefault(r.vehicle_id, []).append(
                AnalysisSample(
                    r.frame_idx,
                    r.t,
                    r.s,
                    r.l,
                    r.speed,
                    x_px=r.x_px,
                    y_px=r.y_px,
                    bbox=box if None not in box else None,
                    confidence=r.confidence,
                    class_raw=r.class_raw,
                )
            )
        classes = dict(
            db.execute(
                select(Vehicle.vehicle_id, Vehicle.vehicle_type).where(Vehicle.job_id == job_id)
            ).all()
        )
        metrics = job.metrics or {}
        result = {
            "histories": histories,
            "classes": classes,
            "fps": metrics["fps"],
            "frames_processed": metrics["frames_processed"],
        }
        job.config = cfg.model_dump()
        db.commit()
        output_dir, has_video, run_info = Path(job.output_dir), job.has_video, job.run_info
    _finish(job_id, result, cfg, output_dir, has_video=has_video, run_info=run_info)


def _finish(
    job_id: str,
    result: dict,
    cfg: PipelineConfig,
    output_dir: Path,
    has_video: bool,
    run_info: dict | None = None,
):
    histories = result["histories"]
    analysis = analyse(histories, result["classes"], result["fps"], result["frames_processed"], cfg)
    with SessionLocal() as db:
        recorded_at = db.get(Job, job_id).recorded_at

    exports.write_excel(output_dir / "speed_predictions.xlsx", histories, analysis, run_info)
    exports.write_wide_csv(output_dir / "speed_predictions.csv", histories, analysis["vehicles"])
    exports.write_trajectories(
        output_dir,
        exports.trajectories_frame(histories, analysis["vehicles"], job_id, recorded_at),
    )
    if run_info:
        exports.write_json(output_dir / "run_info.json", run_info)
    exports.write_json(
        output_dir / "summary.json",
        {
            "metrics": analysis["metrics"],
            "class_summary": analysis["class_summary"],
            "stop_events": analysis["stop_events"],
            "queue_events": analysis["queue_events"],
        },
    )

    with SessionLocal() as db:
        for table in (Vehicle, Sample, StopEvent, QueueEvent):
            db.execute(delete(table).where(table.job_id == job_id))

        vehicle_rows = [
            {"job_id": job_id, "stats": _clean(v), **{k: v.get(k) for k in VEHICLE_COLUMNS}}
            for v in analysis["vehicles"]
        ]
        if vehicle_rows:
            db.execute(insert(Vehicle), vehicle_rows)

        sample_rows = [
            {
                "job_id": job_id,
                "vehicle_id": vid,
                "frame_idx": smp.frame_idx,
                "t": smp.t,
                "s": smp.s,
                "l": smp.l,
                "speed": smp.speed,
                "x_px": smp.x_px,
                "y_px": smp.y_px,
                "bbox_x1": smp.bbox[0] if smp.bbox else None,
                "bbox_y1": smp.bbox[1] if smp.bbox else None,
                "bbox_x2": smp.bbox[2] if smp.bbox else None,
                "bbox_y2": smp.bbox[3] if smp.bbox else None,
                "confidence": smp.confidence,
                "class_raw": smp.class_raw,
            }
            for vid, hist in histories.items()
            for smp in hist
        ]
        for i in range(0, len(sample_rows), 20000):
            db.execute(insert(Sample), sample_rows[i : i + 20000])

        if analysis["stop_events"]:
            db.execute(
                insert(StopEvent), [{"job_id": job_id, **e} for e in analysis["stop_events"]]
            )
        if analysis["queue_events"]:
            db.execute(
                insert(QueueEvent), [{"job_id": job_id, **q} for q in analysis["queue_events"]]
            )

        job = db.get(Job, job_id)
        job.metrics = _clean(analysis["metrics"])
        if run_info:
            job.run_info = _clean(run_info)
        job.has_video = has_video and (output_dir / "annotated.mp4").exists()
        job.status = "completed"
        job.progress = 1.0
        job.frames_done = result["frames_processed"]
        job.frames_total = max(job.frames_total or 0, result["frames_processed"])
        job.finished_at = datetime.now(UTC)
        job.pipeline_version = settings.pipeline_version
        db.commit()


def _clean(d: dict) -> dict:
    """JSON columns cannot store NaN/inf."""
    return exports._json_safe(d)
