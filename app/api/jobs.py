"""Job submission, status and results endpoints."""

import shutil
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.jobs import reanalyse, run_job
from app.models import Job, QueueEvent, Sample, StopEvent, Vehicle
from app.pipeline.analysis import class_summary
from app.pipeline.config import PipelineConfig
from app.pipeline.device import resolve_device
from app.schemas import (
    ClassSummaryOut,
    EventsOut,
    ImportCreate,
    JobCreate,
    JobCreated,
    JobOut,
    SamplesOut,
    VehicleOut,
)

router = APIRouter(prefix="/jobs", tags=["jobs"])

DOWNLOADS = {
    "excel": (
        "speed_predictions.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
    "csv": ("speed_predictions.csv", "text/csv"),
    "summary": ("summary.json", "application/json"),
    "video": ("annotated.mp4", "video/mp4"),
    "trajectories_csv": ("trajectories.csv", "text/csv"),
    "trajectories_parquet": ("trajectories.parquet", "application/vnd.apache.parquet"),
    "run_info": ("run_info.json", "application/json"),
    "frame": ("frame.jpg", "image/jpeg"),
}


def _run_details(req) -> dict:
    # recorded_at is the camera's local clock time; any time zone given is dropped.
    recorded = req.recorded_at.replace(tzinfo=None) if req.recorded_at else None
    return {"location": req.location or None, "recorded_at": recorded}


def safe_file(directory: Path, name: str) -> Path:
    """Resolve `name` inside `directory`, refusing anything that escapes it."""
    path = (directory / name).resolve()
    if path.parent != directory.resolve() or not path.is_file():
        raise HTTPException(404, f"File '{name}' not found")
    return path


def _get_job(db: Session, job_id: str) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


def _dispatch(job_id: str, db: Session) -> None:
    if settings.job_executor == "thread":
        threading.Thread(target=run_job, args=(job_id,), daemon=True).start()
        return
    from app.worker.tasks import process_job

    try:
        process_job.delay(job_id)
    except Exception as exc:  # broker unreachable
        job = db.get(Job, job_id)
        job.status = "failed"
        job.error = f"Could not queue job - is Redis running? ({exc})"
        db.commit()
        raise HTTPException(503, job.error) from exc


def _new_job(db: Session, **fields) -> Job:
    job_id = uuid.uuid4().hex[:12]
    out = settings.runs_dir / job_id
    job = Job(id=job_id, output_dir=str(out), pipeline_version=settings.pipeline_version, **fields)
    db.add(job)
    db.commit()
    return job


@router.post("", status_code=202, response_model=JobCreated)
def create_job(req: JobCreate, db: Session = Depends(get_db)):
    safe_file(settings.videos_dir, req.video)
    if not (settings.calibrations_dir / f"{req.calibration}.json").is_file():
        raise HTTPException(404, f"Calibration '{req.calibration}' not found")
    if not settings.model_path.is_file():
        raise HTTPException(500, f"Model weights not found: {settings.model_path}")
    try:  # reject an unavailable device now, not halfway through the video
        resolve_device(req.config.detection.device)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    job = _new_job(
        db,
        name=req.name or Path(req.video).stem,
        source="pipeline",
        video=req.video,
        calibration=req.calibration,
        config=req.config.model_dump(),
        **_run_details(req),
    )
    _dispatch(job.id, db)
    return JobCreated(job_id=job.id, status=job.status)


@router.post("/import", status_code=202, response_model=JobCreated)
def import_excel(req: ImportCreate, db: Session = Depends(get_db)):
    """Load results from an Excel file produced by 8+8+++_fixed.py (no video processing)."""
    path = safe_file(settings.videos_dir, req.excel_file)
    job = _new_job(
        db,
        name=req.name or f"Imported: {path.stem}",
        source="import",
        video=str(path),
        config=req.config.model_dump(),
        **_run_details(req),
    )
    _dispatch(job.id, db)
    return JobCreated(job_id=job.id, status=job.status)


@router.get("", response_model=list[JobOut])
def list_jobs(status: str | None = None, db: Session = Depends(get_db)):
    q = select(Job).order_by(Job.created_at.desc())
    if status:
        q = q.where(Job.status == status)
    return db.scalars(q).all()


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_db)):
    return _get_job(db, job_id)


@router.post("/{job_id}/reanalyse", response_model=JobOut)
def reanalyse_job(job_id: str, config: PipelineConfig, db: Session = Depends(get_db)):
    """Recompute stops, queues, status, flow and LOS with new thresholds (no video re-run)."""
    job = _get_job(db, job_id)
    if job.status != "completed":
        raise HTTPException(409, "Only completed runs can be re-analysed")
    reanalyse(job_id, config)
    db.expire_all()
    return _get_job(db, job_id)


@router.post("/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: str, db: Session = Depends(get_db)):
    job = _get_job(db, job_id)
    if job.status in ("queued", "running"):
        job.cancel_requested = True
        if job.status == "queued":
            job.status = "cancelled"
        db.commit()
    return job


@router.delete("/{job_id}", status_code=204)
def delete_job(job_id: str, db: Session = Depends(get_db)):
    job = _get_job(db, job_id)
    if job.status == "running":
        raise HTTPException(409, "Cancel the job before deleting it")
    for table in (Sample, Vehicle, StopEvent, QueueEvent):
        db.query(table).filter(table.job_id == job_id).delete()
    out = Path(job.output_dir) if job.output_dir else None
    db.delete(job)
    db.commit()
    if out and out.resolve().parent == settings.runs_dir.resolve():
        shutil.rmtree(out, ignore_errors=True)
    return Response(status_code=204)


@router.get("/{job_id}/preview")
def live_preview(job_id: str, db: Session = Depends(get_db)):
    """Latest annotated frame while the job runs."""
    job = _get_job(db, job_id)
    path = Path(job.output_dir) / "preview.jpg"
    if not path.is_file():
        raise HTTPException(404, "No preview yet")
    return Response(
        path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"}
    )


@router.get("/{job_id}/files/{kind}")
def download(job_id: str, kind: str, db: Session = Depends(get_db)):
    if kind not in DOWNLOADS:
        raise HTTPException(404, f"Unknown file kind '{kind}'")
    job = _get_job(db, job_id)
    name, media = DOWNLOADS[kind]
    path = safe_file(Path(job.output_dir), name)
    return FileResponse(path, media_type=media, filename=f"{job.name}_{name}")


@router.get("/{job_id}/vehicles", response_model=list[VehicleOut])
def vehicles(
    job_id: str,
    vehicle_type: list[str] | None = Query(None),
    status: list[str] | None = Query(None),
    include_low_sample: bool = True,
    db: Session = Depends(get_db),
):
    _get_job(db, job_id)
    q = select(Vehicle).where(Vehicle.job_id == job_id)
    if vehicle_type:
        q = q.where(Vehicle.vehicle_type.in_(vehicle_type))
    if status:
        q = q.where(Vehicle.status.in_(status))
    if not include_low_sample:
        q = q.where(Vehicle.is_low_sample.is_(False))
    return db.scalars(q.order_by(Vehicle.first_seen_sec)).all()


@router.get("/{job_id}/vehicles/{vehicle_id}/stats")
def vehicle_stats(job_id: str, vehicle_id: str, db: Session = Depends(get_db)):
    v = db.scalar(select(Vehicle).where(Vehicle.job_id == job_id, Vehicle.vehicle_id == vehicle_id))
    if v is None:
        raise HTTPException(404, "Vehicle not found")
    return v.stats


@router.get("/{job_id}/class-summary", response_model=list[ClassSummaryOut])
def get_class_summary(job_id: str, db: Session = Depends(get_db)):
    _get_job(db, job_id)
    rows = db.scalars(select(Vehicle.stats).where(Vehicle.job_id == job_id)).all()
    return class_summary(list(rows))


@router.get("/{job_id}/samples", response_model=SamplesOut)
def samples(
    job_id: str,
    vehicle_id: list[str] | None = Query(None),
    step: int = Query(1, ge=1, le=100, description="Keep every n-th sample per vehicle"),
    db: Session = Depends(get_db),
):
    """Per-frame positions and speeds, for space-time diagrams, timelines and video paths."""
    _get_job(db, job_id)
    cols = ("vehicle_id", "t", "s", "l", "speed", "x_px", "y_px", "confidence")
    q = select(*(getattr(Sample, c) for c in cols)).where(Sample.job_id == job_id)
    if vehicle_id:
        q = q.where(Sample.vehicle_id.in_(vehicle_id))
    out = {c: [] for c in cols}
    counters: dict[str, int] = {}
    for row in db.execute(q.order_by(Sample.vehicle_id, Sample.frame_idx)):
        vid = row[0]
        n = counters.get(vid, 0)
        counters[vid] = n + 1
        if n % step:
            continue
        for c, val in zip(cols, row, strict=True):
            out[c].append(val)
    return out


@router.get("/{job_id}/events", response_model=EventsOut)
def events(job_id: str, db: Session = Depends(get_db)):
    _get_job(db, job_id)
    stops = db.scalars(
        select(StopEvent).where(StopEvent.job_id == job_id).order_by(StopEvent.start_sec)
    ).all()
    queues = db.scalars(
        select(QueueEvent).where(QueueEvent.job_id == job_id).order_by(QueueEvent.start_sec)
    ).all()
    return {"stop_events": stops, "queue_events": queues}
