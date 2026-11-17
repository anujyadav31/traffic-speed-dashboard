"""Database tables."""

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


def _now() -> datetime:
    return datetime.now(UTC)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    source: Mapped[str] = mapped_column(String(20))  # "pipeline" | "import"
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    video: Mapped[str | None] = mapped_column(String(500))
    calibration: Mapped[str | None] = mapped_column(String(200))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    frames_done: Mapped[int] = mapped_column(Integer, default=0)
    frames_total: Mapped[int] = mapped_column(Integer, default=0)
    live_stats: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict | None] = mapped_column(JSON)
    output_dir: Mapped[str | None] = mapped_column(String(500))
    has_video: Mapped[bool] = mapped_column(Boolean, default=False)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text)
    pipeline_version: Mapped[str | None] = mapped_column(String(20))
    # Where and when the video was recorded (entered by the user; local camera time).
    location: Mapped[str | None] = mapped_column(String(200))
    recorded_at: Mapped[datetime | None] = mapped_column(DateTime)
    # Everything needed to reproduce the run: video metadata, full calibration copy,
    # model file + hash, tracker and detection settings, software versions.
    run_info: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Vehicle(Base):
    __tablename__ = "vehicles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    vehicle_id: Mapped[str] = mapped_column(String(32))
    vehicle_type: Mapped[str] = mapped_column(String(32), index=True)
    direction: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), index=True)
    first_seen_sec: Mapped[float] = mapped_column(Float)
    last_seen_sec: Mapped[float] = mapped_column(Float)
    track_duration_s: Mapped[float] = mapped_column(Float)
    n_flow_samples: Mapped[int] = mapped_column(Integer)
    n_moving_samples: Mapped[int] = mapped_column(Integer)
    is_low_sample: Mapped[bool] = mapped_column(Boolean)
    has_speed: Mapped[bool] = mapped_column(Boolean)
    estimated_speed: Mapped[float | None] = mapped_column(Float)
    mean_speed: Mapped[float | None] = mapped_column(Float)
    median_raw: Mapped[float | None] = mapped_column(Float)
    p85: Mapped[float | None] = mapped_column(Float)
    min_speed: Mapped[float | None] = mapped_column(Float)
    max_speed: Mapped[float | None] = mapped_column(Float)
    stopped_time_s: Mapped[float] = mapped_column(Float, default=0.0)
    pcu: Mapped[float] = mapped_column(Float)
    stats: Mapped[dict] = mapped_column(JSON)  # the full per-vehicle summary row


class Sample(Base):
    __tablename__ = "samples"
    __table_args__ = (Index("ix_samples_job_vehicle", "job_id", "vehicle_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    vehicle_id: Mapped[str] = mapped_column(String(32))
    frame_idx: Mapped[int] = mapped_column(Integer)
    t: Mapped[float] = mapped_column(Float)
    s: Mapped[float] = mapped_column(Float)
    l: Mapped[float | None] = mapped_column(Float)  # noqa: E741
    speed: Mapped[float] = mapped_column(Float)
    # Raw image measurements (NULL for runs imported from Excel). Kept so the road
    # coordinates can be recomputed with a better calibration without re-processing.
    x_px: Mapped[float | None] = mapped_column(Float)  # point used for speed:
    y_px: Mapped[float | None] = mapped_column(Float)  # bottom-centre of the box
    bbox_x1: Mapped[int | None] = mapped_column(Integer)
    bbox_y1: Mapped[int | None] = mapped_column(Integer)
    bbox_x2: Mapped[int | None] = mapped_column(Integer)
    bbox_y2: Mapped[int | None] = mapped_column(Integer)
    confidence: Mapped[float | None] = mapped_column(Float)
    class_raw: Mapped[str | None] = mapped_column(String(32))  # this frame's class, unsmoothed


class StopEvent(Base):
    __tablename__ = "stop_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    vehicle_id: Mapped[str] = mapped_column(String(32))
    vehicle_type: Mapped[str] = mapped_column(String(32))
    start_frame: Mapped[int] = mapped_column(Integer)
    end_frame: Mapped[int] = mapped_column(Integer)
    start_sec: Mapped[float] = mapped_column(Float)
    end_sec: Mapped[float] = mapped_column(Float)
    duration_s: Mapped[float] = mapped_column(Float)
    position_m: Mapped[float | None] = mapped_column(Float)
    lateral_m: Mapped[float | None] = mapped_column(Float)


class QueueEvent(Base):
    __tablename__ = "queue_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    blocking_vehicle_id: Mapped[str] = mapped_column(String(32))
    blocking_vehicle_type: Mapped[str] = mapped_column(String(32))
    start_frame: Mapped[int] = mapped_column(Integer)
    end_frame: Mapped[int] = mapped_column(Integer)
    start_sec: Mapped[float] = mapped_column(Float)
    end_sec: Mapped[float] = mapped_column(Float)
    duration_s: Mapped[float] = mapped_column(Float)
    max_queue_length: Mapped[int] = mapped_column(Integer)
    max_queue_length_m: Mapped[float] = mapped_column(Float)
    affected_vehicle_ids: Mapped[list] = mapped_column(JSON)
