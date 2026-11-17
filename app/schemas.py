"""Pydantic request / response models for the API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.pipeline.config import PipelineConfig


class RunDetails(BaseModel):
    """Optional description of where and when the video was recorded."""

    location: str | None = Field(None, max_length=200, description="Camera / site name")
    recorded_at: datetime | None = Field(
        None, description="Local camera time of the first video frame"
    )


class JobCreate(RunDetails):
    video: str = Field(description="Video file name inside VIDEOS_DIR")
    calibration: str = Field(description="Calibration name (file stem in calibrations/)")
    name: str | None = None
    config: PipelineConfig = PipelineConfig()


class ImportCreate(RunDetails):
    excel_file: str = Field(description="Excel file written by 8+8+++_fixed.py, inside VIDEOS_DIR")
    name: str | None = None
    config: PipelineConfig = PipelineConfig()


class JobCreated(BaseModel):
    job_id: str
    status: str


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    source: str
    status: str
    video: str | None
    calibration: str | None
    config: dict
    progress: float
    frames_done: int
    frames_total: int
    live_stats: dict
    metrics: dict | None
    has_video: bool
    error: str | None
    pipeline_version: str | None
    location: str | None
    recorded_at: datetime | None
    run_info: dict | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class VehicleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    vehicle_id: str
    vehicle_type: str
    direction: str
    status: str
    first_seen_sec: float
    last_seen_sec: float
    track_duration_s: float
    n_flow_samples: int
    n_moving_samples: int
    is_low_sample: bool
    has_speed: bool
    estimated_speed: float | None
    mean_speed: float | None
    median_raw: float | None
    p85: float | None
    min_speed: float | None
    max_speed: float | None
    stopped_time_s: float
    pcu: float


class SamplesOut(BaseModel):
    """Column-oriented to keep responses small."""

    vehicle_id: list[str]
    t: list[float]
    s: list[float]
    l: list[float | None]  # noqa: E741
    speed: list[float]
    # Pixel position in the video frame (None for runs imported from Excel).
    x_px: list[float | None]
    y_px: list[float | None]
    confidence: list[float | None]


class StopEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    vehicle_id: str
    vehicle_type: str
    start_sec: float
    end_sec: float
    duration_s: float
    position_m: float | None
    lateral_m: float | None


class QueueEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    blocking_vehicle_id: str
    blocking_vehicle_type: str
    start_sec: float
    end_sec: float
    duration_s: float
    max_queue_length: int
    max_queue_length_m: float
    affected_vehicle_ids: list[str]


class EventsOut(BaseModel):
    stop_events: list[StopEventOut]
    queue_events: list[QueueEventOut]


class ClassSummaryOut(BaseModel):
    vehicle_type: str
    count: int
    n_with_speed: int
    mean_speed: float | None
    median_speed: float | None
    std_speed: float | None
    min_speed: float | None
    max_speed: float | None
