"""Pipeline configuration. Defaults reproduce `8+8+++_fixed.py` exactly."""

from pydantic import BaseModel, Field

# best_xl_ITD_v1.2.pt (Indian Traffic Dataset, XL) class taxonomy - NOT COCO.
TRACK_CLASSES: dict[int, str] = {
    0: "two_wheeler",
    1: "autorickshaw",
    2: "car",
    3: "bus",
    4: "LCV",
    5: "truck",
    6: "bicycle",
    7: "pedestrian",
}
PERSON_CLASS_ID = 7
VEHICLE_PERSON_OVERLAP_CLASSES = {0, 1, 2, 3, 4, 5, 6}
NON_VEHICLE_TYPES = {"pedestrian", "unknown"}

# Passenger Car Units for mixed Indian traffic. Values follow the table used in
# traffic-impedance-analysis-main (result_store.py); LCV = 1.4 per IRC:106-1990.
PCU_FACTORS: dict[str, float] = {
    "car": 1.0,
    "two_wheeler": 0.5,
    "bicycle": 0.5,
    "autorickshaw": 1.2,
    "LCV": 1.4,
    "bus": 3.0,
    "truck": 3.0,
    "pedestrian": 0.0,
}

BOTSORT_CONFIG = {
    "tracker_type": "botsort",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.05,
    "new_track_thresh": 0.30,
    "track_buffer": 60,
    "match_thresh": 0.9,
    "fuse_score": True,
    "gmc_method": "sparseOptFlow",
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.25,
    "with_reid": True,
    "model": "auto",
}


class DetectionConfig(BaseModel):
    confidence_threshold: float = Field(0.15, ge=0, le=1)
    iou_threshold: float = Field(0.5, ge=0, le=1)
    imgsz: int = Field(640, ge=320, le=1920)
    # "" = auto: NVIDIA GPU if available, else Apple GPU (mps), else CPU (see device.py).
    # "cuda", "mps" or "cpu" forces one; "cpu" matches 8+8+++_fixed.py on a Mac.
    device: str = ""


class SpeedConfig(BaseModel):
    min_movement_m: float = Field(0.2, ge=0)
    time_window_s: float = Field(1.0, gt=0)
    reid_gap_s: float = Field(5.0, gt=0)
    direction_threshold_m: float = 1.0
    low_sample_min: int = 10


class StopEventConfig(BaseModel):
    """Same state machine and defaults as traffic-impedance-analysis-main."""

    max_speed_for_stop_kmh: float = 2.0
    min_stop_duration_frames: int = 45
    resume_hysteresis_frames: int = 10


class QueueConfig(BaseModel):
    queue_speed_threshold_kmh: float = 2.0
    min_queue_duration_frames: int = 30
    # Only vehicles this far upstream of the blocking vehicle can be part of its queue.
    max_queue_distance_m: float = 60.0


class RoadConfig(BaseModel):
    """Used for flow, V/C ratio and LOS. Leave num_lanes empty to skip V/C."""

    num_lanes: int | None = Field(None, ge=1)
    capacity_per_lane_pcu_hr: float = Field(1000, gt=0)
    parked_speed_threshold_kmh: float = 2.0
    # A parked vehicle must also stay put: vehicles that crawled in from further
    # away are queued traffic, not parked, even if their median speed is ~0.
    parked_max_displacement_m: float = 3.0
    # Distance along the centreline of the line used to count flow.
    # None = middle of the observed road section.
    flow_line_m: float | None = None


class PipelineConfig(BaseModel):
    detection: DetectionConfig = DetectionConfig()
    speed: SpeedConfig = SpeedConfig()
    stop_event: StopEventConfig = StopEventConfig()
    queue: QueueConfig = QueueConfig()
    road: RoadConfig = RoadConfig()
    show_trajectories: bool = True
    # Stop after this many seconds of video (None = whole video). Useful for quick tests.
    max_seconds: float | None = Field(None, gt=0)
    # Save a live preview frame every N frames.
    preview_every_n_frames: int = Field(5, ge=1)
