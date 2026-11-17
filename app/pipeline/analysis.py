"""Post-processing of tracked vehicle histories.

Takes the per-vehicle samples produced by the speed pipeline (or imported from
an Excel file written by 8+8+++_fixed.py) and derives:

* per-vehicle statistics - same formulas as the Excel export of 8+8+++_fixed.py
* per-class statistics
* stop events and queue events - same state machine / defaults as
  traffic-impedance-analysis-main (homography_pipeline.py)
* parked / stopped / moving status
* flow, PCU, V/C ratio and LOS - thresholds from traffic-impedance-analysis-main
  (result_store.py, based on HCM / IRC mid-block criteria)

Everything here is pure Python so it can be unit-tested without a video.
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from app.pipeline.config import NON_VEHICLE_TYPES, PCU_FACTORS, PipelineConfig

PCTS = (10, 25, 40, 50, 60, 75, 85, 90)


@dataclass
class Sample:
    frame_idx: int
    t: float  # seconds from the start of the video
    s: float  # distance along the centreline (m)
    l: float | None  # lateral offset from the centreline (m); None when imported
    speed: float  # km/h (0.0 when displacement < min_movement_m)
    # Raw detection data, only stored - none of it is used in the calculations above.
    # None for runs imported from Excel.
    x_px: float | None = None  # point used for speed (bottom-centre of the box)
    y_px: float | None = None
    bbox: tuple[int, int, int, int] | None = None  # x1, y1, x2, y2 in pixels
    confidence: float | None = None
    class_raw: str | None = None  # this frame's class before smoothing


def percentile(sorted_vals: list[float], q: float) -> float | None:
    """Linear-interpolated percentile of an already-sorted list; q in 0..100."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def track_direction(hist: list[Sample], threshold_m: float) -> str:
    """'going' = along the centreline direction, 'coming' = against it."""
    if len(hist) < 2:
        return "stationary"
    delta_s = hist[-1].s - hist[0].s
    if delta_s > threshold_m:
        return "going"
    if delta_s < -threshold_m:
        return "coming"
    return "stationary"


def sort_ids(ids) -> list[str]:
    """Order export ids like the original script: 1, 1_1, 2, 10 ..."""

    def key(k: str):
        head = k.split("_")[0]
        return (int(head) if head.isdigit() else math.inf, k)

    return sorted(ids, key=key)


# ------------------------------------------------------------------ stop events
def detect_stops(
    histories: dict[str, list[Sample]], classes: dict[str, str], cfg: PipelineConfig
) -> list[dict]:
    """MOVING -> CANDIDATE_STOP -> STOPPED state machine per vehicle."""
    sc = cfg.stop_event
    stops: list[dict] = []
    for vid, hist in histories.items():
        if classes.get(vid, "unknown") in NON_VEHICLE_TYPES:
            continue
        state = "MOVING"
        candidate_start = None
        resume = 0
        active_start = None
        active_pos = []
        for smp in hist:
            if smp.speed <= sc.max_speed_for_stop_kmh:
                resume = 0
                if state == "MOVING":
                    state = "CANDIDATE_STOP"
                    candidate_start = smp
                    active_pos = [smp.s]
                elif state == "CANDIDATE_STOP":
                    active_pos.append(smp.s)
                    if smp.frame_idx - candidate_start.frame_idx >= sc.min_stop_duration_frames:
                        state = "STOPPED"
                        active_start = candidate_start
                else:
                    active_pos.append(smp.s)
            else:
                resume += 1
                if resume >= sc.resume_hysteresis_frames:
                    if state == "STOPPED":
                        stops.append(_stop_record(vid, classes, active_start, smp, active_pos))
                    state = "MOVING"
                    resume = 0
                    active_start = None
        if state == "STOPPED" and active_start is not None:
            stops.append(_stop_record(vid, classes, active_start, hist[-1], active_pos))
    stops.sort(key=lambda e: (e["start_frame"], e["vehicle_id"]))
    return stops


def _stop_record(vid, classes, start: Sample, end: Sample, positions: list[float]) -> dict:
    pos = sorted(positions)
    return {
        "vehicle_id": vid,
        "vehicle_type": classes.get(vid, "unknown"),
        "start_frame": start.frame_idx,
        "end_frame": end.frame_idx,
        "start_sec": start.t,
        "end_sec": end.t,
        "duration_s": end.t - start.t,
        "position_m": percentile(pos, 50),
        "lateral_m": start.l,
    }


# ------------------------------------------------------------------ queue events
def detect_queues(
    stops: list[dict],
    histories: dict[str, list[Sample]],
    classes: dict[str, str],
    status: dict[str, str],
    direction: dict[str, str],
    cfg: PipelineConfig,
) -> list[dict]:
    """For every stop, find slow vehicles queued behind the stopped vehicle.

    "Behind" depends on the queued vehicle's direction of travel: for 'going'
    traffic (along the centreline) it means a smaller distance along the road,
    for 'coming' traffic a larger one. Vehicles with no net direction are
    skipped. When lateral offsets are known, the queued vehicle must also be on
    the same side of the centreline as the stopped one.

    A vehicle counts as queued at frame f when it has been below the queue speed
    threshold for the last `min_queue_duration_frames` frames (a missing sample
    counts as not slow, as in traffic-impedance-analysis-main). Vehicles that are
    themselves parked are excluded - they are parked, not queued.
    """
    qc = cfg.queue
    win = qc.min_queue_duration_frames
    by_frame: dict[int, dict[str, Sample]] = defaultdict(dict)
    for vid, hist in histories.items():
        if (
            classes.get(vid, "unknown") in NON_VEHICLE_TYPES
            or status.get(vid) == "Parked"
            or direction.get(vid) not in ("going", "coming")
        ):
            continue
        for smp in hist:
            by_frame[smp.frame_idx][vid] = smp

    def gap_behind(vid: str, smp: Sample, blocker: Sample) -> float:
        """Distance from the queued vehicle up to the blocker (positive = behind)."""
        if direction[vid] == "going":
            return blocker.s - smp.s
        return smp.s - blocker.s

    def same_side(smp: Sample, blocker: Sample) -> bool:
        if smp.l is None or blocker.l is None:
            return True
        return (smp.l >= 0) == (blocker.l >= 0)

    def slow_for_window(vid: str, f: int) -> bool:
        for t in range(f - win + 1, f + 1):
            smp = by_frame.get(t, {}).get(vid)
            if smp is None or smp.speed >= qc.queue_speed_threshold_kmh:
                return False
        return True

    blocker_at = {vid: {smp.frame_idx: smp for smp in hist} for vid, hist in histories.items()}

    queues = []
    for stop in stops:
        blocker = stop["vehicle_id"]
        affected: set[str] = set()
        max_q = 0
        max_q_len_m = 0.0
        for f in range(stop["start_frame"] + win - 1, stop["end_frame"] + 1):
            b = blocker_at[blocker].get(f)
            if b is None:
                continue
            queued = []
            for vid, smp in by_frame.get(f, {}).items():
                if vid == blocker:
                    continue
                gap = gap_behind(vid, smp, b)
                if (
                    0 < gap <= qc.max_queue_distance_m
                    and same_side(smp, b)
                    and slow_for_window(vid, f)
                ):
                    queued.append((vid, gap))
            affected.update(v for v, _ in queued)
            max_q = max(max_q, len(queued))
            if queued:
                max_q_len_m = max(max_q_len_m, max(g for _, g in queued))
        if affected:
            queues.append(
                {
                    "blocking_vehicle_id": blocker,
                    "blocking_vehicle_type": stop["vehicle_type"],
                    "start_frame": stop["start_frame"],
                    "end_frame": stop["end_frame"],
                    "start_sec": stop["start_sec"],
                    "end_sec": stop["end_sec"],
                    "duration_s": stop["duration_s"],
                    "max_queue_length": max_q,
                    "max_queue_length_m": max_q_len_m,
                    "affected_vehicle_ids": sort_ids(affected),
                }
            )
    return queues


# ------------------------------------------------------------------ LOS
# Based on HCM / IRC mid-block LOS criteria (Prof. Tom V. Mathew, IITB), as used in
# traffic-impedance-analysis-main. Primary: operating speed. Secondary: V/C ratio.
# If both are available, the worse LOS wins.
_LOS_SPEED_THRESHOLDS = [(0, "F"), (15, "E"), (40, "D"), (50, "C"), (60, "B"), (70, "A")]
_LOS_VC_THRESHOLDS = [(0.60, "A"), (0.70, "B"), (0.80, "C"), (0.85, "D"), (0.90, "E"), (1.00, "F")]
_LOS_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}


def los_from_speed(speed_kmh: float) -> str:
    for min_spd, los in reversed(_LOS_SPEED_THRESHOLDS):
        if speed_kmh >= min_spd:
            return los
    return "F"


def los_from_vc(vc: float) -> str:
    for max_vc, los in _LOS_VC_THRESHOLDS:
        if vc <= max_vc:
            return los
    return "F"


def compute_los(
    avg_speed_kmh: float | None,
    total_pcu: float,
    duration_s: float,
    num_lanes: int | None,
    capacity_per_lane_pcu_hr: float,
) -> dict:
    los_speed = los_from_speed(avg_speed_kmh) if avg_speed_kmh is not None else None
    vc_ratio = los_vc = None
    if num_lanes and duration_s > 0 and capacity_per_lane_pcu_hr > 0:
        observed_pcu_hr = total_pcu * 3600.0 / duration_s
        vc_ratio = round(min(observed_pcu_hr / (num_lanes * capacity_per_lane_pcu_hr), 1.5), 3)
        los_vc = los_from_vc(min(vc_ratio, 1.0))
    candidates = [x for x in (los_speed, los_vc) if x]
    final = max(candidates, key=lambda x: _LOS_ORDER[x]) if candidates else None
    return {
        "los_level": final,
        "los_by_speed": los_speed,
        "los_by_vc": los_vc,
        "vc_ratio": vc_ratio,
    }


# ------------------------------------------------------------------ main entry
def analyse(
    histories: dict[str, list[Sample]],
    classes: dict[str, str],
    fps: float,
    frames_processed: int,
    cfg: PipelineConfig,
) -> dict:
    """Return vehicles, class summary, stop events, queue events and run metrics."""
    vehicles = []
    for vid in sort_ids(histories):
        hist = histories[vid]
        if not hist:
            continue
        vehicles.append(_vehicle_summary(vid, classes.get(vid, "unknown"), hist, cfg))

    stops = detect_stops(histories, classes, cfg)
    stop_frames_by_vehicle: dict[str, float] = defaultdict(float)
    for e in stops:
        stop_frames_by_vehicle[e["vehicle_id"]] += e["duration_s"]

    # Status: Parked = slow for most of its visible time AND stayed in one place.
    for v in vehicles:
        displacement = abs(v["end_position_m"] - v["start_position_m"])
        v["stopped_time_s"] = stop_frames_by_vehicle.get(v["vehicle_id"], 0.0)
        if v["vehicle_type"] in NON_VEHICLE_TYPES:
            v["status"] = "Pedestrian" if v["vehicle_type"] == "pedestrian" else "Unknown"
        elif v["is_low_sample"]:
            v["status"] = "Low sample"
        elif (
            v["median_raw"] is not None
            and v["median_raw"] < cfg.road.parked_speed_threshold_kmh
            and v["n_flow_samples"] >= cfg.stop_event.min_stop_duration_frames
            and displacement <= cfg.road.parked_max_displacement_m
        ):
            v["status"] = "Parked"
        elif v["stopped_time_s"] > 0:
            v["status"] = "Stopped"
        else:
            v["status"] = "Moving"
    status = {v["vehicle_id"]: v["status"] for v in vehicles}
    direction = {v["vehicle_id"]: v["direction"] for v in vehicles}

    queues = detect_queues(stops, histories, classes, status, direction, cfg)

    duration_s = frames_processed / fps if fps > 0 else 0.0
    metrics = _metrics(vehicles, stops, queues, fps, frames_processed, duration_s, cfg)
    return {
        "vehicles": vehicles,
        "class_summary": class_summary(vehicles),
        "stop_events": stops,
        "queue_events": queues,
        "metrics": metrics,
    }


def _vehicle_summary(vid: str, cname: str, hist: list[Sample], cfg: PipelineConfig) -> dict:
    """Same fields and formulas as the 'Per Vehicle Summary' sheet of 8+8+++_fixed.py."""
    first_seen, last_seen = hist[0].t, hist[-1].t
    all_speeds = sorted(h.speed for h in hist)
    moving = [v for v in all_speeds if v > 0]
    has_speed = bool(moving)
    if has_speed:
        mean_sp = sum(moving) / len(moving)
        std_sp = math.sqrt(sum((v - mean_sp) ** 2 for v in moving) / len(moving))
        pvals = {f"p{q}": percentile(moving, q) for q in PCTS}
        estimated = percentile(moving, 50)
        iqr = percentile(moving, 75) - percentile(moving, 25)
        coeff_var = (std_sp / mean_sp) if mean_sp > 0 else None
        min_sp, max_sp = moving[0], moving[-1]
    else:
        mean_sp = std_sp = estimated = iqr = coeff_var = min_sp = max_sp = None
        pvals = {f"p{q}": None for q in PCTS}
    return {
        "vehicle_id": vid,
        "vehicle_type": cname,
        "direction": track_direction(hist, cfg.speed.direction_threshold_m),
        "first_seen_sec": first_seen,
        "last_seen_sec": last_seen,
        "track_mid_time": (first_seen + last_seen) / 2.0,
        "track_duration_s": last_seen - first_seen,
        "n_flow_samples": len(hist),
        "n_moving_samples": len(moving),
        "is_low_sample": len(hist) < cfg.speed.low_sample_min,
        "has_speed": has_speed,
        "estimated_speed": estimated,
        "mean_speed": mean_sp,
        "median_raw": percentile(all_speeds, 50),
        "std_speed": std_sp,
        "iqr": iqr,
        "coeff_var": coeff_var,
        **pvals,
        "min_speed": min_sp,
        "max_speed": max_sp,
        "start_position_m": hist[0].s,
        "end_position_m": hist[-1].s,
        "min_position_m": min(h.s for h in hist),
        "max_position_m": max(h.s for h in hist),
        "pcu": PCU_FACTORS.get(cname, 1.0),
    }


def class_summary(vehicles: list[dict]) -> list[dict]:
    """Same as the 'Summary by Class' sheet: stats over per-vehicle estimated_speed."""
    counts: Counter = Counter()
    speeds: dict[str, list[float]] = defaultdict(list)
    for v in vehicles:
        counts[v["vehicle_type"]] += 1
        if v["has_speed"]:
            speeds[v["vehicle_type"]].append(v["estimated_speed"])
    rows = []
    for cname in sorted(counts):
        vals = sorted(speeds[cname])
        if vals:
            mean_v = sum(vals) / len(vals)
            std_v = math.sqrt(sum((x - mean_v) ** 2 for x in vals) / len(vals))
            rows.append(
                {
                    "vehicle_type": cname,
                    "count": counts[cname],
                    "n_with_speed": len(vals),
                    "mean_speed": mean_v,
                    "median_speed": percentile(vals, 50),
                    "std_speed": std_v,
                    "min_speed": vals[0],
                    "max_speed": vals[-1],
                }
            )
        else:
            rows.append(
                {
                    "vehicle_type": cname,
                    "count": counts[cname],
                    "n_with_speed": 0,
                    "mean_speed": None,
                    "median_speed": None,
                    "std_speed": None,
                    "min_speed": None,
                    "max_speed": None,
                }
            )
    return rows


def _metrics(vehicles, stops, queues, fps, frames, duration_s, cfg: PipelineConfig) -> dict:
    real = [v for v in vehicles if v["vehicle_type"] not in NON_VEHICLE_TYPES]
    counted = [v for v in real if not v["is_low_sample"]]
    status_counts = Counter(v["status"] for v in vehicles)
    # Operating speed: vehicles that actually travelled through (not parked, not fragments).
    flowing = [v for v in counted if v["status"] in ("Moving", "Stopped") and v["has_speed"]]
    speeds = sorted(v["estimated_speed"] for v in flowing)
    avg_speed = sum(speeds) / len(speeds) if speeds else None

    # Flow = vehicles crossing a line across the road (a "screen line"), as in a
    # manual classified count. Counting every vehicle seen would include vehicles
    # already in view at the start or leaving at the end, and inflate flow on short clips.
    flow_line = cfg.road.flow_line_m
    if flow_line is None and real:
        lo = min(v["min_position_m"] for v in real)
        hi = max(v["max_position_m"] for v in real)
        flow_line = (lo + hi) / 2.0
    crossing = [
        v
        for v in counted
        if v["status"] in ("Moving", "Stopped")
        and flow_line is not None
        and v["min_position_m"] <= flow_line <= v["max_position_m"]
    ]
    total_pcu = sum(v["pcu"] for v in crossing)
    n_flow = len(crossing)
    los = compute_los(
        avg_speed, total_pcu, duration_s, cfg.road.num_lanes, cfg.road.capacity_per_lane_pcu_hr
    )
    return {
        "fps": fps,
        "frames_processed": frames,
        "duration_s": duration_s,
        "tracks_total": len(vehicles),
        "vehicles_total": len(real),
        "vehicles_counted": len(counted),
        "pedestrians": status_counts.get("Pedestrian", 0),
        "parked": status_counts.get("Parked", 0),
        "stopped": status_counts.get("Stopped", 0),
        "moving": status_counts.get("Moving", 0),
        "low_sample": status_counts.get("Low sample", 0),
        "avg_speed_kmh": avg_speed,
        "median_speed_kmh": percentile(speeds, 50),
        "p85_speed_kmh": percentile(speeds, 85),
        "max_speed_kmh": speeds[-1] if speeds else None,
        "flow_line_m": flow_line,
        "vehicles_crossing": n_flow,
        "flow_veh_hr": n_flow * 3600.0 / duration_s if duration_s > 0 else None,
        "flow_pcu_hr": total_pcu * 3600.0 / duration_s if duration_s > 0 else None,
        "total_pcu": total_pcu,
        **los,
        "stop_events": len(stops),
        "queue_events": len(queues),
        "max_queue_length": max((q["max_queue_length"] for q in queues), default=0),
        "total_stop_time_s": sum(e["duration_s"] for e in stops),
        "going": sum(1 for v in counted if v["direction"] == "going"),
        "coming": sum(1 for v in counted if v["direction"] == "coming"),
    }
