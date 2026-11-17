"""File outputs (CSV / Excel / JSON / Parquet) and import of existing Excel results."""

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook

from app.pipeline.analysis import PCTS, Sample, sort_ids

# Raw detection columns appended to per-frame outputs (empty for imported runs).
RAW_COLUMNS = [
    "x_px",
    "y_px",
    "bbox_x1_px",
    "bbox_y1_px",
    "bbox_x2_px",
    "bbox_y2_px",
    "confidence",
    "class_raw",
]


def raw_values(smp: Sample) -> list:
    """Values for RAW_COLUMNS, in the same order."""
    x1, y1, x2, y2 = smp.bbox if smp.bbox else (None, None, None, None)
    return [smp.x_px, smp.y_px, x1, y1, x2, y2, smp.confidence, smp.class_raw]


def flatten(d: dict, prefix: str = "") -> list[tuple[str, object]]:
    """{"a": {"b": 1}} -> [("a.b", 1)]; lists become JSON text."""
    out = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out += flatten(v, key + ".")
        elif isinstance(v, list | tuple):
            out.append((key, json.dumps(_json_safe(list(v)))))
        else:
            out.append((key, v))
    return out


def write_wide_csv(path: Path, histories: dict[str, list[Sample]], vehicles: list[dict]) -> None:
    """Same layout as speed_predictions_*.csv from 8+8+++_fixed.py:
    ID, class, direction, then one column per timestamp."""
    info = {v["vehicle_id"]: v for v in vehicles}
    all_ts = sorted({smp.t for hist in histories.values() for smp in hist})
    with open(path, "w") as f:
        f.write("ID,class,direction," + ",".join(f"{t:.3f}" for t in all_ts) + "\n")
        for vid in sort_ids(histories):
            v = info.get(vid, {})
            row = f"{vid},{v.get('vehicle_type', 'unknown')},{v.get('direction', '')}"
            speed_map = {smp.t: smp.speed for smp in histories[vid]}
            for ts in all_ts:
                sp = speed_map.get(ts)
                row += f",{sp:.2f}" if sp is not None else ","
            f.write(row + "\n")


def write_excel(
    path: Path, histories: dict[str, list[Sample]], analysis: dict, run_info: dict | None = None
) -> None:
    """The three sheets of 8+8+++_fixed.py plus events, run metrics and run info.

    'Per Frame Samples' keeps the original seven columns first, in the original order,
    so files stay readable by anything written for 8+8+++_fixed.py output.
    """
    vehicles = analysis["vehicles"]
    info = {v["vehicle_id"]: v for v in vehicles}
    wb = Workbook(write_only=True)

    ws = wb.create_sheet("Per Frame Samples")
    ws.append(
        [
            "vehicle_id",
            "vehicle_type",
            "direction",
            "frame_idx",
            "timestamp_sec",
            "arc_length_m",
            "speed_kmh",
            "lateral_m",
        ]
        + RAW_COLUMNS
    )
    for vid in sort_ids(histories):
        v = info.get(vid, {})
        for smp in histories[vid]:
            ws.append(
                [
                    vid,
                    v.get("vehicle_type"),
                    v.get("direction"),
                    smp.frame_idx,
                    smp.t,
                    smp.s,
                    smp.speed,
                    smp.l,
                ]
                + raw_values(smp)
            )

    ws = wb.create_sheet("Per Vehicle Summary")
    cols = (
        [
            "vehicle_id",
            "vehicle_type",
            "direction",
            "first_seen_sec",
            "last_seen_sec",
            "track_mid_time",
            "track_duration_s",
            "n_flow_samples",
            "n_moving_samples",
            "is_low_sample",
            "has_speed",
            "estimated_speed",
            "mean_speed",
            "median_raw",
            "std_speed",
            "iqr",
            "coeff_var",
        ]
        + [f"p{q}" for q in PCTS]
        + ["min_speed", "max_speed", "status", "stopped_time_s", "pcu"]
    )
    ws.append(cols)
    for v in vehicles:
        ws.append([v.get(c) for c in cols])

    ws = wb.create_sheet("Summary by Class")
    cols = [
        "vehicle_type",
        "count",
        "n_with_speed",
        "mean_speed",
        "median_speed",
        "std_speed",
        "min_speed",
        "max_speed",
    ]
    ws.append(cols)
    for row in analysis["class_summary"]:
        ws.append([row[c] for c in cols])

    ws = wb.create_sheet("Stop Events")
    cols = [
        "vehicle_id",
        "vehicle_type",
        "start_frame",
        "end_frame",
        "start_sec",
        "end_sec",
        "duration_s",
        "position_m",
        "lateral_m",
    ]
    ws.append(cols)
    for e in analysis["stop_events"]:
        ws.append([e[c] for c in cols])

    ws = wb.create_sheet("Queue Events")
    ws.append(
        [
            "blocking_vehicle_id",
            "blocking_vehicle_type",
            "start_sec",
            "end_sec",
            "duration_s",
            "max_queue_length",
            "max_queue_length_m",
            "affected_vehicle_ids",
        ]
    )
    for q in analysis["queue_events"]:
        ws.append(
            [
                q["blocking_vehicle_id"],
                q["blocking_vehicle_type"],
                q["start_sec"],
                q["end_sec"],
                q["duration_s"],
                q["max_queue_length"],
                q["max_queue_length_m"],
                ", ".join(q["affected_vehicle_ids"]),
            ]
        )

    ws = wb.create_sheet("Run Metrics")
    ws.append(["metric", "value"])
    for k, val in analysis["metrics"].items():
        ws.append([k, val])

    if run_info:
        ws = wb.create_sheet("Run Info")
        ws.append(["field", "value"])
        for k, val in flatten(run_info):
            ws.append([k, val if not isinstance(val, str) or len(val) < 32000 else val[:32000]])

    wb.save(path)


def trajectories_frame(
    histories: dict[str, list[Sample]],
    vehicles: list[dict],
    run_id: str,
    recorded_at: datetime | None = None,
) -> pd.DataFrame:
    """Long format: one row per vehicle per frame, units in the column names."""
    info = {v["vehicle_id"]: v for v in vehicles}
    rows = []
    for vid in sort_ids(histories):
        v = info.get(vid, {})
        for smp in histories[vid]:
            rows.append(
                [
                    run_id,
                    vid,
                    v.get("vehicle_type"),
                    v.get("direction"),
                    v.get("status"),
                    smp.frame_idx,
                    smp.t,
                    (recorded_at + timedelta(seconds=smp.t)).isoformat(timespec="milliseconds")
                    if recorded_at
                    else None,
                    smp.s,
                    smp.l,
                    smp.speed,
                ]
                + raw_values(smp)
            )
    return pd.DataFrame(
        rows,
        columns=[
            "run_id",
            "vehicle_id",
            "vehicle_type",
            "direction",
            "status",
            "frame_idx",
            "timestamp_s",
            "clock_time",
            "s_m",
            "l_m",
            "speed_kmh",
        ]
        + RAW_COLUMNS,
    )


def write_trajectories(output_dir: Path, df: pd.DataFrame) -> None:
    """trajectories.csv (for sharing) and trajectories.parquet (compact, fast to load)."""
    df.to_csv(output_dir / "trajectories.csv", index=False)
    df.to_parquet(output_dir / "trajectories.parquet", index=False)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(_json_safe(data), indent=2))


def _json_safe(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    return obj


def read_excel_samples(path: Path) -> dict:
    """Load the 'Per Frame Samples' sheet written by 8+8+++_fixed.py (or by this app).

    Returns histories, classes and the fps implied by frame_idx / timestamp_sec.
    """
    wb = load_workbook(path, read_only=True)
    if "Per Frame Samples" not in wb.sheetnames:
        raise ValueError(f"{path.name} has no 'Per Frame Samples' sheet")
    rows = wb["Per Frame Samples"].iter_rows(values_only=True)
    header = [str(h) for h in next(rows)]
    idx = {name: i for i, name in enumerate(header)}
    for needed in (
        "vehicle_id",
        "vehicle_type",
        "frame_idx",
        "timestamp_sec",
        "arc_length_m",
        "speed_kmh",
    ):
        if needed not in idx:
            raise ValueError(f"Column '{needed}' missing in 'Per Frame Samples'")

    def opt(r, col, cast):
        """Optional column; read-only openpyxl drops trailing empty cells, so rows can
        be shorter than the header."""
        i = idx.get(col)
        if i is None or i >= len(r) or r[i] is None:
            return None
        return cast(r[i])

    histories: dict[str, list[Sample]] = defaultdict(list)
    classes: dict[str, str] = {}
    fps_estimates = []
    for r in rows:
        if r[idx["vehicle_id"]] is None:
            continue
        vid = str(r[idx["vehicle_id"]])
        frame_idx = int(r[idx["frame_idx"]])
        t = float(r[idx["timestamp_sec"]])
        smp = Sample(
            frame_idx,
            t,
            float(r[idx["arc_length_m"]]),
            opt(r, "lateral_m", float),
            float(r[idx["speed_kmh"]]),
        )
        if opt(r, "x_px", float) is not None:  # Excel written by this app
            smp.x_px, smp.y_px = opt(r, "x_px", float), opt(r, "y_px", float)
            box = [opt(r, c, int) for c in RAW_COLUMNS[2:6]]
            smp.bbox = tuple(box) if None not in box else None
            smp.confidence, smp.class_raw = opt(r, "confidence", float), opt(r, "class_raw", str)
        histories[vid].append(smp)
        classes[vid] = str(r[idx["vehicle_type"]])
        if t > 1 and len(fps_estimates) < 2000:
            fps_estimates.append(frame_idx / t)
    wb.close()
    if not histories:
        raise ValueError("No samples found")
    for hist in histories.values():
        hist.sort(key=lambda s: s.frame_idx)
    fps_estimates.sort()
    fps = fps_estimates[len(fps_estimates) // 2] if fps_estimates else 15.0
    last_frame = max(s.frame_idx for h in histories.values() for s in h)
    return {
        "histories": dict(histories),
        "classes": classes,
        "fps": fps,
        "frames_processed": last_frame + 1,
    }
