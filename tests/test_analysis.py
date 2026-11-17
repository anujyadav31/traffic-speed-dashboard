import pytest

from app.pipeline.analysis import (
    Sample,
    analyse,
    compute_los,
    detect_stops,
    los_from_speed,
    percentile,
    track_direction,
)
from app.pipeline.config import PipelineConfig

FPS = 15.0


def track(start_frame: int, positions: list[float], speeds: list[float], lat=-2.0):
    return [
        Sample(start_frame + i, (start_frame + i) / FPS, s, lat, v)
        for i, (s, v) in enumerate(zip(positions, speeds, strict=True))
    ]


def moving_track(start_frame, s0, n, kmh, lat=-2.0):
    step = kmh / 3.6 / FPS
    return track(start_frame, [s0 + i * step for i in range(n)], [kmh] * n, lat)


def test_percentile_matches_linear_interpolation():
    vals = [1.0, 2.0, 3.0, 4.0]
    assert percentile(vals, 50) == 2.5
    assert percentile(vals, 0) == 1.0
    assert percentile(vals, 100) == 4.0
    assert percentile([], 50) is None


def test_direction():
    assert track_direction(moving_track(0, 0, 30, 20), 1.0) == "going"
    assert track_direction(track(0, [10, 9, 5], [5, 5, 5]), 1.0) == "coming"
    assert track_direction(track(0, [10, 10.2], [0, 0]), 1.0) == "stationary"


def test_stop_needs_min_duration_and_resume_hysteresis():
    cfg = PipelineConfig()
    # 20 moving, 60 stopped, 20 moving frames -> one stop of ~60 frames
    speeds = [20.0] * 20 + [0.0] * 60 + [20.0] * 20
    hist = track(0, [float(i) for i in range(100)], speeds)
    stops = detect_stops({"1": hist}, {"1": "car"}, cfg)
    assert len(stops) == 1
    assert stops[0]["start_frame"] == 20
    # resume is confirmed after resume_hysteresis_frames (10) moving samples
    assert stops[0]["end_frame"] == 80 + cfg.stop_event.resume_hysteresis_frames - 1

    short = track(0, [0.0] * 30, [0.0] * 30)  # 30 frames < 45 minimum
    assert detect_stops({"2": short}, {"2": "car"}, cfg) == []


def test_pedestrians_never_create_stops():
    hist = track(0, [0.0] * 100, [0.0] * 100)
    assert detect_stops({"p": hist}, {"p": "pedestrian"}, PipelineConfig()) == []


def test_queue_behind_stopped_vehicle_same_direction_and_side():
    cfg = PipelineConfig()
    n = 200
    # Blocker: drives, then stops at s=30 from frame 20 to the end.
    blocker = moving_track(0, 29.0, 20, 18) + track(20, [30.0] * (n - 20), [0.0] * (n - 20))
    # Queued car: approaches from behind, crawls at 1 km/h 10 m behind the blocker.
    queued = moving_track(0, 0.0, 40, 30) + track(
        40, [20 + i * 0.01 for i in range(n - 40)], [1.0] * (n - 40)
    )
    # Same situation but on the other side of the road -> not queued.
    other_side = moving_track(0, 0.0, 40, 30, lat=+3.0) + track(
        40, [20 + i * 0.01 for i in range(n - 40)], [1.0] * (n - 40), lat=+3.0
    )
    # Vehicle ahead of the blocker -> not queued.
    ahead = moving_track(0, 40.0, n, 1)
    histories = {"b": blocker, "q": queued, "o": other_side, "a": ahead}
    classes = {k: "car" for k in histories}
    result = analyse(histories, classes, FPS, n, cfg)
    stop_ids = {e["vehicle_id"] for e in result["stop_events"]}
    assert "b" in stop_ids
    q = [e for e in result["queue_events"] if e["blocking_vehicle_id"] == "b"]
    assert len(q) == 1
    assert q[0]["affected_vehicle_ids"] == ["q"]
    assert q[0]["max_queue_length"] == 1


def test_status_parked_moving_low_sample():
    cfg = PipelineConfig()
    histories = {
        "parked": track(0, [5.0] * 100, [0.0] * 100),
        "moving": moving_track(0, 0.0, 60, 25),
        "frag": moving_track(0, 0.0, 5, 25),
        "ped": moving_track(0, 0.0, 60, 4),
    }
    classes = {"parked": "car", "moving": "bus", "frag": "car", "ped": "pedestrian"}
    result = analyse(histories, classes, FPS, 100, cfg)
    status = {v["vehicle_id"]: v["status"] for v in result["vehicles"]}
    assert status == {
        "parked": "Parked",
        "moving": "Moving",
        "frag": "Low sample",
        "ped": "Pedestrian",
    }
    m = result["metrics"]
    assert m["vehicles_counted"] == 2  # parked + moving (fragment and pedestrian excluded)
    assert m["avg_speed_kmh"] == pytest.approx(25.0)
    assert m["total_pcu"] == 3.0  # only the moving bus counts towards flow


def test_flow_counts_only_vehicles_crossing_the_count_line():
    cfg = PipelineConfig()
    histories = {
        "through": moving_track(0, 0.0, 100, 20),  # 0 -> ~37 m, crosses the middle
        "edge": moving_track(0, 38.0, 30, 10),  # stays near the far end
        "far": moving_track(0, 60.0, 30, 10),  # sets the observed range to ~0-61 m
    }
    classes = {k: "car" for k in histories}
    m = analyse(histories, classes, FPS, 100, cfg)["metrics"]
    assert m["flow_line_m"] == pytest.approx((0 + histories["far"][-1].s) / 2)
    assert m["vehicles_crossing"] == 1
    assert m["flow_veh_hr"] == pytest.approx(1 * 3600 / (100 / FPS))


def test_los():
    assert los_from_speed(75) == "A"
    assert los_from_speed(55) == "C"
    assert los_from_speed(10) == "F"
    # 900 PCU/h on one 1000 PCU/h lane = V/C 0.9 -> E; speed 65 -> B; worse wins.
    res = compute_los(65, total_pcu=225, duration_s=900, num_lanes=1, capacity_per_lane_pcu_hr=1000)
    assert res["vc_ratio"] == pytest.approx(0.9)
    assert (res["los_by_speed"], res["los_by_vc"], res["los_level"]) == ("B", "E", "E")
    assert compute_los(65, 10, 60, None, 1000)["vc_ratio"] is None
