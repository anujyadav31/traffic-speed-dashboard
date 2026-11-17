"""Edie's definitions in a 3-D shape (Saberi et al. 2014, eqs. 9-11) on hand-made trajectories."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
from edie import Box, edie, edie_over_time, segments, smooth_positions  # noqa: E402

FPS = 10


def vehicle(vid, s_start, speed_mps, seconds, l=0.0, t_start=0.0):
    n = int(seconds * FPS) + 1
    return pd.DataFrame(
        {
            "vehicle_id": vid,
            "t": [t_start + i / FPS for i in range(n)],
            "s": [s_start + speed_mps * i / FPS for i in range(n)],
            "l": l,
        }
    )


def test_single_vehicle_constant_speed():
    # 10 m/s for 10 s on a 100 m road observed for 10 s.
    seg = segments(vehicle("a", 0.0, 10.0, 10))
    r = edie(seg, Box(0, 100, -5, 5, 0, 10))
    assert r["V_km_h"] == pytest.approx(36.0, rel=1e-6)  # 10 m/s
    # d = 0.1 km, Lxy * dt = 0.1 km * (10/3600) h  ->  Q = 360 veh/h (one vehicle per 10 s)
    assert r["Q_veh_h"] == pytest.approx(360.0, rel=1e-6)
    # one vehicle present the whole time on 0.1 km -> K = 10 veh/km
    assert r["K_veh_km"] == pytest.approx(10.0, rel=1e-6)
    assert r["K_times_V"] == pytest.approx(r["Q_veh_h"])  # Q = K * V


def test_stopped_vehicle_adds_density_not_flow():
    data = pd.concat([vehicle("a", 0.0, 10.0, 10), vehicle("b", 50.0, 0.0, 10)])
    r = edie(segments(data), Box(0, 100, -5, 5, 0, 10))
    assert r["vehicles"] == 2
    assert r["Q_veh_h"] == pytest.approx(360.0, rel=1e-6)  # only the mover travels
    assert r["K_veh_km"] == pytest.approx(20.0, rel=1e-6)  # both occupy the road
    assert r["V_km_h"] == pytest.approx(18.0, rel=1e-6)  # space-mean speed halves


def test_lanes_and_sideways_limit():
    data = pd.concat([vehicle("a", 0.0, 10.0, 10, l=-2), vehicle("b", 0.0, 10.0, 10, l=3)])
    seg = segments(data)
    both = edie(seg, Box(0, 100, -5, 5, 0, 10), lanes=2)
    assert both["Q_veh_h"] == pytest.approx(360.0, rel=1e-6)  # 2 vehicles / 2 lanes
    one = edie(seg, Box(0, 100, -5, 0, 0, 10), lanes=1, use_lateral=True)
    assert one["vehicles"] == 1


def test_detection_gaps_are_not_counted():
    v = vehicle("a", 0.0, 10.0, 10)
    v = v[(v["t"] < 3) | (v["t"] > 5)]  # missing for 2 s
    r = edie(segments(v), Box(0, 100, -5, 5, 0, 10))
    assert r["time_h"] * 3600 == pytest.approx(7.8)  # seen 0-2.9 s and 5.1-10 s


def test_time_slices_and_smoothing():
    data = vehicle("a", 0.0, 10.0, 10)
    ts = edie_over_time(segments(data), Box(0, 100, -5, 5, 0, 10), 5)
    assert len(ts) == 2 and ts["V_km_h"].round(6).eq(36.0).all()
    # A stopped vehicle at 36 m whose position jumps to 38 m now and then (as #77).
    jitter = data.assign(s=[38.0 if i % 7 == 3 else 36.0 for i in range(len(data))])
    raw = edie(segments(jitter), Box(30, 40, -5, 5, 0, 10))["distance_km"]
    smoothed = edie(segments(smooth_positions(jitter, 5)), Box(30, 40, -5, 5, 0, 10))["distance_km"]
    assert raw > 0.04 and smoothed == 0  # back-and-forth jumps removed
